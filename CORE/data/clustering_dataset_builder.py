"""
clustering_dataset_builder.py

All-in-one dataset preparation tool for the unsupervised clustering pipeline.

Modes
-----
scan  (default)
    Find the highest map-id already present in the processed dataset, then scan
    UPWARD from that id to --max_id (required), downloading and processing any
    new Standard-mode beatmaps found along the way.

patch (--map_ids 123,456,789)
    Download and process only the specified beatmap IDs, then stop.

echosu (--fetch_echosu)
    After the dataset step (scan or patch), refresh the local echosu tag JSON
    by hitting the echosu API.  Can also be used alone with --skip_dataset.

Run from repo root (PowerShell):
    .\\env\\Scripts\\python.exe -m CORE.data.clustering_dataset_builder --max_id 5600000
    .\\env\\Scripts\\python.exe -m CORE.data.clustering_dataset_builder --map_ids 4153835,4200000 --max_id 0
    .\\env\\Scripts\\python.exe -m CORE.data.clustering_dataset_builder --skip_dataset --fetch_echosu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from CORE.paths import DEFAULT_DATASET_ROOT, DEFAULT_ECHOSU_JSON, load_env

try:
    import rosu_pp_py as rosu  # type: ignore
except Exception:
    rosu = None

# ---------------------------------------------------------------------------
# osu! constants
# ---------------------------------------------------------------------------
PLAYFIELD_W, PLAYFIELD_H = 512.0, 384.0
PLAYFIELD_CX = PLAYFIELD_W / 2.0
PLAYFIELD_CY = PLAYFIELD_H / 2.0
# Geometric normalisation constants (used to produce [0, 1] spatial features).
PLAYFIELD_DIAG     = math.hypot(PLAYFIELD_W, PLAYFIELD_H)    # ≈ 640.0 px  — max jump distance
PLAYFIELD_DIST_MAX = math.hypot(PLAYFIELD_CX, PLAYFIELD_CY)  # ≈ 320.0 px  — max dist from centre
PLAYFIELD_EDGE_MAX = PLAYFIELD_CY                             #   192.0 px  — tightest half-dimension
OSU_RAW_URL  = "https://osu.ppy.sh/osu/{id}"
OSU_BASE_URL = "https://osu.ppy.sh/api/v2"
BATCH_SIZE   = 50
USER_AGENT   = "echoCluster-osu-corpus/1.0 (+https://github.com/leggo15/Echosu)"

# ---------------------------------------------------------------------------
# Core network / timing utilities
# ---------------------------------------------------------------------------

def _mono() -> float:
    return time.monotonic()


def _check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and _mono() > float(deadline):
        raise TimeoutError("MAP_TIMEOUT")


def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


_OSU_TOKEN_CACHE: dict | None = None


def get_osu_token(sess: requests.Session, *, timeout: float = 20.0) -> str:
    """Retrieve & cache an OAuth2 client-credentials token for osu! public scope."""
    global _OSU_TOKEN_CACHE
    if _OSU_TOKEN_CACHE and time.time() < float(_OSU_TOKEN_CACHE.get("expires_at", 0)):
        return str(_OSU_TOKEN_CACHE["access_token"])
    cid  = os.getenv("OSU_CLIENT_ID")
    csec = os.getenv("OSU_CLIENT_SECRET")
    if not (cid and csec):
        raise RuntimeError("Missing OSU_CLIENT_ID / OSU_CLIENT_SECRET in ../.env")
    r = sess.post(
        "https://osu.ppy.sh/oauth/token",
        json={"client_id": cid, "client_secret": csec,
              "grant_type": "client_credentials", "scope": "public"},
        timeout=float(timeout),
    )
    r.raise_for_status()
    data = r.json()
    _OSU_TOKEN_CACHE = {
        "access_token": data["access_token"],
        "expires_at":   time.time() + float(data.get("expires_in", 3600)) - 60.0,
    }
    return str(_OSU_TOKEN_CACHE["access_token"])


def osu_get_v2(
    sess: requests.Session,
    endpoint: str,
    params=None,
    *,
    timeout: float = 30.0,
    max_retries: int = 5,
    sleep_base: float = 0.5,
) -> requests.Response:
    """GET osu! v2 API with token refresh and 429 backoff."""
    url = f"{OSU_BASE_URL}{endpoint}"
    for attempt in range(1, int(max_retries) + 1):
        tok = get_osu_token(sess)
        headers = {"Authorization": f"Bearer {tok}"}
        try:
            r = sess.get(url, params=params, headers=headers, timeout=float(timeout))
        except requests.exceptions.ReadTimeout:
            if attempt < int(max_retries):
                time.sleep(float(sleep_base) * attempt)
                continue
            raise
        if r.status_code == 401 and attempt < int(max_retries):
            global _OSU_TOKEN_CACHE
            _OSU_TOKEN_CACHE = None
            time.sleep(float(sleep_base) * attempt)
            continue
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            try:
                wait_s = float(ra) if ra is not None else 10.0
            except Exception:
                wait_s = 10.0
            wait_s = max(wait_s, float(sleep_base) * attempt * 2.0)
            time.sleep(wait_s)
            continue
        return r
    return r  # type: ignore[return-value]


def status_ranked_flag(status: object) -> float:
    try:
        s = str(status or "").strip().lower()
    except Exception:
        s = ""
    return 1.0 if s in {"ranked", "approved", "qualified"} else 0.0


def _is_valid_osu_file(text: str) -> bool:
    return text.lstrip("\ufeff").startswith("osu file format")


def download_osu_raw(
    sess: requests.Session,
    beatmap_id: int,
    dest: Path,
    *,
    timeout: float = 30.0,
    retries: int = 6,
    sleep_base: float = 0.6,
    deadline: Optional[float] = None,
) -> bool:
    """Download a raw .osu file with 429 backoff."""
    if dest.exists() and dest.stat().st_size > 0:
        try:
            if _is_valid_osu_file(dest.read_text(encoding="utf-8", errors="ignore")):
                return True
        except Exception:
            pass
        try:
            dest.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
    url = OSU_RAW_URL.format(id=int(beatmap_id))
    for attempt in range(1, int(retries) + 1):
        if deadline is not None and _mono() > float(deadline):
            return False
        try:
            req_timeout = (
                min(float(timeout), max(0.25, float(deadline) - _mono()))
                if deadline is not None else float(timeout)
            )
            r = sess.get(url, timeout=req_timeout)
        except requests.exceptions.ReadTimeout:
            time.sleep(float(sleep_base) * attempt)
            continue
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            try:
                wait_s = float(ra) if ra is not None else 10.0
            except Exception:
                wait_s = 10.0
            wait_s = max(wait_s, float(sleep_base) * attempt * 2.0)
            if deadline is not None:
                remaining = float(deadline) - _mono()
                if remaining <= 0:
                    return False
                time.sleep(min(wait_s, remaining))
            else:
                time.sleep(wait_s)
            continue
        if r.status_code == 404:
            return False
        if r.status_code == 200 and _is_valid_osu_file(r.text):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(r.text.lstrip("\ufeff"), encoding="utf-8")
            return True
        time.sleep(float(sleep_base) * attempt)
    return False


def ensure_dirs(root: Path) -> None:
    (root / "processed" / "maps_by_id").mkdir(parents=True, exist_ok=True)
    (root / "windows"   / "beat_aligned").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)


def last_logged_id(log_path: Path) -> int:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return 0
    last = ""
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if s:
                last = s
    if not last:
        return 0
    try:
        return int(last.split()[0])
    except Exception:
        return 0


def _write_metadata_jsonl(
    fmeta,
    *,
    scanned_id: int,
    reason: str,
    meta: Optional[Dict[str, object]] = None,
) -> None:
    obj: Dict[str, object] = {"scanned_id": int(scanned_id), "reason": str(reason)}
    if meta:
        obj.update(meta)
    fmeta.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fmeta.flush()


# ---------------------------------------------------------------------------
# .osu file parsing
# ---------------------------------------------------------------------------

_SECTION_HDR = re.compile(r"^\[(.+?)\]\s*$")


def parse_sections(text: str) -> Dict[str, List[str]]:
    cur, data = None, {}
    for ln in text.splitlines():
        s = ln.strip()
        m = _SECTION_HDR.match(s)
        if m:
            cur = m.group(1)
            continue
        if cur:
            data.setdefault(cur, []).append(ln.rstrip("\n"))
    for k in list(data.keys()):
        data[k] = [x.strip() for x in data[k] if x.strip()]
    return data


def parse_kv_section(lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in lines or []:
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def parse_diff_params(text: str) -> Dict[str, float]:
    params, in_diff = {}, False
    for ln in text.splitlines():
        s = ln.strip()
        if s == "[Difficulty]":
            in_diff = True
            continue
        if in_diff and s.startswith("["):
            break
        if in_diff and ":" in s:
            k, v = map(str.strip, s.split(":", 1))
            if k in {"ApproachRate", "CircleSize", "HPDrainRate", "OverallDifficulty",
                     "SliderMultiplier", "SliderTickRate"}:
                try:
                    params[k] = float(v)
                except Exception:
                    pass
    return params


def safe_int(x: str) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def resolve_local_timing(
    timing_lines: List[str],
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Return (uninherited BPM list [(t_ms, bpm)], inherited SV list [(t_ms, sv_mult)])."""
    uninherited, inherited = [], []
    for ln in timing_lines or []:
        tok = [t.strip() for t in ln.split(",")]
        if len(tok) < 2:
            continue
        t = safe_int(tok[0])
        try:
            bl = float(tok[1])
        except Exception:
            bl = None
        unin = 1
        if len(tok) >= 7:
            try:
                unin = int(float(tok[6]))
            except Exception:
                unin = 1
        if t is None or bl is None:
            continue
        if unin == 1 and bl > 0:
            uninherited.append((t, 60000.0 / bl))
        elif unin == 0 and bl < 0:
            inherited.append((t, -100.0 / bl))
    uninherited.sort(key=lambda x: x[0])
    inherited.sort(key=lambda x: x[0])
    return uninherited, inherited


def local_values_at(
    t_ms: int,
    uninherited: List[Tuple[int, float]],
    inherited:   List[Tuple[int, float]],
) -> Tuple[float, float, float]:
    bpm = uninherited[0][1] if uninherited else 120.0
    beat_ms = 60000.0 / bpm if bpm > 0 else 500.0
    for t, b in uninherited:
        if t_ms >= t:
            bpm = b
            beat_ms = 60000.0 / b if b > 0 else beat_ms
        else:
            break
    sv = 1.0
    for t, s in inherited:
        if t_ms >= t:
            sv = s
        else:
            break
    beat_ms = 60000.0 / bpm if (bpm and np.isfinite(bpm) and bpm > 0) else 500.0
    return float(bpm), float(sv), float(beat_ms)


def parse_hitobject(line: str) -> Optional[Dict[str, object]]:
    tok = line.split(",")
    if len(tok) < 4:
        return None
    try:
        x, y, t = float(tok[0]), float(tok[1]), int(float(tok[2]))
    except Exception:
        return None
    try:
        type_int = int(tok[3])
    except Exception:
        type_int = 0
    typ = "slider" if (type_int & 2) else ("spinner" if (type_int & 8) else "circle")

    slider_type = slider_repeats = slider_length = slider_path = None
    anchors_count = None
    slider_fa_x: Optional[float] = None
    slider_fa_y: Optional[float] = None
    slider_la_x: Optional[float] = None
    slider_la_y: Optional[float] = None

    if typ == "slider" and len(tok) > 5:
        slider_path = tok[5]
        slider_type = slider_path.split("|")[0] if slider_path else None
        if len(tok) > 6:
            try:
                slider_repeats = int(float(tok[6]))
            except Exception:
                pass
        if len(tok) > 7:
            try:
                slider_length = float(tok[7])
            except Exception:
                pass
        if slider_path and "|" in slider_path:
            anchor_strs = [p for p in slider_path.split("|")[1:] if ":" in p]
            anchors_count = max(0, len(anchor_strs))
            if anchor_strs:
                try:
                    fx, fy = anchor_strs[0].split(":", 1)
                    slider_fa_x, slider_fa_y = float(fx), float(fy)
                except Exception:
                    pass
                try:
                    lx, ly = anchor_strs[-1].split(":", 1)
                    slider_la_x, slider_la_y = float(lx), float(ly)
                except Exception:
                    pass

    return {
        "t_ms": int(t), "x_px": float(x), "y_px": float(y), "obj_type": typ,
        "slider_curve": slider_type, "slider_repeats": slider_repeats,
        "slider_len_px": slider_length, "slider_path": slider_path,
        "slider_anchors": anchors_count,
        "slider_fa_x": slider_fa_x, "slider_fa_y": slider_fa_y,
        "slider_la_x": slider_la_x, "slider_la_y": slider_la_y,
    }


def angle_deg(
    prev: Dict[str, object], cur: Dict[str, object], nxt: Dict[str, object]
) -> Optional[float]:
    try:
        v1 = (float(cur["x_px"]) - float(prev["x_px"]), float(cur["y_px"]) - float(prev["y_px"]))
        v2 = (float(nxt["x_px"]) - float(cur["x_px"]),  float(nxt["y_px"]) - float(cur["y_px"]))
    except Exception:
        return None
    a = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
    return float((a + 360.0) % 360.0)


def compute_rosu_strains_400ms(
    osu_path: Path,
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]], Optional[int]]:
    if rosu is None:
        return None, None, None
    try:
        bm = rosu.Beatmap(path=str(osu_path))
        s  = rosu.Difficulty().strains(bm)
        sl = int(s.section_length)
        aim = {int(i * sl): float(v) for i, v in enumerate(s.aim)}
        spd = {int(i * sl): float(v) for i, v in enumerate(s.speed)}
        return aim, spd, sl
    except Exception:
        return None, None, None


def strain_at_time(
    t_ms: int, series: Optional[Dict[int, float]], sl_ms: Optional[int]
) -> Optional[float]:
    if not series or not sl_ms:
        return None
    k = (int(t_ms) // int(sl_ms)) * int(sl_ms)
    return float(series.get(k)) if k in series else None


def windows_from_beats(
    t0_ms: int, t_end_ms: int, beat_ms: float, *, beats_per_window: int, overlap: float
):
    step_beats = max(1, int(beats_per_window * (1.0 - overlap)))
    win_len_ms = float(beats_per_window) * float(beat_ms)
    step_ms    = float(step_beats) * float(beat_ms)
    t = float(t0_ms)
    while t + 1 <= float(t_end_ms):
        yield int(t), int(min(t_end_ms, t + win_len_ms))
        t += step_ms


def safe_nanmean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_nanmax(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def _safe_entropy(arr: np.ndarray, n_bins: int = 8) -> float:
    """Shannon entropy (nats) of a discretized 1-D distribution."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    counts, _ = np.histogram(arr, bins=n_bins)
    total = float(counts.sum())
    if total == 0.0:
        return float("nan")
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(probs * np.log(probs)))


# ---------------------------------------------------------------------------
# Core feature extraction: timeseries + windows
# ---------------------------------------------------------------------------

def build_timeseries_and_windows(
    osu_text: str,
    *,
    use_rosu_strains: bool,
    window_beats: int,
    window_overlap: float,
    max_t_ms_sanity: int,
    deadline: Optional[float] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, object]]:
    sect = parse_sections(osu_text)
    _check_deadline(deadline)

    gen      = parse_kv_section(sect.get("General", []))
    mode_raw = gen.get("Mode", "0")
    try:
        mode_int = int(float(mode_raw))
    except Exception:
        mode_int = 0
    if mode_int != 0:
        return None, None, {"mode_int": mode_int, "reason": "NON_STD_MODE"}

    tp_lines = sect.get("TimingPoints", []) or []
    ho_lines = sect.get("HitObjects",   []) or []
    if not ho_lines:
        return None, None, {"mode_int": mode_int, "reason": "NO_HITOBJECTS"}

    diff_params = parse_diff_params(osu_text)
    uninherited, inherited = resolve_local_timing(tp_lines)

    ho: List[Dict[str, object]] = []
    for ln in ho_lines:
        if (len(ho) % 500) == 0:
            _check_deadline(deadline)
        p = parse_hitobject(ln)
        if not p:
            continue
        tms = p.get("t_ms")
        if not isinstance(tms, int):
            continue
        if not (0 <= int(tms) <= int(max_t_ms_sanity)):
            continue
        ho.append(p)
    ho.sort(key=lambda d: int(d["t_ms"]))
    if not ho:
        return None, None, {"mode_int": mode_int, "reason": "NO_VALID_HITOBJECTS"}

    # Map end time
    bpm_ref    = float(uninherited[0][1]) if uninherited else 120.0
    beat_ms_tail = float(min(2000.0, max(50.0, 60000.0 / bpm_ref if bpm_ref > 0 else 500.0)))
    map_end_ms = int(max(int(x["t_ms"]) for x in ho) + beat_ms_tail)

    # AR → approach-circle preempt window
    _ar_val = float(diff_params.get("ApproachRate", 5.0))
    if _ar_val <= 5.0:
        _preempt_ms = 1200.0 + 600.0 * (5.0 - _ar_val) / 5.0
    else:
        _preempt_ms = 1200.0 - 750.0 * (_ar_val - 5.0) / 5.0
    _preempt_ms = float(max(300.0, min(1800.0, _preempt_ms)))

    _ho_t_np = np.array([int(h["t_ms"]) for h in ho], dtype=np.int64)

    # Rolling dt ring for CV + new rhythm features
    _DT_CV_WIN = 8
    _dt_ring: List[float] = []

    # Extra rolling state for new features
    _dt_prev_prev: Optional[float] = None
    _dist_center_ring: List[float] = []
    _DIST_CENTER_WIN  = 8

    # Kinematic running state
    _kin_vel_x: float = float("nan")
    _kin_vel_y: float = float("nan")
    _kin_accel_x: float = float("nan")
    _kin_accel_y: float = float("nan")
    _kin_jerk_x: float = float("nan")
    _kin_jerk_y: float = float("nan")
    _kin_cum_absement_x: float = 0.0
    _kin_cum_absement_y: float = 0.0
    _kin_cum_absity_x:   float = 0.0
    _kin_cum_absity_y:   float = 0.0
    _kin_eff_x: Optional[float] = None
    _kin_eff_y: Optional[float] = None
    _kin_eff_t: Optional[float] = None

    rows = []
    for i, cur in enumerate(ho):
        if (i % 500) == 0:
            _check_deadline(deadline)
        prev = ho[i - 1] if i > 0 else None
        nxt  = ho[i + 1] if i + 1 < len(ho) else None

        bpm, sv, beat_ms = local_values_at(int(cur["t_ms"]), uninherited, inherited)
        if beat_ms and np.isfinite(beat_ms) and beat_ms > 0:
            beat_rel = (
                (int(cur["t_ms"]) - (uninherited[0][0] if uninherited else 0))
                % float(beat_ms)
            ) / float(beat_ms)
        else:
            beat_rel = float("nan")

        dt_prev   = (int(cur["t_ms"]) - int(prev["t_ms"])) if prev else None
        dist_prev = (
            math.hypot(
                float(cur["x_px"]) - float(prev["x_px"]),
                float(cur["y_px"]) - float(prev["y_px"]),
            )
            if prev else None
        )
        ang = angle_deg(prev, cur, nxt) if (prev and nxt) else None

        # ── Jump type ────────────────────────────────────────────────────────
        # -1=no context, 0=linear, 1=back-and-forth, 2=triangle, 3=star/other
        jump_type: int = -1
        if ang is not None:
            if ang < 30.0 or ang > 330.0:
                jump_type = 0
            elif 150.0 <= ang <= 210.0:
                jump_type = 1
            elif 50.0 <= ang <= 130.0:
                jump_type = 2
            else:
                jump_type = 3

        # Rolling angle variance (last 5 notes)
        if i >= 2:
            recent = []
            for j in range(max(1, i - 5), min(i + 1, len(ho) - 1)):
                a = angle_deg(ho[j - 1], ho[j], ho[j + 1])
                if a is not None:
                    recent.append(a)
            ang_var = float(np.var(recent)) if recent else None
        else:
            ang_var = None

        # Polygon squareness proxy (last 4 points)
        poly_sq = None
        if i >= 3:
            p0, p1, p2, p3 = ho[i - 3], ho[i - 2], ho[i - 1], ho[i]

            def _ang(u, v):
                du = math.hypot(*u); dv = math.hypot(*v)
                if du == 0 or dv == 0:
                    return None
                cosv = max(-1.0, min(1.0, (u[0]*v[0] + u[1]*v[1]) / (du * dv)))
                return math.degrees(math.acos(cosv))

            v1 = (float(p0["x_px"]) - float(p1["x_px"]), float(p0["y_px"]) - float(p1["y_px"]))
            v2 = (float(p2["x_px"]) - float(p1["x_px"]), float(p2["y_px"]) - float(p1["y_px"]))
            v3 = (float(p1["x_px"]) - float(p2["x_px"]), float(p1["y_px"]) - float(p2["y_px"]))
            v4 = (float(p3["x_px"]) - float(p2["x_px"]), float(p3["y_px"]) - float(p2["y_px"]))
            a1, a2 = _ang(v1, v2), _ang(v3, v4)
            if a1 is not None and a2 is not None:
                poly_sq = max(0.0, 1.0 - (abs(a1 - 90) + abs(a2 - 90)) / 180.0)

        # Slider features
        slider_vel     = None
        anchor_density = None
        slider_dur_ms: Optional[float] = None
        if str(cur.get("obj_type")) == "slider" and cur.get("slider_len_px") is not None:
            try:
                sm      = float(diff_params.get("SliderMultiplier", 1.4))
                repeats = max(1, int(cur.get("slider_repeats") or 1))
                sv_safe = max(1e-6, float(sv))
                _dur    = (float(cur["slider_len_px"]) / (100.0 * sm * sv_safe)) * float(beat_ms) * repeats
                slider_vel    = float(cur["slider_len_px"]) / max(1.0, float(_dur))
                slider_dur_ms = float(_dur)
            except Exception:
                slider_vel = slider_dur_ms = None
            try:
                if cur.get("slider_anchors") is not None:
                    sm      = float(diff_params.get("SliderMultiplier", 1.4))
                    repeats = max(1, int(cur.get("slider_repeats") or 1))
                    sv_safe = max(1e-6, float(sv))
                    _dur2   = (float(cur["slider_len_px"]) / (100.0 * sm * sv_safe)) * float(beat_ms) * repeats
                    anchor_density = float(cur["slider_anchors"]) / max(1.0, float(_dur2)) * 1000.0
            except Exception:
                anchor_density = None

        slider_vel_beats: Optional[float] = None
        if slider_vel is not None and beat_ms > 0 and np.isfinite(beat_ms):
            slider_vel_beats = float(slider_vel) * float(beat_ms)

        slider_dir_deg: Optional[float] = None
        if str(cur.get("obj_type")) == "slider":
            fa_x = cur.get("slider_fa_x"); fa_y = cur.get("slider_fa_y")
            if fa_x is not None and fa_y is not None:
                dx = float(fa_x) - float(cur["x_px"])
                dy = float(fa_y) - float(cur["y_px"])
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    slider_dir_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        slider_curvature_ratio: Optional[float] = None
        if str(cur.get("obj_type")) == "slider" and cur.get("slider_len_px") is not None:
            la_x = cur.get("slider_la_x"); la_y = cur.get("slider_la_y")
            if la_x is not None and la_y is not None:
                chord = math.hypot(float(la_x) - float(cur["x_px"]), float(la_y) - float(cur["y_px"]))
                arc   = max(1.0, float(cur["slider_len_px"]))
                slider_curvature_ratio = float(min(1.0, chord / arc))

        # Slider end position
        cur_x = float(cur["x_px"]); cur_y = float(cur["y_px"]); cur_t = float(int(cur["t_ms"]))
        slider_end_x_px: Optional[float] = None
        slider_end_y_px: Optional[float] = None
        if str(cur.get("obj_type")) == "slider":
            _reps = max(1, int(cur.get("slider_repeats") or 1))
            _la_x = cur.get("slider_la_x"); _la_y = cur.get("slider_la_y")
            if _reps % 2 == 1 and _la_x is not None and _la_y is not None:
                slider_end_x_px, slider_end_y_px = float(_la_x), float(_la_y)
            else:
                slider_end_x_px, slider_end_y_px = cur_x, cur_y

        # ── Velocity (slider-aware) ───────────────────────────────────────────
        if _kin_eff_x is not None and _kin_eff_t is not None:
            _kin_eff_dt = cur_t - _kin_eff_t
            _kin_eff_dx = cur_x - _kin_eff_x
            _kin_eff_dy = cur_y - _kin_eff_y
            if _kin_eff_dt > 0 and math.isfinite(_kin_eff_dx) and math.isfinite(_kin_eff_dy):
                new_vel_x  = _kin_eff_dx / _kin_eff_dt
                new_vel_y  = _kin_eff_dy / _kin_eff_dt
                new_speed  = math.hypot(new_vel_x, new_vel_y)
            else:
                new_vel_x = new_vel_y = new_speed = float("nan")
                _kin_eff_dt = float("nan")
        else:
            new_vel_x = new_vel_y = new_speed = float("nan")
            _kin_eff_dt = float("nan")

        _kin_dt_ok = math.isfinite(_kin_eff_dt) and _kin_eff_dt > 0

        # ── Acceleration ──────────────────────────────────────────────────────
        if _kin_dt_ok and math.isfinite(_kin_vel_x) and math.isfinite(new_vel_x):
            new_accel_x   = (new_vel_x - _kin_vel_x) / _kin_eff_dt
            new_accel_y   = (new_vel_y - _kin_vel_y) / _kin_eff_dt
            new_accel_mag = math.hypot(new_accel_x, new_accel_y)
        else:
            new_accel_x = new_accel_y = new_accel_mag = float("nan")

        # ── Jerk ──────────────────────────────────────────────────────────────
        if _kin_dt_ok and math.isfinite(_kin_accel_x) and math.isfinite(new_accel_x):
            new_jerk_x   = (new_accel_x - _kin_accel_x) / _kin_eff_dt
            new_jerk_y   = (new_accel_y - _kin_accel_y) / _kin_eff_dt
            new_jerk_mag = math.hypot(new_jerk_x, new_jerk_y)
        else:
            new_jerk_x = new_jerk_y = new_jerk_mag = float("nan")

        # ── Jounce (snap) ─────────────────────────────────────────────────────
        if _kin_dt_ok and math.isfinite(_kin_jerk_x) and math.isfinite(new_jerk_x):
            new_jounce_x   = (new_jerk_x - _kin_jerk_x) / _kin_eff_dt
            new_jounce_y   = (new_jerk_y - _kin_jerk_y) / _kin_eff_dt
            new_jounce_mag = math.hypot(new_jounce_x, new_jounce_y)
        else:
            new_jounce_x = new_jounce_y = new_jounce_mag = float("nan")

        # ── Bearing ───────────────────────────────────────────────────────────
        bearing_deg: float = float("nan")
        if math.isfinite(new_vel_x) and math.isfinite(new_vel_y):
            bearing_deg = (math.degrees(math.atan2(new_vel_y, new_vel_x)) + 360.0) % 360.0

        # ── Distance from playfield centre ────────────────────────────────────
        dist_center_px = math.hypot(cur_x - PLAYFIELD_CX, cur_y - PLAYFIELD_CY)

        # ── Spatial pattern features ──────────────────────────────────────────
        quadrant: int = (1 if cur_x >= PLAYFIELD_CX else 0) + (2 if cur_y >= PLAYFIELD_CY else 0)
        # Edge proximity: normalised [0, 1] — 0 = on the boundary, 1 = at dead centre.
        edge_proximity: float = (
            min(cur_x, PLAYFIELD_W - cur_x, max(0.0, cur_y), PLAYFIELD_H - cur_y)
            / PLAYFIELD_EDGE_MAX
        )
        # dist_center normalised to [0, 1] — 0 = centre, 1 = corner.
        dist_center_norm: float = dist_center_px / PLAYFIELD_DIST_MAX
        _dist_center_ring.append(dist_center_px)
        if len(_dist_center_ring) > _DIST_CENTER_WIN:
            _dist_center_ring.pop(0)
        dist_center_var_rolling: float = (
            float(np.var(_dist_center_ring)) if len(_dist_center_ring) >= 3 else float("nan")
        )

        # ── Visual density ────────────────────────────────────────────────────
        _lo_v = int(np.searchsorted(_ho_t_np, int(cur["t_ms"]) - int(_preempt_ms), side="left"))
        _hi_v = int(np.searchsorted(_ho_t_np, int(cur["t_ms"]),                    side="right"))
        vis_density = _hi_v - _lo_v

        # ── Rolling dt CV ─────────────────────────────────────────────────────
        dt_cv_rolling: float = float("nan")
        if dt_prev is not None and dt_prev > 0:
            _dt_ring.append(float(dt_prev))
            if len(_dt_ring) > _DT_CV_WIN:
                _dt_ring.pop(0)
        if len(_dt_ring) >= 3:
            _dt_a    = np.array(_dt_ring, dtype=np.float64)
            _dt_mean = float(np.mean(_dt_a))
            if _dt_mean > 0:
                dt_cv_rolling = float(np.std(_dt_a) / _dt_mean)

        # ── Temporal rhythm features ──────────────────────────────────────────
        dt_ratio_prev: float = float("nan")
        if dt_prev is not None and _dt_prev_prev is not None and _dt_prev_prev > 0:
            dt_ratio_prev = float(dt_prev) / _dt_prev_prev

        bpm_local_effective: float = float("nan")
        if dt_prev is not None and dt_prev > 0:
            bpm_local_effective = 60000.0 / float(dt_prev)

        is_stream: int = int(dt_prev is not None and dt_prev < 80)

        # is_burst: 1 when this note ends a burst (slow note after fast run)
        is_burst: int = 0
        if len(_dt_ring) >= 4:
            _n_fast = sum(1 for _d in _dt_ring[:-1] if _d < 100.0)
            if _dt_ring[-1] >= 150.0 and _n_fast >= 3:
                is_burst = 1

        # rhythm_complexity: entropy of recent inter-note intervals
        rhythm_complexity: float = float("nan")
        if len(_dt_ring) >= 3:
            rhythm_complexity = _safe_entropy(np.array(_dt_ring, dtype=np.float64), n_bins=6)

        # ── Absement / absity ─────────────────────────────────────────────────
        _abs_dt = float(dt_prev) if (dt_prev is not None and dt_prev > 0) else 0.0
        _kin_cum_absement_x += cur_x * _abs_dt
        _kin_cum_absement_y += cur_y * _abs_dt
        _kin_cum_absity_x   += _kin_cum_absement_x * _abs_dt
        _kin_cum_absity_y   += _kin_cum_absement_y * _abs_dt

        # ── Kinematic state update ────────────────────────────────────────────
        _kin_vel_x,   _kin_vel_y   = new_vel_x,   new_vel_y
        _kin_accel_x, _kin_accel_y = new_accel_x, new_accel_y
        _kin_jerk_x,  _kin_jerk_y  = new_jerk_x,  new_jerk_y
        if str(cur.get("obj_type")) == "slider" and slider_dur_ms is not None and math.isfinite(float(slider_dur_ms)):
            _kin_eff_x = slider_end_x_px if slider_end_x_px is not None else cur_x
            _kin_eff_y = slider_end_y_px if slider_end_y_px is not None else cur_y
            _kin_eff_t = cur_t + float(slider_dur_ms)
        else:
            _kin_eff_x, _kin_eff_y, _kin_eff_t = cur_x, cur_y, cur_t
        if dt_prev is not None:
            _dt_prev_prev = float(dt_prev)

        # ── Log-scale / normalised derivatives ────────────────────────────────
        # log1p transforms eliminate extreme right-skew that z-score cannot fix.
        log_dt_prev_ms: float = (
            math.log1p(float(dt_prev)) if (dt_prev is not None and dt_prev > 0) else float("nan")
        )
        log_speed:     float = math.log1p(new_speed)     if (math.isfinite(new_speed)     and new_speed     >= 0) else float("nan")
        log_accel_mag: float = math.log1p(new_accel_mag) if (math.isfinite(new_accel_mag) and new_accel_mag >= 0) else float("nan")
        log_jerk_mag:  float = math.log1p(new_jerk_mag)  if (math.isfinite(new_jerk_mag)  and new_jerk_mag  >= 0) else float("nan")
        # Distance to previous note normalised to [0, 1] via playfield diagonal.
        dist_prev_norm: float = (
            float(dist_prev) / PLAYFIELD_DIAG if dist_prev is not None else float("nan")
        )

        rows.append({
            "t_ms":      int(cur["t_ms"]),
            "obj_idx":   int(i),
            "x_px":      cur_x,
            "y_px":      cur_y,
            "x_norm":    cur_x / PLAYFIELD_W,
            "y_norm":    cur_y / PLAYFIELD_H,
            "obj_type":  str(cur["obj_type"]),
            "dt_prev_ms":  dt_prev,
            "beat_rel":    float(beat_rel),
            "dist_prev_px": dist_prev,
            "angle_deg":   ang,
            "angle_var_n": ang_var,
            "poly_sq":     poly_sq,
            "slider_len_px":         cur.get("slider_len_px"),
            "slider_vel":            slider_vel,
            "slider_dur_ms":         slider_dur_ms,
            "slider_vel_beats":      slider_vel_beats,
            "slider_dir_deg":        slider_dir_deg,
            "slider_curvature_ratio": slider_curvature_ratio,
            "slider_curve":          cur.get("slider_curve"),
            "slider_repeats":        cur.get("slider_repeats"),
            "slider_tick_count":     None,
            "anchor_density":        anchor_density,
            "aim_strain":            None,
            "speed_strain":          None,
            "local_sv":   float(sv),
            "local_bpm":  float(bpm),
            "AR": float(diff_params["ApproachRate"])      if "ApproachRate"      in diff_params else float("nan"),
            "CS": float(diff_params["CircleSize"])        if "CircleSize"        in diff_params else float("nan"),
            "OD": float(diff_params["OverallDifficulty"]) if "OverallDifficulty" in diff_params else float("nan"),
            "HP": float(diff_params["HPDrainRate"])       if "HPDrainRate"       in diff_params else float("nan"),
            # ── Kinematic ────────────────────────────────────────────────────
            "slider_end_x_px": slider_end_x_px,
            "slider_end_y_px": slider_end_y_px,
            "speed":       new_speed,
            "vel_x":       new_vel_x,
            "vel_y":       new_vel_y,
            "accel_x":     new_accel_x,
            "accel_y":     new_accel_y,
            "accel_mag":   new_accel_mag,
            "jerk_x":      new_jerk_x,
            "jerk_y":      new_jerk_y,
            "jerk_mag":    new_jerk_mag,
            "jounce_x":    new_jounce_x,
            "jounce_y":    new_jounce_y,
            "jounce_mag":  new_jounce_mag,
            "absement_x":  _kin_cum_absement_x,
            "absement_y":  _kin_cum_absement_y,
            "absity_x":    _kin_cum_absity_x,
            "absity_y":    _kin_cum_absity_y,
            # ── Contextual ───────────────────────────────────────────────────
            "bearing_deg":    bearing_deg,
            "dist_center_px": dist_center_px,
            "vis_density":    vis_density,
            "dt_cv_rolling":  dt_cv_rolling,
            # ── Temporal rhythm ───────────────────────────────────────────────
            "dt_ratio_prev":       dt_ratio_prev,
            "bpm_local_effective": bpm_local_effective,
            "is_stream":           is_stream,
            "is_burst":            is_burst,
            "rhythm_complexity":   rhythm_complexity,
            # ── Spatial pattern ───────────────────────────────────────────────
            "jump_type":               jump_type,
            "quadrant":                quadrant,
            "edge_proximity":          edge_proximity,          # [0, 1] — normalised to PLAYFIELD_EDGE_MAX
            "dist_center_norm":        dist_center_norm,        # [0, 1] — normalised to PLAYFIELD_DIST_MAX
            "dist_center_var_rolling": dist_center_var_rolling,
            # ── Log-scale / normalised ────────────────────────────────────────
            "dist_prev_norm":  dist_prev_norm,   # [0, 1] via PLAYFIELD_DIAG
            "log_dt_prev_ms":  log_dt_prev_ms,   # log1p(dt_prev_ms)
            "log_speed":       log_speed,         # log1p(speed px/ms)
            "log_accel_mag":   log_accel_mag,     # log1p(accel_mag)
            "log_jerk_mag":    log_jerk_mag,      # log1p(jerk_mag)
        })

    df_ts = pd.DataFrame(rows)

    num_cols = [
        "t_ms", "obj_idx", "x_px", "y_px", "x_norm", "y_norm",
        "dt_prev_ms", "beat_rel", "dist_prev_px", "angle_deg", "angle_var_n", "poly_sq",
        "slider_len_px", "slider_vel", "slider_dur_ms", "slider_vel_beats",
        "slider_dir_deg", "slider_curvature_ratio", "slider_repeats",
        "slider_tick_count", "anchor_density", "aim_strain", "speed_strain",
        "local_sv", "local_bpm", "AR", "CS", "OD", "HP",
        "slider_end_x_px", "slider_end_y_px",
        "speed", "vel_x", "vel_y",
        "accel_x", "accel_y", "accel_mag",
        "jerk_x", "jerk_y", "jerk_mag",
        "jounce_x", "jounce_y", "jounce_mag",
        "absement_x", "absement_y", "absity_x", "absity_y",
        "bearing_deg", "dist_center_px", "vis_density", "dt_cv_rolling",
        "dt_ratio_prev", "bpm_local_effective", "is_stream", "is_burst", "rhythm_complexity",
        "jump_type", "quadrant", "edge_proximity", "dist_center_norm", "dist_center_var_rolling",
        "dist_prev_norm", "log_dt_prev_ms", "log_speed", "log_accel_mag", "log_jerk_mag",
    ]
    for c in num_cols:
        if c in df_ts.columns:
            df_ts[c] = pd.to_numeric(df_ts[c], errors="coerce")

    # ── Windows (beat-aligned) ────────────────────────────────────────────────
    dfw = None
    if uninherited:
        bpm0     = float(uninherited[0][1])
        beat_ms0 = 60000.0 / bpm0 if bpm0 > 0 else None
        if beat_ms0 and np.isfinite(beat_ms0) and beat_ms0 > 0:
            rows_w: List[dict] = []
            _prev_win_speed_mean:    float = float("nan")
            _prev_win_n_obj:         int   = -1
            _prev_win_dominant_type: int   = -1
            t0 = int(df_ts["t_ms"].min())
            for ws, we in windows_from_beats(
                t0, map_end_ms, beat_ms0,
                beats_per_window=int(window_beats),
                overlap=float(window_overlap),
            ):
                if (len(rows_w) % 200) == 0:
                    _check_deadline(deadline)
                sl = df_ts[(df_ts["t_ms"] >= ws) & (df_ts["t_ms"] < we)]
                if sl.empty:
                    continue

                # Precompute aggregates
                _w_n_obj      = int(sl.shape[0])
                _w_dt_mean    = safe_nanmean(sl["dt_prev_ms"].to_numpy())
                _w_dist_mean  = safe_nanmean(sl["dist_prev_px"].to_numpy())
                _w_speed_arr  = sl["speed"].dropna().to_numpy(dtype=np.float64) if "speed" in sl.columns else np.array([], dtype=np.float64)
                _w_speed_mean = float(np.mean(_w_speed_arr)) if _w_speed_arr.size else float("nan")
                _w_speed_max  = float(np.max(_w_speed_arr))  if _w_speed_arr.size else float("nan")
                _w_accel_mean = safe_nanmean(sl["accel_mag"].to_numpy()) if "accel_mag" in sl.columns else float("nan")
                _w_jerk_mean  = safe_nanmean(sl["jerk_mag"].to_numpy())  if "jerk_mag"  in sl.columns else float("nan")

                # Circular variance of bearing
                _bearing_var = float("nan")
                if "bearing_deg" in sl.columns:
                    _bv = sl["bearing_deg"].dropna().to_numpy(dtype=np.float64)
                    if len(_bv) > 1:
                        _rad = np.deg2rad(_bv)
                        _r   = np.sqrt(np.mean(np.cos(_rad))**2 + np.mean(np.sin(_rad))**2)
                        _bearing_var = float(1.0 - _r)

                # Dominant pattern type
                _dom_type = 2  # mixed
                if math.isfinite(_w_dt_mean) and _w_dt_mean < 80:
                    _dom_type = 0  # stream
                elif math.isfinite(_w_dist_mean) and _w_dist_mean > 200:
                    _dom_type = 1  # jump

                # Window transition features
                speed_delta:    float = _w_speed_mean - _prev_win_speed_mean
                density_delta:  float = float(_w_n_obj - _prev_win_n_obj) if _prev_win_n_obj >= 0 else float("nan")
                pattern_switch: int   = (
                    int(_dom_type != _prev_win_dominant_type)
                    if _prev_win_dominant_type >= 0 else 0
                )
                intensity_slope: float = float("nan")
                if _w_speed_arr.size >= 3:
                    _xs = np.arange(_w_speed_arr.size, dtype=np.float64)
                    _fit_slope, _ = np.polyfit(_xs, _w_speed_arr, 1)
                    intensity_slope = float(_fit_slope)

                _prev_win_speed_mean    = _w_speed_mean
                _prev_win_n_obj         = _w_n_obj
                _prev_win_dominant_type = _dom_type

                rows_w.append({
                    "start_ms":           int(ws),
                    "end_ms":             int(we),
                    "n_obj":              _w_n_obj,
                    "dt_prev_ms_mean":    _w_dt_mean,
                    "dist_prev_px_mean":  _w_dist_mean,
                    "aim_strain_max":     safe_nanmax(sl["aim_strain"].to_numpy()),
                    "speed_strain_max":   safe_nanmax(sl["speed_strain"].to_numpy()),
                    "pct_slider":         float(np.mean((sl["obj_type"] == "slider").to_numpy())),
                    "angle_deg_mean":     safe_nanmean(sl["angle_deg"].to_numpy()),
                    "angle_var_mean":     safe_nanmean(sl["angle_var_n"].to_numpy()),
                    "poly_sq_mean":       safe_nanmean(sl["poly_sq"].to_numpy()),
                    "speed_mean":         _w_speed_mean,
                    "speed_max":          _w_speed_max,
                    "accel_mag_mean":     _w_accel_mean,
                    "jerk_mag_mean":      _w_jerk_mean,
                    "bearing_var":        _bearing_var,
                    "dist_center_mean":   safe_nanmean(sl["dist_center_px"].to_numpy())   if "dist_center_px"   in sl.columns else float("nan"),
                    "vis_density_mean":   safe_nanmean(sl["vis_density"].to_numpy())      if "vis_density"      in sl.columns else float("nan"),
                    "dt_cv_mean":         safe_nanmean(sl["dt_cv_rolling"].to_numpy())    if "dt_cv_rolling"    in sl.columns else float("nan"),
                    # Window transition features
                    "speed_delta":        speed_delta,
                    "density_delta":      density_delta,
                    "pattern_switch":     pattern_switch,
                    "intensity_slope":    intensity_slope,
                    # Log-scale window aggregates (match per-note log features; better for GRU)
                    "log_speed_mean":     math.log1p(_w_speed_mean)  if math.isfinite(_w_speed_mean)  and _w_speed_mean  >= 0 else float("nan"),
                    "log_speed_max":      math.log1p(_w_speed_max)   if math.isfinite(_w_speed_max)   and _w_speed_max   >= 0 else float("nan"),
                    "log_accel_mag_mean": math.log1p(_w_accel_mean)  if math.isfinite(_w_accel_mean)  and _w_accel_mean  >= 0 else float("nan"),
                    "log_jerk_mag_mean":  math.log1p(_w_jerk_mean)   if math.isfinite(_w_jerk_mean)   and _w_jerk_mean   >= 0 else float("nan"),
                    "log_dt_mean":        math.log1p(_w_dt_mean)     if math.isfinite(_w_dt_mean)     and _w_dt_mean     > 0  else float("nan"),
                    "dist_center_norm_mean": safe_nanmean(sl["dist_center_norm"].to_numpy()) if "dist_center_norm" in sl.columns else float("nan"),
                })
            if rows_w:
                dfw = pd.DataFrame(rows_w)

    # ── Map-level aggregate features ──────────────────────────────────────────
    _ts_speed   = pd.to_numeric(df_ts.get("speed",        pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    _ts_jerk    = pd.to_numeric(df_ts.get("jerk_mag",     pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    _ts_angle   = pd.to_numeric(df_ts.get("angle_deg",    pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    _ts_dt_raw  = pd.to_numeric(df_ts.get("dt_prev_ms",   pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
    _ts_dist    = pd.to_numeric(df_ts.get("dist_prev_px", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
    _ts_x       = pd.to_numeric(df_ts.get("x_px",         pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    _ts_y       = pd.to_numeric(df_ts.get("y_px",         pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64)

    _ts_dt_fin   = _ts_dt_raw[np.isfinite(_ts_dt_raw)]
    _ts_dist_fin = _ts_dist[np.isfinite(_ts_dist)]

    _speed_p95       = float(np.percentile(_ts_speed, 95)) if _ts_speed.size else float("nan")
    _jerk_p95        = float(np.percentile(_ts_jerk,  95)) if _ts_jerk.size  else float("nan")
    _angle_entropy   = _safe_entropy(_ts_angle,  n_bins=36)
    _stream_fraction = float(np.mean(_ts_dt_fin  < 80))   if _ts_dt_fin.size  > 0 else float("nan")
    _jump_fraction   = float(np.mean(_ts_dist_fin > 200)) if _ts_dist_fin.size > 0 else float("nan")
    _rhythm_entropy  = _safe_entropy(_ts_dt_fin,  n_bins=20)

    _spatial_entropy = float("nan")
    if _ts_x.size >= 4 and _ts_y.size == _ts_x.size:
        _xb = np.linspace(0.0, PLAYFIELD_W, 9)
        _yb = np.linspace(0.0, PLAYFIELD_H, 7)
        _h2d, _, _ = np.histogram2d(_ts_x, _ts_y, bins=[_xb, _yb])
        _h_total   = float(_h2d.sum())
        if _h_total > 0.0:
            _probs = _h2d[_h2d > 0.0] / _h_total
            _spatial_entropy = float(-np.sum(_probs * np.log(_probs)))

    _slider_complexity_mean = float("nan")
    _slider_mask = df_ts["obj_type"] == "slider"
    if bool(_slider_mask.any()) and "slider_curvature_ratio" in df_ts.columns:
        _sc = pd.to_numeric(df_ts.loc[_slider_mask, "slider_curvature_ratio"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if _sc.size:
            _slider_complexity_mean = float(np.mean(1.0 - _sc))

    _bpm_variance = 0.0
    if len(uninherited) >= 2:
        _bpms = np.array([b for _, b in uninherited], dtype=np.float64)
        _bpm_variance = float(np.var(_bpms))

    meta = {
        "mode_int":               int(mode_int),
        "map_end_time_ms":        int(map_end_ms),
        "count_circles":          int(np.sum(df_ts["obj_type"] == "circle")),
        "count_sliders":          int(np.sum(df_ts["obj_type"] == "slider")),
        "count_spinners":         int(np.sum(df_ts["obj_type"] == "spinner")),
        "note_count":             int(df_ts.shape[0]),
        "bpm0":                   float(uninherited[0][1]) if uninherited else float("nan"),
        "speed_p95":              _speed_p95,
        "jerk_p95":               _jerk_p95,
        "angle_entropy":          _angle_entropy,
        "stream_fraction":        _stream_fraction,
        "jump_fraction":          _jump_fraction,
        "rhythm_entropy":         _rhythm_entropy,
        "spatial_entropy":        _spatial_entropy,
        "slider_complexity_mean": _slider_complexity_mean,
        "bpm_variance":           _bpm_variance,
    }
    return df_ts, dfw, meta


def inject_strains_into_timeseries(df_ts: pd.DataFrame, osu_path: Path) -> pd.DataFrame:
    """Fill aim_strain / speed_strain columns if rosu is available."""
    if rosu is None or df_ts is None or df_ts.empty:
        return df_ts
    aim_s, spd_s, sl_ms = compute_rosu_strains_400ms(osu_path)
    if not aim_s or not spd_s or not sl_ms:
        return df_ts
    try:
        t   = pd.to_numeric(df_ts["t_ms"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        aim = [strain_at_time(int(tt), aim_s, sl_ms) for tt in t]
        spd = [strain_at_time(int(tt), spd_s, sl_ms) for tt in t]
        df_ts = df_ts.copy()
        df_ts["aim_strain"]   = pd.to_numeric(np.asarray(aim, dtype=np.float32), errors="coerce")
        df_ts["speed_strain"] = pd.to_numeric(np.asarray(spd, dtype=np.float32), errors="coerce")
        return df_ts
    except Exception:
        return df_ts


def write_meta(out_dir: Path, map_id: str, meta: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{map_id}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_timeseries(out_dir: Path, map_id: str, df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{map_id}_timeseries.parquet", index=False)


def write_windows(windows_dir: Path, map_id: str, dfw: pd.DataFrame, *, beats_per_window: int) -> None:
    windows_dir.mkdir(parents=True, exist_ok=True)
    dfw.to_parquet(windows_dir / f"{map_id}_w{int(beats_per_window)}b.parquet", index=False)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ECHOSU_BASE     = "https://www.echosu.com"
ECHOSU_API      = f"{ECHOSU_BASE}/api"
ECHOSU_BEATMAPS = f"{ECHOSU_API}/beatmaps/"
ECHOSU_TAG_APPS = f"{ECHOSU_API}/tag-applications/"


# ---------------------------------------------------------------------------
# Helpers — new-map scan
# ---------------------------------------------------------------------------

def find_latest_known_id(processed_root: Path) -> int:
    """Return the highest numeric map ID already present in processed/maps_by_id/."""
    max_id = 0
    if not processed_root.exists():
        return 0
    for d in processed_root.iterdir():
        if not d.is_dir():
            continue
        try:
            mid = int(d.name)
            if mid > max_id:
                max_id = mid
        except ValueError:
            continue
    return max_id


def _chunks(ids: List[int], n: int):
    for i in range(0, len(ids), n):
        yield ids[i : i + n]


# ---------------------------------------------------------------------------
# Update-mode helpers — log parsing and data-completeness checks
# ---------------------------------------------------------------------------



# Required meta keys that must always be present for a well-formed map record.
# Intentionally excluded from this set:
#   ranked_date  — absent/null for graveyard, WIP, pending, loved maps (not a defect)
#   max_combo    — occasionally missing from the API for older approved/loved maps
_REQUIRED_META_KEYS = frozenset({
    "map_id", "ar", "cs", "od", "hp", "bpm",
    "note_count", "count_circles", "status_ranked",
    "star_rating",   # 0.0 is valid; absent means map was built before star_rating was stored
})

# Timeseries columns that every complete parquet must contain.
# Any map whose parquet pre-dates a feature pass will be missing these and will
# be flagged for a full rebuild by --update_dataset.
_REQUIRED_TS_COLS = frozenset({
    "speed", "vel_x", "accel_mag", "absement_x",                        # kinematic
    "bearing_deg", "dist_center_px", "vis_density", "dt_cv_rolling",    # contextual
    "is_stream", "rhythm_complexity", "bpm_local_effective",             # temporal rhythm
    "edge_proximity", "dist_center_var_rolling",                         # spatial (edge_proximity is [0,1])
    "dist_prev_norm", "dist_center_norm",                                # normalised spatial
    "log_dt_prev_ms", "log_speed", "log_accel_mag", "log_jerk_mag",     # log-scale kinematic
})


def _enrich_kinematics_inplace(ts_path: Path) -> bool:
    """Add kinematic columns to an existing timeseries parquet with no network access.

    Reads the parquet, computes velocity / acceleration / jerk / jounce /
    absement / absity from the x_px, y_px, t_ms columns already present, then
    writes the enriched file back to the same path.

    Returns True on success, False on any error (caller should fall back to a
    full rebuild).
    """
    import numpy as _np
    import pandas as _pd

    try:
        df = _pd.read_parquet(ts_path)
    except Exception:
        return False

    if df.empty or not {"t_ms", "x_px", "y_px"}.issubset(df.columns):
        return False

    try:
        df = df.sort_values("t_ms").reset_index(drop=True)
        n = len(df)
        return _enrich_kinematics_inplace_inner(df, ts_path, n)
    except Exception:
        return False


def _enrich_kinematics_inplace_inner(df, ts_path, n):
    import math as _math
    import numpy as _np
    import pandas as _pd

    t_arr   = _pd.to_numeric(df["t_ms"], errors="coerce").to_numpy(dtype=float)
    x_arr   = _pd.to_numeric(df["x_px"], errors="coerce").to_numpy(dtype=float)
    y_arr   = _pd.to_numeric(df["y_px"], errors="coerce").to_numpy(dtype=float)
    obj_arr = df["obj_type"].to_numpy() if "obj_type" in df.columns else _np.full(n, "")
    dur_arr = (
        _pd.to_numeric(df["slider_dur_ms"], errors="coerce").to_numpy(dtype=float)
        if "slider_dur_ms" in df.columns
        else _np.full(n, float("nan"))
    )
    # Use pre-stored slider end positions when available (new-build parquets).
    has_end = "slider_end_x_px" in df.columns and df["slider_end_x_px"].notna().any()
    end_x_stored = (
        _pd.to_numeric(df["slider_end_x_px"], errors="coerce").to_numpy(dtype=float)
        if has_end else None
    )
    end_y_stored = (
        _pd.to_numeric(df["slider_end_y_px"], errors="coerce").to_numpy(dtype=float)
        if has_end else None
    )

    # Output arrays (float32 for derivatives, float64 for accumulating integrals).
    end_x_out = _np.full(n, _np.nan, dtype=_np.float32)
    end_y_out = _np.full(n, _np.nan, dtype=_np.float32)
    speed_out = _np.full(n, _np.nan, dtype=_np.float32)
    vx_out    = _np.full(n, _np.nan, dtype=_np.float32)
    vy_out    = _np.full(n, _np.nan, dtype=_np.float32)
    ax_out    = _np.full(n, _np.nan, dtype=_np.float32)
    ay_out    = _np.full(n, _np.nan, dtype=_np.float32)
    am_out    = _np.full(n, _np.nan, dtype=_np.float32)
    jx_out    = _np.full(n, _np.nan, dtype=_np.float32)
    jy_out    = _np.full(n, _np.nan, dtype=_np.float32)
    jm_out    = _np.full(n, _np.nan, dtype=_np.float32)
    ox_out    = _np.full(n, _np.nan, dtype=_np.float32)  # jounce x
    oy_out    = _np.full(n, _np.nan, dtype=_np.float32)  # jounce y
    om_out    = _np.full(n, _np.nan, dtype=_np.float32)  # jounce mag
    abs_x_out = _np.zeros(n, dtype=_np.float64)
    abs_y_out = _np.zeros(n, dtype=_np.float64)
    abi_x_out = _np.zeros(n, dtype=_np.float64)
    abi_y_out = _np.zeros(n, dtype=_np.float64)

    nan = float("nan")
    kin_vx = kin_vy = nan
    kin_ax = kin_ay = nan
    kin_jx = kin_jy = nan
    cum_ax = cum_ay = cum_ix = cum_iy = 0.0
    eff_x = eff_y = eff_t = None

    for i in range(n):
        cx, cy, ct = x_arr[i], y_arr[i], t_arr[i]
        is_slider = str(obj_arr[i]) == "slider"

        # Slider end position.
        sxe = sye = None
        if is_slider:
            if end_x_stored is not None and _math.isfinite(end_x_stored[i]):
                sxe, sye = float(end_x_stored[i]), float(end_y_stored[i])  # type: ignore[index]
            else:
                sxe, sye = cx, cy  # fallback: head position
            end_x_out[i], end_y_out[i] = sxe, sye

        # Velocity (slider-aware).
        nvx = nvy = nspd = nan
        edt = nan
        if eff_x is not None and _math.isfinite(cx) and _math.isfinite(cy) and _math.isfinite(ct):
            edt = ct - eff_t  # type: ignore[operator]
            if edt > 0:
                dx, dy = cx - eff_x, cy - eff_y  # type: ignore[operator]
                if _math.isfinite(dx) and _math.isfinite(dy):
                    nvx, nvy = dx / edt, dy / edt
                    nspd = _math.hypot(nvx, nvy)

        dt_ok = _math.isfinite(edt) and edt > 0

        # Acceleration.
        nax = nay = nam = nan
        if dt_ok and _math.isfinite(kin_vx) and _math.isfinite(nvx):
            nax = (nvx - kin_vx) / edt
            nay = (nvy - kin_vy) / edt
            nam = _math.hypot(nax, nay)

        # Jerk.
        njx = njy = njm = nan
        if dt_ok and _math.isfinite(kin_ax) and _math.isfinite(nax):
            njx = (nax - kin_ax) / edt
            njy = (nay - kin_ay) / edt
            njm = _math.hypot(njx, njy)

        # Jounce.
        nox = noy = nom = nan
        if dt_ok and _math.isfinite(kin_jx) and _math.isfinite(njx):
            nox = (njx - kin_jx) / edt
            noy = (njy - kin_jy) / edt
            nom = _math.hypot(nox, noy)

        # Absement / absity (head-to-head Δt).
        adt = float(t_arr[i] - t_arr[i - 1]) if i > 0 and _math.isfinite(t_arr[i]) and _math.isfinite(t_arr[i - 1]) and t_arr[i] > t_arr[i - 1] else 0.0
        cum_ax += cx * adt
        cum_ay += cy * adt
        cum_ix += cum_ax * adt
        cum_iy += cum_ay * adt

        # Store.
        speed_out[i] = nspd;  vx_out[i] = nvx;  vy_out[i] = nvy
        ax_out[i]  = nax;    ay_out[i] = nay;   am_out[i] = nam
        jx_out[i]  = njx;    jy_out[i] = njy;   jm_out[i] = njm
        ox_out[i]  = nox;    oy_out[i] = noy;   om_out[i] = nom
        abs_x_out[i] = cum_ax;  abs_y_out[i] = cum_ay
        abi_x_out[i] = cum_ix;  abi_y_out[i] = cum_iy

        # Advance state.
        kin_vx, kin_vy = nvx, nvy
        kin_ax, kin_ay = nax, nay
        kin_jx, kin_jy = njx, njy
        dur = float(dur_arr[i]) if _math.isfinite(dur_arr[i]) else nan
        if is_slider and _math.isfinite(dur) and dur > 0:
            eff_x = sxe if sxe is not None else cx
            eff_y = sye if sye is not None else cy
            eff_t = ct + dur
        else:
            eff_x, eff_y, eff_t = cx, cy, ct

    # Write columns back (only add/overwrite kinematic ones).
    df = df.copy()
    if not has_end:
        df["slider_end_x_px"] = end_x_out
        df["slider_end_y_px"] = end_y_out
    df["speed"]      = speed_out
    df["vel_x"]      = vx_out
    df["vel_y"]      = vy_out
    df["accel_x"]    = ax_out
    df["accel_y"]    = ay_out
    df["accel_mag"]  = am_out
    df["jerk_x"]     = jx_out
    df["jerk_y"]     = jy_out
    df["jerk_mag"]   = jm_out
    df["jounce_x"]   = ox_out
    df["jounce_y"]   = oy_out
    df["jounce_mag"] = om_out
    df["absement_x"] = abs_x_out
    df["absement_y"] = abs_y_out
    df["absity_x"]   = abi_x_out
    df["absity_y"]   = abi_y_out

    try:
        df.to_parquet(ts_path, index=False)
        return True
    except Exception:
        return False


def check_map_completeness(
    beatmap_id: int,
    processed_root: Path,
    windows_dir: Path,
    use_rosu_strains: bool,
    window_beats: int = 4,
) -> Optional[str]:
    """Validate all expected data for *beatmap_id*.

    Returns None when everything looks good, or a short reason string
    describing the first problem found.

    Smart-null rules — these are intentionally absent and are NOT flagged:
    • meta["ranked_date"] null/missing  → normal for non-ranked maps
    • timeseries aim_strain/speed_strain all-NaN → expected when strains not computed
    • timeseries slider_* columns NaN for circle/spinner rows → always the case
    • meta["max_combo"] null → occasionally absent from the API for old maps

    Rosu-strain rule (only when use_rosu_strains=True):
    • aim_strain must not be entirely NaN for maps that have circles, because
      inject_strains_into_timeseries fills every 400 ms bucket.  A map with
      count_circles == 0 (slider/spinner-only) may legitimately have all-NaN strains.
    """
    import pandas as pd  # local import — avoid mandatory top-level dep

    map_dir   = processed_root / str(beatmap_id)
    meta_path = map_dir / f"{beatmap_id}_meta.json"
    ts_path   = map_dir / f"{beatmap_id}_timeseries.parquet"
    win_path  = windows_dir / f"{beatmap_id}_w{window_beats}b.parquet"

    # ── Required files ───────────────────────────────────────────────────────
    if not meta_path.exists():
        return "missing_meta"
    if not ts_path.exists():
        return "missing_timeseries"
    if not win_path.exists():
        return "missing_windows"

    # ── Meta JSON ────────────────────────────────────────────────────────────
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return "corrupt_meta_json"
    if not isinstance(meta, dict):
        return "corrupt_meta_json"

    missing_keys = _REQUIRED_META_KEYS - meta.keys()
    if missing_keys:
        return f"missing_meta_keys:{','.join(sorted(missing_keys))}"

    # ── Timeseries parquet (cheap: read only one column) ────────────────────
    try:
        df_ts_check = pd.read_parquet(ts_path, columns=["t_ms"])
    except Exception:
        return "corrupt_timeseries"
    if df_ts_check.empty:
        return "empty_timeseries"

    # ── Timeseries column completeness ──────────────────────────────────────
    # Detect parquets built before a feature pass (e.g. kinematic columns).
    # Reading only the schema footer is nearly free compared to loading data.
    try:
        import pyarrow.parquet as _pq
        _schema = _pq.read_schema(str(ts_path))
        _ts_col_names = {_schema.field(i).name for i in range(_schema.num_fields)}
        _missing_cols = _REQUIRED_TS_COLS - _ts_col_names
        if _missing_cols:
            return f"missing_ts_cols:{','.join(sorted(_missing_cols))}"
    except Exception:
        pass  # If schema read fails the earlier parquet load already validated the file

    # ── Windows parquet (cheap: read only one column) ───────────────────────
    try:
        df_win_check = pd.read_parquet(win_path, columns=["n_obj"])
    except Exception:
        return "corrupt_windows"
    if df_win_check.empty:
        return "empty_windows"

    # ── Rosu strain completeness (only checked when --use_rosu_strains) ──────
    if use_rosu_strains:
        try:
            df_strain = pd.read_parquet(ts_path, columns=["aim_strain"])
        except Exception:
            # Column missing entirely — file pre-dates strain injection
            return "missing_strain_column"

        n_circles = int(meta.get("count_circles") or 0)
        if n_circles > 0 and bool(df_strain["aim_strain"].isna().all()):
            # Map has circles but aim_strain was never filled in
            return "missing_rosu_strains"

    return None  # all checks passed


# ---------------------------------------------------------------------------
# Core batch-processing logic (shared between scan and patch modes)
# ---------------------------------------------------------------------------

def _make_log_fn(log_path: Path, meta_path: Path, write_lock: threading.Lock):
    def _log(scanned_id: int, msg: str, meta: Optional[Dict] = None) -> None:
        with write_lock:
            with log_path.open("a", encoding="utf-8") as flog, \
                 meta_path.open("a", encoding="utf-8") as fmeta:
                flog.write(f"{int(scanned_id)} {msg}\n")
                _write_metadata_jsonl(
                    fmeta, scanned_id=int(scanned_id), reason=str(msg), meta=meta
                )
    return _log


def process_batch(
    batch: List[int],
    sess: requests.Session,
    processed_root: Path,
    windows_dir: Path,
    maps_dir: Path,
    log_fn,
    *,
    api_timeout: float,
    raw_timeout: float,
    raw_retries: int,
    sleep_api: float,
    sleep_raw: float,
    max_seconds_per_map: float,
    keep_osu: bool,
    use_rosu_strains: bool,
    window_beats: int,
    window_overlap: float,
    max_t_ms_sanity: int,
) -> None:
    """Fetch API metadata for *batch*, download .osu files, build timeseries + windows."""
    to_query = []
    for mid in batch:
        out_dir = processed_root / str(mid)
        ts_path = out_dir / f"{mid}_timeseries.parquet"
        if ts_path.exists():
            log_fn(mid, "EXISTS")
            continue
        to_query.append(int(mid))

    if not to_query:
        return

    beatmaps: Dict[int, dict] = {}
    params = [("ids[]", str(bid)) for bid in to_query[:BATCH_SIZE]]
    resp = osu_get_v2(sess, "/beatmaps", params=params, timeout=api_timeout)
    if resp.status_code == 200:
        for bm in (resp.json() or {}).get("beatmaps", []) or []:
            try:
                beatmaps[int(bm["id"])] = bm
            except Exception:
                continue
    else:
        log_fn(to_query[0], f"API_HTTP_{resp.status_code}")
        time.sleep(sleep_api)
        return
    time.sleep(sleep_api)

    for scanned_id in to_query:
        t0 = _mono()
        deadline = t0 + max_seconds_per_map if max_seconds_per_map > 0 else None
        bm = beatmaps.get(int(scanned_id))
        if bm is None:
            log_fn(scanned_id, "404")
            continue
        if str(bm.get("mode") or "").lower() != "osu":
            log_fn(scanned_id, "NON_STD_MODE", {"mode": bm.get("mode")})
            continue

        beatmap_id = int(bm.get("id"))
        out_dir  = processed_root / str(beatmap_id)
        ts_path  = out_dir / f"{beatmap_id}_timeseries.parquet"
        if ts_path.exists():
            log_fn(scanned_id, "EXISTS")
            continue

        osu_path = maps_dir / f"{beatmap_id}.osu"
        ok = download_osu_raw(
            sess, beatmap_id, osu_path,
            timeout=raw_timeout, retries=raw_retries, deadline=deadline,
        )
        if not ok:
            reason = "TIMEOUT_RAW" if (deadline is not None and _mono() > deadline) else "RAW_404"
            log_fn(scanned_id, reason)
            continue

        try:
            _check_deadline(deadline)
            text = osu_path.read_text(encoding="utf-8", errors="ignore")
            df_ts, dfw, meta_extra = build_timeseries_and_windows(
                text,
                use_rosu_strains=use_rosu_strains,
                window_beats=window_beats,
                window_overlap=window_overlap,
                max_t_ms_sanity=max_t_ms_sanity,
                deadline=deadline,
            )
            if df_ts is None or df_ts.empty:
                log_fn(scanned_id, str((meta_extra or {}).get("reason") or "SKIP"))
                continue

            if use_rosu_strains:
                _check_deadline(deadline)
                df_ts = inject_strains_into_timeseries(df_ts, osu_path)

            bmset      = bm.get("beatmapset") or {}
            status_raw = bm.get("status") or bm.get("ranked_status") or bmset.get("status")
            meta = {
                "map_id":          str(beatmap_id),
                "mode":            str(bm.get("mode") or ""),
                "version":         bm.get("version") or "",
                "artist":          bmset.get("artist") or "",
                "title":           bmset.get("title") or "",
                "creator":         bmset.get("creator") or "",
                "star_rating":     bm.get("difficulty_rating"),
                "difficulty_rating": bm.get("difficulty_rating"),
                "ar":              bm.get("ar"),
                "cs":              bm.get("cs"),
                "od":              bm.get("accuracy"),
                "hp":              bm.get("drain"),
                "bpm":             bm.get("bpm"),
                "length_total":    bm.get("total_length"),
                "length_drain":    bm.get("hit_length"),
                "count_circles":   bm.get("count_circles"),
                "count_sliders":   bm.get("count_sliders"),
                "count_spinners":  bm.get("count_spinners"),
                "note_count":      (
                    bm.get("count_circles", 0)
                    + bm.get("count_sliders", 0)
                    + bm.get("count_spinners", 0)
                ),
                "max_combo":       bm.get("max_combo"),
                "ranked_date":     (bmset.get("ranked_date") if isinstance(bmset, dict) else None),
                "status_ranked":   status_ranked_flag(status_raw),
                "map_end_time_ms": int(
                    (meta_extra or {}).get("map_end_time_ms", int(df_ts["t_ms"].max()))
                ),
                # ── Map-level computed features ────────────────────────────
                "speed_p95":              (meta_extra or {}).get("speed_p95"),
                "jerk_p95":               (meta_extra or {}).get("jerk_p95"),
                "angle_entropy":          (meta_extra or {}).get("angle_entropy"),
                "stream_fraction":        (meta_extra or {}).get("stream_fraction"),
                "jump_fraction":          (meta_extra or {}).get("jump_fraction"),
                "rhythm_entropy":         (meta_extra or {}).get("rhythm_entropy"),
                "spatial_entropy":        (meta_extra or {}).get("spatial_entropy"),
                "slider_complexity_mean": (meta_extra or {}).get("slider_complexity_mean"),
                "bpm_variance":           (meta_extra or {}).get("bpm_variance"),
            }
            write_meta(out_dir, str(beatmap_id), meta)
            write_timeseries(out_dir, str(beatmap_id), df_ts)
            if dfw is not None and not dfw.empty:
                write_windows(
                    windows_dir, str(beatmap_id), dfw,
                    beats_per_window=window_beats,
                )
            log_fn(scanned_id, "OK", {"beatmap_id": int(beatmap_id)})

        except TimeoutError:
            log_fn(scanned_id, "TIMEOUT")
        except Exception as e:
            log_fn(scanned_id, f"ERR:{type(e).__name__}")
        finally:
            if not keep_osu:
                try:
                    osu_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            time.sleep(sleep_raw)


# ---------------------------------------------------------------------------
# Scan mode  — ascending from latest known ID to --max_id
# ---------------------------------------------------------------------------

def run_scan(args, root: Path, sess: requests.Session, log_fn) -> None:
    processed_root = root / "processed" / "maps_by_id"
    windows_dir    = root / "windows" / "beat_aligned"
    maps_dir       = (root / "maps") if args.keep_osu else (root / "tmp_osu")
    maps_dir.mkdir(parents=True, exist_ok=True)

    max_id = int(args.max_id)

    if getattr(args, "resume", False):
        # Fast resume: read the last line of the log instead of scanning all dirs.
        log_path = root / "logs" / "build_log_asc.txt"
        last_log = last_logged_id(log_path)
        if last_log > 0:
            start_id = last_log + 1
            print(f"[builder/scan] --resume: last logged id={last_log:,} → resuming from {start_id:,}")
        else:
            # Log missing or empty — fall back to disk scan.
            latest   = find_latest_known_id(processed_root)
            start_id = max(1, latest + 1)
            print(f"[builder/scan] --resume: log empty, falling back to disk scan (latest id={latest:,})")
    else:
        latest   = find_latest_known_id(processed_root)
        start_id = max(1, latest + 1)
        print(f"[builder/scan] latest known id={latest:,}")

    if start_id > max_id:
        print(f"[builder/scan] start_id={start_id:,} already at or above --max_id={max_id:,}. Nothing to do.")
        return

    ids = list(range(start_id, max_id + 1))
    print(f"[builder/scan] scanning {start_id:,}–{max_id:,} ({len(ids):,} IDs in {len(ids)//BATCH_SIZE+1:,} batches)")

    kw = dict(
        api_timeout=float(args.api_timeout),
        raw_timeout=float(args.raw_timeout),
        raw_retries=int(args.raw_retries),
        sleep_api=float(args.sleep_api),
        sleep_raw=float(args.sleep_raw),
        max_seconds_per_map=float(args.max_seconds_per_map),
        keep_osu=bool(args.keep_osu),
        use_rosu_strains=bool(args.use_rosu_strains),
        window_beats=int(args.window_beats),
        window_overlap=float(args.window_overlap),
        max_t_ms_sanity=int(args.max_t_ms_sanity),
    )

    for batch in tqdm(list(_chunks(ids, BATCH_SIZE)), unit="batch", desc="scan ↑"):
        process_batch(
            batch, sess, processed_root, windows_dir, maps_dir, log_fn, **kw
        )


# ---------------------------------------------------------------------------
# Patch mode  — specific map IDs only
# ---------------------------------------------------------------------------

def run_patch(args, root: Path, sess: requests.Session, log_fn) -> None:
    processed_root = root / "processed" / "maps_by_id"
    windows_dir    = root / "windows" / "beat_aligned"
    maps_dir       = (root / "maps") if args.keep_osu else (root / "tmp_osu")
    maps_dir.mkdir(parents=True, exist_ok=True)

    raw_ids = str(args.map_ids).replace(" ", ",").split(",")
    ids = []
    for r in raw_ids:
        r = r.strip()
        if r:
            try:
                ids.append(int(r))
            except ValueError:
                print(f"[builder/patch] ignoring non-integer id '{r}'")

    if not ids:
        print("[builder/patch] no valid map IDs provided; nothing to do.")
        return

    print(f"[builder/patch] processing {len(ids)} specified map IDs: {ids[:10]}{'…' if len(ids) > 10 else ''}")

    kw = dict(
        api_timeout=float(args.api_timeout),
        raw_timeout=float(args.raw_timeout),
        raw_retries=int(args.raw_retries),
        sleep_api=float(args.sleep_api),
        sleep_raw=float(args.sleep_raw),
        max_seconds_per_map=float(args.max_seconds_per_map),
        keep_osu=bool(args.keep_osu),
        use_rosu_strains=bool(args.use_rosu_strains),
        window_beats=int(args.window_beats),
        window_overlap=float(args.window_overlap),
        max_t_ms_sanity=int(args.max_t_ms_sanity),
    )

    for batch in tqdm(list(_chunks(ids, BATCH_SIZE)), unit="batch", desc="patch"):
        process_batch(
            batch, sess, processed_root, windows_dir, maps_dir, log_fn, **kw
        )


# ---------------------------------------------------------------------------
# Update mode  — validate + repair every map in processed/maps_by_id/,
#                then optionally continue scanning beyond the highest known ID.
# ---------------------------------------------------------------------------

def run_update(args, root: Path, sess: requests.Session) -> None:
    """Validate + repair every map found on disk, then optionally scan forward.

    Flow
    ----
    1. Check for a saved repair queue (logs/update_repair_queue.json).
       If found (and --revalidate is not set), skip enumeration + validation
       and jump straight to repair using the saved list.
    2. Otherwise enumerate all sub-directories of processed/maps_by_id/ and
       validate each map via check_map_completeness().
         • missing_ts_cols  → enrich in-place (no network access).
         • anything else    → delete data and queue for full rebuild.
       After validation, save the repair queue so the process can be resumed
       if interrupted.
    3. Re-process queued maps in BATCH_SIZE batches.  process_batch() already
       skips maps whose timeseries parquet exists, so restarting is safe —
       only maps that haven't been rebuilt yet incur a download.
    4. If --max_id > highest known on-disk ID, continue ascending scan.
    """
    processed_root = root / "processed" / "maps_by_id"
    windows_dir    = root / "windows" / "beat_aligned"
    maps_dir       = (root / "maps") if args.keep_osu else (root / "tmp_osu")
    maps_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    queue_file = logs_dir / "update_repair_queue.json"

    kw = dict(
        api_timeout        = float(args.api_timeout),
        raw_timeout        = float(args.raw_timeout),
        raw_retries        = int(args.raw_retries),
        sleep_api          = float(args.sleep_api),
        sleep_raw          = float(args.sleep_raw),
        max_seconds_per_map= float(args.max_seconds_per_map),
        keep_osu           = bool(args.keep_osu),
        use_rosu_strains   = bool(args.use_rosu_strains),
        window_beats       = int(args.window_beats),
        window_overlap     = float(args.window_overlap),
        max_t_ms_sanity    = int(args.max_t_ms_sanity),
    )

    needs_repair: List[int] = []
    enriched_count = 0
    max_known_id = 0

    revalidate       = bool(getattr(args, "revalidate",        False))
    skip_validation  = bool(getattr(args, "skip_validation",   False))
    start_id         = int(getattr(args,  "start_id",          0))

    # ── Resume check ─────────────────────────────────────────────────────────
    # Use the saved queue if it exists unless the user explicitly wants a fresh
    # validation pass (--revalidate) or wants to bypass validation entirely
    # (--skip_validation, which will re-enumerate without validating).
    if queue_file.exists() and not revalidate and not skip_validation:
        try:
            saved = json.loads(queue_file.read_text(encoding="utf-8"))
            needs_repair   = [int(x) for x in saved.get("needs_repair", [])]
            enriched_count = int(saved.get("enriched_count", 0))
            max_known_id   = int(saved.get("max_known_id", 0))
            if start_id > 0:
                before = len(needs_repair)
                needs_repair = [bid for bid in needs_repair if bid >= start_id]
                print(
                    f"[update] Resuming from saved queue ({queue_file.name}): "
                    f"{len(needs_repair):,} maps to rebuild (filtered from "
                    f"{before:,} by --start_id={start_id:,}), "
                    f"max_known_id={max_known_id:,}"
                )
            else:
                print(
                    f"[update] Resuming from saved queue ({queue_file.name}): "
                    f"{len(needs_repair):,} maps to rebuild, "
                    f"{enriched_count:,} already enriched in-place, "
                    f"max_known_id={max_known_id:,}"
                )
            print("[update] Run with --revalidate to discard this queue and re-scan from scratch.")
        except Exception as exc:
            print(f"[update] Could not load queue file ({exc}); running fresh validation.")
            needs_repair = []
            max_known_id = 0

    # ── Steps 1 + 2: Enumerate + validate (skipped when resuming) ────────────
    if not needs_repair and max_known_id == 0:
        # Step 1 — enumerate maps currently on disk
        print("[update] Enumerating processed/maps_by_id …")
        all_map_ids: List[int] = []
        if processed_root.exists():
            for d in processed_root.iterdir():
                if d.is_dir():
                    try:
                        all_map_ids.append(int(d.name))
                    except ValueError:
                        continue
        disk_id_set = set(all_map_ids)

        # Step 1b — recover IDs that were previously built (appear as "OK" in
        # any build_log_*.txt) but have since been deleted from disk.
        # This handles the case where a prior --update_dataset run deleted maps
        # during its validation pass before being interrupted.
        ghost_ids: List[int] = []
        log_max_id = 0
        for log_path in sorted(logs_dir.glob("build_log_*.txt")):
            try:
                with log_path.open("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] == "OK":
                            try:
                                bid = int(parts[0])
                            except ValueError:
                                continue
                            if bid > log_max_id:
                                log_max_id = bid
                            if bid not in disk_id_set:
                                ghost_ids.append(bid)
            except Exception as exc:
                print(f"[update] Warning: could not read {log_path.name}: {exc}")

        if ghost_ids:
            ghost_ids.sort()
            print(
                f"[update] Found {len(ghost_ids):,} map IDs in build logs that are "
                f"missing from disk — queuing for re-download."
            )

        all_map_ids.sort()
        # max_known_id should reflect the broadest known ID range so Step 4
        # (forward scan) starts correctly above everything that's been seen.
        max_known_id = max(
            all_map_ids[-1] if all_map_ids else 0,
            log_max_id,
        )
        print(f"[update] Found {len(all_map_ids):,} map directories  |  max_id={max_known_id:,}")

        # Apply --start_id filter before any further work.
        if start_id > 0:
            all_map_ids = [bid for bid in all_map_ids if bid >= start_id]
            ghost_ids   = [bid for bid in ghost_ids   if bid >= start_id]
            print(
                f"[update] --start_id={start_id:,}: "
                f"{len(all_map_ids):,} on-disk + {len(ghost_ids):,} log-recovered maps in scope."
            )

        if skip_validation:
            # Step 2 skipped — queue every in-scope map and rely on
            # process_batch()'s EXISTS check to skip already-complete ones.
            # Map data is NOT deleted, so complete maps are never re-downloaded.
            needs_repair = sorted(set(all_map_ids) | set(ghost_ids))
            print(
                f"[update] --skip_validation: {len(needs_repair):,} maps queued for repair "
                f"(completeness unchecked; already-complete maps will be skipped by EXISTS check)."
            )
        else:
            # Step 2 — full completeness validation of on-disk maps.
            # Ghost IDs (deleted from disk) are added directly to needs_repair
            # — there is nothing to validate since no files exist for them.
            needs_repair.extend(ghost_ids)

            enrich_failed: List[int] = []
            if all_map_ids:
                print(f"[update] Checking completeness of {len(all_map_ids):,} maps …")
                for bid in tqdm(all_map_ids, desc="validate", unit="map"):
                    reason = check_map_completeness(
                        bid, processed_root, windows_dir,
                        args.use_rosu_strains, args.window_beats,
                    )
                    if reason is None:
                        continue

                    if reason.startswith("missing_ts_cols:"):
                        # Fast path: compute columns in-place, no network access.
                        ts_path = processed_root / str(bid) / f"{bid}_timeseries.parquet"
                        if _enrich_kinematics_inplace(ts_path):
                            enriched_count += 1
                            continue
                        enrich_failed.append(bid)

                    needs_repair.append(bid)

            needs_repair.sort()
            on_disk_failed = len(needs_repair) - len(ghost_ids)
            summary_parts = [
                f"{len(all_map_ids) - on_disk_failed - enriched_count:,} on-disk complete"
            ]
            if ghost_ids:
                summary_parts.append(f"{len(ghost_ids):,} recovered from logs")
            if enriched_count:
                summary_parts.append(f"{enriched_count:,} enriched in-place")
            if enrich_failed:
                summary_parts.append(f"{len(enrich_failed):,} enrich-failed → full rebuild")
            print(f"[update] {len(needs_repair):,} maps queued for rebuild  "
                  f"({', '.join(summary_parts)}).")

        # Save queue so the process can be resumed if interrupted during repair.
        try:
            queue_file.write_text(
                json.dumps({
                    "max_known_id":   max_known_id,
                    "enriched_count": enriched_count,
                    "needs_repair":   needs_repair,
                }),
                encoding="utf-8",
            )
            print(f"[update] Repair queue saved → {queue_file}")
            print("[update] If this run is interrupted, restart with the same command "
                  "to resume repair without re-validating.")
        except Exception as exc:
            print(f"[update] Warning: could not save queue file: {exc}")

    # ── Step 3: Re-process incomplete maps ───────────────────────────────────
    if needs_repair:
        def _noop_log(scanned_id: int, msg: str, meta: Optional[Dict] = None) -> None:
            pass  # repairs are silent; existing log entries are never touched

        print(f"[update] Re-processing {len(needs_repair):,} maps in "
              f"{(len(needs_repair) - 1) // BATCH_SIZE + 1:,} batches …")
        for batch in tqdm(list(_chunks(needs_repair, BATCH_SIZE)), unit="batch", desc="repair"):
            # Delete stale data immediately before rebuilding — not during
            # validation — so an interrupted run never leaves maps missing.
            for bid in batch:
                map_dir = processed_root / str(bid)
                if map_dir.exists():
                    shutil.rmtree(map_dir, ignore_errors=True)
                win_file = windows_dir / f"{bid}_w{args.window_beats}b.parquet"
                win_file.unlink(missing_ok=True)  # type: ignore[arg-type]
            process_batch(batch, sess, processed_root, windows_dir, maps_dir, _noop_log, **kw)

    # Queue file is no longer needed once repair completes successfully.
    queue_file.unlink(missing_ok=True)  # type: ignore[arg-type]

    # ── Step 4: Continue ascending scan beyond the highest known ID ───────────
    user_max_id = int(args.max_id)
    if user_max_id > max_known_id:
        asc_log_path  = logs_dir / "build_log_asc.txt"
        asc_meta_path = logs_dir / "build_metadata_asc.jsonl"
        write_lock = threading.Lock()
        log_fn = _make_log_fn(asc_log_path, asc_meta_path, write_lock)

        fwd_start_id = max_known_id + 1
        new_ids  = list(range(fwd_start_id, user_max_id + 1))
        print(f"[update] Scanning new IDs {fwd_start_id:,}–{user_max_id:,} ({len(new_ids):,} IDs) …")
        for batch in tqdm(list(_chunks(new_ids, BATCH_SIZE)), unit="batch", desc="scan ↑"):
            process_batch(batch, sess, processed_root, windows_dir, maps_dir, log_fn, **kw)
    elif user_max_id > 0:
        print(f"[update] --max_id={user_max_id:,} ≤ highest known id={max_known_id:,}; "
              f"no new scan needed.")


# ---------------------------------------------------------------------------
# echosu tag refresh
# ---------------------------------------------------------------------------

def _echosu_get(url: str, token: str, params=None, timeout: float = 30.0) -> requests.Response:
    hdr = {"Authorization": f"Token {token}", "Accept": "application/json"}
    return requests.get(url, headers=hdr, params=params, timeout=timeout)


def run_fetch_echosu(args) -> None:
    """Download/update the echosu tag JSON used by label_clusters.py."""
    token = os.getenv("ECHOSU_TOKEN")
    if not token:
        raise RuntimeError("Missing ECHOSU_TOKEN in .env — cannot fetch echosu tags.")

    out_path = Path(str(args.echosu_json)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sleep_s = float(args.sleep_api)

    print("[builder/echosu] fetching beatmap list …")
    url    = ECHOSU_BEATMAPS
    params: dict | None = {"page_size": 500}
    beatmaps = []
    while url:
        r = _echosu_get(url, token, params=params)
        if r.status_code != 200:
            raise RuntimeError(f"echosu beatmaps HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if isinstance(data, dict) and "results" in data:
            beatmaps.extend(data.get("results", []))
            url    = data.get("next")
            params = None
        elif isinstance(data, list):
            beatmaps = data
            url = None
        else:
            url = None
        time.sleep(sleep_s)
    print(f"[builder/echosu] {len(beatmaps)} beatmaps found")

    # Resume support: skip IDs already written
    done: set[str] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open("r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        mid = obj.get("map_id")
                        if mid:
                            done.add(str(mid))
                    except Exception:
                        continue
        except Exception:
            done = set()

    with out_path.open("a", encoding="utf-8") as fout:
        for bm in tqdm(beatmaps, desc="echosu tags", unit="map"):
            mid = str(bm.get("beatmap_id") or bm.get("id") or "").strip()
            if not mid or mid in done:
                continue
            # Fetch per-map tag counts (exclude predictions)
            r2 = _echosu_get(
                ECHOSU_TAG_APPS, token,
                params={"beatmap_id": mid, "include": "tag_counts"},
            )
            counts: Dict[str, int] = {}
            if r2.status_code == 200:
                for item in (r2.json() or []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("is_predicted") is True:
                        continue
                    if item.get("is_negative") is True:
                        continue
                    tag  = item.get("tag") or {}
                    name = tag.get("name") if isinstance(tag, dict) else None
                    if not name:
                        continue
                    cnt = tag.get("count")
                    if isinstance(cnt, int):
                        counts[str(name)] = max(int(cnt), counts.get(str(name), 0))
            fout.write(json.dumps({"map_id": mid, "tags": counts}, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(mid)
            time.sleep(sleep_s)

    print(f"[builder/echosu] wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode flags ─────────────────────────────────────────────────────────
    ap.add_argument(
        "--max_id", type=int, required=False, default=0,
        metavar="ID",
        help=(
            "Required for scan mode: scan upward from the latest known ID up to "
            "this value.  Pass 0 (or omit) when using --map_ids only or "
            "--skip_dataset.  In --update_dataset mode this is optional: if "
            "provided and greater than the highest ID in build_log_desc.txt, the "
            "script will continue scanning upward after the repair pass."
        ),
    )
    ap.add_argument(
        "--map_ids", type=str, default="",
        metavar="ID[,ID,…]",
        help=(
            "Patch mode: comma-separated list of specific beatmap IDs to download "
            "and process. The scan is skipped; only these IDs are fetched."
        ),
    )
    ap.add_argument(
        "--skip_dataset", action="store_true",
        help="Skip both scan and patch; jump straight to --fetch_echosu.",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume an interrupted ascending scan.  Reads the last line of "
            "logs/build_log_asc.txt and continues from that ID + 1.  Much faster "
            "than the default behaviour (which scans every on-disk directory to "
            "find the highest map ID).  Falls back to the disk scan if the log "
            "is absent or empty.  Has no effect in --update_dataset or patch mode."
        ),
    )
    ap.add_argument(
        "--fetch_echosu", action="store_true",
        help="After the dataset step, refresh the local echosu tag JSON.",
    )
    ap.add_argument(
        "--update_dataset", action="store_true",
        help=(
            "Update / repair mode.  Enumerates every directory under "
            "processed/maps_by_id/ and validates each map for completeness "
            "(files present, non-empty, required meta keys including star_rating, "
            "required timeseries columns, and — when --use_rosu_strains is set — "
            "non-null strain columns for circle-heavy maps).  Maps that fail are "
            "deleted and re-downloaded silently.  Fields that are legitimately "
            "absent (e.g. ranked_date for unranked maps, slider columns for circle "
            "rows) are never flagged.  After the repair pass, if --max_id exceeds "
            "the highest on-disk ID, a normal ascending scan continues from there "
            "and is written to build_log_asc.txt.  The repair queue is saved to "
            "logs/update_repair_queue.json after validation; if the run is "
            "interrupted, restarting with the same command resumes repair without "
            "re-validating.  Use --revalidate to force a fresh scan."
        ),
    )
    ap.add_argument(
        "--revalidate", action="store_true",
        help=(
            "Only meaningful with --update_dataset.  Discards any saved repair "
            "queue (logs/update_repair_queue.json) and re-runs the full "
            "enumeration + validation pass before repairing."
        ),
    )
    ap.add_argument(
        "--skip_validation", action="store_true",
        help=(
            "Only meaningful with --update_dataset.  Skips the completeness "
            "validation pass entirely and queues every map found on disk for "
            "repair.  Maps that are already complete are skipped automatically "
            "by the EXISTS check in process_batch(), so no data is deleted and "
            "no complete map is ever re-downloaded.  Useful when you know "
            "validation is unnecessary (e.g. after a partial run where the "
            "repair queue was never saved)."
        ),
    )
    ap.add_argument(
        "--start_id", type=int, default=0,
        help=(
            "Only meaningful with --update_dataset.  Skip all map IDs below "
            "this value during enumeration, validation, and repair.  When "
            "resuming from a saved queue the filter is applied to the loaded "
            "list.  Useful for continuing a run that was interrupted partway "
            "through a known ID range."
        ),
    )

    # ── Dataset / output paths ──────────────────────────────────────────────
    ap.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    ap.add_argument(
        "--echosu_json", type=str, default=str(DEFAULT_ECHOSU_JSON),
        help="Path to write/update the echosu tag JSONL used by label_clusters.py.",
    )
    ap.add_argument(
        "--keep_osu", action="store_true",
        help="Retain downloaded .osu files under <dataset_root>/maps.",
    )

    # ── API / network ───────────────────────────────────────────────────────
    ap.add_argument("--sleep_api",  type=float, default=0.2,  help="Sleep (s) between osu! v2 batch requests.")
    ap.add_argument("--sleep_raw",  type=float, default=0.2,  help="Sleep (s) after each raw .osu download.")
    ap.add_argument("--api_timeout", type=float, default=30.0)
    ap.add_argument("--raw_timeout", type=float, default=30.0)
    ap.add_argument("--raw_retries", type=int,   default=6)
    ap.add_argument("--max_seconds_per_map", type=float, default=3.0,
                    help="Hard per-map time budget; exceeded → TIMEOUT and skip.")

    # ── Feature extraction ──────────────────────────────────────────────────
    ap.add_argument("--window_beats",     type=int,   default=4)
    ap.add_argument("--window_overlap",   type=float, default=0.5)
    ap.add_argument("--max_t_ms_sanity",  type=int,   default=100_000_000)
    ap.add_argument("--use_rosu_strains", action="store_true",
                    help="Compute rosu strain series (slower, optional).")

    args = ap.parse_args()

    update_mode = bool(args.update_dataset)
    patch_mode  = bool(str(args.map_ids).strip())
    scan_mode   = not patch_mode and not bool(args.skip_dataset) and not update_mode

    # Validate: plain scan mode requires --max_id > 0
    if scan_mode and int(args.max_id) <= 0:
        ap.error("--max_id is required for scan mode (provide the highest beatmap ID to check, e.g. --max_id 5600000)")

    # --update_dataset is mutually exclusive with patch / skip_dataset modes
    if update_mode and patch_mode:
        ap.error("--update_dataset and --map_ids cannot be used together")
    if update_mode and bool(args.skip_dataset):
        ap.error("--update_dataset and --skip_dataset cannot be used together")

    # Load .env for OSU_CLIENT_ID / OSU_CLIENT_SECRET / ECHOSU_TOKEN
    load_env()

    root = Path(str(args.dataset_root)).resolve()
    ensure_dirs(root)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sess = _requests_session()

    # ── Dataset step ────────────────────────────────────────────────────────
    if update_mode:
        run_update(args, root, sess)
    elif not bool(args.skip_dataset):
        log_path  = logs_dir / "build_log_asc.txt"
        meta_path = logs_dir / "build_metadata_asc.jsonl"
        write_lock = threading.Lock()
        log_fn = _make_log_fn(log_path, meta_path, write_lock)
        if patch_mode:
            run_patch(args, root, sess, log_fn)
        else:
            run_scan(args, root, sess, log_fn)

    # ── echosu step ─────────────────────────────────────────────────────────
    if bool(args.fetch_echosu):
        run_fetch_echosu(args)

    print("[builder] done.")


if __name__ == "__main__":
    main()
