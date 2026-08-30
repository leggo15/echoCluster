from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def normalize_map_id(value: object) -> str:
    s = str(value).strip().lstrip("\ufeff").strip('"').strip("'")
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        return s
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_cluster_map(clusters_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(clusters_csv, usecols=["map_id", "cluster"], dtype={"map_id": "string"}, keep_default_na=False)
    df["map_id"] = df["map_id"].map(normalize_map_id).astype(str)
    df["cluster"] = df["cluster"].astype(int)
    return df


def load_embeddings_from_chunks(run_dir: Path) -> Tuple[np.ndarray, List[str]]:
    state_path = run_dir / "cluster_progress" / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing progress state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    chunks = state.get("chunks") or []
    embs: List[np.ndarray] = []
    ids: List[str] = []
    for ch in chunks:
        emb_path = run_dir / "cluster_progress" / str(ch["emb"])
        ids_path = run_dir / "cluster_progress" / str(ch["ids"])
        if not emb_path.exists() or not ids_path.exists():
            continue
        arr = np.load(emb_path)
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if arr.shape[0] != len(chunk_ids):
            n = min(arr.shape[0], len(chunk_ids))
            arr = arr[:n]
            chunk_ids = chunk_ids[:n]
        embs.append(arr.astype(np.float32))
        ids.extend([normalize_map_id(x) for x in chunk_ids])
    if not embs:
        raise RuntimeError("No embedding chunks found.")
    return np.concatenate(embs, axis=0), ids


def load_embeddings(run_dir: Path) -> Tuple[np.ndarray, List[str]]:
    emb_path = run_dir / "embeddings.npy"
    clu_path = run_dir / "clusters.csv"
    if emb_path.exists() and clu_path.exists():
        emb = np.load(emb_path).astype(np.float32)
        dfc = pd.read_csv(clu_path, usecols=["map_id"], dtype={"map_id": "string"}, keep_default_na=False)
        ids = [normalize_map_id(x) for x in dfc["map_id"].tolist()]
        if emb.shape[0] == len(ids):
            return emb, ids
    # fallback to chunk checkpoints
    return load_embeddings_from_chunks(run_dir)


def infer_dataset_root(run_dir: Path) -> Path:
    st = run_dir / "cluster_progress" / "state.json"
    if st.exists():
        try:
            obj = json.loads(st.read_text(encoding="utf-8"))
            dr = obj.get("dataset_root")
            if dr:
                return Path(str(dr)).resolve()
        except Exception:
            pass
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            obj = json.loads(cfg.read_text(encoding="utf-8"))
            dr = obj.get("dataset_root")
            if dr:
                return Path(str(dr)).resolve()
        except Exception:
            pass
    try:
        from CORE.paths import DEFAULT_DATASET_ROOT
        fallback = DEFAULT_DATASET_ROOT
    except ImportError:
        fallback = (Path(__file__).resolve().parents[1] / "CORE" / "data" / "dataset_full_osu").resolve()
    print(f"[webgl] WARNING: could not infer dataset_root from state/config; falling back to hardcoded path: {fallback}")
    return fallback


def reduce_2d(emb: np.ndarray, seed: int) -> np.ndarray:
    # PCA chosen for speed on >1M points.
    pca = PCA(n_components=2, random_state=int(seed))
    xy = pca.fit_transform(emb).astype(np.float32)
    return xy


def reduce_3d_sphere(emb: np.ndarray, seed: int, radius: float = 1.0) -> np.ndarray:
    # Build a stable 3D PCA basis, then project to a fixed-radius sphere.
    pca = PCA(n_components=3, random_state=int(seed))
    xyz = pca.fit_transform(emb).astype(np.float32)
    norms = np.linalg.norm(xyz, axis=1).astype(np.float32)
    safe = np.where(norms > 1e-8, norms, 1.0).astype(np.float32)
    r = float(max(1e-4, radius))
    xyz = (xyz / safe[:, None]) * r
    return xyz.astype(np.float32)


def density_alpha(xy: np.ndarray, bins: int = 240) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax <= xmin or ymax <= ymin:
        return np.full((xy.shape[0],), 120, dtype=np.uint8)
    H, xedges, yedges = np.histogram2d(x, y, bins=int(bins), range=[[xmin, xmax], [ymin, ymax]])
    ix = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, H.shape[1] - 1)
    d = H[ix, iy].astype(np.float32)
    if np.max(d) > np.min(d):
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
    else:
        d = np.zeros_like(d)
    alpha = (35.0 + 210.0 * d).astype(np.uint8)  # 35..245
    return alpha


def local_density_values(xy: np.ndarray, bins: int = 240) -> np.ndarray:
    """Return per-point local occupancy count from a 2D histogram bin lookup."""
    x = xy[:, 0]
    y = xy[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax <= xmin or ymax <= ymin:
        return np.zeros((xy.shape[0],), dtype=np.float32)
    H, xedges, yedges = np.histogram2d(x, y, bins=int(bins), range=[[xmin, xmax], [ymin, ymax]])
    ix = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, H.shape[1] - 1)
    d = H[ix, iy].astype(np.float32)
    return d


def local_density_values_3d(xyz: np.ndarray, n_bins_lat: int = 120) -> np.ndarray:
    """Per-point density estimated from a spherical (lat/lon) histogram.

    Uses the 3D unit-sphere positions so the density is coherent with the 3D
    view, avoiding the PCA-distortion artefacts that come from applying 2D
    histogram density to a sphere surface.
    """
    if xyz.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    norms = np.linalg.norm(xyz, axis=1).astype(np.float32)
    safe = np.where(norms > 1e-8, norms, 1.0).astype(np.float32)
    u = (xyz / safe[:, None]).astype(np.float32)
    lat = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))           # -pi/2 .. pi/2
    lon = np.arctan2(u[:, 1], u[:, 0])                      # -pi .. pi
    n_lat = int(max(4, n_bins_lat))
    n_lon = int(max(4, n_lat * 2))
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1, dtype=np.float32)
    lon_edges = np.linspace(-np.pi, np.pi, n_lon + 1, dtype=np.float32)
    H, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
    H = H.astype(np.float32)
    i_lat = np.clip(np.searchsorted(lat_edges, lat, side="right") - 1, 0, n_lat - 1)
    i_lon = np.clip(np.searchsorted(lon_edges, lon, side="right") - 1, 0, n_lon - 1)
    # Normalise each bin by its solid angle so polar bins (which are smaller)
    # don't appear artificially empty.  Solid angle ∝ cos(lat_centre) * dlat * dlon.
    dlat = np.pi / n_lat
    dlon = 2.0 * np.pi / n_lon
    lat_centres = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    solid_angle = np.maximum(np.cos(lat_centres) * dlat * dlon, 1e-8).astype(np.float32)
    H_norm = H / solid_angle[:, None]
    d = H_norm[i_lat, i_lon].astype(np.float32)
    return d


def load_echosu_positive_tags(echosu_json: Path) -> Dict[str, List[Dict[str, object]]]:
    """Load positive tags from echosu export.

    The file may be:
      (a) A plain JSON array of {map_id, tags} objects with full tag metadata.
      (b) A JSON array (format a) concatenated with JSONL records using the
          simpler {map_id, tags: {name: count}} format.
      (c) Pure JSONL (format b).
    Both tag-value formats are handled; entries from all sections are merged
    per map_id (max count wins; richer metadata is preferred).
    """
    if not echosu_json.exists():
        return {}

    raw = echosu_json.read_text(encoding="utf-8")
    items: list = []

    # Try to peel off a leading JSON array (handles formats a and b).
    try:
        decoder = json.JSONDecoder()
        first, end = decoder.raw_decode(raw)
        if isinstance(first, list):
            items.extend(first)
        raw_tail = raw[end:].strip()
    except Exception:
        raw_tail = raw

    # Parse any remaining content as JSONL (handles formats b and c).
    for line in raw_tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("map_id"):
                items.append(obj)
        except Exception:
            pass

    # Accumulate and merge tags per map_id across all sources.
    # merged[mid][tag_name] = (count, category)
    merged: Dict[str, Dict[str, Tuple[int, str]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("map_id") or "").strip()
        if not mid:
            continue
        tags = item.get("tags") or {}
        if not isinstance(tags, dict):
            continue
        tag_acc = merged.setdefault(mid, {})
        for name, meta in tags.items():
            name = str(name)
            if isinstance(meta, dict):
                if bool(meta.get("is_negative", False)):
                    continue
                try:
                    cnt = int(meta.get("count", 0) or 0)
                except Exception:
                    cnt = 0
                cat = str(meta.get("category") or "").strip().lower()
            elif isinstance(meta, (int, float)):
                # Simple format: {tag_name: count}
                cnt = int(meta)
                cat = ""
            else:
                continue
            if cnt <= 0:
                continue
            existing = tag_acc.get(name)
            if existing is None:
                tag_acc[name] = (cnt, cat)
            else:
                # Prefer higher count; prefer richer category string.
                tag_acc[name] = (max(cnt, existing[0]), cat or existing[1])

    out: Dict[str, List[Dict[str, object]]] = {}
    for mid, tag_acc in merged.items():
        if not tag_acc:
            continue
        rows = sorted(tag_acc.items(), key=lambda kv: (-kv[1][0], kv[0].lower()))
        out[mid] = [{"name": k, "count": v, "category": cat} for k, (v, cat) in rows]
    return out



def load_meta_fields(dataset_root: Path, map_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Return map_id -> scalar metadata needed by WebGL filters."""
    out: Dict[str, Dict[str, float]] = {}
    processed = dataset_root / "processed" / "maps_by_id"
    for mid in map_ids:
        mid_n = normalize_map_id(mid)
        p = processed / mid_n / f"{mid_n}_meta.json"
        if not p.exists() and str(mid).strip() != mid_n:
            # Fallback for unexpected folder naming in older exports.
            raw_mid = str(mid).strip()
            p = processed / raw_mid / f"{raw_mid}_meta.json"
        if not p.exists():
            out[mid_n] = {
                "star": float("nan"),
                "status_ranked": -1.0,
                "hp": float("nan"),
                "od": float("nan"),
                "cs": float("nan"),
                "ar": float("nan"),
                "bpm": float("nan"),
                "length": float("nan"),
                "artist": "",
                "title": "",
                "version": "",
                "creator": "",
            }
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            out[mid_n] = {
                "star": float("nan"),
                "status_ranked": -1.0,
                "hp": float("nan"),
                "od": float("nan"),
                "cs": float("nan"),
                "ar": float("nan"),
                "bpm": float("nan"),
                "length": float("nan"),
                "artist": "",
                "title": "",
                "version": "",
                "creator": "",
            }
            continue

        # star fallback candidates
        star = float("nan")
        for k in ("star_rating", "difficulty_rating", "stars", "star", "sr_total"):
            v = m.get(k)
            try:
                if v is not None:
                    star = float(v)
                    break
            except Exception:
                continue

        # ranked fallback candidates
        ranked = -1
        if "status_ranked" in m:
            try:
                ranked = 1 if float(m.get("status_ranked")) >= 0.5 else 0
            except Exception:
                ranked = -1
        else:
            s = str(m.get("status") or m.get("ranked_status") or "").strip().lower()
            if s:
                ranked = 1 if s in {"ranked", "approved", "qualified"} else 0

        def _first_float(keys: List[str]) -> float:
            for k in keys:
                v = m.get(k)
                try:
                    if v is not None:
                        return float(v)
                except Exception:
                    continue
            return float("nan")

        def _first_str(keys: List[str]) -> str:
            for k in keys:
                v = m.get(k)
                s = str(v).strip() if v is not None else ""
                if s:
                    return s
            return ""

        out[mid_n] = {
            "star": float(star),
            "status_ranked": float(ranked),
            "hp": _first_float(["hp", "drain"]),
            "od": _first_float(["od", "accuracy"]),
            "cs": _first_float(["cs"]),
            "ar": _first_float(["ar"]),
            "bpm": _first_float(["bpm"]),
            "length": _first_float(["length_total", "length_drain", "total_length", "hit_length", "length", "duration", "drain_time"]),
            "artist": _first_str(["artist_unicode", "artist"]),
            "title": _first_str(["title_unicode", "title"]),
            "version": _first_str(["version", "difficulty_name", "diff_name"]),
            "creator": _first_str(["creator"]),
        }
    return out


def _fill_nan_from_neighbors(field: np.ndarray, passes: int = 10) -> np.ndarray:
    out = field.astype(np.float32).copy()
    for _ in range(max(1, int(passes))):
        nan_mask = ~np.isfinite(out)
        if not np.any(nan_mask):
            break
        p = np.pad(out, 1, mode="edge")
        neighbors = np.stack(
            [
                p[:-2, :-2],
                p[:-2, 1:-1],
                p[:-2, 2:],
                p[1:-1, :-2],
                p[1:-1, 2:],
                p[2:, :-2],
                p[2:, 1:-1],
                p[2:, 2:],
            ],
            axis=0,
        )
        valid = np.isfinite(neighbors)
        count = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, neighbors, 0.0), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = sums / np.maximum(count, 1)
        fill_mask = nan_mask & np.isfinite(est)
        out[fill_mask] = est[fill_mask]
    nan_mask = ~np.isfinite(out)
    if np.any(nan_mask):
        med = float(np.nanmedian(out))
        out[nan_mask] = med if np.isfinite(med) else 0.0
    return out


def _box_blur_3x3(field: np.ndarray, passes: int = 3) -> np.ndarray:
    out = field.astype(np.float32).copy()
    for _ in range(max(0, int(passes))):
        p = np.pad(out, 1, mode="edge")
        out = (
            p[:-2, :-2]
            + p[:-2, 1:-1]
            + p[:-2, 2:]
            + p[1:-1, :-2]
            + p[1:-1, 1:-1]
            + p[1:-1, 2:]
            + p[2:, :-2]
            + p[2:, 1:-1]
            + p[2:, 2:]
        ) / 9.0
    return out


def _compute_boost_weights(
    s: np.ndarray,
    *,
    hv_thr: float,
    hv_k: float,
    hv_p: float,
    hv_cap: "float | None",
    lv_thr: float,
    lv_k: float,
    lv_p: float,
) -> np.ndarray:
    """Per-point weight multipliers for high- and low-value emphasis boosts."""
    w = np.ones_like(s, dtype=np.float32)
    if hv_k > 0.0:
        if hv_cap is not None and float(hv_cap) > hv_thr:
            # Asymptotic curve: flat below threshold, tanh ramp that plateaus at cap.
            # 0→thr: boost=0; thr→cap: tanh(p*t) where t=(s-thr)/(cap-thr); >cap: plateau.
            t = np.clip((s - hv_thr) / (float(hv_cap) - hv_thr), 0.0, 1.0)
            w = w * (1.0 + hv_k * np.tanh(hv_p * t))
        else:
            w = w * (1.0 + hv_k * np.maximum(0.0, s - hv_thr) ** hv_p)
    if lv_k > 0.0:
        w = w * (1.0 + lv_k * np.maximum(0.0, lv_thr - s) ** lv_p)
    return w.astype(np.float32)


def _marching_squares_segments(field: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, level: float) -> List[List[List[float]]]:
    nx, ny = field.shape
    if nx < 2 or ny < 2:
        return []
    case_edges = {
        0: [],
        1: [(3, 0)],
        2: [(0, 1)],
        3: [(3, 1)],
        4: [(1, 2)],
        5: [(3, 2), (0, 1)],
        6: [(0, 2)],
        7: [(3, 2)],
        8: [(2, 3)],
        9: [(0, 2)],
        10: [(0, 3), (1, 2)],
        11: [(1, 2)],
        12: [(1, 3)],
        13: [(0, 1)],
        14: [(3, 0)],
        15: [],
    }
    segments: List[List[List[float]]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            c0 = (i, j, float(field[i, j]))
            c1 = (i + 1, j, float(field[i + 1, j]))
            c2 = (i + 1, j + 1, float(field[i + 1, j + 1]))
            c3 = (i, j + 1, float(field[i, j + 1]))
            corners = [c0, c1, c2, c3]
            idx = 0
            idx |= 1 if c0[2] >= level else 0
            idx |= 2 if c1[2] >= level else 0
            idx |= 4 if c2[2] >= level else 0
            idx |= 8 if c3[2] >= level else 0
            if idx in (0, 15):
                continue
            pairs = case_edges.get(idx, [])
            if not pairs:
                continue
            edge_corners = {
                0: (0, 1),  # c0-c1
                1: (1, 2),  # c1-c2
                2: (2, 3),  # c2-c3
                3: (3, 0),  # c3-c0
            }

            def edge_point(edge_id: int) -> Tuple[float, float]:
                ia, ib = edge_corners[edge_id]
                a = corners[ia]
                b = corners[ib]
                va = a[2]
                vb = b[2]
                if not np.isfinite(va) or not np.isfinite(vb):
                    return (float("nan"), float("nan"))
                if abs(vb - va) < 1e-12:
                    t = 0.5
                else:
                    t = (level - va) / (vb - va)
                    t = float(np.clip(t, 0.0, 1.0))
                x = float((1.0 - t) * x_grid[a[0]] + t * x_grid[b[0]])
                y = float((1.0 - t) * y_grid[a[1]] + t * y_grid[b[1]])
                return (x, y)

            for e0, e1 in pairs:
                p0 = edge_point(int(e0))
                p1 = edge_point(int(e1))
                if not (np.isfinite(p0[0]) and np.isfinite(p0[1]) and np.isfinite(p1[0]) and np.isfinite(p1[1])):
                    continue
                if abs(p0[0] - p1[0]) < 1e-9 and abs(p0[1] - p1[1]) < 1e-9:
                    continue
                segments.append(
                    [
                        [round(float(p0[0]), 4), round(float(p0[1]), 4)],
                        [round(float(p1[0]), 4), round(float(p1[1]), 4)],
                    ]
                )
    return segments


def _simplify_segments(
    segments: List[List[List[float]]],
    quant_step: float = 0.0,
    min_segment_len: float = 0.0,
    keep_every: int = 1,
) -> List[List[List[float]]]:
    if not segments:
        return []
    q = float(max(0.0, quant_step))
    min_len2 = float(max(0.0, min_segment_len)) ** 2
    k_every = int(max(1, int(keep_every)))
    out: List[List[List[float]]] = []
    seen: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for idx, seg in enumerate(segments):
        if k_every > 1 and (idx % k_every) != 0:
            continue
        if not isinstance(seg, list) or len(seg) != 2:
            continue
        p0, p1 = seg[0], seg[1]
        if not (isinstance(p0, list) and isinstance(p1, list) and len(p0) == 2 and len(p1) == 2):
            continue
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx = x1 - x0
        dy = y1 - y0
        if (dx * dx + dy * dy) < min_len2:
            continue
        if q > 0.0:
            a = (int(round(x0 / q)), int(round(y0 / q)))
            b = (int(round(x1 / q)), int(round(y1 / q)))
            edge = (a, b) if a <= b else (b, a)
            if edge in seen:
                continue
            seen.add(edge)
        out.append([[round(x0, 4), round(y0, 4)], [round(x1, 4), round(y1, 4)]])
    return out


def build_value_contours(
    xy: np.ndarray,
    values: np.ndarray,
    level_count: int = 15,
    level_min: float = 1.0,
    level_max: float = 15.0,
    grid_size: int = 180,
    smooth_passes: int = 3,
    simplify_quant_step: float = 0.0,
    min_segment_len: float = 0.0,
    segment_keep_every: int = 1,
    high_value_boost_threshold: float = 1e9,
    high_value_boost_strength: float = 0.0,
    high_value_boost_power: float = 2.0,
    high_value_boost_cap: float | None = None,
    low_value_boost_threshold: float = -1e9,
    low_value_boost_strength: float = 0.0,
    low_value_boost_power: float = 2.0,
) -> Dict[str, object]:
    if xy.shape[0] == 0:
        return {"contours": [], "n_segments": 0}
    x = xy[:, 0].astype(np.float32)
    y = xy[:, 1].astype(np.float32)
    s = values.astype(np.float32)
    valid = np.isfinite(s)
    if not np.any(valid):
        return {"contours": [], "n_segments": 0}

    x = x[valid]
    y = y[valid]
    s = s[valid]
    s = np.clip(s, float(level_min), float(level_max))
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax <= xmin or ymax <= ymin:
        return {"contours": [], "n_segments": 0}

    gs = int(max(40, min(512, int(grid_size))))
    ix = np.clip(((x - xmin) / (xmax - xmin + 1e-12) * (gs - 1)).astype(np.int32), 0, gs - 1)
    iy = np.clip(((y - ymin) / (ymax - ymin + 1e-12) * (gs - 1)).astype(np.int32), 0, gs - 1)
    flat = ix * gs + iy
    w = _compute_boost_weights(
        s,
        hv_thr=float(high_value_boost_threshold),
        hv_k=float(max(0.0, high_value_boost_strength)),
        hv_p=float(max(1.0, high_value_boost_power)),
        hv_cap=high_value_boost_cap,
        lv_thr=float(low_value_boost_threshold),
        lv_k=float(max(0.0, low_value_boost_strength)),
        lv_p=float(max(1.0, low_value_boost_power)),
    )
    sum_star = np.bincount(flat, weights=(s * w), minlength=gs * gs).reshape(gs, gs).astype(np.float32)
    count = np.bincount(flat, weights=w, minlength=gs * gs).reshape(gs, gs).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_star = sum_star / count
    mean_star[count <= 0] = np.nan
    mean_star = _fill_nan_from_neighbors(mean_star, passes=12)
    mean_star = _box_blur_3x3(mean_star, passes=int(max(0, smooth_passes)))

    x_grid = np.linspace(xmin, xmax, gs, dtype=np.float32)
    y_grid = np.linspace(ymin, ymax, gs, dtype=np.float32)
    levels = np.linspace(float(level_min), float(level_max), int(max(1, level_count)), dtype=np.float32)

    contours = []
    total_segments = 0
    for lvl in levels.tolist():
        segs = _marching_squares_segments(mean_star, x_grid, y_grid, float(lvl))
        segs = _simplify_segments(
            segs,
            quant_step=float(simplify_quant_step),
            min_segment_len=float(min_segment_len),
            keep_every=int(segment_keep_every),
        )
        if not segs:
            continue
        total_segments += len(segs)
        contours.append({"level": round(float(lvl), 3), "segments": segs, "n_segments": int(len(segs))})
    return {
        "contours": contours,
        "n_segments": int(total_segments),
        "n_levels": int(level_count),
        "level_min": float(level_min),
        "level_max": float(level_max),
        "grid_size": int(gs),
        "smooth_passes": int(smooth_passes),
        "field_min": float(np.min(mean_star)),
        "field_max": float(np.max(mean_star)),
    }


def _normalize_rows(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return out / norms


def build_icosphere(subdivisions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    t = float((1.0 + np.sqrt(5.0)) / 2.0)
    verts = np.asarray(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int32,
    )
    verts = _normalize_rows(verts)
    for _ in range(max(0, int(subdivisions))):
        verts_list: List[List[float]] = verts.tolist()
        midpoint_cache: Dict[Tuple[int, int], int] = {}
        next_faces: List[List[int]] = []

        def midpoint_index(i0: int, i1: int) -> int:
            key = (int(i0), int(i1)) if int(i0) < int(i1) else (int(i1), int(i0))
            if key in midpoint_cache:
                return midpoint_cache[key]
            p = np.asarray(verts_list[key[0]], dtype=np.float32) + np.asarray(verts_list[key[1]], dtype=np.float32)
            p = p / np.maximum(np.linalg.norm(p), 1e-8)
            idx = len(verts_list)
            verts_list.append([float(p[0]), float(p[1]), float(p[2])])
            midpoint_cache[key] = idx
            return idx

        for tri in faces.tolist():
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = midpoint_index(a, b)
            bc = midpoint_index(b, c)
            ca = midpoint_index(c, a)
            next_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        verts = np.asarray(verts_list, dtype=np.float32)
        faces = np.asarray(next_faces, dtype=np.int32)
    verts = _normalize_rows(verts)
    # Enforce outward winding for every face so rasterization is stable.
    fixed_faces = faces.astype(np.int32).copy()
    for i in range(fixed_faces.shape[0]):
        a, b, c = int(fixed_faces[i, 0]), int(fixed_faces[i, 1]), int(fixed_faces[i, 2])
        va = verts[a]
        vb = verts[b]
        vc = verts[c]
        n = np.cross(vb - va, vc - va)
        centroid = (va + vb + vc) / 3.0
        if float(np.dot(n, centroid)) < 0.0:
            fixed_faces[i, 1], fixed_faces[i, 2] = fixed_faces[i, 2], fixed_faces[i, 1]
    return verts, fixed_faces


def build_sphere_contour_cache(
    xyz: np.ndarray,
    radius: float,
    grid_size: int = 180,
    candidate_neighbors: int = 48,
    mesh_subdivisions: int | None = None,
) -> Dict[str, np.ndarray]:
    norms = np.linalg.norm(xyz, axis=1).astype(np.float32)
    good = norms > 1e-8
    if not np.any(good):
        return {}
    unit_xyz = _normalize_rows(xyz[good])
    subdivisions = int(mesh_subdivisions) if mesh_subdivisions is not None else (4 if int(grid_size) >= 140 else 3)
    subdivisions = int(max(1, min(7, subdivisions)))
    mesh_vertices_unit, mesh_faces = build_icosphere(subdivisions=subdivisions)
    mesh_vertices = mesh_vertices_unit * float(radius)
    k = int(max(1, min(int(candidate_neighbors), unit_xyz.shape[0])))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(unit_xyz)
    dists, local_idx = nn.kneighbors(mesh_vertices_unit, return_distance=True)
    source_idx = np.flatnonzero(good).astype(np.int32)
    full_idx = source_idx[local_idx.astype(np.int32)]
    edges = np.concatenate([mesh_faces[:, [0, 1]], mesh_faces[:, [1, 2]], mesh_faces[:, [2, 0]]], axis=0).astype(np.int32)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return {
        "mesh_vertices": mesh_vertices.astype(np.float32),
        "mesh_faces": mesh_faces.astype(np.int32),
        "neighbor_idx": full_idx.astype(np.int32),
        "neighbor_dist": dists.astype(np.float32),
        "edges": edges.astype(np.int32),
    }


def _smooth_mesh_field(values: np.ndarray, edges: np.ndarray, passes: int = 2) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if out.size == 0 or edges.size == 0:
        return out
    src = edges[:, 0].astype(np.int32)
    dst = edges[:, 1].astype(np.int32)
    n = int(out.shape[0])
    for _ in range(max(0, int(passes))):
        sums = out.copy()
        counts = np.ones((n,), dtype=np.float32)
        sums += np.bincount(src, weights=out[dst], minlength=n).astype(np.float32)
        sums += np.bincount(dst, weights=out[src], minlength=n).astype(np.float32)
        counts += np.bincount(src, minlength=n).astype(np.float32)
        counts += np.bincount(dst, minlength=n).astype(np.float32)
        out = sums / np.maximum(counts, 1.0)
    return out.astype(np.float32)


def _slerp_segment_path(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    max_step_radians: float = 0.05,
) -> List[List[float]]:
    u0 = np.asarray(p0, dtype=np.float32)
    u1 = np.asarray(p1, dtype=np.float32)
    u0 = u0 / np.maximum(np.linalg.norm(u0), 1e-8)
    u1 = u1 / np.maximum(np.linalg.norm(u1), 1e-8)
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    omega = float(np.arccos(dot))
    if omega < 1e-5:
      a = (u0 * float(radius)).astype(np.float32)
      b = (u1 * float(radius)).astype(np.float32)
      return [[round(float(a[0]), 4), round(float(a[1]), 4), round(float(a[2]), 4)], [round(float(b[0]), 4), round(float(b[1]), 4), round(float(b[2]), 4)]]
    sin_omega = float(np.sin(omega))
    n_steps = int(max(2, min(24, np.ceil(omega / max_step_radians) + 1)))
    out: List[List[float]] = []
    for t in np.linspace(0.0, 1.0, n_steps, dtype=np.float32).tolist():
        a = float(np.sin((1.0 - t) * omega) / np.maximum(sin_omega, 1e-8))
        b = float(np.sin(t * omega) / np.maximum(sin_omega, 1e-8))
        p = (a * u0 + b * u1).astype(np.float32)
        p = p / np.maximum(np.linalg.norm(p), 1e-8) * float(radius)
        out.append([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
    return out


def build_sphere_value_contours(
    xyz: np.ndarray,
    values: np.ndarray,
    level_count: int = 15,
    level_min: float = 1.0,
    level_max: float = 15.0,
    grid_size: int = 180,
    smooth_passes: int = 3,
    simplify_quant_step: float = 0.0,
    min_segment_len: float = 0.0,
    segment_keep_every: int = 1,
    high_value_boost_threshold: float = 1e9,
    high_value_boost_strength: float = 0.0,
    high_value_boost_power: float = 2.0,
    high_value_boost_cap: float | None = None,
    low_value_boost_threshold: float = -1e9,
    low_value_boost_strength: float = 0.0,
    low_value_boost_power: float = 2.0,
    sphere_cache: Dict[str, np.ndarray] | None = None,
    mesh_subdivisions: int | None = None,
) -> Dict[str, object]:
    if xyz.shape[0] == 0:
        return {"contours": [], "n_segments": 0}
    s = values.astype(np.float32)
    valid = np.isfinite(s)
    if not np.any(valid):
        return {"contours": [], "n_segments": 0}
    cache = sphere_cache or build_sphere_contour_cache(
        xyz,
        radius=float(max(1e-5, np.median(np.linalg.norm(xyz, axis=1)))),
        grid_size=int(grid_size),
        mesh_subdivisions=mesh_subdivisions,
    )
    if not cache:
        return {"contours": [], "n_segments": 0}
    mesh_vertices = np.asarray(cache["mesh_vertices"], dtype=np.float32)
    mesh_faces = np.asarray(cache["mesh_faces"], dtype=np.int32)
    neighbor_idx = np.asarray(cache["neighbor_idx"], dtype=np.int32)
    neighbor_dist = np.asarray(cache["neighbor_dist"], dtype=np.float32)
    edges = np.asarray(cache["edges"], dtype=np.int32)
    contour_radius = float(max(1e-5, np.median(np.linalg.norm(mesh_vertices, axis=1))) * 1.01)

    s = np.clip(s, float(level_min), float(level_max))
    w = _compute_boost_weights(
        s,
        hv_thr=float(high_value_boost_threshold),
        hv_k=float(max(0.0, high_value_boost_strength)),
        hv_p=float(max(1.0, high_value_boost_power)),
        hv_cap=high_value_boost_cap,
        lv_thr=float(low_value_boost_threshold),
        lv_k=float(max(0.0, low_value_boost_strength)),
        lv_p=float(max(1.0, low_value_boost_power)),
    )
    nbr_vals = s[neighbor_idx]
    # Gaussian kernel: sigma = median nearest-neighbour distance so each point
    # influences ~one neighbourhood width.  Avoids the extreme weight ratios
    # produced by pure 1/d² when one neighbour is much closer than the rest.
    sigma = float(np.median(neighbor_dist[:, 0])) * 1.5
    sigma = max(sigma, 1e-4)
    nbr_weights = w[neighbor_idx] * np.exp(-(neighbor_dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    nbr_valid = np.isfinite(nbr_vals)
    weighted_sum = np.sum(np.where(nbr_valid, nbr_vals * nbr_weights, 0.0), axis=1).astype(np.float32)
    weight_sum = np.sum(np.where(nbr_valid, nbr_weights, 0.0), axis=1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_field = weighted_sum / np.maximum(weight_sum, 1e-8)
    fill_value = float(np.nanmedian(s[valid]))
    if not np.isfinite(fill_value):
        fill_value = float(level_min)
    mean_field = np.where(np.isfinite(mean_field), mean_field, fill_value).astype(np.float32)
    mean_field = _smooth_mesh_field(mean_field, edges, passes=int(max(0, smooth_passes)))
    levels = np.linspace(float(level_min), float(level_max), int(max(1, level_count)), dtype=np.float32)

    contours = []
    total_segments = 0
    for lvl in levels.tolist():
        segs_xyz: List[List[List[float]]] = []
        for tri in mesh_faces.tolist():
            ids = [int(tri[0]), int(tri[1]), int(tri[2])]
            tri_vals = mean_field[ids]
            tri_pts = mesh_vertices[ids]
            intersections: List[np.ndarray] = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                va = float(tri_vals[a])
                vb = float(tri_vals[b])
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                d0 = va - float(lvl)
                d1 = vb - float(lvl)
                if abs(d0) < 1e-7 and abs(d1) < 1e-7:
                    continue
                if (d0 > 0.0 and d1 > 0.0) or (d0 < 0.0 and d1 < 0.0):
                    continue
                denom = vb - va
                t = 0.5 if abs(denom) < 1e-8 else float(np.clip((float(lvl) - va) / denom, 0.0, 1.0))
                p = ((1.0 - t) * tri_pts[a] + t * tri_pts[b]).astype(np.float32)
                p = p / np.maximum(np.linalg.norm(p), 1e-8) * contour_radius
                is_dup = False
                for q in intersections:
                    if float(np.linalg.norm(p - q)) < 1e-5:
                        is_dup = True
                        break
                if not is_dup:
                    intersections.append(p)
            if len(intersections) == 2:
                segs_xyz.append(_slerp_segment_path(intersections[0], intersections[1], contour_radius, max_step_radians=0.05))
        if not segs_xyz:
            continue
        if float(min_segment_len) > 0.0:
            min_len2 = float(min_segment_len) ** 2
            segs_xyz = [
                path
                for path in segs_xyz
                if float(np.sum((np.asarray(path[-1], dtype=np.float32) - np.asarray(path[0], dtype=np.float32)) ** 2)) >= min_len2
            ]
        if int(segment_keep_every) > 1:
            segs_xyz = [path for idx, path in enumerate(segs_xyz) if (idx % int(segment_keep_every)) == 0]
        if not segs_xyz:
            continue
        total_segments += len(segs_xyz)
        contours.append({"level": round(float(lvl), 3), "segments": segs_xyz, "n_segments": int(len(segs_xyz))})
    return {
        "contours": contours,
        "n_segments": int(total_segments),
        "n_levels": int(level_count),
        "level_min": float(level_min),
        "level_max": float(level_max),
        "grid_size": int(mesh_vertices.shape[0]),
        "smooth_passes": int(smooth_passes),
        "field_min": float(np.min(mean_field)),
        "field_max": float(np.max(mean_field)),
    }


def build_sphere_surface_metric(
    xyz: np.ndarray,
    values: np.ndarray,
    level_min: float,
    level_max: float,
    grid_size: int = 180,
    smooth_passes: int = 3,
    high_value_boost_threshold: float = 1e9,
    high_value_boost_strength: float = 0.0,
    high_value_boost_power: float = 2.0,
    high_value_boost_cap: float | None = None,
    low_value_boost_threshold: float = -1e9,
    low_value_boost_strength: float = 0.0,
    low_value_boost_power: float = 2.0,
    sphere_cache: Dict[str, np.ndarray] | None = None,
    mesh_subdivisions: int | None = None,
) -> Dict[str, object]:
    s = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(s)
    if xyz.shape[0] == 0 or not np.any(valid):
        return {"face_values": [], "level_min": float(level_min), "level_max": float(level_max)}
    cache = sphere_cache or build_sphere_contour_cache(
        xyz,
        radius=float(max(1e-5, np.median(np.linalg.norm(xyz, axis=1)))),
        grid_size=int(grid_size),
        mesh_subdivisions=mesh_subdivisions,
    )
    if not cache:
        return {"face_values": [], "level_min": float(level_min), "level_max": float(level_max)}
    mesh_faces = np.asarray(cache["mesh_faces"], dtype=np.int32)
    neighbor_idx = np.asarray(cache["neighbor_idx"], dtype=np.int32)
    neighbor_dist = np.asarray(cache["neighbor_dist"], dtype=np.float32)
    edges = np.asarray(cache["edges"], dtype=np.int32)

    s = np.clip(s, float(level_min), float(level_max))
    w = _compute_boost_weights(
        s,
        hv_thr=float(high_value_boost_threshold),
        hv_k=float(max(0.0, high_value_boost_strength)),
        hv_p=float(max(1.0, high_value_boost_power)),
        hv_cap=high_value_boost_cap,
        lv_thr=float(low_value_boost_threshold),
        lv_k=float(max(0.0, low_value_boost_strength)),
        lv_p=float(max(1.0, low_value_boost_power)),
    )

    nbr_vals = s[neighbor_idx]
    sigma = float(np.median(neighbor_dist[:, 0])) * 1.5
    sigma = max(sigma, 1e-4)
    nbr_weights = w[neighbor_idx] * np.exp(-(neighbor_dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    nbr_valid = np.isfinite(nbr_vals)
    weighted_sum = np.sum(np.where(nbr_valid, nbr_vals * nbr_weights, 0.0), axis=1).astype(np.float32)
    weight_sum = np.sum(np.where(nbr_valid, nbr_weights, 0.0), axis=1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_field = weighted_sum / np.maximum(weight_sum, 1e-8)
    fill_value = float(np.nanmedian(s[valid]))
    if not np.isfinite(fill_value):
        fill_value = float(level_min)
    mean_field = np.where(np.isfinite(mean_field), mean_field, fill_value).astype(np.float32)
    mean_field = _smooth_mesh_field(mean_field, edges, passes=int(max(0, smooth_passes)))

    face_values = np.mean(mean_field[mesh_faces], axis=1).astype(np.float32)
    return {
        "face_values": [round(float(v), 4) for v in face_values.tolist()],
        "level_min": float(level_min),
        "level_max": float(level_max),
    }


def _boost_kwargs(metric_name: str, spec: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    """Return the boost keyword arguments for a contour/surface function call."""
    smin = float(spec["min"])  # type: ignore[arg-type]
    smax = float(spec["max"])  # type: ignore[arg-type]
    return dict(
        high_value_boost_threshold=(
            8.0 if metric_name == "star"
            else smin + 0.72 * (smax - smin)
        ),
        high_value_boost_strength=1.4 if metric_name == "star" else 0.65,
        high_value_boost_power=float(args.star_boost_power) if metric_name == "star" else 1.6,
        high_value_boost_cap=(
            float(args.star_boost_cap) if metric_name == "star" and float(args.star_boost_cap) > 0 else None
        ),
        low_value_boost_threshold=smin + 0.28 * (smax - smin) if metric_name == "bpm" else -1e9,
        low_value_boost_strength=0.65 if metric_name == "bpm" else 0.0,
        low_value_boost_power=float(args.bpm_low_boost_power) if metric_name == "bpm" else 2.0,
    )


def compute_cluster_stats(
    clusters: np.ndarray,
    metric_arrays: Dict[str, np.ndarray],
    sigma_threshold: float = 2.5,
) -> List[Dict]:
    """Per-cluster outlier-trimmed mean for each metric.

    Algorithm (per metric per cluster):
      1. Collect finite values for the cluster.
      2. Compute mean and std over all of them (first pass).
      3. Discard values outside mean ± sigma_threshold * std (outlier trim).
      4. Return the mean of the remaining values.

    Values that are NaN/Inf are ignored.  If fewer than 2 values survive the
    trim the original unfiltered mean is used as fallback.
    """
    unique_clusters = sorted(set(int(c) for c in clusters.tolist()))
    result: List[Dict] = []
    for c in unique_clusters:
        mask = clusters == int(c)
        if not np.any(mask):
            continue
        row: Dict[str, object] = {"cluster": int(c)}
        for name, arr in metric_arrays.items():
            vals = arr[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[name] = None
                continue
            mu1 = float(np.mean(vals))
            sigma1 = float(np.std(vals))
            if sigma1 > 1e-6:
                inlier = np.abs(vals - mu1) <= sigma_threshold * sigma1
                trimmed = vals[inlier]
            else:
                trimmed = vals
            # If the trim removed everything (highly unlikely), fall back to full mean.
            final_mean = float(np.mean(trimmed)) if trimmed.size > 0 else mu1
            row[name] = round(final_mean, 4)
        result.append(row)
    return result


def _auto_range(
    arr: np.ndarray,
    plo: float = 10.0,
    phi: float = 99.0,
    fallback_min: float = 0.0,
    fallback_max: float = 1.0,
) -> Tuple[float, float]:
    """Return (min, max) from percentiles of the finite values, with a safe fallback."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return (fallback_min, fallback_max)
    lo = float(np.percentile(valid, plo))
    hi = float(np.percentile(valid, phi))
    if hi <= lo:
        hi = float(np.max(valid))
        lo = float(np.min(valid))
        if hi <= lo:
            hi = lo + 1.0
    return (lo, hi)


def main() -> None:
    try:
        from CORE.paths import DEFAULT_ECHOSU_JSON, OUTPUTS_DIR
        default_run = str(OUTPUTS_DIR / "20260309_UNSUP_083351")
        default_tags = str(DEFAULT_ECHOSU_JSON)
    except ImportError:
        root = Path(__file__).resolve().parents[1]
        default_run = str(root / "CORE" / "models" / "outputs" / "20260309_UNSUP_083351")
        default_tags = str(root / "CORE" / "data" / "raw" / "tag_data_with_ids.json")

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=default_run)
    ap.add_argument("--webgl_dir", type=str, default="")
    ap.add_argument("--dataset_root", type=str, default="", help="Optional override for dataset root containing processed/maps_by_id")
    ap.add_argument("--max_points", type=int, default=0, help="0 => keep all points")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--echosu_json", type=str, default=default_tags)
    ap.add_argument("--contours", type=int, default=1, help="1 => generate offline contour overlays")
    ap.add_argument("--contour_grid", type=int, default=180, help="Grid resolution used for contour generation")
    ap.add_argument("--contour_smooth", type=int, default=3, help="Smoothing passes for 2D flat contours")
    ap.add_argument("--contour_smooth_sphere", type=int, default=6, help="Graph-Laplacian smoothing passes for 3D sphere surface/contours (higher = less streaking)")
    ap.add_argument("--star_boost_power", type=float, default=2.0, help="Power (shape) of high-SR weighting curve")
    ap.add_argument("--star_boost_cap", type=float, default=15.0, help="SR value at which the boost asymptotes; set 0 to use legacy unbounded power law")
    ap.add_argument("--bpm_low_boost_power", type=float, default=1.6, help="Power of low-BPM weighting curve")
    ap.add_argument("--perf_low_points", type=int, default=25000, help="Performance mode low-zoom point cap")
    ap.add_argument("--perf_mid_points", type=int, default=100000, help="Performance mode mid-zoom point cap")
    ap.add_argument("--perf_high_points", type=int, default=250000, help="Performance mode high-zoom point cap")
    ap.add_argument("--sphere_radius", type=float, default=1.0, help="Radius for 3D spherical projection coordinates")
    ap.add_argument("--sphere_mesh_subdivisions", type=int, default=7, help="Icosphere subdivision level for 3D contour/surface mesh")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    webgl_dir = Path(args.webgl_dir).resolve() if str(args.webgl_dir).strip() else (run_dir / "WebGL")
    data_dir = webgl_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[webgl] run_dir={run_dir}")
    dataset_root = Path(args.dataset_root).resolve() if str(args.dataset_root).strip() else infer_dataset_root(run_dir)
    print(f"[webgl] dataset_root={dataset_root}")
    emb, ids = load_embeddings(run_dir)
    if emb.shape[0] != len(ids):
        n = min(emb.shape[0], len(ids))
        print(f"[webgl] WARNING: embedding/id count mismatch ({emb.shape[0]} embeddings vs {len(ids)} ids); truncating to {n} rows.")
        emb = emb[:n]
        ids = ids[:n]

    # join cluster ids
    dfc = load_cluster_map(run_dir / "clusters.csv")
    cluster_map = dict(zip([normalize_map_id(x) for x in dfc["map_id"].tolist()], dfc["cluster"].tolist()))

    rows_idx: List[int] = []
    rows_cluster: List[int] = []
    for i, mid in enumerate(ids):
        mid_n = normalize_map_id(mid)
        c = cluster_map.get(mid_n)
        if c is None:
            continue
        rows_idx.append(i)
        rows_cluster.append(int(c))

    if not rows_idx:
        raise RuntimeError("No overlap between embeddings and clusters.csv map_ids.")

    emb = emb[np.asarray(rows_idx, dtype=np.int64)]
    ids = [normalize_map_id(ids[i]) for i in rows_idx]
    clusters = np.asarray(rows_cluster, dtype=np.int32)

    # optional downsample
    n = emb.shape[0]
    if int(args.max_points) > 0 and n > int(args.max_points):
        rng = np.random.default_rng(int(args.seed))
        pick = rng.choice(n, size=int(args.max_points), replace=False)
        emb = emb[pick]
        clusters = clusters[pick]
        ids = [normalize_map_id(ids[int(i)]) for i in pick]
        n = int(args.max_points)
    print(f"[webgl] points={n}")

    xy = reduce_2d(emb, seed=int(args.seed))
    xyz = reduce_3d_sphere(emb, seed=int(args.seed), radius=float(args.sphere_radius))
    alpha = density_alpha(xy)
    density_vals = local_density_values(xy, bins=240)         # used for 2D contours
    density_vals_3d = local_density_values_3d(xyz, n_bins_lat=120)  # used for 3D surface/contours
    tag_map = load_echosu_positive_tags(Path(args.echosu_json).resolve())
    labeled_set = set(tag_map.keys())
    labeled = np.asarray([1 if mid in labeled_set else 0 for mid in ids], dtype=np.uint8)
    meta_map = load_meta_fields(dataset_root, ids)
    ranked_known_pre = int(sum(1 for _mid, d in meta_map.items() if int(d.get("status_ranked", -1)) in (0, 1)))
    star_known_pre = int(sum(1 for _mid, d in meta_map.items() if np.isfinite(float(d.get("star", float("nan"))))))
    print(f"[webgl] metadata coverage: stars={star_known_pre}/{len(ids)} ranked={ranked_known_pre}/{len(ids)}")

    # compact array-of-arrays for browser:
    # [x,y,map_id,cluster,is_labeled,alpha,star,status_ranked,hp,od,cs,ar,bpm,length]
    points = [
        [
            float(xy[i, 0]),
            float(xy[i, 1]),
            ids[i],
            int(clusters[i]),
            int(labeled[i]),
            int(alpha[i]),
            (float(meta_map.get(ids[i], {}).get("star", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("star", float("nan")))) else None),
            int(meta_map.get(ids[i], {}).get("status_ranked", -1)),
            (float(meta_map.get(ids[i], {}).get("hp", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("hp", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("od", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("od", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("cs", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("cs", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("ar", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("ar", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("bpm", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("bpm", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("length", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("length", float("nan")))) else None),
        ]
        for i in range(n)
    ]

    # Cluster details for richer WebGL UI lists.
    cluster_details = []
    cluster_plot_labels: Dict[int, str] = {}
    unique_clusters = sorted(set(int(x) for x in clusters.tolist()))
    for c in unique_clusters:
        idxs = np.where(clusters == int(c))[0]
        if idxs.size == 0:
            continue
        n_maps = int(idxs.size)
        n_labeled = 0
        n_ranked = 0
        tag_union: set[str] = set()
        labeled_maps = []
        for i in idxs.tolist():
            mid = str(ids[int(i)])
            rank_v = int(points[int(i)][7])
            if rank_v == 1:
                n_ranked += 1
            tag_rows = tag_map.get(mid, [])
            if tag_rows:
                n_labeled += 1
                for trow in tag_rows:
                    tname = str(trow.get("name") or "")
                    if tname:
                        tag_union.add(tname)
                labeled_maps.append(
                    {
                        "map_id": mid,
                        "tags": [str(trow.get("name") or "") for trow in tag_rows if str(trow.get("name") or "").strip()],
                        "star": points[int(i)][6],
                        "status_ranked": rank_v,
                    }
                )
        labeled_maps.sort(key=lambda r: str(r["map_id"]))
        # Build requested cluster display name:
        # c{cluster}: {top2 genre | top2 pattern | top1 other}
        genre_counts: Dict[str, int] = {}
        pattern_counts: Dict[str, int] = {}
        other_counts: Dict[str, int] = {}
        for i in idxs.tolist():
            mid = str(ids[int(i)])
            tag_rows = tag_map.get(mid, [])
            for trow in tag_rows:
                tname = str(trow.get("name") or "").strip()
                if not tname:
                    continue
                cnt = int(trow.get("count") or 0)
                cat = str(trow.get("category") or "").strip().lower()
                if cat == "mapping_genre":
                    genre_counts[tname] = genre_counts.get(tname, 0) + cnt
                elif cat == "pattern_type":
                    pattern_counts[tname] = pattern_counts.get(tname, 0) + cnt
                else:
                    other_counts[tname] = other_counts.get(tname, 0) + cnt

        def _topn(d: Dict[str, int], n_top: int) -> List[str]:
            return [k for k, _v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0].lower()))[: int(n_top)]]

        genre_top = _topn(genre_counts, 2)
        pattern_top = _topn(pattern_counts, 2)
        other_top = _topn(other_counts, 1)
        display_name = f"c{int(c)}: {(' / '.join(genre_top) if genre_top else '-')} | {(' / '.join(pattern_top) if pattern_top else '-')} | {(' / '.join(other_top) if other_top else '-')}"
        ordered = list(genre_top) + list(pattern_top) + list(other_top)
        first_two = ordered[:2]
        plot_label = f"c{int(c)}: {(' / '.join(first_two) if first_two else '-')}"
        cluster_plot_labels[int(c)] = str(plot_label)

        cluster_details.append(
            {
                "cluster": int(c),
                "label": str(plot_label),
                "display_name": str(display_name),
                "n_maps": int(n_maps),
                "n_labeled": int(n_labeled),
                "n_ranked": int(n_ranked),
                "n_tags": int(len(tag_union)),
                "labeled_maps": labeled_maps,
            }
        )

    label_points = []
    # cluster centroids based on plotted points
    for c in sorted(set(int(x) for x in clusters.tolist())):
        mask = clusters == int(c)
        if not np.any(mask):
            continue
        cx = float(np.mean(xy[mask, 0]))
        cy = float(np.mean(xy[mask, 1]))
        txt = cluster_plot_labels.get(int(c), f"c{c}")
        label_points.append({"x": cx, "y": cy, "text": txt, "cluster": int(c), "n": int(np.sum(mask))})

    points_path = data_dir / "points.json"
    points3d_path = data_dir / "points3d.json"
    labels_path = data_dir / "labels.json"
    labels3d_path = data_dir / "labels3d.json"
    meta_path = data_dir / "meta.json"
    details_path = data_dir / "cluster_details.json"
    contours_path = data_dir / "contours.json"
    contours3d_path = data_dir / "contours3d.json"
    surface3d_path = data_dir / "surface3d.json"
    lods_path = data_dir / "lods.json"
    star_contours_path = data_dir / "star_contours.json"  # backward compatibility for existing viewers
    map_ids_path = data_dir / "map_ids.json"
    map_hover_path = data_dir / "map_hover.json"
    points_bin_path = data_dir / "points.f32.bin"
    points3d_bin_path = data_dir / "points3d.f32.bin"
    points_path.write_text(json.dumps(points, separators=(",", ":")), encoding="utf-8")
    points3d = [
        [
            float(xyz[i, 0]),
            float(xyz[i, 1]),
            float(xyz[i, 2]),
            ids[i],
            int(clusters[i]),
            int(labeled[i]),
            int(alpha[i]),
            (float(meta_map.get(ids[i], {}).get("star", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("star", float("nan")))) else None),
            int(meta_map.get(ids[i], {}).get("status_ranked", -1)),
            (float(meta_map.get(ids[i], {}).get("hp", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("hp", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("od", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("od", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("cs", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("cs", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("ar", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("ar", float("nan")))) else None),
            (float(meta_map.get(ids[i], {}).get("bpm", float("nan"))) if np.isfinite(float(meta_map.get(ids[i], {}).get("bpm", float("nan")))) else None),
        ]
        for i in range(n)
    ]
    points3d_path.write_text(json.dumps(points3d, separators=(",", ":")), encoding="utf-8")
    labels_path.write_text(json.dumps(label_points, ensure_ascii=False), encoding="utf-8")
    label_points_3d = []
    for c in sorted(set(int(x) for x in clusters.tolist())):
        mask = clusters == int(c)
        if not np.any(mask):
            continue
        center = np.mean(xyz[mask], axis=0).astype(np.float32)
        cn = float(np.linalg.norm(center))
        if cn > 1e-8:
            center = center / cn * float(args.sphere_radius)
        txt = cluster_plot_labels.get(int(c), f"c{c}")
        label_points_3d.append(
            {"x": float(center[0]), "y": float(center[1]), "z": float(center[2]), "text": txt, "cluster": int(c), "n": int(np.sum(mask))}
        )
    labels3d_path.write_text(json.dumps(label_points_3d, ensure_ascii=False), encoding="utf-8")
    details_path.write_text(json.dumps(cluster_details, ensure_ascii=False), encoding="utf-8")
    map_ids_path.write_text(json.dumps(ids, separators=(",", ":")), encoding="utf-8")
    map_hover = []
    for mid in ids:
        mm = meta_map.get(mid, {})
        artist = str(mm.get("artist") or "").strip()
        title = str(mm.get("title") or "").strip()
        version = str(mm.get("version") or "").strip()
        creator = str(mm.get("creator") or "").strip()
        if artist and title and version:
            disp = f"{artist} - {title} [{version}]"
        elif artist and title:
            disp = f"{artist} - {title}"
        elif title and version:
            disp = f"{title} [{version}]"
        elif title:
            disp = title
        else:
            disp = str(mid)
        map_hover.append([disp, creator])
    map_hover_path.write_text(json.dumps(map_hover, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Binary numeric payload for faster browser load on low-end hardware.
    # Row schema: [x, y, cluster, is_labeled, alpha, star, ranked, hp, od, cs, ar, bpm, length]
    points_f32 = np.zeros((n, 13), dtype=np.float32)
    points_f32[:, 0] = xy[:, 0].astype(np.float32)
    points_f32[:, 1] = xy[:, 1].astype(np.float32)
    points_f32[:, 2] = clusters.astype(np.float32)
    points_f32[:, 3] = labeled.astype(np.float32)
    points_f32[:, 4] = alpha.astype(np.float32)
    points_f32[:, 5] = np.asarray([float(p[6]) if p[6] is not None else np.nan for p in points], dtype=np.float32)
    points_f32[:, 6] = np.asarray([float(p[7]) for p in points], dtype=np.float32)
    points_f32[:, 7] = np.asarray([float(p[8]) if p[8] is not None else np.nan for p in points], dtype=np.float32)
    points_f32[:, 8] = np.asarray([float(p[9]) if p[9] is not None else np.nan for p in points], dtype=np.float32)
    points_f32[:, 9] = np.asarray([float(p[10]) if p[10] is not None else np.nan for p in points], dtype=np.float32)
    points_f32[:, 10] = np.asarray([float(p[11]) if p[11] is not None else np.nan for p in points], dtype=np.float32)
    points_f32[:, 11] = np.asarray([float(p[12]) if p[12] is not None else np.nan for p in points], dtype=np.float32)
    points_f32[:, 12] = np.asarray([float(p[13]) if p[13] is not None else np.nan for p in points], dtype=np.float32)
    points_f32.tofile(points_bin_path)
    # 3D binary payload.
    # Row schema: [x, y, z, cluster, is_labeled, alpha, star, ranked, hp, od, cs, ar, bpm, length]
    points3d_f32 = np.zeros((n, 14), dtype=np.float32)
    points3d_f32[:, 0] = xyz[:, 0].astype(np.float32)
    points3d_f32[:, 1] = xyz[:, 1].astype(np.float32)
    points3d_f32[:, 2] = xyz[:, 2].astype(np.float32)
    points3d_f32[:, 3] = clusters.astype(np.float32)
    points3d_f32[:, 4] = labeled.astype(np.float32)
    points3d_f32[:, 5] = alpha.astype(np.float32)
    points3d_f32[:, 6] = np.asarray([float(p[6]) if p[6] is not None else np.nan for p in points], dtype=np.float32)
    points3d_f32[:, 7] = np.asarray([float(p[7]) for p in points], dtype=np.float32)
    points3d_f32[:, 8] = np.asarray([float(p[8]) if p[8] is not None else np.nan for p in points], dtype=np.float32)
    points3d_f32[:, 9] = np.asarray([float(p[9]) if p[9] is not None else np.nan for p in points], dtype=np.float32)
    points3d_f32[:, 10] = np.asarray([float(p[10]) if p[10] is not None else np.nan for p in points], dtype=np.float32)
    points3d_f32[:, 11] = np.asarray([float(p[11]) if p[11] is not None else np.nan for p in points], dtype=np.float32)
    points3d_f32[:, 12] = np.asarray([float(p[12]) if p[12] is not None else np.nan for p in points], dtype=np.float32)
    points3d_f32[:, 13] = points_f32[:, 12]  # length (mirror from 2D binary)
    points3d_f32.tofile(points3d_bin_path)

    # Precomputed LOD index sets for performance mode.
    rng_lod = np.random.default_rng(int(args.seed))
    perm = np.arange(n, dtype=np.int32)
    rng_lod.shuffle(perm)
    low_n = int(min(n, max(1000, int(args.perf_low_points))))
    lod_mid_n = int(min(n, max(low_n, int(args.perf_mid_points))))
    high_n = int(min(n, max(lod_mid_n, int(args.perf_high_points))))
    lods = {
        "low": perm[:low_n].tolist(),
        "mid": perm[:lod_mid_n].tolist(),
        "high": perm[:high_n].tolist(),
        "all_n": int(n),
    }
    lods_path.write_text(json.dumps(lods, separators=(",", ":")), encoding="utf-8")

    # ── Metric arrays (needed for both cluster_stats and optional contours) ──────
    stars_arr  = np.asarray([float(p[6])  if p[6]  is not None else np.nan for p in points], dtype=np.float32)
    hp_arr     = np.asarray([float(p[8])  if p[8]  is not None else np.nan for p in points], dtype=np.float32)
    od_arr     = np.asarray([float(p[9])  if p[9]  is not None else np.nan for p in points], dtype=np.float32)
    cs_arr     = np.asarray([float(p[10]) if p[10] is not None else np.nan for p in points], dtype=np.float32)
    ar_arr     = np.asarray([float(p[11]) if p[11] is not None else np.nan for p in points], dtype=np.float32)
    bpm_arr    = np.asarray([float(p[12]) if p[12] is not None else np.nan for p in points], dtype=np.float32)
    length_arr = np.asarray([float(meta_map.get(mid, {}).get("length", float("nan"))) for mid in ids], dtype=np.float32)

    # ── Cluster stats (outlier-trimmed means, written to cluster_stats.json) ──────
    cluster_stats_path = data_dir / "cluster_stats.json"
    cluster_stats = compute_cluster_stats(
        clusters,
        {"star": stars_arr, "hp": hp_arr, "od": od_arr, "cs": cs_arr,
         "ar": ar_arr, "bpm": bpm_arr, "length": length_arr},
        sigma_threshold=2.5,
    )
    cluster_stats_path.write_text(json.dumps(cluster_stats, separators=(",", ":")), encoding="utf-8")
    print(f"[webgl] wrote {cluster_stats_path} ({len(cluster_stats)} clusters)")

    contours_info: Dict[str, object] = {"metrics": {}, "enabled_metrics": [], "total_segments": 0}
    contours3d_info: Dict[str, object] = {"metrics": {}, "enabled_metrics": [], "total_segments": 0}
    surface3d_info: Dict[str, object] = {"vertices": [], "faces": [], "metrics": {}}
    star_contour_info: Dict[str, object] = {"contours": [], "n_segments": 0}
    if int(args.contours) == 1:
        # stars_arr, hp_arr, od_arr, cs_arr, ar_arr, bpm_arr, length_arr
        # are already defined above; reused directly here.
        bpm_valid = bpm_arr[np.isfinite(bpm_arr)]
        if bpm_valid.size > 0:
            bpm_min_auto = float(np.clip(np.percentile(bpm_valid, 2.0), 30.0, 280.0))
            bpm_max_raw = float(np.clip(np.percentile(bpm_valid, 98.0), 60.0, 300.0))
            bpm_max_auto = max(bpm_min_auto + 10.0, bpm_max_raw)
            bpm_max_auto = float(min(300.0, bpm_max_auto))
            if bpm_max_auto <= bpm_min_auto:
                bpm_max_auto = float(min(300.0, bpm_min_auto + 10.0))
        else:
            bpm_min_auto = 60.0
            bpm_max_auto = 220.0

        metric_specs = {
            "star": {"values": stars_arr, "levels": 15, "min": 1.0, "max": 15.0},
            "ar": {"values": ar_arr, "levels": 11, "min": 0.0, "max": 10.0},
            "od": {"values": od_arr, "levels": 11, "min": 0.0, "max": 10.0},
            "cs": {"values": cs_arr, "levels": 11, "min": 0.0, "max": 10.0},
            "hp": {"values": hp_arr, "levels": 11, "min": 0.0, "max": 10.0},

            "bpm": {"values": bpm_arr, "levels": 16, "min": bpm_min_auto, "max": bpm_max_auto},
        }
        dmin, dmax = _auto_range(density_vals, plo=10.0, phi=99.0)
        dmin3d, dmax3d = _auto_range(density_vals_3d, plo=10.0, phi=99.0, fallback_min=dmin, fallback_max=dmax)
        metric_specs["density"] = {
            "values": density_vals,       # 2D histogram — used for flat contours only
            "values_3d": density_vals_3d, # solid-angle-normalised spherical histogram
            "levels": 14,
            "min": dmin,
            "max": dmax,
            "min_3d": dmin3d,
            "max_3d": dmax3d,
        }
        lmin, lmax = _auto_range(length_arr, plo=5.0, phi=98.0, fallback_min=30.0, fallback_max=300.0)
        metric_specs["length"] = {"values": length_arr, "levels": 14, "min": lmin, "max": lmax}

        metrics_out: Dict[str, Dict[str, object]] = {}
        sphere_cache = build_sphere_contour_cache(
            xyz,
            radius=float(args.sphere_radius),
            grid_size=int(args.contour_grid),
            mesh_subdivisions=int(args.sphere_mesh_subdivisions),
        )
        if sphere_cache:
            surface3d_info["vertices"] = np.round(np.asarray(sphere_cache["mesh_vertices"], dtype=np.float32), 4).tolist()
            surface3d_info["faces"] = np.asarray(sphere_cache["mesh_faces"], dtype=np.int32).tolist()
        enabled_metrics: List[str] = []
        total_segments = 0
        for metric_name, spec in metric_specs.items():
            info = build_value_contours(
                xy,
                np.asarray(spec["values"], dtype=np.float32),
                level_count=int(spec["levels"]),
                level_min=float(spec["min"]),
                level_max=float(spec["max"]),
                grid_size=int(args.contour_grid),
                smooth_passes=int(args.contour_smooth),
                simplify_quant_step=0.0,
                min_segment_len=0.0,
                segment_keep_every=1,
                **_boost_kwargs(metric_name, spec, args),
            )
            n_seg = int(info.get("n_segments", 0) or 0)
            metrics_out[str(metric_name)] = info
            if n_seg > 0:
                enabled_metrics.append(str(metric_name))
            total_segments += n_seg
            print(f"[webgl] contour metric={metric_name} segments={n_seg}")
            # For density use the 3D spherical histogram values so the surface
            # colours reflect actual angular density instead of the 2D PCA layout.
            sphere_values = np.asarray(
                spec.get("values_3d", spec["values"]), dtype=np.float32
            )
            sphere_lmin = float(spec.get("min_3d", spec["min"]))
            sphere_lmax = float(spec.get("max_3d", spec["max"]))
            sphere_smooth = int(args.contour_smooth_sphere)
            info3d = build_sphere_value_contours(
                xyz,
                sphere_values,
                level_count=int(spec["levels"]),
                level_min=sphere_lmin,
                level_max=sphere_lmax,
                grid_size=int(args.contour_grid),
                smooth_passes=sphere_smooth,
                simplify_quant_step=0.0,
                min_segment_len=0.0,
                segment_keep_every=1,
                **_boost_kwargs(metric_name, spec, args),
                sphere_cache=sphere_cache,
                mesh_subdivisions=int(args.sphere_mesh_subdivisions),
            )
            n_seg_3d = int(info3d.get("n_segments", 0) or 0)
            contours3d_info["metrics"][str(metric_name)] = info3d
            surface3d_info["metrics"][str(metric_name)] = build_sphere_surface_metric(
                xyz,
                sphere_values,
                level_min=sphere_lmin,
                level_max=sphere_lmax,
                grid_size=int(args.contour_grid),
                smooth_passes=sphere_smooth,
                **_boost_kwargs(metric_name, spec, args),
                sphere_cache=sphere_cache,
                mesh_subdivisions=int(args.sphere_mesh_subdivisions),
            )
            if n_seg_3d > 0:
                contours3d_info["enabled_metrics"].append(str(metric_name))
            contours3d_info["total_segments"] = int(contours3d_info.get("total_segments", 0) or 0) + n_seg_3d
            print(f"[webgl] contour3d metric={metric_name} segments={n_seg_3d}")

        contours_info = {
            "metrics": metrics_out,
            "enabled_metrics": enabled_metrics,
            "total_segments": int(total_segments),
        }
        contours_path.write_text(json.dumps(contours_info, separators=(",", ":")), encoding="utf-8")
        contours3d_path.write_text(json.dumps(contours3d_info, separators=(",", ":")), encoding="utf-8")
        surface3d_path.write_text(json.dumps(surface3d_info, separators=(",", ":")), encoding="utf-8")
        # Keep legacy file for compatibility with older frontends.
        star_contour_info = metrics_out.get("star", {"contours": [], "n_segments": 0})
        star_contours_path.write_text(json.dumps(star_contour_info, separators=(",", ":")), encoding="utf-8")
    meta = {
        "n_points": int(n),
        "schema": ["x", "y", "map_id", "cluster", "is_labeled", "alpha", "star", "status_ranked", "hp", "od", "cs", "ar", "bpm"],
        "run_dir": str(run_dir),
        "labeled_count": int(np.sum(labeled)),
        "star_known_count": int(np.sum(np.asarray([1 if p[6] is not None else 0 for p in points], dtype=np.int32))),
        "ranked_known_count": int(np.sum(np.asarray([1 if int(p[7]) in (0, 1) else 0 for p in points], dtype=np.int32))),
        "hp_known_count": int(np.sum(np.asarray([1 if p[8] is not None else 0 for p in points], dtype=np.int32))),
        "od_known_count": int(np.sum(np.asarray([1 if p[9] is not None else 0 for p in points], dtype=np.int32))),
        "cs_known_count": int(np.sum(np.asarray([1 if p[10] is not None else 0 for p in points], dtype=np.int32))),
        "ar_known_count": int(np.sum(np.asarray([1 if p[11] is not None else 0 for p in points], dtype=np.int32))),
        "bpm_known_count": int(np.sum(np.asarray([1 if p[12] is not None else 0 for p in points], dtype=np.int32))),
        "length_known_count": int(np.sum(np.asarray([1 if np.isfinite(float(meta_map.get(mid, {}).get("length", float("nan")))) else 0 for mid in ids], dtype=np.int32))),
        "contours_enabled": int(args.contours) == 1,
        "contour_total_segments": int(contours_info.get("total_segments", 0) or 0),
        "contour_metrics": list(contours_info.get("enabled_metrics", [])),
        "star_contour_segments": int(star_contour_info.get("n_segments", 0) or 0),
        "binary_points_schema": ["x", "y", "cluster", "is_labeled", "alpha", "star", "ranked", "hp", "od", "cs", "ar", "bpm", "length"],
        "binary_points_stride": 13,
        "binary_points_count": int(n),
        "binary_points3d_schema": ["x", "y", "z", "cluster", "is_labeled", "alpha", "star", "ranked", "hp", "od", "cs", "ar", "bpm", "length"],
        "binary_points3d_stride": 14,
        "binary_points3d_count": int(n),
        "contour3d_total_segments": int(contours3d_info.get("total_segments", 0) or 0),
        "contour3d_metrics": list(contours3d_info.get("enabled_metrics", [])),
        "surface3d_metrics": list(surface3d_info.get("metrics", {}).keys()),
        "sphere_radius": float(args.sphere_radius),
        "sphere_mesh_subdivisions": int(args.sphere_mesh_subdivisions),
        "lod_sizes": {"low": int(low_n), "mid": int(lod_mid_n), "high": int(high_n)},
        "generated_at": datetime.now().strftime("%d.%m.%Y"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[webgl] wrote {points_path}")
    print(f"[webgl] wrote {points_bin_path}")
    print(f"[webgl] wrote {points3d_path}")
    print(f"[webgl] wrote {points3d_bin_path}")
    print(f"[webgl] wrote {map_ids_path}")
    print(f"[webgl] wrote {map_hover_path}")
    print(f"[webgl] wrote {lods_path}")
    print(f"[webgl] wrote {labels_path}")
    print(f"[webgl] wrote {labels3d_path}")
    print(f"[webgl] wrote {details_path}")
    if int(args.contours) == 1:
        print(f"[webgl] wrote {contours_path} (segments={int(contours_info.get('total_segments', 0) or 0)})")
        print(f"[webgl] wrote {contours3d_path} (segments={int(contours3d_info.get('total_segments', 0) or 0)})")
        print(f"[webgl] wrote {surface3d_path}")
        print(f"[webgl] wrote {star_contours_path} (segments={int(star_contour_info.get('n_segments', 0) or 0)})")
    print(f"[webgl] wrote {meta_path}")


if __name__ == "__main__":
    main()

