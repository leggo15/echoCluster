from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset

# ------------------------------- Paths ---------------------------------
from CORE.paths import DEFAULT_DATASET_ROOT

DATASET_ROOT = DEFAULT_DATASET_ROOT
WINDOWS_DIR = DATASET_ROOT / "windows" / "beat_aligned"
PROCESSED_DIR = DATASET_ROOT / "processed" / "maps_by_id"
SPLITS_DIR = DATASET_ROOT / "splits"
ONTOLOGY_PATH = DATASET_ROOT / "ontology" / "tags.yml"


@dataclass
class SequenceItem:
    map_id: str
    X: np.ndarray               # shape [T, F]
    Y: np.ndarray               # shape [T, C]
    N: np.ndarray               # shape [T, C] true-negative mask (1 where explicitly negative at map-level)
    frames: pd.DataFrame        # ['map_id','start_ms','end_ms', <features...>]


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_splits() -> Dict[str, List[str]]:
    return {
        "train": _read_lines(SPLITS_DIR / "train.txt"),
        "val": _read_lines(SPLITS_DIR / "val.txt"),
        "test": _read_lines(SPLITS_DIR / "test.txt"),
    }


def load_ontology_tags() -> List[str]:
    if not ONTOLOGY_PATH.exists():
        return []
    data = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8")) or {}
    tags = [t.get("name") for t in (data.get("tags") or []) if isinstance(t, dict) and t.get("name")]
    return sorted(set(tags), key=lambda s: s.lower())


def _load_map_windows(map_id: str) -> Optional[pd.DataFrame]:
    path = WINDOWS_DIR / f"{map_id}_w4b.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df.copy()
    df["map_id"] = str(map_id)
    return df


def _load_timeseries(map_id: str) -> Optional[pd.DataFrame]:
    path = PROCESSED_DIR / str(map_id) / f"{map_id}_timeseries.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "t_ms" in df.columns:
        df = df.sort_values("t_ms").reset_index(drop=True)
    return df


def _load_meta_numeric(map_id: str) -> Dict[str, float]:
    path = PROCESSED_DIR / str(map_id) / f"{map_id}_meta.json"
    if not path.exists():
        return {}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    numeric_keys = [
        "ar","cs","od","hp","bpm","length_total","length_drain",
        "count_circles","count_sliders","count_spinners","note_count","map_end_time_ms",
        "status_ranked",
    ]
    out = {}
    for k in numeric_keys:
        v = meta.get(k)
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def _derive_slider_features(df_ts: pd.DataFrame) -> pd.DataFrame:
    """Add derived slider columns that can be computed from existing timeseries data.

    These are backward-compatible enrichments: they are only added when the
    corresponding source columns exist and the derived column is not already present.
    All derived values are NaN for non-slider rows (same convention as the raw columns).

    New columns produced:
      slider_dur_ms      — total slider duration incl. repeats (ms)
      slider_vel_beats   — slider velocity in px per beat (BPM-normalised)
      slider_dir_circ_x  — cos of slider_dir_deg (for circular stats in windows)
      slider_dir_circ_y  — sin of slider_dir_deg (for circular stats in windows)
    """
    df_ts = df_ts.copy()

    _f32_max = float(np.finfo(np.float32).max)

    # slider_dur_ms = len_px / vel  (vel already encodes repeats)
    if "slider_dur_ms" not in df_ts.columns:
        if "slider_len_px" in df_ts.columns and "slider_vel" in df_ts.columns:
            len_px = pd.to_numeric(df_ts["slider_len_px"], errors="coerce")
            vel = pd.to_numeric(df_ts["slider_vel"], errors="coerce")
            df_ts["slider_dur_ms"] = (
                (len_px / vel.where(vel > 1e-6)).clip(-_f32_max, _f32_max).astype(np.float32)
            )

    # slider_vel_beats = vel * beat_ms  (px per beat, BPM-normalised)
    if "slider_vel_beats" not in df_ts.columns:
        if "slider_vel" in df_ts.columns and "local_bpm" in df_ts.columns:
            vel = pd.to_numeric(df_ts["slider_vel"], errors="coerce")
            bpm = pd.to_numeric(df_ts["local_bpm"], errors="coerce")
            beat_ms = (60000.0 / bpm.where(bpm > 0.1))
            df_ts["slider_vel_beats"] = (
                (vel * beat_ms).clip(-_f32_max, _f32_max).astype(np.float32)
            )

    # Decompose slider_dir_deg into unit-circle components for circular window stats.
    # slider_dir_deg is only present in enriched / newly-built timeseries; skip if absent.
    if "slider_dir_deg" in df_ts.columns:
        if "slider_dir_circ_x" not in df_ts.columns:
            ang_rad = np.deg2rad(pd.to_numeric(df_ts["slider_dir_deg"], errors="coerce"))
            df_ts["slider_dir_circ_x"] = np.cos(ang_rad).astype(np.float32)
            df_ts["slider_dir_circ_y"] = np.sin(ang_rad).astype(np.float32)

    return df_ts


def _aggregate_timeseries_over_windows(df_ts: pd.DataFrame, dfw: pd.DataFrame) -> pd.DataFrame:
    if df_ts is None or df_ts.empty or dfw is None or dfw.empty:
        return pd.DataFrame(index=dfw.index)

    # Enrich with derived slider features before aggregation.
    df_ts = _derive_slider_features(df_ts)

    t = pd.to_numeric(df_ts.get("t_ms"), errors="coerce").to_numpy()
    order = np.argsort(np.nan_to_num(t, nan=-1))
    t = t[order]
    # Identify numeric columns to aggregate (exclude none to maximize coverage)
    num_cols = [c for c in df_ts.columns if pd.api.types.is_numeric_dtype(df_ts[c])]
    # Exclude circular decomposition columns from standard aggregation — they get
    # special treatment below to produce a single circular-variance feature.
    _circ_cols = {"slider_dir_circ_x", "slider_dir_circ_y"}
    num_cols_std = [c for c in num_cols if c not in _circ_cols]
    arrays: Dict[str, np.ndarray] = {}
    for c in num_cols_std:
        arrays[c] = pd.to_numeric(df_ts[c], errors="coerce").to_numpy()[order]
    obj_type = (df_ts.get("obj_type") if "obj_type" in df_ts.columns else pd.Series([None]*len(df_ts))).to_numpy()[order]
    slider_curve = (df_ts.get("slider_curve") if "slider_curve" in df_ts.columns else pd.Series([None]*len(df_ts))).to_numpy()[order]

    # Circular-direction arrays (present only when slider_dir_deg column exists).
    _has_circ = "slider_dir_circ_x" in df_ts.columns and "slider_dir_circ_y" in df_ts.columns
    if _has_circ:
        _circ_x = pd.to_numeric(df_ts["slider_dir_circ_x"], errors="coerce").to_numpy()[order]
        _circ_y = pd.to_numeric(df_ts["slider_dir_circ_y"], errors="coerce").to_numpy()[order]

    starts = pd.to_numeric(dfw["start_ms"], errors="coerce").to_numpy()
    ends = pd.to_numeric(dfw["end_ms"], errors="coerce").to_numpy()

    feat_cols: List[str] = []
    for c in num_cols_std:
        for s in ("mean","std","min","max","median"):
            feat_cols.append(f"ts_{c}_{s}")
    feat_cols += ["ts_frac_circle","ts_frac_slider","ts_frac_spinner"]
    # slider curve type fractions (L=Linear, B=Bezier, C=Catmull, P=Perfect)
    feat_cols += [
        "ts_frac_curve_L",
        "ts_frac_curve_B",
        "ts_frac_curve_C",
        "ts_frac_curve_P",
        # Shannon entropy of curve-type distribution within the window (higher = more variety)
        "ts_curve_type_entropy",
    ]
    # Circular direction variance for slider_dir_deg (0=all same direction, 1=uniform)
    if _has_circ:
        feat_cols += ["ts_slider_dir_circ_var", "ts_slider_dir_mean_deg"]

    out = np.full((dfw.shape[0], len(feat_cols)), np.nan, dtype=np.float32)
    clip_abs = 1_000_000.0

    for i in range(dfw.shape[0]):
        ws, we = int(starts[i]), int(ends[i])
        i0 = int(np.searchsorted(t, ws, side="left"))
        i1 = int(np.searchsorted(t, we, side="left"))
        if i1 <= i0:
            continue
        col_idx = 0
        for c in num_cols_std:
            arr = arrays.get(c)
            sl = arr[i0:i1]
            if sl.size == 0:
                vals = [float("nan")]*5
            else:
                valid = sl[np.isfinite(sl)]
                if valid.size == 0:
                    vals = [float("nan")]*5
                else:
                    m = float(np.mean(valid)); sdev = float(np.std(valid)); mn = float(np.min(valid)); mx = float(np.max(valid)); med = float(np.median(valid))
                    # clamp
                    m  = float(np.clip(m,  -clip_abs, clip_abs))
                    sdev= float(np.clip(sdev,-clip_abs, clip_abs))
                    mn = float(np.clip(mn, -clip_abs, clip_abs))
                    mx = float(np.clip(mx, -clip_abs, clip_abs))
                    med= float(np.clip(med,-clip_abs, clip_abs))
                    vals = [m, sdev, mn, mx, med]
            out[i, col_idx:col_idx+5] = vals
            col_idx += 5
        # Obj type fractions
        slice_types = obj_type[i0:i1]
        total = max(1, slice_types.size)
        c_cnt = np.sum(slice_types == "circle") if slice_types.size else 0
        s_cnt = np.sum(slice_types == "slider") if slice_types.size else 0
        sp_cnt = np.sum(slice_types == "spinner") if slice_types.size else 0
        # place obj type fractions
        off_types = len(feat_cols) - (8 + (2 if _has_circ else 0))
        out[i, off_types:off_types+3] = [c_cnt/total, s_cnt/total, sp_cnt/total]
        # slider curve fractions + curve entropy
        sc = slider_curve[i0:i1]
        mask_slider = (slice_types == "slider")
        slider_denom = max(1, int(np.sum(mask_slider)))
        if sc.size:
            curv = sc
            l_cnt = int(np.sum((curv == "L") & mask_slider)) if curv.size else 0
            b_cnt = int(np.sum((curv == "B") & mask_slider)) if curv.size else 0
            c_cnt2 = int(np.sum((curv == "C") & mask_slider)) if curv.size else 0
            p_cnt = int(np.sum((curv == "P") & mask_slider)) if curv.size else 0
            out[i, off_types+3:off_types+7] = [l_cnt/slider_denom, b_cnt/slider_denom, c_cnt2/slider_denom, p_cnt/slider_denom]
            # Shannon entropy over the 4 curve-type fractions
            probs = np.array([l_cnt, b_cnt, c_cnt2, p_cnt], dtype=np.float32) / float(slider_denom)
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
            out[i, off_types+7] = entropy
        else:
            out[i, off_types+3:off_types+8] = 0.0

        # Circular direction variance for slider objects in this window
        if _has_circ:
            circ_off = off_types + 8
            sx = _circ_x[i0:i1][mask_slider]
            sy = _circ_y[i0:i1][mask_slider]
            sx_valid = sx[np.isfinite(sx) & np.isfinite(sy[: sx.size])]
            sy_valid = sy[np.isfinite(sx) & np.isfinite(sy[: sx.size])]
            if sx_valid.size >= 2:
                # Circular variance: 1 − R̄  where R̄ = magnitude of mean unit vector
                mean_r = float(np.sqrt(np.mean(sx_valid)**2 + np.mean(sy_valid)**2))
                circ_var = float(np.clip(1.0 - mean_r, 0.0, 1.0))
                mean_deg = float((np.degrees(np.arctan2(float(np.mean(sy_valid)), float(np.mean(sx_valid)))) + 360.0) % 360.0)
                out[i, circ_off] = circ_var
                out[i, circ_off + 1] = mean_deg
            elif sx_valid.size == 1:
                out[i, circ_off] = 0.0
                out[i, circ_off + 1] = float((np.degrees(np.arctan2(float(sy_valid[0]), float(sx_valid[0]))) + 360.0) % 360.0)

    df_out = pd.DataFrame(out, columns=feat_cols, index=dfw.index)
    return df_out


def build_features_for_map(map_id: str, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    dfw = _load_map_windows(map_id)
    if dfw is None or dfw.empty:
        return pd.DataFrame(), ([] if feature_cols is None else feature_cols)
    dfts = _load_timeseries(map_id)
    if dfts is not None and not dfts.empty:
        dfts = _derive_slider_features(dfts)
    meta_num = _load_meta_numeric(map_id)

    # Numeric columns from windows (exclude time boundaries)
    base_num_cols = [c for c in dfw.columns if pd.api.types.is_numeric_dtype(dfw[c]) and c not in {"start_ms","end_ms"}]
    df_base = dfw[["map_id","start_ms","end_ms"] + base_num_cols].copy()

    # Aggregate ALL numeric timeseries columns over each window
    df_tsagg = _aggregate_timeseries_over_windows(dfts, dfw)

    # Meta features per row
    for k, v in meta_num.items():
        df_base[f"meta_{k}"] = float(v)

    df_full = pd.concat([df_base, df_tsagg], axis=1)

    # Determine feature columns
    if feature_cols is None:
        feature_cols = [c for c in df_full.columns if c not in {"map_id","start_ms","end_ms"} and pd.api.types.is_numeric_dtype(df_full[c])]
    # Ensure all requested feature cols present
    for c in feature_cols:
        if c not in df_full.columns:
            df_full[c] = np.nan
    df_full = df_full[["map_id","start_ms","end_ms"] + feature_cols]
    return df_full, feature_cols


def _load_spans(map_id: str) -> List[Tuple[int, int, str]]:
    path = PROCESSED_DIR / str(map_id) / f"{map_id}_spans.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    req = {"start_ms", "end_ms", "label"}
    if not req.issubset(set(df.columns)):
        return []
    rows: List[Tuple[int, int, str]] = []
    for _, r in df.iterrows():
        try:
            a = int(r["start_ms"]) if pd.notna(r["start_ms"]) else None
            b = int(r["end_ms"]) if pd.notna(r["end_ms"]) else None
            lab = str(r["label"]).strip()
        except Exception:
            continue
        if not (lab and isinstance(a, int) and isinstance(b, int) and a < b):
            continue
        rows.append((a, b, lab))
    return rows


def _load_weak_tags(map_id: str) -> List[str]:
    path = PROCESSED_DIR / str(map_id) / f"{map_id}_maplevel_tags.json"
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [k for k, v in (obj or {}).items() if v]


def _load_negative_tags(map_id: str, allowed_tags: List[str]) -> List[str]:
    """Load map-level true-negative tags from processed tags.json if available.
    Returns list of tag names (subset of allowed_tags) marked as negative for this map.
    """
    path = PROCESSED_DIR / str(map_id) / "tags.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    negs: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                name = str(k)
                if name in allowed_tags and bool((v or {}).get("is_negative", False)):
                    negs.append(name)
            except Exception:
                continue
    return negs


def label_windows_for_map(dfw: pd.DataFrame, allowed_tags: List[str]) -> Dict[int, List[str]]:
    tags_set = set(allowed_tags)
    spans = _load_spans(str(dfw["map_id"].iloc[0]))

    by_tag: Dict[str, List[Tuple[int, int]]] = {}
    for a, b, lab in spans:
        if lab in tags_set:
            by_tag.setdefault(lab, []).append((a, b))

    labels_by_row: Dict[int, List[str]] = {}
    starts = dfw["start_ms"].to_numpy()
    ends = dfw["end_ms"].to_numpy()
    for i in range(dfw.shape[0]):
        ws, we = int(starts[i]), int(ends[i])
        labs: List[str] = []
        for tag, intervals in by_tag.items():
            for a, b in intervals:
                if not (we <= a or ws >= b):
                    labs.append(tag)
                    break
        labels_by_row[i] = sorted(set(labs)) if labs else []
    return labels_by_row


def select_tags(train_Y: np.ndarray, tags: List[str]) -> Tuple[List[str], np.ndarray]:
    if train_Y.size == 0:
        return [], np.array([], dtype=int)
    pos_counts = train_Y.sum(axis=0).astype(int)
    keep_idx = np.where(pos_counts >= 5)[0]
    kept_tags = [tags[i] for i in keep_idx]
    return kept_tags, keep_idx
def probe_feature_columns(train_map_ids: List[str]) -> Tuple[List[str], Optional[str]]:
    """Find the first train map that produces features and return its feature columns.

    Returns (feature_columns, probe_map_id)
    """
    for mid in train_map_ids:
        df_full, feat_cols = build_features_for_map(mid, None)
        if df_full is not None and not df_full.empty and feat_cols:
            return feat_cols, mid
    return [], None


def build_sequences(
    map_ids: List[str],
    all_tags: List[str],
    feature_cols: List[str],
) -> List[SequenceItem]:
    tag_to_idx = {t: i for i, t in enumerate(all_tags)}
    items: List[SequenceItem] = []
    for mid in map_ids:
        df_full, _ = build_features_for_map(mid, feature_cols)
        if df_full is None or df_full.empty:
            continue
        # labels
        dfw = df_full[["map_id","start_ms","end_ms"]].copy()
        labels = label_windows_for_map(dfw, all_tags)
        Y = np.zeros((dfw.shape[0], len(all_tags)), dtype=np.float32)
        for i, labs in labels.items():
            for lab in labs:
                idx = tag_to_idx.get(lab)
                if idx is not None:
                    Y[i, idx] = 1.0
        # map-level true negatives: set negative mask for all windows at those tags
        neg_list = _load_negative_tags(str(dfw["map_id"].iloc[0]), all_tags)
        N = np.zeros((dfw.shape[0], len(all_tags)), dtype=np.float32)
        for lab in neg_list:
            j = tag_to_idx.get(lab)
            if j is not None:
                N[:, j] = 1.0
        X = df_full[feature_cols].to_numpy(dtype=np.float32)
        items.append(SequenceItem(map_id=str(mid), X=X, Y=Y.astype(np.float32), N=N, frames=df_full))
    return items


def compute_kept_tags(train_items: List[SequenceItem], all_tags: List[str], min_pos_windows: int = 5) -> Tuple[List[str], np.ndarray]:
    if not train_items:
        return [], np.array([], dtype=int)
    Y_all = np.concatenate([it.Y for it in train_items], axis=0)
    # Reuse selection rule
    kept_tags, keep_idx = select_tags(Y_all, all_tags)
    # Enforce minimum explicitly if provided
    if min_pos_windows is not None and len(kept_tags) > 0:
        pos = Y_all.sum(axis=0).astype(int)
        keep_idx = np.array([i for i in keep_idx if pos[i] >= int(min_pos_windows)], dtype=int)
        kept_tags = [all_tags[i] for i in keep_idx]
    return kept_tags, keep_idx


def standardize_inplace(items: List[SequenceItem], median: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    med = median.astype(np.float32)
    mu = mean.astype(np.float32)
    sigma = np.where((std <= 1e-8) | ~np.isfinite(std), 1.0, std).astype(np.float32)
    for it in items:
        X = it.X
        X = np.where(np.isfinite(X), X, np.nan)
        X = np.where(np.isnan(X), med[None, :], X)
        X = (X - mu[None, :]) / sigma[None, :]
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
        it.X = X


def fit_standardizer(train_items: List[SequenceItem]) -> Dict[str, List[float]]:
    if not train_items:
        return {"median": [], "mean": [], "std": []}
    Xcat = np.concatenate([it.X for it in train_items], axis=0)
    Xcat = np.where(np.isfinite(Xcat), Xcat, np.nan)
    med = np.nanmedian(Xcat, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Ximp = np.where(np.isnan(Xcat), med[None, :], Xcat)
    mu = np.nanmean(Ximp, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.nanstd(Ximp, axis=0)
    sigma = np.where((~np.isfinite(sigma)) | (sigma <= 1e-8), 1.0, sigma)
    return {"median": med.astype(float).tolist(), "mean": mu.astype(float).tolist(), "std": sigma.astype(float).tolist()}


class SequenceDataset(Dataset):
    def __init__(self, items: List[SequenceItem], keep_idx: Optional[np.ndarray] = None):
        self.items = items
        self.keep_idx = keep_idx

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        X = it.X
        Y = it.Y if self.keep_idx is None else it.Y[:, self.keep_idx]
        N = it.N if self.keep_idx is None else it.N[:, self.keep_idx]
        length = int(X.shape[0])
        return {
            "x": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(Y, dtype=torch.float32),
            "neg": torch.tensor(N, dtype=torch.float32),
            "len": torch.tensor(length, dtype=torch.long),
            "map_id": it.map_id,
            "start_ms": torch.tensor(it.frames["start_ms"].to_numpy(), dtype=torch.long),
            "end_ms": torch.tensor(it.frames["end_ms"].to_numpy(), dtype=torch.long),
        }


def pad_collate(batch: List[dict]):
    # lengths
    lengths = torch.stack([b["len"] for b in batch])
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    feat_dim = int(batch[0]["x"].shape[1]) if batch else 0
    num_tags = int(batch[0]["y"].shape[1]) if batch else 0

    x_pad = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    y_pad = torch.zeros((len(batch), max_len, num_tags), dtype=torch.float32)
    n_pad = torch.zeros((len(batch), max_len, num_tags), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    start_pad = torch.zeros((len(batch), max_len), dtype=torch.long)
    end_pad = torch.zeros((len(batch), max_len), dtype=torch.long)
    map_ids = []

    for i, b in enumerate(batch):
        L = int(b["len"].item())
        x_pad[i, :L] = b["x"]
        y_pad[i, :L] = b["y"]
        n_pad[i, :L] = b.get("neg", torch.zeros_like(b["y"]))
        mask[i, :L] = True
        start_pad[i, :L] = b["start_ms"]
        end_pad[i, :L] = b["end_ms"]
        map_ids.append(b["map_id"]) 

    return {
        "x": x_pad,
        "y": y_pad,
        "neg": n_pad,
        "mask": mask,
        "lengths": lengths,
        "map_ids": map_ids,
        "start_ms": start_pad,
        "end_ms": end_pad,
    }



