from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'pandas'.\n\n"
        "You are likely running the wrong Python interpreter.\n"
        "Use the repo virtualenv:\n"
        "  .\\env\\Scripts\\python.exe -m CORE.models.unsupervised_clustering.train ...\n"
        "or activate it:\n"
        "  .\\env\\Scripts\\Activate.ps1\n"
    ) from e
import torch
from torch.utils.data import Dataset

import CORE.models.embedding_model.dataset as emb_dataset
from CORE.models.embedding_model.dataset import build_features_for_map
from CORE.paths import DEFAULT_DATASET_ROOT

DATASET_ROOT = DEFAULT_DATASET_ROOT
PROCESSED_DIR = DATASET_ROOT / "processed" / "maps_by_id"


def configure_embedding_dataset_root(dataset_root: Path) -> None:
    """Point embedding_model.dataset helpers at a custom dataset root."""
    root = Path(dataset_root).resolve()
    emb_dataset.DATASET_ROOT = root
    emb_dataset.WINDOWS_DIR = root / "windows" / "beat_aligned"
    emb_dataset.PROCESSED_DIR = root / "processed" / "maps_by_id"
    emb_dataset.SPLITS_DIR = root / "splits"
    emb_dataset.ONTOLOGY_PATH = root / "ontology" / "tags.yml"


def list_map_ids_from_processed(processed_dir: Path = PROCESSED_DIR) -> List[str]:
    if not processed_dir.exists():
        return []
    out = []
    for d in processed_dir.iterdir():
        if d.is_dir():
            out.append(d.name)
    # Stable sort numeric-ish
    def _k(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)
    out.sort(key=_k)
    return out


@dataclass
class Standardizer:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def to_jsonable(self) -> Dict[str, List[float]]:
        return {
            "median": self.median.astype(float).tolist(),
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }

    @staticmethod
    def from_jsonable(obj: Dict) -> "Standardizer":
        med = np.asarray(obj.get("median", []), dtype=np.float32)
        mean = np.asarray(obj.get("mean", []), dtype=np.float32)
        std = np.asarray(obj.get("std", []), dtype=np.float32)
        return Standardizer(median=med, mean=mean, std=std)


def fit_standardizer_from_maps(
    map_ids: List[str],
    feature_cols: List[str],
    *,
    sample_maps: int = 2000,
    max_windows_per_map: int = 64,
    seed: int = 2025,
) -> Standardizer:
    rng = np.random.default_rng(int(seed))
    ids = list(map_ids)
    if sample_maps > 0 and len(ids) > sample_maps:
        ids = [ids[i] for i in rng.choice(len(ids), size=int(sample_maps), replace=False)]

    chunks: List[np.ndarray] = []
    for mid in ids:
        try:
            df, _ = build_features_for_map(str(mid), feature_cols)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        X = df[feature_cols].to_numpy(dtype=np.float32)
        X = np.where(np.isfinite(X), X, np.nan)
        if max_windows_per_map and X.shape[0] > int(max_windows_per_map):
            idx = rng.choice(X.shape[0], size=int(max_windows_per_map), replace=False)
            X = X[idx]
        chunks.append(X)

    if not chunks:
        # fallback: identity-ish
        F = len(feature_cols)
        return Standardizer(
            median=np.zeros((F,), dtype=np.float32),
            mean=np.zeros((F,), dtype=np.float32),
            std=np.ones((F,), dtype=np.float32),
        )

    Xcat = np.concatenate(chunks, axis=0)
    clip_abs = 1_000_000.0
    Xcat = np.clip(Xcat, -clip_abs, clip_abs)
    all_nan_cols = np.all(np.isnan(Xcat), axis=0)
    if np.any(all_nan_cols):
        Xcat[:, all_nan_cols] = 0.0
    med = np.nanmedian(Xcat, axis=0).astype(np.float32)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    Ximp = np.where(np.isnan(Xcat), med[None, :], Xcat)
    mean = np.nanmean(Ximp, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.nanstd(Ximp, axis=0).astype(np.float32)
    std = np.where((~np.isfinite(std)) | (std <= 1e-8), 1.0, std).astype(np.float32)
    return Standardizer(median=med, mean=mean, std=std)


def standardize(X: np.ndarray, st: Standardizer) -> np.ndarray:
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    med = st.median.astype(np.float32)
    mu = st.mean.astype(np.float32)
    sd = np.where((st.std <= 1e-8) | ~np.isfinite(st.std), 1.0, st.std).astype(np.float32)
    X = np.where(np.isnan(X), med[None, :], X)
    X = (X - mu[None, :]) / sd[None, :]
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    # Keep dynamic range bounded to avoid fp16 overflow in encoder matmuls.
    X = np.clip(X, -50.0, 50.0).astype(np.float32)
    return X


def augment_view(
    X: np.ndarray,
    *,
    crop_len: int,
    rng: np.random.Generator,
    feat_dropout: float = 0.10,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Create one view for contrastive training.

    - random contiguous crop (or pad if too short)
    - feature dropout (column-wise mask)
    - small gaussian noise
    """
    T, F = X.shape
    L = int(crop_len)
    if L <= 0:
        raise ValueError("crop_len must be > 0")

    if T <= 0:
        out = np.zeros((L, F), dtype=np.float32)
    elif T >= L:
        start = int(rng.integers(0, T - L + 1))
        out = X[start : start + L].copy()
    else:
        out = np.zeros((L, F), dtype=np.float32)
        out[:T] = X

    if feat_dropout and feat_dropout > 0:
        keep = rng.random(F) >= float(feat_dropout)
        out[:, ~keep] = 0.0

    if noise_std and noise_std > 0:
        out = out + rng.normal(0.0, float(noise_std), size=out.shape).astype(np.float32)

    return out.astype(np.float32)


class ContrastiveMapDataset(Dataset):
    """Returns two augmented views of the same map's window-feature sequence."""

    def __init__(
        self,
        map_ids: List[str],
        feature_cols: List[str],
        standardizer: Standardizer,
        *,
        crop_len: int = 256,
        feat_dropout: float = 0.10,
        noise_std: float = 0.01,
        seed: int = 2025,
    ):
        self.map_ids = list(map_ids)
        self.feature_cols = list(feature_cols)
        self.st = standardizer
        self.crop_len = int(crop_len)
        self.feat_dropout = float(feat_dropout)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.map_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        mid = self.map_ids[idx]
        df, _ = build_features_for_map(str(mid), self.feature_cols)
        if df is None or df.empty:
            # return dummy; collate drops invalid
            return {"map_id": str(mid), "x1": None, "x2": None}
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        X = standardize(X, self.st)

        rng = np.random.default_rng(self.seed + int(idx) * 9973)
        x1 = augment_view(X, crop_len=self.crop_len, rng=rng, feat_dropout=self.feat_dropout, noise_std=self.noise_std)
        x2 = augment_view(X, crop_len=self.crop_len, rng=rng, feat_dropout=self.feat_dropout, noise_std=self.noise_std)

        return {
            "map_id": str(mid),
            "x1": torch.tensor(x1, dtype=torch.float32),
            "x2": torch.tensor(x2, dtype=torch.float32),
            "len": torch.tensor(int(self.crop_len), dtype=torch.long),
        }


def collate_contrastive(batch: List[Dict[str, object]]) -> Dict[str, object]:
    batch = [b for b in batch if isinstance(b.get("x1"), torch.Tensor) and isinstance(b.get("x2"), torch.Tensor)]
    if not batch:
        return {"map_id": [], "x1": torch.zeros((1, 1, 1)), "x2": torch.zeros((1, 1, 1)), "lengths": torch.tensor([1])}
    x1 = torch.stack([b["x1"] for b in batch], dim=0)  # type: ignore[list-item]
    x2 = torch.stack([b["x2"] for b in batch], dim=0)  # type: ignore[list-item]
    lengths = torch.stack([b["len"] for b in batch], dim=0)  # type: ignore[list-item]
    mids = [str(b["map_id"]) for b in batch]
    return {"map_id": mids, "x1": x1, "x2": x2, "lengths": lengths}


def save_run_artifacts(run_dir: Path, *, feature_cols: List[str], standardizer: Standardizer, config: Dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_columns.json").write_text(json.dumps(list(feature_cols), indent=2), encoding="utf-8")
    (run_dir / "preprocess.json").write_text(json.dumps(standardizer.to_jsonable(), indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

