from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from CORE.paths import DEFAULT_DATASET_ROOT

from .dataset import (
    ContrastiveMapDataset,
    collate_contrastive,
    configure_embedding_dataset_root,
    fit_standardizer_from_maps,
    list_map_ids_from_processed,
    save_run_artifacts,
)
from .model import EncoderConfig, GRUEncoder, info_nce_loss


OUTPUTS_DIR = (Path(__file__).resolve().parents[1] / "outputs").resolve()


def ensure_run_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now():%Y%m%d}_UNSUP_{datetime.now():%H%M%S}"
    rd = OUTPUTS_DIR / run_id
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def probe_feature_columns(map_ids: List[str]) -> List[str]:
    """Use the existing feature builder to infer feature columns from a map."""
    from CORE.models.embedding_model.dataset import build_features_for_map

    for mid in map_ids:
        try:
            df, cols = build_features_for_map(str(mid), None)
        except Exception:
            continue
        if df is not None and not df.empty and cols:
            return list(cols)
    return []


def main() -> None:
    # Hide warning spam from large numeric pipelines.
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")

    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--crop_len", type=int, default=256)
    p.add_argument("--feat_dropout", type=float, default=0.10)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--max_maps", type=int, default=20000)
    p.add_argument("--fit_norm_maps", type=int, default=2000)
    p.add_argument("--fit_norm_max_windows_per_map", type=int, default=64)
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    dataset_root = Path(args.dataset_root).resolve()
    processed_dir = dataset_root / "processed" / "maps_by_id"
    configure_embedding_dataset_root(dataset_root)

    all_ids = list_map_ids_from_processed(processed_dir)
    if not all_ids:
        raise RuntimeError(f"No map directories found under {processed_dir}")

    # Keep only maps that already have the expected window parquet.
    windows_dir = dataset_root / "windows" / "beat_aligned"
    window_ids = set()
    if windows_dir.exists():
        for fp in windows_dir.glob("*_w4b.parquet"):
            stem = fp.stem
            if stem.endswith("_w4b"):
                window_ids.add(stem[:-4])
    map_ids_full = [mid for mid in all_ids if mid in window_ids]
    if not map_ids_full:
        raise RuntimeError(f"No usable *_w4b.parquet files found under {windows_dir}")

    # Optional cap for faster iteration
    if int(args.max_maps) > 0 and len(map_ids_full) > int(args.max_maps):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(map_ids_full), size=int(args.max_maps), replace=False)
        map_ids = [map_ids_full[i] for i in idx]
    else:
        map_ids = map_ids_full

    feature_cols = probe_feature_columns(map_ids)
    if not feature_cols:
        raise RuntimeError("Could not probe feature columns from any map.")

    st = fit_standardizer_from_maps(
        map_ids,
        feature_cols,
        sample_maps=int(args.fit_norm_maps),
        max_windows_per_map=int(args.fit_norm_max_windows_per_map),
        seed=int(args.seed),
    )

    ds = ContrastiveMapDataset(
        map_ids,
        feature_cols,
        st,
        crop_len=int(args.crop_len),
        feat_dropout=float(args.feat_dropout),
        noise_std=float(args.noise_std),
        seed=int(args.seed),
    )
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_contrastive)

    device = torch.device(str(args.device))
    enc_cfg = EncoderConfig(
        input_dim=len(feature_cols),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        proj_dim=int(args.proj_dim),
    )
    model = GRUEncoder(enc_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_dir = ensure_run_dir()
    config: Dict[str, object] = {
        "dataset_root": str(dataset_root),
        "dataset_processed_dir": str(processed_dir),
        "n_maps": int(len(map_ids)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "temperature": float(args.temperature),
        "crop_len": int(args.crop_len),
        "feat_dropout": float(args.feat_dropout),
        "noise_std": float(args.noise_std),
        "seed": int(args.seed),
        "encoder": asdict(enc_cfg),
    }
    save_run_artifacts(run_dir, feature_cols=feature_cols, standardizer=st, config=config)

    print(f"[unsup] run_dir={run_dir}")
    print(f"[unsup] maps={len(map_ids)} (from {len(all_ids)} dirs) | features={len(feature_cols)} | device={device}")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        skipped_batches = 0
        nonfinite_batches = 0
        for batch in dl:
            x1 = batch["x1"].to(device, non_blocking=True)
            x2 = batch["x2"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)

            # collate_contrastive emits a tiny dummy tensor when an entire batch is invalid
            # (e.g. maps missing usable features). Skip those safely.
            if int(x1.shape[-1]) != int(len(feature_cols)) or int(x2.shape[-1]) != int(len(feature_cols)):
                skipped_batches += 1
                continue
            if x1.shape[0] <= 0:
                skipped_batches += 1
                continue

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=(device.type == "cuda"), dtype=torch.float16):
                z1, _ = model(x1, lengths)
                z2, _ = model(x2, lengths)
                loss = info_nce_loss(z1, z2, temperature=float(args.temperature))

            if not torch.isfinite(loss):
                nonfinite_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"[unsup] epoch {epoch:02d}/{int(args.epochs)} | "
            f"loss={mean_loss:.5f} | skipped_batches={skipped_batches} | nonfinite_batches={nonfinite_batches}"
        )

        # Save checkpoint each epoch (small model)
        ckpt = {
            "encoder_config": asdict(enc_cfg),
            "feature_cols": feature_cols,
            "preprocess": st.to_jsonable(),
            "model": model.state_dict(),
            "epoch": int(epoch),
            "loss": float(mean_loss),
        }
        torch.save(ckpt, run_dir / "model.pt")

    metrics = {"final_loss": float(mean_loss)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[unsup] done")


if __name__ == "__main__":
    main()

