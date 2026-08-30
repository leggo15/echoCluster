from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'pandas'.\n\n"
        "Use the repo virtualenv:\n"
        "  .\\env\\Scripts\\python.exe -m CORE.models.unsupervised_clustering.elbow_k ...\n"
    ) from e
from sklearn.cluster import KMeans, MiniBatchKMeans

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'matplotlib'.\n\n"
        "Install in the repo virtualenv if needed:\n"
        "  .\\env\\Scripts\\python.exe -m pip install matplotlib\n"
    ) from e


def parse_k_values(k_values: str, k_min: int, k_max: int, k_step: int) -> List[int]:
    txt = str(k_values or "").strip()
    if txt:
        out: List[int] = []
        for token in txt.split(","):
            s = token.strip()
            if not s:
                continue
            out.append(int(s))
        out = sorted(set([k for k in out if k >= 2]))
        if not out:
            raise ValueError("No valid k values were parsed from --k_values.")
        return out
    if int(k_min) < 2:
        raise ValueError("--k_min must be >= 2")
    if int(k_max) < int(k_min):
        raise ValueError("--k_max must be >= --k_min")
    if int(k_step) <= 0:
        raise ValueError("--k_step must be > 0")
    return list(range(int(k_min), int(k_max) + 1, int(k_step)))


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
        ids.extend([str(x) for x in chunk_ids])
    if not embs:
        raise RuntimeError("No embedding chunks found under cluster_progress.")
    return np.concatenate(embs, axis=0), ids


def load_embeddings(run_dir: Path) -> Tuple[np.ndarray, List[str]]:
    emb_path = run_dir / "embeddings.npy"
    clu_path = run_dir / "clusters.csv"
    if emb_path.exists() and clu_path.exists():
        emb = np.load(emb_path).astype(np.float32)
        dfc = pd.read_csv(clu_path, usecols=["map_id"], dtype={"map_id": "string"}, keep_default_na=False)
        ids = [str(x) for x in dfc["map_id"].tolist()]
        n = min(emb.shape[0], len(ids))
        if n > 0:
            return emb[:n], ids[:n]
    return load_embeddings_from_chunks(run_dir)


def sample_embeddings(emb: np.ndarray, ids: Sequence[str], sample_size: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    n = int(emb.shape[0])
    if int(sample_size) <= 0 or n <= int(sample_size):
        return emb, list(ids)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(sample_size), replace=False)
    idx = np.asarray(idx, dtype=np.int64)
    return emb[idx], [str(ids[int(i)]) for i in idx.tolist()]


def fit_inertia(
    X: np.ndarray,
    k: int,
    *,
    use_minibatch: bool,
    batch_size: int,
    seed: int,
    max_iter: int,
    n_init: int,
) -> float:
    if bool(use_minibatch):
        km = MiniBatchKMeans(
            n_clusters=int(k),
            batch_size=int(batch_size),
            random_state=int(seed),
            n_init=int(n_init),
            max_iter=int(max_iter),
        )
    else:
        km = KMeans(
            n_clusters=int(k),
            random_state=int(seed),
            n_init=int(n_init),
            max_iter=int(max_iter),
        )
    km.fit(X)
    return float(km.inertia_)


def percent_drop(prev: float, cur: float) -> float:
    if not np.isfinite(prev) or prev <= 0:
        return 0.0
    return float((prev - cur) / prev)


def save_plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9.5, 6.0))
    ax1.plot(df["k"], df["inertia"], marker="o")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("inertia (SSE)")
    ax1.set_title("Elbow Method")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["delta_pct"], marker="x", linestyle="--")
    ax2.set_ylabel("relative inertia drop")
    ax2.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--k_values", type=str, default="", help="Comma-separated list, e.g. 200,400,800,1200")
    ap.add_argument("--k_min", type=int, default=100)
    ap.add_argument("--k_max", type=int, default=3000)
    ap.add_argument("--k_step", type=int, default=100)
    ap.add_argument("--sample_size", type=int, default=200000, help="0 => use all embeddings")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--minibatch", action="store_true", help="Use MiniBatchKMeans for speed")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--n_init", type=int, default=5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Missing run dir: {run_dir}")

    k_list = parse_k_values(str(args.k_values), int(args.k_min), int(args.k_max), int(args.k_step))
    emb, ids = load_embeddings(run_dir)
    if emb.shape[0] <= 2:
        raise SystemExit("Not enough embeddings to run elbow method.")

    X, ids_s = sample_embeddings(emb, ids, int(args.sample_size), int(args.seed))
    print(f"[elbow] run_dir={run_dir}")
    print(f"[elbow] embeddings={emb.shape[0]} sample={X.shape[0]} dim={X.shape[1]}")
    print(f"[elbow] k_values={k_list}")

    rows = []
    prev_inertia = float("nan")
    for k in k_list:
        inertia = fit_inertia(
            X,
            int(k),
            use_minibatch=bool(args.minibatch),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            max_iter=int(args.max_iter),
            n_init=int(args.n_init),
        )
        drop = percent_drop(prev_inertia, inertia)
        rows.append(
            {
                "k": int(k),
                "inertia": float(inertia),
                "delta_abs": (float(prev_inertia - inertia) if np.isfinite(prev_inertia) else float("nan")),
                "delta_pct": float(drop),
                "n_points": int(X.shape[0]),
                "dim": int(X.shape[1]),
                "algo": ("MiniBatchKMeans" if bool(args.minibatch) else "KMeans"),
            }
        )
        prev_inertia = float(inertia)
        print(f"[elbow] k={k:5d} inertia={inertia:.6e} drop={drop:.4%}")

    df = pd.DataFrame(rows)
    out_csv = run_dir / "elbow_k.csv"
    out_json = run_dir / "elbow_k.json"
    out_png = run_dir / "elbow_k.png"
    df.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "k_values": [int(k) for k in k_list],
                "sample_size": int(X.shape[0]),
                "embedding_count": int(emb.shape[0]),
                "dim": int(X.shape[1]),
                "algorithm": ("MiniBatchKMeans" if bool(args.minibatch) else "KMeans"),
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_plot(df, out_png)

    # Heuristic suggestion: first k where relative drop falls below 5%.
    suggested_k = None
    for _, r in df.iterrows():
        if float(r.get("delta_pct", 1.0)) < 0.05:
            suggested_k = int(r["k"])
            break
    if suggested_k is None and len(df) > 0:
        suggested_k = int(df.iloc[int(np.argmax(df["delta_pct"].fillna(0.0).to_numpy()))]["k"])

    print(f"[elbow] wrote {out_csv}")
    print(f"[elbow] wrote {out_json}")
    print(f"[elbow] wrote {out_png}")
    if suggested_k is not None:
        print(f"[elbow] heuristic_k={suggested_k} (manual review recommended)")
    print(f"[elbow] sample_first_ids={ids_s[:5]}")


if __name__ == "__main__":
    main()

