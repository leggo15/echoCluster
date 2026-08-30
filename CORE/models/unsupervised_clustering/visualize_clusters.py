from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'matplotlib'.\n\n"
        "Use the repo virtualenv and install if needed:\n"
        "  .\\env\\Scripts\\python.exe -m pip install matplotlib\n"
    ) from e


def reduce_2d(emb: np.ndarray, method: str, seed: int) -> np.ndarray:
    method = str(method).strip().lower()
    if method == "umap":
        try:
            import umap  # type: ignore
        except Exception:
            method = "pca"
        else:
            reducer = umap.UMAP(n_components=2, random_state=int(seed), n_neighbors=30, min_dist=0.15, metric="euclidean")
            return reducer.fit_transform(emb).astype(np.float32)

    if method == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            random_state=int(seed),
            perplexity=30,
            init="pca",
            learning_rate="auto",
            metric="euclidean",
        )
        return tsne.fit_transform(emb).astype(np.float32)

    # default/fallback
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=int(seed))
    return pca.fit_transform(emb).astype(np.float32)


def _density_alpha(xy: np.ndarray, bins: int = 220) -> np.ndarray:
    """Map each point to local 2D-bin density => alpha in [0.05, 0.65]."""
    x = xy[:, 0]
    y = xy[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax <= xmin or ymax <= ymin:
        return np.full((xy.shape[0],), 0.35, dtype=np.float32)

    H, xedges, yedges = np.histogram2d(x, y, bins=int(bins), range=[[xmin, xmax], [ymin, ymax]])
    ix = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, H.shape[1] - 1)
    d = H[ix, iy].astype(np.float32)
    if np.max(d) > np.min(d):
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
    else:
        d = np.zeros_like(d)
    alpha = 0.05 + 0.60 * d
    return alpha.astype(np.float32)


def _cluster_label_map(run_dir: Path, top_tag_per_cluster: int = 2) -> Dict[int, str]:
    p = run_dir / "cluster_tag_summary.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    if not {"cluster", "tag"}.issubset(set(df.columns)):
        return {}
    out: Dict[int, str] = {}
    for c in sorted(df["cluster"].dropna().astype(int).unique().tolist()):
        sub = df[df["cluster"].astype(int) == int(c)].copy()
        if "enrichment_log" in sub.columns:
            sub = sub.sort_values(["enrichment_log", "tag_count_in_cluster"], ascending=[False, False])
        else:
            sub = sub.sort_values(["tag_count_in_cluster"], ascending=[False])
        tags = [str(t) for t in sub["tag"].head(int(top_tag_per_cluster)).tolist()]
        if tags:
            out[int(c)] = f"c{c}: " + " / ".join(tags)
        else:
            out[int(c)] = f"c{c}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--method", type=str, default="umap", choices=["umap", "pca", "tsne"])
    ap.add_argument("--max_points", type=int, default=200000, help="Max points to plot (sampled if larger).")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--fig_w", type=float, default=16.0)
    ap.add_argument("--fig_h", type=float, default=11.0)
    ap.add_argument("--point_size", type=float, default=3.0)
    ap.add_argument("--annotate_top_clusters", type=int, default=35, help="Label this many largest clusters.")
    ap.add_argument("--top_tag_per_cluster", type=int, default=2)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    emb_path = run_dir / "embeddings.npy"
    clu_path = run_dir / "clusters.csv"
    if not emb_path.exists():
        raise SystemExit(f"Missing embeddings file: {emb_path}")
    if not clu_path.exists():
        raise SystemExit(f"Missing clusters file: {clu_path}")

    emb = np.load(emb_path)
    dfc = pd.read_csv(clu_path)
    if not {"map_id", "cluster"}.issubset(set(dfc.columns)):
        raise SystemExit("clusters.csv must contain columns: map_id, cluster")
    clusters = dfc["cluster"].to_numpy(dtype=np.int64)

    n = min(int(emb.shape[0]), int(clusters.shape[0]))
    emb = emb[:n]
    clusters = clusters[:n]
    if n == 0:
        raise SystemExit("No points to visualize.")

    # Optional sampling for speed/readability
    if int(args.max_points) > 0 and n > int(args.max_points):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(n, size=int(args.max_points), replace=False)
        emb = emb[idx]
        clusters = clusters[idx]
        n = int(emb.shape[0])

    print(f"[viz] run_dir={run_dir}")
    print(f"[viz] points={n} | method={args.method}")

    xy = reduce_2d(emb, method=str(args.method), seed=int(args.seed))
    alpha = _density_alpha(xy, bins=220)

    # Colors by cluster id
    uniq = np.unique(clusters)
    cmin, cmax = int(np.min(uniq)), int(np.max(uniq))
    norm = (clusters - cmin) / max(1, (cmax - cmin))
    cmap = plt.get_cmap("turbo")
    rgba = cmap(norm)
    rgba[:, 3] = alpha  # density-driven intensity

    fig, ax = plt.subplots(figsize=(float(args.fig_w), float(args.fig_h)))
    ax.scatter(xy[:, 0], xy[:, 1], s=float(args.point_size), c=rgba, linewidths=0, rasterized=True)
    ax.set_title("Unsupervised map clusters in 2D")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")

    # Annotate cluster centers (largest clusters only)
    labels = _cluster_label_map(run_dir, top_tag_per_cluster=int(args.top_tag_per_cluster))
    counts = pd.Series(clusters).value_counts().sort_values(ascending=False)
    top_clusters = [int(c) for c in counts.head(int(args.annotate_top_clusters)).index.tolist()]
    for c in top_clusters:
        mask = clusters == int(c)
        if not np.any(mask):
            continue
        cx = float(np.mean(xy[mask, 0]))
        cy = float(np.mean(xy[mask, 1]))
        txt = labels.get(int(c), f"c{c}")
        ax.text(
            cx,
            cy,
            txt,
            fontsize=8,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.2"),
        )

    out_png = run_dir / "clusters_2d.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[viz] wrote {out_png}")


if __name__ == "__main__":
    main()

