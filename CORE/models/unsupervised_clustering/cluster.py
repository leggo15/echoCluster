from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'pandas'.\n\n"
        "You are likely running the wrong Python interpreter.\n"
        "Use the repo virtualenv:\n"
        "  .\\env\\Scripts\\python.exe -m CORE.models.unsupervised_clustering.cluster ...\n"
    ) from e
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from CORE.paths import DEFAULT_DATASET_ROOT

from .dataset import Standardizer, configure_embedding_dataset_root, list_map_ids_from_processed, standardize
from .model import EncoderConfig, GRUEncoder


# ---------------------------------------------------------------------------
# Agglomerative clustering (two-stage: over-cluster → Ward merge)
# ---------------------------------------------------------------------------

def fit_agglomerative(
    embs: np.ndarray,
    k: int,
    pre_k: int,
    linkage_method: str,
    batch_size: int,
    linkage_out_path: Path | None = None,
) -> np.ndarray:
    """Two-stage agglomerative clustering for large datasets.

    Stage 1 — Over-cluster with MiniBatch K-Means (pre_k >> k).
    Stage 2 — Merge the pre_k centroids with hierarchical Ward linkage, cut at k.

    The full linkage matrix is saved to *linkage_out_path* (if given) so the
    dendrogram can be re-cut at a different k without any re-run.
    """
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore

    n = embs.shape[0]
    actual_pre_k = min(pre_k, n)
    actual_k     = min(k,     actual_pre_k)

    print(f"[cluster/agg] stage-1: MiniBatchKMeans pre_k={actual_pre_k} …")
    pre_km = MiniBatchKMeans(
        n_clusters=actual_pre_k,
        batch_size=int(batch_size),
        random_state=2025,
        n_init="auto",
    )
    pre_labels = pre_km.fit_predict(embs)
    centroids  = pre_km.cluster_centers_.astype(np.float64)
    print(f"[cluster/agg] stage-1 done. centroids shape={centroids.shape}")

    print(f"[cluster/agg] stage-2: {linkage_method} linkage on {actual_pre_k} centroids …")
    Z = linkage(centroids, method=linkage_method, metric="euclidean")
    print(f"[cluster/agg] stage-2 done.")

    if linkage_out_path is not None:
        np.save(linkage_out_path, Z.astype(np.float64))
        print(f"[cluster/agg] linkage matrix saved → {linkage_out_path}")

    # Cut the dendrogram at k clusters and map each point through its proto-cluster.
    proto_to_final = fcluster(Z, t=actual_k, criterion="maxclust") - 1  # 0-indexed
    labels = proto_to_final[pre_labels]
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Recursive agglomerative: HDBSCAN within proto-clusters → Ward merge
# ---------------------------------------------------------------------------

def fit_recursive_agglomerative(
    embs: np.ndarray,
    pre_k: int,
    min_cluster_size: int,
    k: int,
    final_threshold: float,
    batch_size: int,
    linkage_method: str,
    max_stage3: int,
    linkage_out_path: Path | None = None,
) -> np.ndarray:
    """Three-stage clustering for large datasets with natural cluster sizes.

    Stage 1 — MiniBatch K-Means into pre_k proto-clusters.
    Stage 2 — HDBSCAN within each proto-cluster: finds dense sub-groups and
               marks genuinely isolated maps as noise (→ singleton clusters).
    Stage 3 — Ward agglomerative on all sub-cluster centroids + singletons
               collected across all proto-clusters. Representatives from
               different proto-clusters merge freely at this stage, so partial
               overlaps between adjacent proto-clusters and proto-clusters
               whose noise has been expelled can still join up.
               If the representative count exceeds max_stage3, a consolidation
               MiniBatch K-Means pass is applied first to keep Ward tractable.

    Cut the final dendrogram via distance threshold (final_threshold > 0) or
    by fixing the cluster count (final_threshold == 0, uses --k).
    """
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore

    try:
        from sklearn.cluster import HDBSCAN as _HDBSCAN  # type: ignore
    except ImportError:
        raise ImportError(
            "HDBSCAN requires scikit-learn >= 1.3. "
            "Upgrade: pip install -U scikit-learn"
        )

    n = embs.shape[0]
    actual_pre_k = min(pre_k, n)

    # --- Stage 1 ---
    print(f"[cluster/rec] stage-1: MiniBatchKMeans pre_k={actual_pre_k} …")
    pre_km = MiniBatchKMeans(
        n_clusters=actual_pre_k, batch_size=batch_size, random_state=2025, n_init="auto"
    )
    proto_labels = pre_km.fit_predict(embs)
    print(f"[cluster/rec] stage-1 done.")

    # --- Stage 2: HDBSCAN within each proto-cluster ---
    print(
        f"[cluster/rec] stage-2: HDBSCAN per proto-cluster "
        f"(min_cluster_size={min_cluster_size}) …"
    )
    rep_centroids: List[np.ndarray] = []
    rep_point_lists: List[List[int]] = []  # original indices per representative

    for proto_id in range(actual_pre_k):
        ptrs = np.where(proto_labels == proto_id)[0]
        if len(ptrs) == 0:
            continue

        sub_embs = embs[ptrs]

        # Proto-cluster too small to run HDBSCAN meaningfully — keep as singletons.
        if len(ptrs) < max(2, min_cluster_size):
            for j, orig_idx in enumerate(ptrs):
                rep_centroids.append(sub_embs[j])
                rep_point_lists.append([int(orig_idx)])
            continue

        hdb = _HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=1, n_jobs=-1)
        sub_labels = hdb.fit_predict(sub_embs)

        # Dense sub-clusters (label >= 0): one centroid per sub-cluster.
        for lbl in set(sub_labels) - {-1}:
            sm = np.where(sub_labels == lbl)[0]
            rep_centroids.append(sub_embs[sm].mean(axis=0))
            rep_point_lists.append(ptrs[sm].tolist())

        # Noise points (label == -1): each becomes its own singleton representative.
        for j in np.where(sub_labels == -1)[0]:
            rep_centroids.append(sub_embs[j])
            rep_point_lists.append([int(ptrs[j])])

        if (proto_id + 1) % 500 == 0:
            print(
                f"[cluster/rec] stage-2: {proto_id+1}/{actual_pre_k} proto-clusters done, "
                f"{len(rep_centroids)} representatives so far"
            )

    n_rep = len(rep_centroids)
    print(f"[cluster/rec] stage-2 done. {n_rep} representatives (sub-clusters + singletons).")
    rep_arr = np.stack(rep_centroids, axis=0).astype(np.float32)

    # --- Stage 3 consolidation (only if reps exceed Ward memory budget) ---
    # Ward on n_rep points needs ~n_rep²/2 float64 entries. At 20 000 reps
    # that is ~3.2 GB which is comfortable; above ~30 000 it becomes risky.
    if n_rep > max_stage3:
        print(
            f"[cluster/rec] stage-3 consolidation: {n_rep} reps → {max_stage3} "
            f"via MiniBatchKMeans …"
        )
        consol_km = MiniBatchKMeans(
            n_clusters=max_stage3, batch_size=batch_size, random_state=2025, n_init="auto"
        )
        consol_labels = consol_km.fit_predict(rep_arr)
        new_rep_indices: List[List[int]] = [[] for _ in range(max_stage3)]
        for ri, cl in enumerate(consol_labels):
            new_rep_indices[cl].extend(rep_point_lists[ri])
        rep_arr = consol_km.cluster_centers_.astype(np.float32)
        rep_point_lists = new_rep_indices
        print(f"[cluster/rec] stage-3 consolidation done. {max_stage3} consolidated reps.")

    # --- Ward agglomerative on final representative set ---
    n_final_rep = rep_arr.shape[0]
    print(f"[cluster/rec] Ward linkage on {n_final_rep} representatives …")
    Z = linkage(rep_arr.astype(np.float64), method=linkage_method, metric="euclidean")
    print(f"[cluster/rec] Ward linkage done.")

    if linkage_out_path is not None:
        np.save(linkage_out_path, Z.astype(np.float64))
        print(f"[cluster/rec] linkage matrix saved → {linkage_out_path}")

    if final_threshold > 0.0:
        cut_labels = fcluster(Z, t=final_threshold, criterion="distance") - 1
        n_final = len(set(cut_labels.tolist()))
        print(f"[cluster/rec] distance cut={final_threshold:.4f} → {n_final} final clusters")
    else:
        actual_k = min(k, n_final_rep)
        cut_labels = fcluster(Z, t=actual_k, criterion="maxclust") - 1
        n_final = len(set(cut_labels.tolist()))
        print(f"[cluster/rec] maxclust k={actual_k} → {n_final} final clusters")

    # Map each original point through its representative to the final cluster label.
    labels = np.empty(n, dtype=np.int32)
    for ri, orig_indices in enumerate(rep_point_lists):
        fl = int(cut_labels[ri])
        for oi in orig_indices:
            labels[oi] = fl

    return labels


def load_run(run_dir: Path) -> Tuple[GRUEncoder, List[str], Standardizer, Dict]:
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    cfg = state.get("encoder_config") or {}
    enc_cfg = EncoderConfig(
        input_dim=int(cfg.get("input_dim", len(state.get("feature_cols") or []))),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        proj_dim=int(cfg.get("proj_dim", 128)),
    )
    model = GRUEncoder(enc_cfg)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    feature_cols = state.get("feature_cols") or json.loads((run_dir / "feature_columns.json").read_text(encoding="utf-8"))
    st = Standardizer.from_jsonable(state.get("preprocess") or json.loads((run_dir / "preprocess.json").read_text(encoding="utf-8")))
    return model, list(feature_cols), st, state


@torch.no_grad()
def embed_map(model: GRUEncoder, X: np.ndarray, device: torch.device) -> np.ndarray:
    # Use full sequence (capped for safety) and mean pooling via attention inside the model.
    T = int(X.shape[0])
    if T <= 0:
        return np.zeros((model.cfg.proj_dim,), dtype=np.float32)
    x = torch.tensor(X[None, ...], dtype=torch.float32, device=device)
    lengths = torch.tensor([T], dtype=torch.long, device=device)
    z, _ = model(x, lengths)
    return z[0].detach().cpu().numpy().astype(np.float32)


def _patch_embed(args) -> None:
    """Embed a small set of maps and assign them to the nearest existing cluster.

    Reads: run_dir/embeddings.npy, run_dir/clusters.csv, run_dir/model.pt
    Writes: updated embeddings.npy, updated clusters.csv, new chunk in cluster_progress/
    """
    run_dir = Path(args.run_dir).resolve()
    print(f"[cluster/patch] run_dir={run_dir}")

    # Parse the supplied IDs
    raw = str(args.map_ids).replace(" ", ",").split(",")
    target_ids: List[str] = []
    for r in raw:
        r = r.strip()
        if r:
            try:
                target_ids.append(str(int(r)))
            except ValueError:
                print(f"[cluster/patch] ignoring non-integer id '{r}'")
    if not target_ids:
        print("[cluster/patch] no valid map IDs; nothing to do.")
        return
    print(f"[cluster/patch] will embed {len(target_ids)} map(s): {target_ids[:10]}{'…' if len(target_ids)>10 else ''}")

    # Load existing clusters.csv + embeddings.npy
    clusters_csv = run_dir / "clusters.csv"
    emb_npy      = run_dir / "embeddings.npy"
    if not clusters_csv.exists() or not emb_npy.exists():
        raise FileNotFoundError(
            "clusters.csv / embeddings.npy not found in run_dir — "
            "run a full cluster pass first before using --map_ids."
        )
    df_existing   = pd.read_csv(clusters_csv, dtype={"map_id": str, "cluster": int})
    embs_existing = np.load(emb_npy).astype(np.float32)
    print(f"[cluster/patch] loaded {len(df_existing)} existing maps, {embs_existing.shape}")

    # Compute per-cluster centroids from existing embeddings.
    # Agglomerative runs don't save cluster_centroids.npy, so we always
    # recompute here to be safe.
    labels_existing = df_existing["cluster"].to_numpy(dtype=np.int32)
    n_clusters      = int(labels_existing.max()) + 1
    dim             = embs_existing.shape[1]
    centroids       = np.zeros((n_clusters, dim), dtype=np.float32)
    counts          = np.zeros(n_clusters, dtype=np.int32)
    for c in range(n_clusters):
        mask = labels_existing == c
        if mask.any():
            centroids[c] = embs_existing[mask].mean(axis=0)
            counts[c]    = int(mask.sum())
    print(f"[cluster/patch] computed {n_clusters} cluster centroids")

    # Load model
    model, feature_cols, st, _state = load_run(run_dir)
    device = torch.device(str(args.device))
    model  = model.to(device)

    if str(args.dataset_root).strip():
        dataset_root = Path(str(args.dataset_root)).resolve()
    else:
        cfg: Dict = {}
        try:
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        dataset_root = Path(cfg.get("dataset_root", str(DEFAULT_DATASET_ROOT))).resolve()
    configure_embedding_dataset_root(dataset_root)

    from CORE.models.embedding_model.dataset import build_features_for_map

    # Embed each requested map
    new_ids:  List[str]       = []
    new_embs: List[np.ndarray] = []
    for mid in target_ids:
        try:
            df, _ = build_features_for_map(str(mid), feature_cols)
        except Exception as e:
            print(f"[cluster/patch] {mid}: feature build failed ({e}); skipping")
            continue
        if df is None or df.empty:
            print(f"[cluster/patch] {mid}: empty features; skipping")
            continue
        X = df[feature_cols].to_numpy(dtype=np.float32)
        X = standardize(X, st)
        z = embed_map(model, X, device)
        new_ids.append(str(mid))
        new_embs.append(z)
        print(f"[cluster/patch] embedded {mid}")

    if not new_ids:
        print("[cluster/patch] no maps successfully embedded; nothing written.")
        return

    new_embs_arr = np.stack(new_embs, axis=0).astype(np.float32)

    # Assign to nearest centroid (cosine-like: L2 on unit-normed embeddings)
    norms   = np.linalg.norm(new_embs_arr, axis=1, keepdims=True)
    emb_n   = new_embs_arr / np.where(norms > 1e-9, norms, 1.0)
    cnorm   = np.linalg.norm(centroids, axis=1, keepdims=True)
    cent_n  = centroids / np.where(cnorm > 1e-9, cnorm, 1.0)
    sims    = emb_n @ cent_n.T                  # (n_new, n_clusters)
    new_labels = sims.argmax(axis=1).astype(np.int32)

    for mid, c in zip(new_ids, new_labels.tolist()):
        print(f"[cluster/patch] {mid} → cluster {c}")

    # Save new chunk to cluster_progress/
    progress_dir = run_dir / "cluster_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    state_path = progress_dir / "state.json"
    state: Dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    chunk_count = int(state.get("chunk_count", 0))
    emb_chunk   = progress_dir / f"emb_chunk_{chunk_count:06d}_patch.npy"
    ids_chunk   = progress_dir / f"ids_chunk_{chunk_count:06d}_patch.json"
    np.save(emb_chunk, new_embs_arr)
    ids_chunk.write_text(json.dumps(new_ids, ensure_ascii=False), encoding="utf-8")
    print(f"[cluster/patch] saved chunk → {emb_chunk.name}")

    # Append to clusters.csv
    df_new    = pd.DataFrame({"map_id": new_ids, "cluster": new_labels.astype(int)})
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    # Deduplicate: keep the latest entry for any repeated map_id
    df_merged = df_merged.drop_duplicates(subset=["map_id"], keep="last")
    df_merged.to_csv(clusters_csv, index=False)
    print(f"[cluster/patch] clusters.csv updated → {len(df_merged)} total rows")

    # Update embeddings.npy: rebuild from scratch to stay in sync with clusters.csv
    # (ensures row order matches map_id order in the CSV)
    id_to_old = {str(mid): i for i, mid in enumerate(df_existing["map_id"].tolist())}
    id_to_new = {str(mid): i for i, mid in enumerate(new_ids)}
    all_embs_list: List[np.ndarray] = []
    for mid in df_merged["map_id"].tolist():
        if str(mid) in id_to_new:
            all_embs_list.append(new_embs_arr[id_to_new[str(mid)]])
        elif str(mid) in id_to_old:
            all_embs_list.append(embs_existing[id_to_old[str(mid)]])
        else:
            # Fallback: zero vector (shouldn't happen)
            all_embs_list.append(np.zeros(dim, dtype=np.float32))
    embs_final = np.stack(all_embs_list, axis=0).astype(np.float32)
    np.save(emb_npy, embs_final)
    print(f"[cluster/patch] embeddings.npy updated → shape {embs_final.shape}")
    print(f"[cluster/patch] done — added {len(new_ids)} map(s).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_maps", type=int, default=0)
    p.add_argument("--k", type=int, default=200)
    # --- clustering method ---
    p.add_argument("--method", type=str, default="minibatch_kmeans",
                   choices=["kmeans", "minibatch_kmeans", "agglomerative", "recursive_agglomerative"],
                   help="Clustering algorithm. 'agglomerative' = two-stage Ward. "
                        "'recursive_agglomerative' = HDBSCAN within proto-clusters then Ward merge.")
    p.add_argument("--minibatch", action="store_true",
                   help="Shorthand for --method minibatch_kmeans (legacy flag, kept for compatibility).")
    p.add_argument("--batch_size", type=int, default=2048)
    # --- agglomerative / recursive-agglomerative ---
    p.add_argument("--pre_k", type=int, default=8000,
                   help="Number of proto-clusters for agglomerative methods.")
    p.add_argument("--linkage", type=str, default="ward",
                   choices=["ward", "average", "complete", "single"],
                   help="Linkage criterion for agglomerative / recursive_agglomerative.")
    p.add_argument("--min_cluster_size", type=int, default=5,
                   help="(recursive_agglomerative) Minimum maps to form an HDBSCAN sub-cluster; "
                        "smaller groups are expelled as singletons.")
    p.add_argument("--final_threshold", type=float, default=0.0,
                   help="(recursive_agglomerative) Distance threshold for the final Ward cut. "
                        "0 = use --k instead.")
    p.add_argument("--max_stage3", type=int, default=20000,
                   help="(recursive_agglomerative) Max representatives passed to Ward. "
                        "If exceeded, a consolidation K-Means pass reduces them first.")
    # --- embedding ---
    p.add_argument("--skip_embedding", action="store_true",
                   help="Skip re-embedding and load embeddings from existing chunk files / embeddings.npy.")
    p.add_argument("--resume", action="store_true",
                   help="Resume embedding from cluster_progress/state.json if present.")
    p.add_argument("--save_every", type=int, default=5000, help="Checkpoint interval in processed maps.")
    p.add_argument(
        "--map_ids", type=str, default="",
        metavar="ID[,ID,…]",
        help=(
            "Patch-embed mode: comma-separated beatmap IDs to embed and append to an "
            "existing clusters.csv + embeddings.npy.  Each new map is assigned to its "
            "nearest existing cluster centroid.  No full re-clustering is performed."
        ),
    )
    args = p.parse_args()

    # ── Patch-embed mode ────────────────────────────────────────────────────
    # When --map_ids is given, skip the full scan/cluster cycle and instead:
    # 1. Embed only the specified maps.
    # 2. Assign each to its nearest existing cluster centroid.
    # 3. Append rows to clusters.csv and extend embeddings.npy.
    if str(args.map_ids).strip():
        _patch_embed(args)
        return

    # --minibatch is a legacy shorthand
    if args.minibatch and args.method == "minibatch_kmeans":
        pass  # already the default, no-op
    elif args.minibatch:
        args.method = "minibatch_kmeans"

    run_dir = Path(args.run_dir).resolve()
    progress_dir = run_dir / "cluster_progress"

    print(f"[cluster] run_dir={run_dir}")
    print(f"[cluster] method={args.method} | k={int(args.k)} | skip_embedding={args.skip_embedding}")

    # ------------------------------------------------------------------
    # Embedding phase (skippable when embeddings already exist)
    # ------------------------------------------------------------------
    if args.skip_embedding:
        # Load embeddings from existing chunk files (same path the normal run writes).
        # Falls back to embeddings.npy if chunk files are absent.
        state_path = progress_dir / "state.json"
        all_ids: List[str] = []
        all_embs: List[np.ndarray] = []
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for ch in state.get("chunks", []):
                emb_path = progress_dir / str(ch["emb"])
                ids_path = progress_dir / str(ch["ids"])
                if not emb_path.exists() or not ids_path.exists():
                    continue
                arr = np.load(emb_path)
                ids = json.loads(ids_path.read_text(encoding="utf-8"))
                if arr.size == 0 or not ids:
                    continue
                all_embs.append(arr.astype(np.float32))
                all_ids.extend([str(x) for x in ids])

        if not all_embs:
            # Fallback: load the monolithic embeddings.npy + clusters.csv for IDs.
            emb_npy = run_dir / "embeddings.npy"
            clus_csv = run_dir / "clusters.csv"
            if not emb_npy.exists():
                raise FileNotFoundError(
                    f"--skip_embedding set but no chunk files and no embeddings.npy found in {run_dir}"
                )
            all_embs = [np.load(emb_npy).astype(np.float32)]
            df_ids = pd.read_csv(clus_csv, usecols=["map_id"], dtype={"map_id": str})
            all_ids = df_ids["map_id"].tolist()

        embs_ok = np.concatenate(all_embs, axis=0)
        ids_ok  = all_ids
        print(f"[cluster] loaded {len(ids_ok)} existing embeddings (shape {embs_ok.shape})")

    else:
        # Full embedding run.
        model, feature_cols, st, _state = load_run(run_dir)
        device = torch.device(str(args.device))
        model = model.to(device)

        if str(args.dataset_root).strip():
            dataset_root = Path(str(args.dataset_root)).resolve()
        else:
            cfg: Dict = {}
            try:
                cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            dataset_root = Path(cfg.get("dataset_root", str(DEFAULT_DATASET_ROOT))).resolve()

        configure_embedding_dataset_root(dataset_root)
        processed_dir = dataset_root / "processed" / "maps_by_id"
        map_ids = list_map_ids_from_processed(processed_dir)
        if int(args.max_maps) > 0 and len(map_ids) > int(args.max_maps):
            rng = np.random.default_rng(2025)
            idx = rng.choice(len(map_ids), size=int(args.max_maps), replace=False)
            map_ids = [map_ids[i] for i in idx]

        print(f"[cluster] maps={len(map_ids)} | features={len(feature_cols)} | device={device}")
        print(f"[cluster] dataset_root={dataset_root}")

        from CORE.models.embedding_model.dataset import build_features_for_map

        progress_dir.mkdir(parents=True, exist_ok=True)
        state_path = progress_dir / "state.json"

        def _new_state() -> Dict:
            return {
                "next_index": 0,
                "chunk_count": 0,
                "chunks": [],
                "map_count": int(len(map_ids)),
                "dataset_root": str(dataset_root),
                "k": int(args.k),
                "method": str(args.method),
                "batch_size": int(args.batch_size),
            }

        state: Dict
        if bool(args.resume) and state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if int(state.get("map_count", -1)) != int(len(map_ids)):
                print("[cluster] resume state map_count mismatch; restarting from scratch.")
                state = _new_state()
        else:
            state = _new_state()
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        def _flush_chunk(ids_buf: List[str], embs_buf: List[np.ndarray], state_obj: Dict) -> Tuple[List[str], List[np.ndarray], Dict]:
            if not ids_buf:
                return ids_buf, embs_buf, state_obj
            cidx = int(state_obj.get("chunk_count", 0))
            emb_path = progress_dir / f"emb_chunk_{cidx:06d}.npy"
            ids_path = progress_dir / f"ids_chunk_{cidx:06d}.json"
            arr = np.stack(embs_buf, axis=0).astype(np.float32)
            np.save(emb_path, arr)
            ids_path.write_text(json.dumps(ids_buf, ensure_ascii=False), encoding="utf-8")
            chunks = list(state_obj.get("chunks", []))
            chunks.append({"emb": emb_path.name, "ids": ids_path.name, "n": int(arr.shape[0])})
            state_obj["chunks"] = chunks
            state_obj["chunk_count"] = cidx + 1
            state_path.write_text(json.dumps(state_obj, indent=2), encoding="utf-8")
            print(f"[cluster] checkpoint chunk={cidx} n={arr.shape[0]} next_index={state_obj.get('next_index', 0)}")
            return [], [], state_obj

        start_idx = int(state.get("next_index", 0))
        ids_buf: List[str] = []
        embs_buf: List[np.ndarray] = []
        processed_since_ckpt = 0
        if start_idx > 0:
            print(f"[cluster] resuming from index {start_idx}/{len(map_ids)}")

        for i in range(start_idx, len(map_ids)):
            mid = map_ids[i]
            state["next_index"] = i + 1
            try:
                df, _ = build_features_for_map(str(mid), feature_cols)
            except Exception:
                processed_since_ckpt += 1
                if processed_since_ckpt >= int(args.save_every):
                    ids_buf, embs_buf, state = _flush_chunk(ids_buf, embs_buf, state)
                    processed_since_ckpt = 0
                continue
            if df is None or df.empty:
                processed_since_ckpt += 1
                if processed_since_ckpt >= int(args.save_every):
                    ids_buf, embs_buf, state = _flush_chunk(ids_buf, embs_buf, state)
                    processed_since_ckpt = 0
                continue
            X = df[feature_cols].to_numpy(dtype=np.float32)
            X = standardize(X, st)
            z = embed_map(model, X, device)
            ids_buf.append(str(mid))
            embs_buf.append(z)

            processed_since_ckpt += 1
            if (i + 1) % 2000 == 0:
                print(f"[cluster] embedded {i+1}/{len(map_ids)}")
            if processed_since_ckpt >= int(args.save_every):
                ids_buf, embs_buf, state = _flush_chunk(ids_buf, embs_buf, state)
                processed_since_ckpt = 0

        # Final flush
        ids_buf, embs_buf, state = _flush_chunk(ids_buf, embs_buf, state)
        state["next_index"] = int(len(map_ids))
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        # Collect all chunks
        all_ids = []
        all_embs = []
        for ch in state.get("chunks", []):
            emb_path = progress_dir / str(ch["emb"])
            ids_path = progress_dir / str(ch["ids"])
            if not emb_path.exists() or not ids_path.exists():
                continue
            arr = np.load(emb_path)
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if arr.size == 0 or not ids:
                continue
            all_embs.append(arr.astype(np.float32))
            all_ids.extend([str(x) for x in ids])

        if not all_embs or not all_ids:
            raise RuntimeError("No embeddings computed (are feature files present?)")

        embs_ok = np.concatenate(all_embs, axis=0)
        ids_ok  = all_ids

    # ------------------------------------------------------------------
    # Clustering phase
    # ------------------------------------------------------------------
    print(f"[cluster] clustering {len(ids_ok)} maps with method={args.method} k={args.k}")

    if args.method == "agglomerative":
        linkage_path = run_dir / "cluster_linkage.npy"
        labels = fit_agglomerative(
            embs_ok,
            k=int(args.k),
            pre_k=int(args.pre_k),
            linkage_method=str(args.linkage),
            batch_size=int(args.batch_size),
            linkage_out_path=linkage_path,
        )
        config_extra = {
            "method": "agglomerative",
            "pre_k": int(args.pre_k),
            "linkage": str(args.linkage),
            "linkage_matrix": linkage_path.name,
        }
        centroids = None
    elif args.method == "recursive_agglomerative":
        linkage_path = run_dir / "cluster_linkage.npy"
        labels = fit_recursive_agglomerative(
            embs_ok,
            pre_k=int(args.pre_k),
            min_cluster_size=int(args.min_cluster_size),
            k=int(args.k),
            final_threshold=float(args.final_threshold),
            batch_size=int(args.batch_size),
            linkage_method=str(args.linkage),
            max_stage3=int(args.max_stage3),
            linkage_out_path=linkage_path,
        )
        config_extra = {
            "method": "recursive_agglomerative",
            "pre_k": int(args.pre_k),
            "min_cluster_size": int(args.min_cluster_size),
            "linkage": str(args.linkage),
            "final_threshold": float(args.final_threshold),
            "max_stage3": int(args.max_stage3),
            "linkage_matrix": linkage_path.name,
        }
        centroids = None
    elif args.method == "kmeans":
        km = KMeans(n_clusters=int(args.k), random_state=2025, n_init="auto")
        labels = km.fit_predict(embs_ok)
        config_extra = {"method": "kmeans"}
        centroids = km.cluster_centers_.astype(np.float32)
    else:  # minibatch_kmeans (default)
        km = MiniBatchKMeans(n_clusters=int(args.k), batch_size=int(args.batch_size), random_state=2025, n_init="auto")
        labels = km.fit_predict(embs_ok)
        config_extra = {"method": "minibatch_kmeans"}
        centroids = km.cluster_centers_.astype(np.float32)

    out_df = pd.DataFrame({"map_id": ids_ok, "cluster": labels.astype(int)})
    out_path = run_dir / "clusters.csv"
    out_df.to_csv(out_path, index=False)

    np.save(run_dir / "embeddings.npy", embs_ok)
    (run_dir / "cluster_config.json").write_text(
        json.dumps(
            {
                "k": int(args.k),
                "batch_size": int(args.batch_size),
                "n_maps_clustered": int(len(ids_ok)),
                # legacy keys kept for backward compatibility
                "minibatch": args.method == "minibatch_kmeans",
                **config_extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if centroids is not None:
        np.save(run_dir / "cluster_centroids.npy", centroids)

    print(f"[cluster] wrote {out_path}")


if __name__ == "__main__":
    main()

