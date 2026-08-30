# Unsupervised clustering pipeline — commands

End-to-end flow: **dataset (osu + windows)** → **contrastive encoder training** → **(optional k sweep)** → **embed + cluster** → **echosu tag summaries** → **2D plot (optional)** → **WebGL data export**.

Run these from the **repo root** `c:\Projects\echoCluster` with the venv interpreter. Ensure the project root is on `PYTHONPATH` (Cursor/IDEs often do this; otherwise `set PYTHONPATH=c:\Projects\echoCluster` before `python -m …`).

**Secrets (`.env` next to `CORE`, i.e. `c:\Projects\echoCluster\.env` or as loaded by the scripts):**


| Variable                             | Used by                                     |
| ------------------------------------ | ------------------------------------------- |
| `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` | `clustering_dataset_builder` (osu! API)     |
| `ECHOSU_TOKEN`                       | `clustering_dataset_builder --fetch_echosu` |


---

## 1. Dataset: `clustering_dataset_builder.py`

**Module:** `CORE.data.clustering_dataset_builder`  
**Defaults:** `--dataset_root` → `C:/Projects/echoCluster/CORE/data/dataset_full_osu`, `--echosu_json` → `C:/Projects/echoCluster/CORE/data/raw/tag_data_with_ids.json`


| Flag                    | Type  | Default     | Role                                                                                                              |
| ----------------------- | ----- | ----------- | ----------------------------------------------------------------------------------------------------------------- |
| `--max_id`              | int   | `0`         | **Required for scan mode:** highest beatmap ID to scan up to. Use `0` with `--map_ids` or `--skip_dataset`.       |
| `--map_ids`             | str   | `""`        | Comma-separated IDs: patch mode only (no full scan).                                                              |
| `--skip_dataset`        | flag  | off         | Skip download/process; use with `--fetch_echosu` only.                                                            |
| `--resume`              | flag  | off         | Resume ascending scan from `logs/build_log_asc.txt`.                                                              |
| `--fetch_echosu`        | flag  | off         | After dataset step, refresh local echosu tag JSON (`--echosu_json`).                                              |
| `--update_dataset`      | flag  | off         | Repair/validate all maps under `processed/maps_by_id/`, optionally continue scan if `--max_id` above max on disk. |
| `--revalidate`          | flag  | off         | With `--update_dataset`: ignore saved repair queue, re-validate all.                                              |
| `--skip_validation`     | flag  | off         | With `--update_dataset`: skip validation pass (queue all for repair check).                                       |
| `--start_id`            | int   | `0`         | With `--update_dataset`: ignore IDs below this.                                                                   |
| `--dataset_root`        | str   | see above   | Dataset root.                                                                                                     |
| `--echosu_json`         | str   | see above   | Output path for echosu tags (used later by `label_clusters` / WebGL).                                             |
| `--keep_osu`            | flag  | off         | Keep downloaded `.osu` under `<dataset_root>/maps`.                                                               |
| `--sleep_api`           | float | `0.2`       | Pause between osu! v2 batch requests (s).                                                                         |
| `--sleep_raw`           | float | `0.2`       | Pause after each raw `.osu` download (s).                                                                         |
| `--api_timeout`         | float | `30`        | osu! v2 HTTP timeout (s).                                                                                         |
| `--raw_timeout`         | float | `30`        | Raw `.osu` HTTP timeout (s).                                                                                      |
| `--raw_retries`         | int   | `6`         | Retries for raw download.                                                                                         |
| `--max_seconds_per_map` | float | `3.0`       | Per-map time budget; exceeded → skip.                                                                             |
| `--window_beats`        | int   | `4`         | Beat-aligned window length.                                                                                       |
| `--window_overlap`      | float | `0.5`       | Window overlap fraction.                                                                                          |
| `--max_t_ms_sanity`     | int   | `100000000` | Sanity cap on timestamps (ms).                                                                                    |
| `--use_rosu_strains`    | flag  | off         | Compute rosu strain series (slower).                                                                              |


---

## 2. Train encoder: `unsupervised_clustering/train.py`

**Module:** `CORE.models.unsupervised_clustering.train`  
Creates a **new** folder under `CORE/models/outputs/` named like `YYYYMMDD_UNSUP_HHMMSS` and writes `model.pt`, `metrics.json`, and run config artifacts. **Note the printed `[unsup] run_dir=...` for later steps.**


| Flag                             | Type  | Default                                            |
| -------------------------------- | ----- | -------------------------------------------------- |
| `--device`                       | str   | `cuda` if available else `cpu`                     |
| `--dataset_root`                 | str   | `C:/Projects/echoCluster/CORE/data/dataset_full_osu` |
| `--epochs`                       | int   | `10`                                               |
| `--batch_size`                   | int   | `128`                                              |
| `--lr`                           | float | `3e-4`                                             |
| `--temperature`                  | float | `0.1`                                              |
| `--crop_len`                     | int   | `256`                                              |
| `--feat_dropout`                 | float | `0.10`                                             |
| `--noise_std`                    | float | `0.01`                                             |
| `--hidden_dim`                   | int   | `256`                                              |
| `--num_layers`                   | int   | `1`                                                |
| `--dropout`                      | float | `0.1`                                              |
| `--proj_dim`                     | int   | `128`                                              |
| `--max_maps`                     | int   | `20000`                                            |
| `--fit_norm_maps`                | int   | `2000`                                             |
| `--fit_norm_max_windows_per_map` | int   | `64`                                               |
| `--seed`                         | int   | `2025`                                             |


---

## 3. Optional: elbow / k sweep — `elbow_k.py`

**Module:** `CORE.models.unsupervised_clustering.elbow_k`  
Reads embeddings from `--run_dir`, writes `elbow_k.csv`, `elbow_k.json`, `elbow_k.png`.


| Flag            | Type | Default                                        |
| --------------- | ---- | ---------------------------------------------- |
| `--run_dir`     | str  | **required**                                   |
| `--k_values`    | str  | `""` — if empty, uses `k_min`/`k_max`/`k_step` |
| `--k_min`       | int  | `100`                                          |
| `--k_max`       | int  | `3000`                                         |
| `--k_step`      | int  | `100`                                          |
| `--sample_size` | int  | `200000` (`0` = all points)                    |
| `--seed`        | int  | `2025`                                         |
| `--minibatch`   | flag | off                                            |
| `--batch_size`  | int  | `8192`                                         |
| `--max_iter`    | int  | `200`                                          |
| `--n_init`      | int  | `5`                                            |


---

## 4. Embed + cluster: `cluster.py`

**Module:** `CORE.models.unsupervised_clustering.cluster`  
Uses `--run_dir` from training (checkpoint + config). Writes `embeddings.npy`, `clusters.csv`, and chunked progress under `cluster_progress/` (unless patch-only mode).


| Flag                 | Type  | Default                                                                                                                       |
| -------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--run_dir`          | str   | **required**                                                                                                                  |
| `--dataset_root`     | str   | `""` (infer from run if empty)                                                                                                |
| `--device`           | str   | `cuda` if available else `cpu`                                                                                                |
| `--max_maps`         | int   | `0` = no cap                                                                                                                  |
| `--k`                | int   | `200`                                                                                                                         |
| `--method`           | str   | `minibatch_kmeans` — choices: `kmeans`, `minibatch_kmeans`, `agglomerative`, `recursive_agglomerative`                        |
| `--minibatch`        | flag  | off (legacy shorthand → `minibatch_kmeans`)                                                                                   |
| `--batch_size`       | int   | `2048`                                                                                                                        |
| `--pre_k`            | int   | `8000` (proto-clusters for agglomerative methods)                                                                             |
| `--linkage`          | str   | `ward` — choices: `ward`, `average`, `complete`, `single`                                                                     |
| `--min_cluster_size` | int   | `5` (`recursive_agglomerative` / HDBSCAN stage)                                                                               |
| `--final_threshold`  | float | `0.0` (`0` = cut by `--k`; else Ward distance threshold)                                                                      |
| `--max_stage3`       | int   | `20000`                                                                                                                       |
| `--skip_embedding`   | flag  | off — reuse existing chunk files / `embeddings.npy`                                                                           |
| `--resume`           | flag  | off — resume embedding from `cluster_progress/state.json`                                                                     |
| `--save_every`       | int   | `5000`                                                                                                                        |
| `--map_ids`          | str   | `""` — if set: **patch mode** (embed only these IDs, assign to nearest centroid, append to `clusters.csv` / `embeddings.npy`) |


---

## 5. Cluster ↔ echosu tags: `label_clusters.py`

**Module:** `CORE.models.unsupervised_clustering.label_clusters`  
Writes `cluster_tag_summary.csv` and `cluster_tag_summary.json` under `--run_dir`.


| Flag             | Type | Default                                                      |
| ---------------- | ---- | ------------------------------------------------------------ |
| `--run_dir`      | str  | **required**                                                 |
| `--top_k`        | int  | `20`                                                         |
| `--label_source` | str  | `echosu_json` — choices: `echosu_json`, `dataset_artifacts`  |
| `--echosu_json`  | str  | `C:/Projects/echoCluster/CORE/data/raw/tag_data_with_ids.json` |


---

## 6. Optional static plot: `visualize_clusters.py`

**Module:** `CORE.models.unsupervised_clustering.visualize_clusters`  
Writes `clusters_2d.png` under `--run_dir`.


| Flag                      | Type  | Default                                 |
| ------------------------- | ----- | --------------------------------------- |
| `--run_dir`               | str   | **required**                            |
| `--method`                | str   | `umap` — choices: `umap`, `pca`, `tsne` |
| `--max_points`            | int   | `200000`                                |
| `--seed`                  | int   | `2025`                                  |
| `--fig_w`                 | float | `16`                                    |
| `--fig_h`                 | float | `11`                                    |
| `--point_size`            | float | `3`                                     |
| `--annotate_top_clusters` | int   | `35`                                    |
| `--top_tag_per_cluster`   | int   | `2`                                     |


---

## 7. WebGL export: `generate_webgl_data.py`

Canonical copy: `webgl/generate_webgl_data.py`. A copy is also kept under each `<run_dir>/WebGL/` next to `index.html`. Always pass `--run_dir`.

Writes JSON/binary under `<run_dir>/WebGL/data/` (or `--webgl_dir` if set).


| Flag                         | Type  | Default                                                      |
| ---------------------------- | ----- | ------------------------------------------------------------ |
| `--run_dir`                  | str   | *(hardcoded fallback in each copy — always pass explicitly)* |
| `--webgl_dir`                | str   | `""` → `<run_dir>/WebGL`                                     |
| `--dataset_root`             | str   | `""` — infer from run config                                 |
| `--max_points`               | int   | `0` = keep all                                               |
| `--seed`                     | int   | `2025`                                                       |
| `--echosu_json`              | str   | `C:/Projects/echoCluster/CORE/data/raw/tag_data_with_ids.json` |
| `--contours`                 | int   | `1`                                                          |
| `--contour_grid`             | int   | `180`                                                        |
| `--contour_smooth`           | int   | `3`                                                          |
| `--contour_smooth_sphere`    | int   | `6`                                                          |
| `--star_boost_power`         | float | `2.0`                                                        |
| `--star_boost_cap`           | float | `15.0`                                                       |
| `--bpm_low_boost_power`      | float | `1.6`                                                        |
| `--perf_low_points`          | int   | `25000`                                                      |
| `--perf_mid_points`          | int   | `100000`                                                     |
| `--perf_high_points`         | int   | `250000`                                                     |
| `--sphere_radius`            | float | `1.0`                                                        |
| `--sphere_mesh_subdivisions` | int   | `7`                                                          |


---

## Bare-bones (single-line) example

Replace `RUN_DIR` with the directory printed after training (e.g. `CORE/models/outputs/20260223_UNSUP_234612`).

```text
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --max_id 5600000 --fetch_echosu
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.train --dataset_root C:/Projects/echoCluster/CORE/data/dataset_full_osu --epochs 10 --max_maps 20000
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.cluster --run_dir RUN_DIR --k 200
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.label_clusters --run_dir RUN_DIR
.\env\Scripts\python.exe webgl\generate_webgl_data.py --run_dir RUN_DIR
```

---

## Full (single-line) example

Paths match the defaults in code and an example run id; adjust `20260223_UNSUP_234612` and `--max_id` to your machine.

```text
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --dataset_root C:/Projects/echoCluster/CORE/data/dataset_full_osu --max_id 5600000 --fetch_echosu --echosu_json C:/Projects/echoCluster/CORE/data/raw/tag_data_with_ids.json --sleep_api 0.2 --sleep_raw 0.2 --window_beats 4 --window_overlap 0.5
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.train --device cuda --dataset_root C:/Projects/echoCluster/CORE/data/dataset_full_osu --epochs 10 --batch_size 128 --lr 0.0003 --temperature 0.1 --crop_len 256 --feat_dropout 0.1 --noise_std 0.01 --hidden_dim 256 --num_layers 1 --dropout 0.1 --proj_dim 128 --max_maps 20000 --fit_norm_maps 2000 --fit_norm_max_windows_per_map 64 --seed 2025
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.cluster --run_dir C:/Projects/echoCluster/CORE/models/outputs/20260223_UNSUP_234612 --dataset_root C:/Projects/echoCluster/CORE/data/dataset_full_osu --device cuda --max_maps 0 --k 200 --method minibatch_kmeans --batch_size 2048 --save_every 5000
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.elbow_k --run_dir C:/Projects/echoCluster/CORE/models/outputs/20260223_UNSUP_234612 --k_min 100 --k_max 3000 --k_step 100 --sample_size 200000 --seed 2025 --minibatch --batch_size 8192 --max_iter 200 --n_init 5
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.label_clusters --run_dir C:/Projects/echoCluster/CORE/models/outputs/20260223_UNSUP_234612 --top_k 20 --label_source echosu_json --echosu_json C:/Projects/echoCluster/CORE/data/raw/tag_data_with_ids.json
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.visualize_clusters --run_dir C:/Projects/echoCluster/CORE/models/outputs/20260223_UNSUP_234612 --method umap --max_points 200000 --seed 2025 --fig_w 16 --fig_h 11 --point_size 3 --annotate_top_clusters 35 --top_tag_per_cluster 2
.\env\Scripts\python.exe C:/Projects/echoCluster/webgl/generate_webgl_data.py --run_dir C:/Projects/echoCluster/CORE/models/outputs/20260223_UNSUP_234612 --webgl_dir C:/Projects/echoCluster/CORE/models/outputs/20260223_UNSUP_234612/WebGL --dataset_root C:/Projects/echoCluster/CORE/data/dataset_full_osu --max_points 0 --seed 2025 --echosu_json C:/Projects/echoCluster/CORE/data/raw/tag_data_with_ids.json --contours 1 --contour_grid 180 --contour_smooth 3 --contour_smooth_sphere 6 --star_boost_power 2.0 --star_boost_cap 15.0 --bpm_low_boost_power 1.6 --perf_low_points 25000 --perf_mid_points 100000 --perf_high_points 250000 --sphere_radius 1.0 --sphere_mesh_subdivisions 7
```

---

## Related

- Large **osu-only** corpus (alternative builder): `CORE/data/README_full_osu_dataset.md` (`build_osu_full_dataset.py`, `collect_echosu_tag_counts.py`).

