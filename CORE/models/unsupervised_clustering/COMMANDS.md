# Pipeline Commands — Unsupervised Clustering

All commands are run from the **repo root** (`C:\Projects\echoCluster`) with the project virtualenv.
Replace `<RUN_DIR>` with your actual output folder, e.g.
`C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351`.

---

## Pipeline order

```
dataset_builder  →  train  →  cluster (embed + cluster)  →  label_clusters  →  generate_webgl_data
                                         ↑
                            (elbow_k and visualize_clusters are optional helpers)
```

---

## 0 · Build / update the dataset

Scan for new maps above the highest ID already in the dataset (ascending), or process a specific list of map IDs.  Also handles refreshing the local echosu tag file.

```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--max_id` | *(required for scan mode)* | Scan upward from latest known ID to this value |
| `--map_ids` | — | Patch mode: comma-separated map IDs to download and process (skips scan) |
| `--skip_dataset` | flag | Skip dataset step; jump straight to `--fetch_echosu` |
| `--resume` | flag | Resume an interrupted ascending scan from the last logged ID (much faster than the default disk scan) |
| `--fetch_echosu` | flag | After dataset step, refresh the local echosu tag JSONL |
| `--update_dataset` | flag | **Update/repair mode** — see full description below |
| `--revalidate` | flag | *(update mode only)* Discard any saved repair queue and re-run full validation from scratch |
| `--skip_validation` | flag | *(update mode only)* Skip completeness checks; queue all maps for repair (safe — EXISTS check skips complete maps) |
| `--start_id` | `0` | *(update mode only)* Skip all map IDs below this value during enumeration, validation, and repair |
| `--dataset_root` | hardcoded | Override the processed-maps root |
| `--echosu_json` | hardcoded | Path to write/update the echosu tag JSONL |
| `--keep_osu` | flag | Retain downloaded `.osu` files under `<dataset_root>/maps` |
| `--sleep_api` | `0.2` | Sleep (s) between osu! v2 batch requests |
| `--sleep_raw` | `0.2` | Sleep (s) after each raw `.osu` download |
| `--api_timeout` | `30.0` | osu! API request timeout (s) |
| `--raw_timeout` | `30.0` | Raw `.osu` download timeout (s) |
| `--raw_retries` | `6` | Download retry attempts per map |
| `--max_seconds_per_map` | `3.0` | Hard per-map time budget; exceeded → skip |
| `--window_beats` | `4` | Beats per sliding window in the windows parquet |
| `--window_overlap` | `0.5` | Window overlap fraction (0 = no overlap, 0.5 = 50%) |
| `--max_t_ms_sanity` | `100000000` | Reject any hit object with `t_ms` above this value |
| `--use_rosu_strains` | flag | Compute rosu strain series (slower, optional) |

**Scan for new maps up to ID 5 800 000:**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --max_id 5800000
```

**Resume an interrupted scan (reads last logged ID instead of scanning all directories):**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --max_id 5800000 --resume
```

**Add specific maps + refresh echosu tags:**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --map_ids 4153835,4200001 --max_id 0 --fetch_echosu
```

### What each map produces

**`_meta.json`** — API metadata plus map-level computed features:
`star_rating`, `ar`, `cs`, `od`, `hp`, `bpm`, `length_total`, `length_drain`, `note_count`, `count_circles`, `count_sliders`, `count_spinners`, `max_combo`, `status_ranked`, `ranked_date`, `map_end_time_ms`, `speed_p95`, `jerk_p95`, `angle_entropy`, `stream_fraction`, `jump_fraction`, `rhythm_entropy`, `spatial_entropy`, `slider_complexity_mean`, `bpm_variance`

**`_timeseries.parquet`** — one row per hit object with ~60 features across four groups:
- *Positional / object*: `t_ms`, `x_px`, `y_px`, `x_norm`, `y_norm`, `obj_type`, `beat_rel`, `dist_prev_px`, `angle_deg`, `poly_sq`, `vis_density`
- *Slider*: `slider_len_px`, `slider_vel`, `slider_vel_beats`, `slider_dir_deg`, `slider_curvature_ratio`, `slider_dur_ms`, `anchor_density`
- *Kinematic*: `speed`, `vel_x/y`, `accel_x/y/mag`, `jerk_x/y/mag`, `jounce_x/y/mag`, `absement_x/y`, `absity_x/y`, `bearing_deg`
- *Temporal rhythm*: `dt_prev_ms`, `dt_cv_rolling`, `dt_ratio_prev`, `bpm_local_effective`, `is_stream`, `is_burst`, `rhythm_complexity`
- *Spatial pattern*: `dist_center_px`, `dist_center_var_rolling`, `edge_proximity`, `quadrant`, `jump_type`

**`_w4b.parquet`** — one row per 4-beat sliding window (50% overlap) with aggregated features including `speed_delta`, `density_delta`, `pattern_switch`, and `intensity_slope` for tracking transitions between windows.

### `--update_dataset` — repair existing data + optional forward scan

Enumerates every directory under `processed/maps_by_id/` **and** scans all `build_log_*.txt` files to recover map IDs that were previously built but have since been deleted from disk (ghost IDs).  Each on-disk map is validated for completeness; failures are silently deleted and re-queued for re-download.  After the repair pass, if `--max_id` exceeds the highest known ID, a normal ascending scan continues from there and is written to `build_log_asc.txt`.

The repair queue is saved to `logs/update_repair_queue.json` after the validation pass — if the run is interrupted, restarting with the same command resumes repair without re-validating.  Use `--revalidate` to discard the saved queue and start fresh.

**What is checked per map:**

| Check | Failure condition | Smart-null rule (NOT flagged) |
|---|---|---|
| `_meta.json` | File missing or not valid JSON | — |
| Required meta keys | `ar`, `cs`, `od`, `hp`, `bpm`, `note_count`, `count_circles`, `status_ranked`, `star_rating` absent | `ranked_date` absent/null for non-ranked maps is expected; `max_combo` absent for old approved/loved maps |
| `_timeseries.parquet` | File missing, unreadable, or zero rows | Slider columns (`slider_*`) NaN for circle/spinner rows |
| Timeseries column completeness | Any of the required columns absent — parquet pre-dates a feature pass. **Kinematic:** `speed`, `vel_x`, `accel_mag`, `absement_x`. **Contextual:** `bearing_deg`, `dist_center_px`, `vis_density`, `dt_cv_rolling`. **Temporal rhythm:** `is_stream`, `rhythm_complexity`, `bpm_local_effective`. **Spatial:** `edge_proximity`, `dist_center_var_rolling` | — |
| `_w4b.parquet` | File missing, unreadable, or zero rows | — |
| Rosu strains *(only with `--use_rosu_strains`)* | `aim_strain` column missing **or** all-NaN when `count_circles > 0` | Slider/spinner-only maps (`count_circles == 0`) allowed to have all-NaN strains |

**Validate + repair existing data, then scan new IDs up to 5 800 000:**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --update_dataset --max_id 5800000
```

**Validate + repair with rosu strains (rebuilds any map missing strain data):**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --update_dataset --use_rosu_strains --max_id 5800000
```

**Validate only (no new scan):**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --update_dataset
```

**Resume an interrupted repair pass (skips re-validation, uses saved queue):**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --update_dataset --max_id 5800000
```

**Force fresh re-validation (discards saved queue):**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --update_dataset --revalidate --max_id 5800000
```

**Resume repair from a specific ID (e.g. if interrupted mid-range):**
```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --update_dataset --start_id 3000000 --max_id 5800000
```

---

## 1 · Train the GRU encoder

Trains a new SimCLR encoder from scratch. Always creates a **new timestamped run folder** under `CORE/models/outputs/`.

```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.train [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--max_maps` | `20000` | Maps to train on; `0` = all available |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `128` | Contrastive pairs per batch |
| `--lr` | `3e-4` | AdamW learning rate |
| `--hidden_dim` | `256` | GRU hidden size |
| `--num_layers` | `1` | GRU stacked layers |
| `--proj_dim` | `128` | Projection head output dim (InfoNCE loss) |
| `--temperature` | `0.1` | InfoNCE temperature |
| `--crop_len` | `256` | Window crop length for augmentation |
| `--feat_dropout` | `0.1` | Per-feature dropout during augmentation |
| `--noise_std` | `0.01` | Gaussian noise std during augmentation |
| `--fit_norm_maps` | `2000` | Maps sampled to fit the feature standardiser |
| `--seed` | `2025` | RNG seed |
| `--device` | auto | `cuda` or `cpu` |
| `--dataset_root` | hardcoded | Override the processed-maps root |

> **Note:** Training always creates a fresh run folder; there is no `--resume` flag.

**Train on all maps, 10 epochs:**
```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.train --max_maps 0 --epochs 10 --batch_size 128 --lr 3e-4 --hidden_dim 256 --num_layers 1 --proj_dim 128 --temperature 0.1 --crop_len 256 --feat_dropout 0.1 --noise_std 0.01 --seed 2025
```

---

## 2 · Embed all maps & cluster

Embeds every map through the trained encoder, then runs recursive agglomerative clustering.

```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.cluster --run_dir "<RUN_DIR>" [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--run_dir` | *(required)* | Output folder from `train.py` |
| `--method` | `minibatch_kmeans` | `minibatch_kmeans`, `kmeans`, `agglomerative`, `recursive_agglomerative` |
| `--pre_k` | `8000` | Stage-1 k-means centroids before agglomerative merging |
| `--min_cluster_size` | `5` | Clusters smaller than this are dissolved |
| `--final_threshold` | `0.0` | Agglomerative merge distance threshold (0 = use `--k` hard target) |
| `--k` | `200` | Final cluster count when not using a threshold |
| `--linkage` | `ward` | Linkage criterion for agglomerative stage |
| `--max_stage3` | `20000` | Max centroid-level agglomerative nodes |
| `--batch_size` | `2048` | Mini-batch size during embedding |
| `--max_maps` | `0` | Embed only N maps; `0` = all |
| `--save_every` | `5000` | Checkpoint interval (maps processed) |
| `--resume` | flag | Resume an interrupted embedding pass from last checkpoint |
| `--skip_embedding` | flag | Skip embedding; use existing `embeddings.npy` and re-cluster |
| `--map_ids` | — | Patch-embed: embed specific maps and assign to nearest existing cluster (no re-clustering) |
| `--device` | auto | `cuda` or `cpu` |

**Resume an interrupted embedding + cluster:**
```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.cluster --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --resume --method recursive_agglomerative --pre_k 8000 --min_cluster_size 2 --final_threshold 0.8
```

**Re-cluster without re-embedding:**
```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.cluster --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --skip_embedding --method recursive_agglomerative --pre_k 8000 --min_cluster_size 2 --final_threshold 0.8
```

**Patch-embed specific maps into an existing run:**
```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.cluster --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --map_ids 4153835,4200001
```

---

## 3 · Label clusters with echosu tags

Reads `clusters.csv` and the echosu tag JSON to compute per-cluster tag summaries.
Required before `generate_webgl_data.py` can show tag information.

```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.label_clusters --run_dir "<RUN_DIR>" [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--run_dir` | *(required)* | Run folder containing `clusters.csv` |
| `--top_k` | `20` | Top-N tags to retain per cluster |
| `--label_source` | `echosu_json` | `echosu_json` or `dataset_artifacts` |
| `--echosu_json` | hardcoded | Path to the echosu tag JSONL |

```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.label_clusters --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --echosu_json "C:\Projects\echoCluster\CORE\data\raw\tag_data_with_ids.json"
```

---

## 4 · Generate WebGL viewer data

Produces all binary/JSON data files consumed by `WebGL/index.html`.

```powershell
.\env\Scripts\python.exe webgl\generate_webgl_data.py --run_dir "<RUN_DIR>" [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--run_dir` | hardcoded | Run folder with `clusters.csv`, `embeddings.npy`, etc. |
| `--webgl_dir` | `<run_dir>/WebGL` | Override output location for generated data |
| `--echosu_json` | hardcoded | echosu tag JSONL for labeling |
| `--max_points` | `0` | Downsample output; `0` = keep all |
| `--contours` | `1` | `1` = generate contour overlays |
| `--contour_grid` | `180` | Contour grid resolution |
| `--contour_smooth` | `3` | 2D flat contour smoothing passes |
| `--contour_smooth_sphere` | `6` | 3D sphere contour smoothing passes |
| `--sphere_radius` | `1.0` | Radius for 3D spherical projection |
| `--sphere_mesh_subdivisions` | `7` | Icosphere subdivision level |
| `--star_boost_power` | `2.0` | High-SR weighting curve power |
| `--seed` | `2025` | RNG seed |

```powershell
.\env\Scripts\python.exe webgl\generate_webgl_data.py --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --echosu_json "C:\Projects\echoCluster\CORE\data\raw\tag_data_with_ids.json" --max_points 0 --seed 2026 --contours 1 --contour_grid 360 --sphere_mesh_subdivisions 9
```

---

## 5 · Top Scores proxy (local only)

Allows the WebGL viewer's **Top Scores** search to query the osu! API v2.
Requires a registered osu! OAuth application (client credentials grant).

```powershell
python "<RUN_DIR>\WebGL\top_scores_proxy.py" --client-id <ID> --client-secret <SECRET> [--port 7373]
```

| Flag | Default | Description |
|---|---|---|
| `--client-id` | *(required)* | osu! OAuth application client_id |
| `--client-secret` | *(required)* | osu! OAuth application client_secret |
| `--port` | `7373` | Port the proxy listens on |

```powershell
python "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351\WebGL\top_scores_proxy.py" --client-id 12345 --client-secret your_secret_here
```

---

## 6 · (Optional) Elbow-k analysis

Sweeps a range of k values to help pick the right cluster count.

```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.elbow_k --run_dir "<RUN_DIR>" [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--run_dir` | *(required)* | Run folder with `embeddings.npy` |
| `--k_values` | — | Explicit k list, e.g. `200,400,800` |
| `--k_min/max/step` | `100/3000/100` | Range sweep when `--k_values` is omitted |
| `--sample_size` | `200000` | Embeddings to sample; `0` = all |
| `--minibatch` | flag | Use MiniBatchKMeans for speed |
| `--seed` | `2025` | RNG seed |

```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.elbow_k --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --k_min 100 --k_max 3000 --k_step 100 --sample_size 200000
```

---

## 7 · (Optional) 2D visualisation

Renders a static cluster map image (`clusters_2d.png`).

```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.visualize_clusters --run_dir "<RUN_DIR>" [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--run_dir` | *(required)* | Run folder |
| `--method` | `umap` | `umap`, `pca`, or `tsne` |
| `--max_points` | `200000` | Points to plot (sampled if larger) |
| `--annotate_top_clusters` | `35` | How many large clusters to label |
| `--seed` | `2025` | RNG seed |

```powershell
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.visualize_clusters --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --method umap --max_points 200000
```

---

## Files to manually transfer from old run folder

When moving to a new run folder, copy the WebGL **app files** and regenerate all data:

| File | Action |
|---|---|
| `WebGL/index.html` | Copy |
| `WebGL/generate_webgl_data.py` | Copy |
| `WebGL/top_scores_proxy.py` | Copy |
| `WebGL/.gitignore` | Copy |
| `WebGL/.gitattributes` | Copy |
| `WebGL/LICENSE` | Copy |
| `WebGL/README.md` | Copy *(update run-dir references inside)* |
| `WebGL/data/` | **Regenerate** via `generate_webgl_data.py` |
| `cluster_tag_summary.json/csv` | **Regenerate** via `label_clusters.py` |

### 8 — Serve locally

Binary data files must be served over HTTP (browsers block direct `file://` access to `ArrayBuffer` fetches).

```bat
cd CORE\models\outputs\20260309_UNSUP_083351\WebGL
..\..\..\..\..\env\Scripts\python.exe -m http.server 8160
```

Open `http://localhost:8160` in any modern browser.
