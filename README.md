# osu! Maps Cluster Model and Visualization

## Layout

| Path | Role |
|---|---|
| `CORE/data/clustering_dataset_builder.py` | Download `.osu` files, build window features, optional echosu tags |
| `CORE/data/collect_echosu_tag_counts.py` | Tag-count JSONL for post-hoc cluster labels |
| `CORE/models/unsupervised_clustering/` | Contrastive GRU encoder, clustering, labeling, static 2D plot |
| `CORE/models/embedding_model/dataset.py` | Shared window-feature builder used by training/clustering |
| `webgl/` | Canonical Cluster Explorer (`index.html`, data export, top-scores proxy) |
| `CORE/data/dataset_full_osu/` | Junction to the existing echoModel corpus (not a second copy) |
| `CORE/data/raw/` | Junction to existing echosu tag JSON |
| `CORE/models/outputs/20260309_UNSUP_083351/` | Junction to the latest trained run + explorer data |
| `CLUSTERING_PIPELINE.md` | Full CLI flag reference |
| `CORE/models/unsupervised_clustering/COMMANDS.md` | Day-to-day command cookbook |

## Setup

```powershell
cd C:\Projects\echoCluster
python -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

Rename `.env.example` to `.env` and fill in `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET`, and `ECHOSU_TOKEN`. 

Run all `python -m CORE...` commands from this repo root so `CORE` is importable.

## Pipeline

```
dataset_builder → train → cluster → label_clusters → generate_webgl_data
```

See `CLUSTERING_PIPELINE.md` for flags. Bare-bones:

```powershell
.\env\Scripts\python.exe -m CORE.data.clustering_dataset_builder --max_id 5600000 --fetch_echosu
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.train --epochs 10 --max_maps 20000
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.cluster --run_dir RUN_DIR --k 200
.\env\Scripts\python.exe -m CORE.models.unsupervised_clustering.label_clusters --run_dir RUN_DIR
.\env\Scripts\python.exe webgl\generate_webgl_data.py --run_dir RUN_DIR
```

## Run Locally

```powershell
cd CORE\models\outputs\20260309_UNSUP_083351\WebGL
..\..\..\..\..\env\Scripts\python.exe -m http.server 8160
```

Then open `http://localhost:8160`. Use the **2D** / **3D sphere** toggle in the left panel.

The editable explorer template lives in `webgl/`. After a new clustering run, copy those app files into `<run_dir>/WebGL/` (or pass `--webgl_dir`) and regenerate data.
