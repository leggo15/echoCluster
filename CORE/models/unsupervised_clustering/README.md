## Unsupervised clustering over beatmap window features

This module adds a **tag-free** training path:

1. **Embed** each map into a fixed-length vector using only window feature sequences (the same features used by the supervised models).
2. **Cluster** embeddings (e.g., KMeans / MiniBatchKMeans).
3. **Optionally label clusters** by joining to the echosu dataset (tags/spans) after-the-fact.

### Why this exists

This path learns a **latent “map-style” space** from window features only (no tags during training), then studies how echosu tags sit on that space after clustering.

### Files

- `dataset.py`: load map window sequences and sample two augmented views for contrastive training.
- `model.py`: GRU encoder + projection head.
- `train.py`: self-supervised training (InfoNCE).
- `cluster.py`: embed maps and fit a clustering model; save per-map cluster ids.
- `label_clusters.py`: compute cluster→tag summaries using echosu-derived labels.
- `visualize_clusters.py`: project embeddings to 2D and render density-colored cluster map with optional tag labels.

### Quickstart (PowerShell)

Train an unsupervised embedder on maps present under `CORE/data/dataset/processed/maps_by_id`:

```powershell
python -m CORE.models.unsupervised_clustering.train --max_maps 20000 --epochs 10
```

Cluster embeddings:

```powershell
python -m CORE.models.unsupervised_clustering.cluster --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --k 200
```

Label clusters using echosu tags (only maps that have tag artifacts contribute):

```powershell
python -m CORE.models.unsupervised_clustering.label_clusters --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351"
```

Render a 2D visualization after clustering + labeling:

```powershell
python -m CORE.models.unsupervised_clustering.visualize_clusters --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" --method umap --max_points 200000
```

