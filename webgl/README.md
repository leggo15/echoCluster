# echoCluster — WebGL Cluster Explorer

An interactive browser-based visualisation of osu! beatmap clusters produced by the **echoCluster** unsupervised pipeline. Every beatmap is plotted as a point whose position encodes its learned embedding, projected to either a flat 2D plane or a textured 3D sphere. Contour overlays show how musical and difficulty properties vary across the embedding space.

---

## What this shows

echoCluster processes every `.osu` file into a sequence of beat-aligned 4-beat window feature vectors. Each window captures timing, note density, movement patterns, slider shape and speed, curve-type distribution, and difficulty settings. A **bidirectional GRU encoder** with attention pooling is then trained on these sequences using **SimCLR-style contrastive learning** (InfoNCE loss): two randomly augmented views of the same map are pushed together in embedding space while different maps are pushed apart. The result is a 128-dimensional L2-normalised embedding per map that encodes playing style rather than just difficulty.

Maps are then grouped by running **recursive agglomerative clustering** on the full embedding set:
1. MiniBatch K-Means partitions all maps into rough proto-clusters.
2. HDBSCAN finds dense sub-groups within each proto-cluster; isolated maps become singletons.
3. Ward hierarchical linkage merges all sub-cluster centroids into final clusters, cut at a distance threshold.

Alternative methods (K-Means, MiniBatch K-Means, two-stage agglomerative) are also supported. This viewer lets you explore the resulting latent space — maps that play similarly are close together; maps that differ are far apart.

---

## Quickstart

### 1 — Generate the browser data

Run `generate_webgl_data.py` from the repo root, pointing it at the run directory that contains `clusters.csv` and `cluster_progress/`.

```bat
"C:\Projects\echoCluster\env\Scripts\python.exe" ^
  "C:\Projects\echoCluster\webgl\generate_webgl_data.py" ^
  --run_dir "C:\Projects\echoCluster\CORE\models\outputs\20260309_UNSUP_083351" ^
  --max_points 0 ^
  --sphere_mesh_subdivisions 8 ^
  --contour_smooth_sphere 8
```

`--max_points 0` means *use all points*. Set a small number (e.g. `10000`) for a fast preview run.

### 2 — Serve locally

Binary data files must be served over HTTP (browsers block direct `file://` access to `ArrayBuffer` fetches).

```bat
cd CORE\models\outputs\20260223_UNSUP_234612\WebGL
..\..\..\..\..\env\Scripts\python.exe -m http.server 8160
```

Open `http://localhost:8160` in any modern browser.

### 3 — Hosting on GitHub Pages

The viewer is fully static. Copy the `WebGL/` folder into a repository, enable GitHub Pages for that repo, and `index.html` serves directly. No server-side code required.

---

## Data files (`data/`)

All files are written by `generate_webgl_data.py`. The viewer loads them over HTTP at startup.

| File | Format | Description |
|---|---|---|
| `points.f32.bin` | Binary Float32 | 2D PCA coordinates + per-point attributes (stride in `meta.json`). Fields: `x, y, cluster, is_labeled, alpha, star, status_ranked, hp, od, cs, ar, bpm, length`. |
| `points3d.f32.bin` | Binary Float32 | 3D unit-sphere coordinates for the same points, stride in `meta.json`. |
| `map_ids.json` | JSON array | Parallel array of osu! map IDs. Used for `map_id → index` lookups and to open beatmap URLs on click. |
| `map_hover.json` | JSON array | Rich hover text per map (title, mapper). Loaded in the background after the rest of the UI is ready. |
| `meta.json` | JSON object | Run-level metadata: binary strides, total point count, source run directory, generation timestamp. |
| `labels.json` | JSON array | 2D `[x, y, text, cluster]` label positions for cluster name overlays. |
| `labels3d.json` | JSON array | Equivalent labels projected onto the sphere surface (`x, y, z`). |
| `cluster_details.json` | JSON array | Per-cluster statistics: total map count, labeled map count, ranked map count, representative tags. |
| `contours.json` | JSON object | Precomputed 2D contour line segments for all metrics. |
| `contours3d.json` | JSON object | Same structure as `contours.json` but paths are `[x, y, z]` points on the sphere surface. |
| `surface3d.json` | JSON object | Shared icosphere mesh (`vertices`, `faces`) plus per-face scalar values for each metric. Used for the smooth gradient surface in 3D. |

> `points.json` / `points3d.json` are JSON equivalents written for debugging and are **not loaded** by the viewer.

### Binary point layout

#### `points.f32.bin` — stride 13

| Offset | Field | Notes |
|---|---|---|
| 0 | `x` | 2D PCA coordinate |
| 1 | `y` | 2D PCA coordinate |
| 2 | `cluster` | Integer cluster ID |
| 3 | `is_labeled` | 0.0 or 1.0 |
| 4 | `alpha` | Display opacity (35–245), derived from 2D local density |
| 5 | `star` | Star rating (NaN if unknown) |
| 6 | `status_ranked` | 1 = ranked/qualified, 0 = unranked, −1 = unknown |
| 7 | `hp` | HP drain |
| 8 | `od` | Overall Difficulty |
| 9 | `cs` | Circle Size |
| 10 | `ar` | Approach Rate |
| 11 | `bpm` | Beats per minute |
| 12 | `length` | Drain length in seconds (NaN if unknown) |

#### `points3d.f32.bin` — stride 14

Offsets 0–2 hold the `x, y, z` unit-sphere position. The remaining eleven fields mirror the 2D layout from offset 3 onward (cluster through length).

---

## How the data is generated (`generate_webgl_data.py`)

### Embeddings and dimensionality reduction

Raw embeddings are loaded from `embeddings.npy` (preferred) or assembled from chunk checkpoints in `cluster_progress/`.

- **2D projection** — PCA to 2 components. Axes preserve the dominant variance directions.
- **3D projection** — PCA to 3 components, then each point is L2-normalised onto a unit sphere. Proximity on the sphere reflects the same similarity structure as the 2D view, from a third axis of variance.

### Per-point metadata

`load_meta_fields()` reads each map's `{map_id}_meta.json` from the dataset's `processed/maps_by_id/` tree and returns star rating, BPM, AR, OD, CS, HP, drain length, and ranked status. Maps with no metadata file are assigned NaN for numeric fields and −1 for status.

### Density

- **2D alpha** — `density_alpha()` bins points into a 240×240 histogram and maps bin occupancy to an opacity value in [35, 245]. Higher density → more opaque.
- **2D contour values** — `local_density_values()` returns raw bin counts for the density contour metric.
- **3D contour values** — `local_density_values_3d()` converts points to latitude/longitude, bins them on the sphere, and normalises each bin by its solid angle (`cos(lat) · dlat · dlon`) to prevent polar compression from creating false density hotspots.

### Contour generation

#### 2D contours

`build_value_contours()` places a regular grid over the 2D embedding extent, queries the k-nearest neighbours of each grid node, and interpolates a weighted scalar value using a Gaussian kernel (σ = 1.5 × median neighbour distance). The grid is smoothed with Laplacian passes, then contour isolines are extracted by Marching Squares.

#### 3D sphere contours

`build_sphere_value_contours()` projects a subdivided icosphere mesh onto the unit sphere, queries kNN for each mesh vertex using the 3D sphere positions, and interpolates values with the same Gaussian kernel. After Laplacian smoothing, contours are extracted by intersecting each triangle edge pair; crossing points are joined with **spherical linear interpolation (slerp)** arcs to keep them on the sphere surface.

#### Star-rating boost

High-star maps are weighted more heavily so regions with difficult content are visible even when numerically sparse. Weight is flat from 0–8★, then scales via a `tanh`-shaped curve that asymptotes at 15★. Controlled by `--star_boost_threshold`, `--star_boost_strength`, `--star_boost_power`, and `--star_boost_cap`.

#### BPM boost

Low-BPM maps use a power-curve boost (`--bpm_low_boost_strength`, `--bpm_low_boost_power`) so slow-BPM structure is not washed out by the numerically dominant mid-range BPM population.

### Sphere surface mesh

`build_sphere_surface_metric()` assigns each icosphere face a scalar value (same kNN + Gaussian interpolation used for contours). Mesh geometry and per-face values are written to `surface3d.json` for all metrics. The viewer performs per-vertex averaging and colour-mapping at load time in JavaScript for smooth GPU-interpolated shading.

### Key CLI arguments

| Argument | Default | Effect |
|---|---|---|
| `--run_dir` | *(required)* | Path to the clustering run output folder |
| `--max_points` | `500000` | Cap on total points; `0` = all |
| `--sphere_mesh_subdivisions` | `5` | Icosphere subdivision depth (8 recommended for publication) |
| `--contour_smooth_sphere` | `6` | Laplacian smoothing passes for 3D contours/surface |
| `--contour_smooth` | `3` | Laplacian smoothing passes for 2D contours |
| `--contour_grid` | `180` | 2D contour grid resolution |
| `--contours` | `1` | Set to `0` to skip contour generation |
| `--seed` | `42` | Random seed for PCA reproducibility |

---

## How the viewer works (`index.html`)

The viewer is a single self-contained HTML file. It uses [deck.gl 8.9](https://deck.gl) for WebGL rendering and [noUiSlider 15](https://refreshless.com/nouislider/) for range filter controls.

### Architecture overview

All logic runs inside a single async `init()` function. Data is loaded in parallel at startup; the deck.gl instance is created once and updated by calling `deckgl.setProps({layers: buildLayers()})` whenever any filter or display setting changes.

### Projection modes

**2D (`OrthographicView`)** — Points are drawn in the raw PCA coordinate space. Zoom and pan are unrestricted within soft limits. Cluster label `TextLayer` instances are rendered in pixel units with `sizeMinPixels`/`sizeMaxPixels` clamping so labels remain legible across zoom levels.

**3D sphere (`OrbitView` + `OrbitController`)** — The camera orbits a fixed centre. Scroll zoom is mapped to camera distance (half the default deck.gl speed). Minimum/maximum zoom distances prevent clipping through or flying too far from the sphere.

### Rendering layers

| Layer | Type | Purpose |
|---|---|---|
| Surface | `SmoothSphereSurfaceLayer` | Smooth per-vertex gradient mesh on the sphere (3D only) |
| Contours 3D | `PathLayer` | Contour arcs as `[x,y,z]` polylines on the sphere surface (3D only) |
| Contours 2D | `PathLayer` | Contour polylines in PCA space (2D only) |
| Points | `SimpleMeshLayer` | Per-point flat disc meshes, tangent to the sphere in 3D |
| Highlight rings | `SimpleMeshLayer` (3D) / `ScatterplotLayer` (2D) | Annulus ring around each highlighted/searched map |
| Labels | `TextLayer` | Cluster name overlays |

### `SmoothSphereSurfaceLayer`

A subclass of `SimpleMeshLayer` that injects a custom `vtxColor` attribute (`vec3`, values in [0, 1]) into the vertex shader via the `DECKGL_FILTER_COLOR` hook. The GPU interpolates `vtxColor` linearly across each triangle, producing fully smooth shading with no band boundaries regardless of zoom level.

### Point rendering

Each point is a 16-segment flat disc mesh (`DISC_MESH`). In 3D mode, `sphereTangentMatrix(i)` computes a 4×4 rotation matrix that aligns the disc's local Z axis with the outward sphere normal, making each disc lie tangent to the sphere surface. Points are scaled by star rating so higher-star maps render on top of lower-star maps at any viewing angle.

### Highlight rings

Search results and selected-cluster points get a hollow annulus ring (`RING_MESH`). In 3D, rings lie flush with the sphere surface, lifted slightly above it. Points in highlighted clusters are lifted an additional offset so they visually pop out from the surrounding surface.

### Filtering

All filtering is evaluated per-point in `render()`. A point passes if:
- `labeledOnly` is off, or the point is labeled.
- `rankedMode` matches the point's ranked status.
- Star, BPM, AR, OD, CS, HP, and length values are within the active slider range (or NaN when `keepUnknown` is set — currently `false` for all metric filters).

All range slider value displays are **click-to-edit**: clicking the value text activates an inline input pre-filled with the current range in `lo-hi` format. Length accepts both `m:ss-m:ss` and plain seconds. Press Enter or click away to apply; press Escape to cancel.

Star and BPM sliders support open-ended upper bounds (`15+` and `300+`).

### Search

Two modes are available via the **Search** row:

- **Map ID** — finds the single map matching the typed ID and selects its cluster.
- **Mapper** — maps a mapper name to all their map indices. All matching maps are highlighted; matching clusters are listed in the collapsible top-right panel.

### Cluster interaction

Clicking a highlighted point opens `https://osu.ppy.sh/beatmaps/{map_id}` in a new tab. Clicking elsewhere deselects. Press **Escape** to clear any selection or search.

### Contour colour mapping

- **Star** — hand-crafted ramp matching osu! difficulty colours: blue → cyan → green → yellow → red → purple → white (0–12+★).
- **All other metrics** — two-stop linear ramp with unique colours per metric.

---

## Source files

| File | Role |
|---|---|
| `index.html` | Self-contained viewer (HTML + CSS + JS, no build step) |
| `generate_webgl_data.py` | Python data generation pipeline |
| `data/` | Output directory written by `generate_webgl_data.py` |

**Runtime dependencies (CDN):**
- [deck.gl 8.9.36](https://unpkg.com/deck.gl@8.9.36/dist.min.js)
- [noUiSlider 15.8.1](https://cdn.jsdelivr.net/npm/nouislider@15.8.1/)

**Python dependencies:** `numpy`, `pandas`, `scikit-learn` (PCA, NearestNeighbors), `requests`.
