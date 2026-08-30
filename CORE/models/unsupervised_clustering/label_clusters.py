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
        "  .\\env\\Scripts\\python.exe -m CORE.models.unsupervised_clustering.label_clusters ...\n"
    ) from e

from CORE.paths import DEFAULT_ECHOSU_JSON

from .dataset import DATASET_ROOT


PROCESSED_DIR = (DATASET_ROOT / "processed" / "maps_by_id").resolve()


def read_spans(map_id: str) -> List[Tuple[int, int, str]]:
    f = PROCESSED_DIR / str(map_id) / f"{map_id}_spans.csv"
    if not f.exists():
        return []
    try:
        df = pd.read_csv(f)
    except Exception:
        return []
    out: List[Tuple[int, int, str]] = []
    for _, r in df.iterrows():
        try:
            s = int(r["start_ms"])
            e = int(r["end_ms"])
            lab = str(r["label"]).strip()
        except Exception:
            continue
        if lab and e > s:
            out.append((s, e, lab))
    return out


def read_weak_tags(map_id: str) -> List[str]:
    f = PROCESSED_DIR / str(map_id) / f"{map_id}_maplevel_tags.json"
    if not f.exists():
        return []
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(k) for k, v in (obj or {}).items() if v]


def map_level_positive_tags(map_id: str) -> List[str]:
    """A map is positive for a tag if it appears in spans OR weak map-level tags.

    This mirrors the repo’s global-label logic used in the supervised models.
    """
    tags = set()
    for _s, _e, lab in read_spans(map_id):
        if lab:
            tags.add(str(lab))
    for lab in read_weak_tags(map_id):
        if lab:
            tags.add(str(lab))
    return sorted(tags)


def load_echosu_tag_sets(json_path: Path) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Load positive/negative tag sets per map from an echosu tag file.

    Supports two formats:

    1. JSONL  (one object per line) — produced by collect_echosu_tag_counts.py /
       clustering_dataset_builder.py.  Tags are already filtered to positive
       human-applied ones, stored as plain int counts:
         {"map_id": "123", "tags": {"aim": 4, "streams": 2}}

    2. JSON array — older format where each tag entry is a metadata dict:
         [{"map_id": "123", "tags": {"aim": {"count": 4, "is_negative": false}}}]
    """
    if not json_path.exists():
        raise SystemExit(f"Missing echosu json file: {json_path}")

    # Try to load as a standard JSON document first; fall back to JSONL.
    raw = json_path.read_text(encoding="utf-8")
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pass  # likely JSONL — parse line by line below

    if data is None:
        # JSONL: each non-empty line is a separate JSON object.
        data = []
        for lineno, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[label_clusters] skipping malformed line {lineno}: {e}")

    if not isinstance(data, list):
        # Single-object file — wrap it.
        data = [data] if isinstance(data, dict) else []

    pos_by_map: Dict[str, set] = {}
    neg_by_map: Dict[str, set] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("map_id") or "").strip()
        if not mid:
            continue
        tags = item.get("tags") or {}
        if not isinstance(tags, dict):
            continue
        pos: set = set()
        neg: set = set()
        for name, meta in tags.items():
            t = str(name).strip()
            if not t:
                continue
            if isinstance(meta, dict):
                # Older format: {"count": N, "is_negative": bool, ...}
                is_neg = bool(meta.get("is_negative", False))
                cnt    = int(meta.get("count", 0) or 0)
                if is_neg:
                    neg.add(t)
                elif cnt > 0:
                    pos.add(t)
            else:
                # JSONL format: tag value is a plain int count.
                # Already filtered for positive human-applied tags at write time.
                try:
                    cnt = int(meta)
                except (TypeError, ValueError):
                    cnt = 0
                if cnt > 0:
                    pos.add(t)
        if pos:
            pos_by_map[mid] = pos
        if neg:
            neg_by_map[mid] = neg
    return pos_by_map, neg_by_map


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument(
        "--label_source",
        type=str,
        default="echosu_json",
        choices=["echosu_json", "dataset_artifacts"],
        help="Where to read map->tags labels from.",
    )
    p.add_argument(
        "--echosu_json",
        type=str,
        default=str(DEFAULT_ECHOSU_JSON),
        help="Path to collect_echosu output json when label_source=echosu_json.",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    clusters_path = run_dir / "clusters.csv"
    if not clusters_path.exists():
        raise SystemExit(f"Missing clusters file: {clusters_path}")

    dfc = pd.read_csv(clusters_path)
    if not {"map_id", "cluster"}.issubset(set(dfc.columns)):
        raise SystemExit("clusters.csv must contain columns: map_id, cluster")

    pos_by_map: Dict[str, set] = {}
    if str(args.label_source) == "echosu_json":
        pos_by_map, _neg_by_map = load_echosu_tag_sets(Path(args.echosu_json).resolve())

    # Build tag counts per cluster
    cluster_tag_counts: Dict[int, Dict[str, int]] = {}
    cluster_sizes: Dict[int, int] = {}
    total_tag_counts: Dict[str, int] = {}

    n_with_tags = 0
    for _, row in dfc.iterrows():
        mid = str(row["map_id"])
        c = int(row["cluster"])
        if str(args.label_source) == "echosu_json":
            tags = sorted(pos_by_map.get(mid, set()))
        else:
            tags = map_level_positive_tags(mid)
        if not tags:
            continue
        n_with_tags += 1
        cluster_sizes[c] = cluster_sizes.get(c, 0) + 1
        ct = cluster_tag_counts.setdefault(c, {})
        for t in tags:
            ct[t] = ct.get(t, 0) + 1
            total_tag_counts[t] = total_tag_counts.get(t, 0) + 1

    if n_with_tags <= 0:
        if str(args.label_source) == "echosu_json":
            raise SystemExit(
                f"No clustered maps matched positive tags in echosu json: {Path(args.echosu_json).resolve()}"
            )
        raise SystemExit("No maps in clusters.csv had tag artifacts (spans/weak tags).")

    # Summaries
    rows = []
    total_maps_tagged = int(n_with_tags)
    for c, size in sorted(cluster_sizes.items(), key=lambda kv: kv[0]):
        ct = cluster_tag_counts.get(c, {})
        if not ct:
            continue
        # Rank tags by within-cluster frequency and PMI-ish score
        top = sorted(ct.items(), key=lambda kv: kv[1], reverse=True)
        top = top[: int(args.top_k)]
        for tag, cnt in top:
            p_tag = total_tag_counts.get(tag, 0) / max(1, total_maps_tagged)
            p_tag_in_c = cnt / max(1, size)
            # log(p(tag|c)/p(tag)) as a simple enrichment score
            enrich = float(np.log((p_tag_in_c + 1e-9) / (p_tag + 1e-9)))
            rows.append(
                {
                    "cluster": int(c),
                    "cluster_tagged_maps": int(size),
                    "tag": str(tag),
                    "tag_count_in_cluster": int(cnt),
                    "tag_frac_in_cluster": float(p_tag_in_c),
                    "tag_frac_global": float(p_tag),
                    "enrichment_log": float(enrich),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.sort_values(["cluster", "enrichment_log", "tag_count_in_cluster"], ascending=[True, False, False], inplace=True)
    out_df.to_csv(run_dir / "cluster_tag_summary.csv", index=False)

    # Convenience: one JSON per cluster with top enriched tags
    cluster_json: Dict[str, List[Dict[str, object]]] = {}
    for c in sorted(cluster_sizes.keys()):
        sub = out_df[out_df["cluster"] == int(c)].head(int(args.top_k))
        cluster_json[str(c)] = sub.to_dict(orient="records")
    (run_dir / "cluster_tag_summary.json").write_text(json.dumps(cluster_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[label] tagged_maps_used={n_with_tags} / {len(dfc)} clustered maps")
    print(f"[label] wrote {(run_dir / 'cluster_tag_summary.csv')}")


if __name__ == "__main__":
    main()

