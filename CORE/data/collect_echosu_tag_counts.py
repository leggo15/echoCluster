"""
collect_echosu_tag_counts.py

Fetch *only* what we need from echosu for post-hoc labeling:
- beatmap_id list
- per-beatmap tag counts (how many times applied), excluding predictions

Output:
  <OUT>.jsonl with lines:
    {"map_id":"12345","tags":{"aim":4,"streams":2,...}}

Run (PowerShell, using repo venv):
  cd C:\Projects\echoCluster
  .\env\Scripts\python.exe CORE\data\collect_echosu_tag_counts.py --out "C:\Projects\echoCluster\CORE\data\dataset_full_osu\raw\echosu_tag_counts.jsonl"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm.auto import tqdm

from CORE.paths import load_env


ECHOSU_BASE = "https://www.echosu.com"
ECHOSU_API = f"{ECHOSU_BASE}/api"
ECHOSU_BEATMAPS = f"{ECHOSU_API}/beatmaps/"
ECHOSU_TAG_APPS = f"{ECHOSU_API}/tag-applications/"


def _load_env() -> None:
    load_env()


def fetch_all_beatmaps(token: str, *, page_size: int = 500, sleep_s: float = 0.2) -> List[dict]:
    hdr = {"Authorization": f"Token {token}", "Accept": "application/json"}
    url = ECHOSU_BEATMAPS
    params = {"page_size": int(page_size)}
    out: List[dict] = []
    while url:
        r = requests.get(url, headers=hdr, params=(params if "?" not in url else None), timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"echosu beatmaps HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if isinstance(data, dict) and "results" in data:
            out.extend(data.get("results", []))
            url = data.get("next")
            params = None
        elif isinstance(data, list):
            out = data
            url = None
        else:
            url = None
        time.sleep(float(sleep_s))
    return out


def fetch_tag_counts(token: str, beatmap_id: str, *, sleep_s: float = 0.2) -> Dict[str, int]:
    hdr = {"Authorization": f"Token {token}", "Accept": "application/json"}
    params = {"beatmap_id": str(beatmap_id), "include": "tag_counts"}
    r = requests.get(ECHOSU_TAG_APPS, headers=hdr, params=params, timeout=30)
    if r.status_code != 200:
        return {}
    items = r.json() or []
    counts: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("is_predicted") is True:
            continue
        if item.get("is_negative") is True:
            continue
        tag = item.get("tag") or {}
        name = (tag.get("name") if isinstance(tag, dict) else None)
        if not name:
            continue
        cnt = tag.get("count")
        if isinstance(cnt, int):
            counts[str(name)] = max(int(cnt), counts.get(str(name), 0))
    time.sleep(float(sleep_s))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--page_size", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="Debug: limit number of beatmaps (0 = no limit)")
    args = ap.parse_args()

    _load_env()
    token = os.getenv("ECHOSU_TOKEN")
    if not token:
        raise RuntimeError("Missing ECHOSU_TOKEN in ../.env")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[echosu] Fetching beatmap list …")
    beatmaps = fetch_all_beatmaps(token, page_size=int(args.page_size), sleep_s=float(args.sleep))
    if int(args.limit) > 0:
        beatmaps = beatmaps[: int(args.limit)]
    print(f"[echosu] beatmaps={len(beatmaps)}")

    # Resume support: build a set of already written IDs (cheap for jsonl)
    done = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open("r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        mid = obj.get("map_id")
                        if mid:
                            done.add(str(mid))
                    except Exception:
                        continue
        except Exception:
            done = set()

    with out_path.open("a", encoding="utf-8") as fout:
        for bm in tqdm(beatmaps, desc="echosu tag counts", unit="map"):
            mid = str(bm.get("beatmap_id") or bm.get("id") or "").strip()
            if not mid or mid in done:
                continue
            tags = fetch_tag_counts(token, mid, sleep_s=float(args.sleep))
            fout.write(json.dumps({"map_id": mid, "tags": tags}, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[echosu] wrote {out_path}")


if __name__ == "__main__":
    main()

