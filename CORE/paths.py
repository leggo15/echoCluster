"""Project-root paths and .env loading for echoCluster."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (PROJECT_ROOT / "CORE" / "data" / "dataset_full_osu").resolve()
DEFAULT_ECHOSU_JSON = (PROJECT_ROOT / "CORE" / "data" / "raw" / "tag_data_with_ids.json").resolve()
WEBGL_DIR = (PROJECT_ROOT / "webgl").resolve()
OUTPUTS_DIR = (PROJECT_ROOT / "CORE" / "models" / "outputs").resolve()


def load_env() -> None:
    from dotenv import find_dotenv, load_dotenv

    for candidate in (PROJECT_ROOT / ".env", PROJECT_ROOT / "CORE" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=True)
            return
    load_dotenv(find_dotenv(), override=True)
