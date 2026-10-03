"""Repo-relative paths. Every module gets its directories from here."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dir(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else default


CACHE_DIR = _dir("VERIMEM_CACHE_DIR", REPO_ROOT / "cache")
DATA_DIR = _dir("VERIMEM_DATA_DIR", REPO_ROOT / "data")
RESULTS_DIR = _dir("VERIMEM_RESULTS_DIR", REPO_ROOT / "results")
ATTACKS_DIR = REPO_ROOT / "attacks"


def run_dir(name: str, date: str | None = None) -> Path:
    """`results/<YYYY-MM-DD>_<name>/` — rule 7. Created on first use."""
    from datetime import date as _date

    stamp = date or _date.today().isoformat()
    path = RESULTS_DIR / f"{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path
