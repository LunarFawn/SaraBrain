"""Default paths and configuration for Sara Brain."""

from __future__ import annotations

import os
from pathlib import Path


def default_db_path() -> str:
    """Return the default persistent DB path: ~/.sara_brain/sara.db"""
    sara_dir = Path.home() / ".sara_brain"
    sara_dir.mkdir(exist_ok=True)
    return str(sara_dir / "sara.db")
