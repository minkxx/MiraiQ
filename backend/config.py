from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_CACHE_DIR = str(PROJECT_ROOT / ".cache")

HOST = os.getenv("MIRAIQ_HOST", "127.0.0.1")
PORT = int(os.getenv("MIRAIQ_PORT", "5000"))
DEBUG = os.getenv("MIRAIQ_DEBUG", "false").lower() in ("1", "true", "yes")

CACHE_DIR = os.getenv("MIRAIQ_CACHE_DIR", DEFAULT_CACHE_DIR)
