from __future__ import annotations

from flask_cors import CORS


def init_extensions(app) -> None:
    CORS(app)
