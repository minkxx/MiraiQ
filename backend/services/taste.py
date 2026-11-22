from __future__ import annotations

from typing import Any

from .base import BaseService


class TasteService(BaseService):
    def drivers(self, username: str, model_name: str) -> dict[str, Any]:
        drivers = self.cm(username).load_taste_drivers(
            username, model_name=model_name
        )
        return {
            "model": model_name,
            "count": len(drivers),
            "drivers": [d.model_dump() for d in drivers],
        }
