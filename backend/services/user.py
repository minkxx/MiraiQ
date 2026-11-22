from __future__ import annotations

from typing import Any

from miraiq.anilist.sync import run_fetch_catalog, run_sync

from .base import BaseService


class UserService(BaseService):
    def info(self, username: str) -> dict[str, Any]:
        return self.cm(username).get_cache_info(username)

    def models(self, username: str) -> dict[str, Any]:
        return {"models": self.cm(username).list_available_models(username)}

    def sync(self, username: str, payload: dict[str, Any]) -> dict[str, Any]:
        force = bool(payload.get("force", False))
        metadata = self.run_async(
            run_sync(username, cache_dir=self.cache_dir, force=force)
        )
        return {
            "user_id": metadata.user_id,
            "total_user_entries": metadata.total_user_entries,
            "synced_at": metadata.synced_at,
        }

    def catalog(self, payload: dict[str, Any]) -> dict[str, Any]:
        limit = int(payload.get("limit", 2000))
        metadata = self.run_async(
            run_fetch_catalog(cache_dir=self.cache_dir, limit=limit)
        )
        return {"total_cached_media": metadata.total_cached_media}
