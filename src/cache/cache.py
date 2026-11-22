"""Local JSON cache for AniList data and trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from src.anilist.models import (
    CachedMedia,
    CachedUserEntry,
    CacheMetadata,
    TasteDriver,
)


class CacheManager:
    """Manages local JSON cache for user data, anime catalog, and model artifacts."""

    GLOBAL_CATALOG_FILE = "media_catalog.json"

    def __init__(self, cache_dir: Path | str = "~/.miraiq/cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.user_dir = self.cache_dir / "users"
        self.user_dir.mkdir(exist_ok=True)

        self.model_dir = self.cache_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

    def _user_cache_dir(self, username: str) -> Path:
        """Get user-specific cache directory."""
        safe_name = username.lower().replace(" ", "_")
        user_dir = self.user_dir / safe_name
        user_dir.mkdir(exist_ok=True)
        return user_dir

    def _global_catalog_file(self) -> Path:
        """Get global shared catalog file path."""
        return self.cache_dir / self.GLOBAL_CATALOG_FILE

    def _user_entries_file(self, username: str) -> Path:
        return self._user_cache_dir(username) / "user_entries.json"

    def _metadata_file(self, username: str) -> Path:
        return self._user_cache_dir(username) / "metadata.json"

    def _model_file(self, username: str) -> Path:
        return self.model_dir / f"{username.lower().replace(' ', '_')}_model.joblib"

    def _features_file(self, username: str) -> Path:
        return self.model_dir / f"{username.lower().replace(' ', '_')}_features.json"

    def _taste_drivers_file(self, username: str) -> Path:
        return self.model_dir / f"{username.lower().replace(' ', '_')}_taste.json"

    # --- Media Catalog (Global, shared across users) ---

    def save_media_catalog(self, username: str, media_list: list[CachedMedia]) -> None:
        """Save anime catalog to global cache (shared across all users)."""
        file_path = self._global_catalog_file()
        data = [media.model_dump() for media in media_list]
        file_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def load_media_catalog(self, username: str) -> list[CachedMedia]:
        """Load anime catalog from global cache."""
        file_path = self._global_catalog_file()
        if not file_path.exists():
            return []
        data = orjson.loads(file_path.read_bytes())
        return [CachedMedia(**item) for item in data]

    def global_catalog_exists(self) -> bool:
        """Check if global catalog exists."""
        return self._global_catalog_file().exists()

    def clear_global_catalog(self) -> None:
        """Clear global catalog."""
        file_path = self._global_catalog_file()
        if file_path.exists():
            file_path.unlink()

    # --- User Entries ---

    def save_user_entries(self, username: str, entries: list[CachedUserEntry]) -> None:
        """Save user's anime list entries to cache."""
        file_path = self._user_entries_file(username)
        data = [entry.model_dump() for entry in entries]
        file_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def load_user_entries(self, username: str) -> list[CachedUserEntry]:
        """Load user's anime list entries from cache."""
        file_path = self._user_entries_file(username)
        if not file_path.exists():
            return []
        data = orjson.loads(file_path.read_bytes())
        return [CachedUserEntry(**item) for item in data]

    # --- Metadata ---

    def save_metadata(self, metadata: CacheMetadata) -> None:
        """Save cache metadata."""
        file_path = self._metadata_file(metadata.username)
        file_path.write_bytes(orjson.dumps(metadata.model_dump(), option=orjson.OPT_INDENT_2))

    def load_metadata(self, username: str) -> CacheMetadata | None:
        """Load cache metadata."""
        file_path = self._metadata_file(username)
        if not file_path.exists():
            return None
        data = orjson.loads(file_path.read_bytes())
        return CacheMetadata(**data)

    # --- Model Artifacts ---

    def save_model(self, username: str, model: Any, feature_names: list[str]) -> None:
        """Save trained model and feature names."""
        import joblib

        model_file = self._model_file(username)
        features_file = self._features_file(username)

        joblib.dump(model, model_file)
        features_file.write_bytes(
            orjson.dumps({"feature_names": feature_names}, option=orjson.OPT_INDENT_2)
        )

    def load_model(self, username: str) -> tuple[Any, list[str]] | None:
        """Load trained model and feature names."""
        import joblib

        model_file = self._model_file(username)
        features_file = self._features_file(username)

        if not model_file.exists() or not features_file.exists():
            return None

        model = joblib.load(model_file)
        features_data = orjson.loads(features_file.read_bytes())
        # Handle old format (just feature names list) and new format (dict)
        if isinstance(features_data, dict):
            feature_names = features_data.get("feature_names", [])
        else:
            feature_names = features_data
        return model, feature_names

    def save_taste_drivers(self, username: str, drivers: list[TasteDriver]) -> None:
        """Save learned taste drivers."""
        file_path = self._taste_drivers_file(username)
        data = [driver.model_dump() for driver in drivers]
        file_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def load_taste_drivers(self, username: str) -> list[TasteDriver]:
        """Load learned taste drivers."""
        file_path = self._taste_drivers_file(username)
        if not file_path.exists():
            return []
        data = orjson.loads(file_path.read_bytes())
        return [TasteDriver(**item) for item in data]

    # --- Utility ---

    def cache_exists(self, username: str) -> bool:
        """Check if user cache exists (global catalog + user entries)."""
        return self.global_catalog_exists() and self._user_entries_file(username).exists()

    def model_exists(self, username: str) -> bool:
        """Check if trained model exists."""
        return self._model_file(username).exists() and self._features_file(username).exists()

    def clear_user_cache(self, username: str, clear_global: bool = False) -> None:
        """Clear all cache for a user."""
        import shutil

        user_dir = self._user_cache_dir(username)
        if user_dir.exists():
            shutil.rmtree(user_dir)

        # Also clear model files
        for suffix in ["_model.joblib", "_features.json", "_taste.json"]:
            model_file = self.model_dir / f"{username.lower().replace(' ', '_')}{suffix}"
            if model_file.exists():
                model_file.unlink()

        # Optionally clear global catalog (only if explicitly requested)
        if clear_global:
            self.clear_global_catalog()

    def get_cache_info(self, username: str) -> dict[str, Any]:
        """Get cache statistics."""
        user_dir = self._user_cache_dir(username)
        if not user_dir.exists() and not self.global_catalog_exists():
            return {"exists": False}

        global_media_file = self._global_catalog_file()
        entries_file = self._user_entries_file(username)
        metadata_file = self._metadata_file(username)
        model_file = self._model_file(username)

        info = {
            "exists": True,
            "cache_dir": str(user_dir),
            "media_catalog": global_media_file.exists(),
            "user_entries": entries_file.exists(),
            "metadata": metadata_file.exists(),
            "model": model_file.exists(),
            "global_catalog": global_media_file.exists(),
        }

        if global_media_file.exists():
            info["media_catalog_size_mb"] = global_media_file.stat().st_size / (1024 * 1024)
        if entries_file.exists():
            info["user_entries_size_mb"] = entries_file.stat().st_size / (1024 * 1024)

        metadata = self.load_metadata(username)
        if metadata:
            info["synced_at"] = metadata.synced_at
            info["total_user_entries"] = metadata.total_user_entries
            info["total_cached_media"] = metadata.total_cached_media

        return info
