from __future__ import annotations

from typing import Any

from miraiq.model.recommender import RecommendationFilters, load_engine

from .base import BaseService


class RecommendService(BaseService):
    def recommend(self, username: str, payload: dict[str, Any]) -> dict[str, Any]:
        available = self.cm(username).list_available_models(username)
        if not available:
            raise ValueError("No trained models yet. Train one first.")

        model_name = payload.get("model_name") or available[0]
        if model_name not in available:
            raise ValueError(
                f"Model '{model_name}' not available. Available: {available}"
            )

        engine = load_engine(
            username, cache_dir=self.cache_dir, model_name=model_name
        )
        filters = RecommendationFilters(
            include_genres=payload.get("include_genres"),
            exclude_genres=payload.get("exclude_genres"),
            include_tags=payload.get("include_tags"),
            exclude_tags=payload.get("exclude_tags"),
            min_tag_rank=int(payload.get("min_tag_rank", 50)),
            include_planning=bool(payload.get("include_planning", True)),
            hide_sequels_without_prequel=not bool(
                payload.get("show_sequels", False)
            ),
            sort_by=payload.get("sort_by", "predicted"),
            limit=int(payload.get("limit", 20)),
        )
        results = engine.recommend(filters)
        return {
            "model": model_name,
            "count": len(results),
            "recommendations": [r.to_dict() for r in results],
        }
