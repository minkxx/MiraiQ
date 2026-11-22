"""Recommendation engine for generating personalized anime suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.anilist.models import (
    CachedMedia,
    CachedUserEntry,
    RecommendationResult,
)
from src.cache.cache import CacheManager
from src.model.trainer import FeatureExtractor, TasteModel

# Known Chinese/Korean studio name patterns (for filtering)
CHINESE_KOREAN_STUDIO_PATTERNS = [
    "studio lan",
    "paper plane",
    "pb animation",
    "sharefun",
    "aha entertainment",
    "bug films",
    "cmc media",
    "haoliners",
    "bilibili",
    "tencent",
    "col group",
    "park",
    "studio elle",
    "l2 studio",
    "sparkly key",
    "seven stone",
    "b.cmay",
    "big fire bird",
    "studio hibari",
    "studio flad",
    "studio nanahoshi",
]


@dataclass
class RecommendationFilters:
    """Filters for recommendation query."""

    include_genres: list[str] | None = None
    exclude_genres: list[str] | None = None
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    min_tag_rank: int = 50
    formats: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    hide_sequels_without_prequel: bool = True
    include_planning: bool = False
    released_only: bool = True
    exclude_chinese_korean: bool = True
    sort_by: str = "predicted"
    limit: int = 50


class RecommendationEngine:
    """Generates personalized anime recommendations."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.model: TasteModel | None = None
        self.username: str | None = None
        self._user_entries: list[CachedUserEntry] = []
        self._media_catalog: list[CachedMedia] = []
        self._media_by_id: dict[int, CachedMedia] = {}
        self._watched_ids: set[int] = set()
        self._completed_ids: set[int] = set()
        self._planning_ids: set[int] = set()

    def load(self, username: str, model_name: str | None = None) -> None:
        """Load model and data for a user."""
        self.username = username

        # Load model
        model_data = self.cache_manager.load_model(username, model_name=model_name)
        if not model_data:
            available = self.cache_manager.list_available_models(username)
            hint = f" Available models: {available}." if available else ""
            raise ValueError(
                f"No trained model '{model_name or 'gbr'}' found for {username}. "
                f"Run 'miraiq train' first.{hint}"
            )

        pipeline, feature_names = model_data
        self.model = TasteModel()
        self.model.pipeline = pipeline
        self.model.feature_extractor = FeatureExtractor()
        self.model.feature_extractor._feature_names = feature_names
        self.model._is_trained = True

        # Load data
        self._user_entries = self.cache_manager.load_user_entries(username)
        self._media_catalog = self.cache_manager.load_media_catalog(username)
        self._media_by_id = {m.id: m for m in self._media_catalog}

        # Build watched/completed/planning sets
        for entry in self._user_entries:
            self._watched_ids.add(entry.media_id)
            if entry.status == "COMPLETED":
                self._completed_ids.add(entry.media_id)
            elif entry.status == "PLANNING":
                self._planning_ids.add(entry.media_id)

    def _passes_filters(self, media: CachedMedia, filters: RecommendationFilters) -> bool:
        """Check if anime passes all filters."""
        # Released filter (exclude not-yet-released)
        if filters.released_only:
            current_year = datetime.now().year
            if media.status == "NOT_YET_RELEASED":
                return False
            if media.season_year and media.season_year > current_year:
                return False

        # Chinese/Korean filter
        if filters.exclude_chinese_korean and self._is_chinese_korean(media):
            return False

        # Genre filters (ALL specified genres must match)
        if filters.include_genres and not all(g in media.genres for g in filters.include_genres):
            return False
        if filters.exclude_genres and any(g in media.genres for g in filters.exclude_genres):
            return False

        # Tag filters (check tag rank threshold)
        tag_ranks = media.tag_ranks
        if filters.include_tags and not any(
            tag in tag_ranks and tag_ranks[tag] >= filters.min_tag_rank
            for tag in filters.include_tags
        ):
            return False
        if filters.exclude_tags and any(
            tag in tag_ranks and tag_ranks[tag] >= filters.min_tag_rank
            for tag in filters.exclude_tags
        ):
            return False

        # Format filter
        if filters.formats and media.format not in filters.formats:
            return False

        # Year filters
        if filters.year_min and media.season_year and media.season_year < filters.year_min:
            return False
        if filters.year_max and media.season_year and media.season_year > filters.year_max:
            return False

        # Sequel filter
        if filters.hide_sequels_without_prequel and self._is_sequel_without_prequel(media):
            return False

        # Planning filter
        return not (not filters.include_planning and media.id in self._planning_ids)

    def _is_chinese_korean(self, media: CachedMedia) -> bool:
        """Check if anime is likely Chinese or Korean based on studio names."""
        for studio in media.all_studios:
            studio_lower = studio.lower()
            for pattern in CHINESE_KOREAN_STUDIO_PATTERNS:
                if pattern in studio_lower:
                    return True
        return False

    def _is_sequel_without_prequel(self, media: CachedMedia) -> bool:
        """Check if anime is a sequel whose prequel isn't in completed list."""
        for relation in media.relations:
            # Block sequels whose prequel hasn't been completed
            if relation.relation_type == "PREQUEL" and relation.id not in self._completed_ids:
                return True
        return False

    def _get_sequel_prequel_title(self, media: CachedMedia) -> str | None:
        """Get the prequel title if this is a sequel without prequel watched."""
        for relation in media.relations:
            if relation.relation_type == "PREQUEL" and relation.id not in self._completed_ids:
                return relation.title_english or relation.title_romaji
        return None

    def _compute_similarity(self, media1: CachedMedia, media2: CachedMedia) -> dict:
        """Compute similarity between two anime based on shared genres, tags, studios."""
        genres1 = set(media1.genres)
        genres2 = set(media2.genres)
        tags1 = set(media1.tag_ranks.keys())
        tags2 = set(media2.tag_ranks.keys())
        studios1 = set(media1.main_studios)
        studios2 = set(media2.main_studios)

        shared_genres = genres1 & genres2
        shared_tags = tags1 & tags2
        shared_studios = studios1 & studios2

        # Jaccard-like similarity for each dimension
        genre_score = len(shared_genres) / max(len(genres1 | genres2), 1)
        tag_score = len(shared_tags) / max(len(tags1 | tags2), 1)
        studio_score = len(shared_studios) / max(len(studios1 | studios2), 1)

        # Weighted combination
        similarity = 0.5 * genre_score + 0.3 * tag_score + 0.2 * studio_score

        return {
            "shared_genres": sorted(shared_genres),
            "shared_tags": sorted(shared_tags),
            "shared_studios": sorted(shared_studios),
            "similarity": round(similarity, 3),
        }

    def _get_similar_watched(self, media: CachedMedia, top_k: int = 3) -> list[dict]:
        """Find watched anime most similar to the given media."""
        rated_watched = [
            (entry, self._media_by_id[entry.media_id])
            for entry in self._user_entries
            if entry.is_rated and entry.media_id in self._media_by_id and entry.media_id != media.id
        ]

        similarities = []
        for entry, watched_media in rated_watched:
            sim = self._compute_similarity(media, watched_media)
            if sim["similarity"] > 0:
                similarities.append(
                    {
                        "watched_media_id": watched_media.id,
                        "watched_title": watched_media.display_title,
                        "watched_score": entry.score,
                        "shared_genres": sim["shared_genres"],
                        "shared_tags": sim["shared_tags"],
                        "shared_studios": sim["shared_studios"],
                        "similarity": sim["similarity"],
                    }
                )

        # Sort by similarity * watched_score (favor similar AND highly rated)
        similarities.sort(key=lambda x: x["similarity"] * (x["watched_score"] or 5), reverse=True)
        return similarities[:top_k]

    def recommend(self, filters: RecommendationFilters | None = None) -> list[RecommendationResult]:
        """Generate recommendations based on filters."""
        if not self.model or not self.username:
            raise RuntimeError("Engine not loaded. Call load(username) first.")

        filters = filters or RecommendationFilters()

        # Filter candidate anime
        candidates = []
        for media in self._media_catalog:
            if media.id in self._watched_ids and media.id not in self._planning_ids:
                continue  # Skip already watched (but include planning list anime)
            if not self._passes_filters(media, filters):
                continue
            candidates.append(media)

        if not candidates:
            return []

        # Predict scores
        predicted_scores = self.model.predict_batch(candidates)

        # Build watched anime lookup for similarity boost
        # Boost score for anime similar to highly-rated watched anime
        boosted_scores = []
        for media, pred_score in zip(candidates, predicted_scores, strict=False):
            boost = 0.0
            similar_watched = self._get_similar_watched(media, top_k=5)
            for sim in similar_watched:
                if sim["watched_score"] and sim["watched_score"] >= 7.0:
                    # Boost up to +0.5 based on similarity to highly-rated watched anime
                    boost += sim["similarity"] * 0.5 * (sim["watched_score"] / 10.0)
            boost = min(boost, 0.5)  # Cap the boost
            boosted_scores.append(min(max(pred_score + boost, 0.0), 10.0))  # Clamp to 0-10

        # Build results
        results = []
        for media, _, boosted_score in zip(
            candidates, predicted_scores, boosted_scores, strict=False
        ):
            community_score = media.community_score
            score_diff = boosted_score - community_score if community_score else 0.0

            is_planning = media.id in self._planning_ids
            is_sequel = self._is_sequel_without_prequel(media)
            prequel_title = self._get_sequel_prequel_title(media) if is_sequel else None

            # Get similar watched anime for explanation
            similar_to = self._get_similar_watched(media, top_k=3)

            results.append(
                RecommendationResult(
                    media=media,
                    predicted_score=round(boosted_score, 2),
                    community_score=round(community_score, 2) if community_score else None,
                    score_difference=round(score_diff, 2),
                    is_planning=is_planning,
                    is_sequel_without_prequel=is_sequel,
                    prequel_title=prequel_title,
                    similar_to=similar_to,
                )
            )

        if filters.sort_by == "predicted":
            results.sort(key=lambda r: r.predicted_score, reverse=True)
        elif filters.sort_by == "community":
            results.sort(key=lambda r: r.community_score or 0, reverse=True)
        elif filters.sort_by == "difference":
            results.sort(key=lambda r: r.score_difference, reverse=True)

        return results[: filters.limit]

    def get_available_genres(self) -> list[str]:
        """Get all unique genres in catalog."""
        genres = set()
        for media in self._media_catalog:
            genres.update(media.genres)
        return sorted(genres)

    def get_available_tags(self, min_rank: int = 50) -> list[tuple[str, int]]:
        """Get all unique tags with their max rank across catalog."""
        tag_max_rank: dict[str, int] = {}
        for media in self._media_catalog:
            for tag, rank in media.tag_ranks.items():
                if rank >= min_rank:
                    tag_max_rank[tag] = max(tag_max_rank.get(tag, 0), rank)
        return sorted(tag_max_rank.items(), key=lambda x: -x[1])

    def get_available_studios(self) -> list[str]:
        """Get all unique main studios in catalog."""
        studios = set()
        for media in self._media_catalog:
            studios.update(media.main_studios)
        return sorted(studios)

    def get_available_formats(self) -> list[str]:
        """Get all unique formats in catalog."""
        formats = set()
        for media in self._media_catalog:
            if media.format:
                formats.add(media.format)
        return sorted(formats)

    def get_year_range(self) -> tuple[int, int]:
        """Get min/max years in catalog."""
        years = [m.season_year for m in self._media_catalog if m.season_year]
        return (min(years), max(years)) if years else (2000, 2025)

    def search_tags(self, query: str, min_rank: int = 50) -> list[tuple[str, int]]:
        """Search tags by name."""
        query = query.lower()
        all_tags = self.get_available_tags(min_rank)
        return [(tag, rank) for tag, rank in all_tags if query in tag.lower()]


def load_engine(
    username: str, cache_dir: str = "~/.miraiq/cache", model_name: str | None = None
) -> RecommendationEngine:
    """Convenience function to load recommendation engine."""
    cache_manager = CacheManager(cache_dir)
    engine = RecommendationEngine(cache_manager)
    engine.load(username, model_name=model_name)
    return engine
