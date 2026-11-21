"""AniList GraphQL API data models."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class MediaFormat(str, Enum):
    TV = "TV"
    TV_SHORT = "TV_SHORT"
    MOVIE = "MOVIE"
    OVA = "OVA"
    ONA = "ONA"
    SPECIAL = "SPECIAL"
    MUSIC = "MUSIC"
    MANGA = "MANGA"
    NOVEL = "NOVEL"
    ONE_SHOT = "ONE_SHOT"


class MediaStatus(str, Enum):
    FINISHED = "FINISHED"
    RELEASING = "RELEASING"
    NOT_YET_RELEASED = "NOT_YET_RELEASED"
    CANCELLED = "CANCELLED"
    HIATUS = "HIATUS"


class MediaSeason(str, Enum):
    WINTER = "WINTER"
    SPRING = "SPRING"
    SUMMER = "SUMMER"
    FALL = "FALL"


class MediaSource(str, Enum):
    ORIGINAL = "ORIGINAL"
    MANGA = "MANGA"
    LIGHT_NOVEL = "LIGHT_NOVEL"
    VISUAL_NOVEL = "VISUAL_NOVEL"
    VIDEO_GAME = "VIDEO_GAME"
    OTHER = "OTHER"
    NOVEL = "NOVEL"
    DOUJINSHI = "DOUJINSHI"
    ANIME = "ANIME"
    WEB_NOVEL = "WEB_NOVEL"
    LIVE_ACTION = "LIVE_ACTION"
    GAME = "GAME"
    COMIC = "COMIC"
    MULTIMEDIA_PROJECT = "MULTIMEDIA_PROJECT"
    PICTURE_BOOK = "PICTURE_BOOK"


class MediaRelationType(str, Enum):
    SEQUEL = "SEQUEL"
    PREQUEL = "PREQUEL"
    ALTERNATIVE = "ALTERNATIVE"
    PARENT = "PARENT"
    SIDE_STORY = "SIDE_STORY"
    CHARACTER = "CHARACTER"
    SUMMARY = "SUMMARY"
    OTHER = "OTHER"
    SOURCE = "SOURCE"
    COMPILATION = "COMPILATION"
    CONTAINS = "CONTAINS"
    ADAPTATION = "ADAPTATION"
    SPIN_OFF = "SPIN_OFF"


class MediaListStatus(str, Enum):
    CURRENT = "CURRENT"
    PLANNING = "PLANNING"
    COMPLETED = "COMPLETED"
    DROPPED = "DROPPED"
    PAUSED = "PAUSED"
    REPEATING = "REPEATING"


class FuzzyDate(BaseModel):
    year: int | None = None
    month: int | None = None
    day: int | None = None


class MediaTitle(BaseModel):
    romaji: str | None = None
    english: str | None = None
    native: str | None = None
    userPreferred: str | None = None

    @property
    def display(self) -> str:
        return self.english or self.romaji or self.native or "Unknown"


class MediaTag(BaseModel):
    id: int
    name: str
    description: str | None = None
    category: str | None = None
    rank: int = 0  # 0-100 relevance percentage
    isGeneralSpoiler: bool = False
    isMediaSpoiler: bool = False
    isAdult: bool = False


class MediaGenre(str, Enum):
    ACTION = "Action"
    ADVENTURE = "Adventure"
    COMEDY = "Comedy"
    DRAMA = "Drama"
    ECCHI = "Ecchi"
    FANTASY = "Fantasy"
    HORROR = "Horror"
    MAHOU_SHOUJO = "Mahou Shoujo"
    MECHA = "Mecha"
    MUSIC = "Music"
    MYSTERY = "Mystery"
    PSYCHOLOGICAL = "Psychological"
    ROMANCE = "Romance"
    SCI_FI = "Sci-Fi"
    SLICE_OF_LIFE = "Slice of Life"
    SPORTS = "Sports"
    SUPERNATURAL = "Supernatural"
    THRILLER = "Thriller"


class Studio(BaseModel):
    id: int
    name: str
    isAnimationStudio: bool = True
    siteUrl: str | None = None


class MediaStudioConnection(BaseModel):
    edges: list[MediaStudioEdge] = Field(default_factory=list)
    nodes: list[Studio] = Field(default_factory=list)


class MediaStudioEdge(BaseModel):
    id: int
    isMain: bool = False
    node: Studio


class MediaRelation(BaseModel):
    id: int
    relationType: MediaRelationType
    node: Optional["Media"] = None


class MediaExternalLink(BaseModel):
    id: int
    url: str
    site: str
    siteId: int
    type: str
    language: str | None = None
    color: str | None = None
    icon: str | None = None
    notes: str | None = None


class MediaRecommendation(BaseModel):
    id: int
    rating: int | None = None
    media: Media
    mediaRecommendation: Media | None = None
    userRating: int | None = None


class MediaEdge(BaseModel):
    node: Media


class Media(BaseModel):
    id: int
    idMal: int | None = None
    title: MediaTitle
    format: MediaFormat | None = None
    status: MediaStatus | None = None
    description: str | None = None
    season: MediaSeason | None = None
    seasonYear: int | None = None
    seasonInt: int | None = None
    episodes: int | None = None
    duration: int | None = None
    chapters: int | None = None
    volumes: int | None = None
    countryOfOrigin: str | None = None
    isLicensed: bool = False
    source: MediaSource | None = None
    hashtag: str | None = None
    trailer: Any | None = None
    updatedAt: int | None = None
    startDate: FuzzyDate | None = None
    endDate: FuzzyDate | None = None
    synonyms: list[str] = Field(default_factory=list)
    tags: list[MediaTag] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    studios: MediaStudioConnection | None = None
    relations: MediaRelationConnection | None = None
    recommendations: MediaRecommendationConnection | None = None
    externalLinks: list[MediaExternalLink] = Field(default_factory=list)
    averageScore: int | None = None  # 0-100
    meanScore: int | None = None
    popularity: int | None = None
    trending: int | None = None
    favourites: int | None = None
    siteUrl: str | None = None
    bannerImage: str | None = None
    coverImage: MediaCoverImage | None = None
    isAdult: bool = False
    nextAiringEpisode: Any | None = None
    stats: Any | None = None

    @property
    def year(self) -> int | None:
        return self.seasonYear or (self.startDate.year if self.startDate else None)

    @property
    def main_studios(self) -> list[str]:
        if not self.studios:
            return []
        return [edge.node.name for edge in self.studios.edges if edge.isMain]

    @property
    def all_studios(self) -> list[str]:
        if not self.studios:
            return []
        return [edge.node.name for edge in self.studios.edges]

    @property
    def tag_ranks(self) -> dict[str, int]:
        return {tag.name: tag.rank for tag in self.tags if tag.rank > 0}

    @property
    def community_score(self) -> float | None:
        if self.averageScore is not None:
            return self.averageScore / 10.0
        return None


class MediaCoverImage(BaseModel):
    extraLarge: str | None = None
    large: str | None = None
    medium: str | None = None
    color: str | None = None


class MediaRelationConnection(BaseModel):
    edges: list[MediaRelationEdge] = Field(default_factory=list)
    nodes: list[Media] = Field(default_factory=list)


class MediaRelationEdge(BaseModel):
    id: int
    relationType: MediaRelationType
    node: Media


class MediaRecommendationConnection(BaseModel):
    edges: list[MediaRecommendationEdge] = Field(default_factory=list)
    nodes: list[MediaRecommendation] = Field(default_factory=list)


class MediaRecommendationEdge(BaseModel):
    node: MediaRecommendation


class MediaListEntry(BaseModel):
    id: int
    mediaId: int
    status: str | None = None  # Can be standard status, custom list name, or null
    score: float | None = None  # Raw score from AniList (1-5 for POINT_5, 1-100 for POINT_100)
    progress: int = 0
    progressVolumes: int | None = 0
    repeat: int = 0
    priority: int = 0
    private: bool = False
    notes: str | None = None
    startedAt: FuzzyDate | None = None
    completedAt: FuzzyDate | None = None
    updatedAt: int | None = None
    media: Media | None = None

    @property
    def personal_rating(self) -> float | None:
        """Convert AniList score to 0-10 scale."""
        if self.score is not None:
            # Score format is handled at sync time based on user's mediaListOptions
            return self.score
        return None

    @property
    def is_rated(self) -> bool:
        return self.score is not None and self.score > 0

    @property
    def is_negative_signal(self) -> bool:
        return self.status == "DROPPED"


class User(BaseModel):
    id: int
    name: str
    avatar: UserAvatar | None = None
    about: str | None = None
    siteUrl: str | None = None
    mediaListOptions: MediaListOptions | None = None


class UserAvatar(BaseModel):
    large: str | None = None
    medium: str | None = None


class MediaListOptions(BaseModel):
    scoreFormat: str | None = None
    rowOrder: str | None = None


class PageInfo(BaseModel):
    total: int
    perPage: int
    currentPage: int
    lastPage: int
    hasNextPage: bool


class MediaListCollection(BaseModel):
    user: User
    lists: list[MediaListGroup] = Field(default_factory=list)


class MediaListGroup(BaseModel):
    status: str | None = None  # Can be standard status, custom list name, or null
    entries: list[MediaListEntry] = Field(default_factory=list)


class Page(BaseModel):
    pageInfo: PageInfo
    media: list[Media] = Field(default_factory=list)
    mediaList: list[MediaListEntry] = Field(default_factory=list)


# GraphQL Response wrappers
class UserResponse(BaseModel):
    User: Optional["User"] = None


class MediaListResponse(BaseModel):
    MediaListCollection: Optional["MediaListCollection"] = None


class PageResponse(BaseModel):
    Page: Optional["Page"] = None


class MediaResponse(BaseModel):
    Media: Optional["Media"] = None


# Simplified models for caching
class CachedMedia(BaseModel):
    """Lightweight media for caching."""
    id: int
    title_romaji: str
    title_english: str | None = None
    title_native: str | None = None
    format: str | None = None
    status: str | None = None
    season_year: int | None = None
    season: str | None = None
    episodes: int | None = None
    duration: int | None = None
    genres: list[str] = Field(default_factory=list)
    tag_ranks: dict[str, int] = Field(default_factory=dict)
    main_studios: list[str] = Field(default_factory=list)
    all_studios: list[str] = Field(default_factory=list)
    average_score: int | None = None
    popularity: int | None = None
    favourites: int | None = None
    is_adult: bool = False
    source: str | None = None
    description: str | None = None
    cover_image: str | None = None
    banner_image: str | None = None
    site_url: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    relations: list[CachedRelation] = Field(default_factory=list)

    @property
    def display_title(self) -> str:
        return self.title_english or self.title_romaji or self.title_native or f"ID:{self.id}"

    @property
    def community_score(self) -> float | None:
        if self.average_score is not None:
            return self.average_score / 10.0
        return None


class CachedRelation(BaseModel):
    id: int
    relation_type: str
    title_romaji: str | None = None
    title_english: str | None = None


class CachedUserEntry(BaseModel):
    id: int
    media_id: int
    status: str
    score: float | None = None  # 0-10
    progress: int = 0
    started_at: dict | None = None
    completed_at: dict | None = None
    updated_at: int | None = None

    @property
    def is_rated(self) -> bool:
        return self.score is not None and self.score > 0

    @property
    def is_negative_signal(self) -> bool:
        return self.status == "DROPPED"


class CacheMetadata(BaseModel):
    username: str
    user_id: int
    synced_at: str
    total_user_entries: int
    total_cached_media: int
    model_version: str = "1.0"
    anilist_api_version: str = "2024-01-01"


# Feature extraction models
class AnimeFeatures(BaseModel):
    """Extracted features for ML model."""
    media_id: int
    genres: list[str] = Field(default_factory=list)
    tag_ranks: dict[str, int] = Field(default_factory=dict)
    main_studios: list[str] = Field(default_factory=list)
    format: str | None = None
    season_year: int | None = None
    episodes: int | None = None
    duration: int | None = None
    source: str | None = None
    popularity: int | None = None
    community_score: float | None = None

    def to_vector(self, feature_names: list[str]) -> np.ndarray:
        """Convert to feature vector matching feature_names order."""
        vector = []
        for name in feature_names:
            if name.startswith("genre_"):
                # Handle genre interaction features (e.g., genre_Romance_x_Comedy)
                if "_x_" in name:
                    parts = name[6:].split("_x_")
                    if len(parts) == 2:
                        g1, g2 = parts
                        vector.append(1.0 if (g1 in self.genres and g2 in self.genres) else 0.0)
                    else:
                        vector.append(0.0)
                else:
                    genre = name[6:]
                    vector.append(1.0 if genre in self.genres else 0.0)
            elif name.startswith("tag_"):
                tag = name[4:]
                vector.append(float(self.tag_ranks.get(tag, 0)) / 100.0)
            elif name.startswith("studio_"):
                studio = name[7:]
                vector.append(1.0 if studio in self.main_studios else 0.0)
            elif name == "format_TV":
                vector.append(1.0 if self.format == "TV" else 0.0)
            elif name == "format_MOVIE":
                vector.append(1.0 if self.format == "MOVIE" else 0.0)
            elif name == "format_OVA":
                vector.append(1.0 if self.format == "OVA" else 0.0)
            elif name == "format_ONA":
                vector.append(1.0 if self.format == "ONA" else 0.0)
            elif name == "season_year":
                vector.append(float(self.season_year or 0) / 2025.0)
            elif name == "episodes":
                vector.append(float(self.episodes or 0) / 100.0)
            elif name == "duration":
                vector.append(float(self.duration or 0) / 60.0)
            elif name == "source_original":
                vector.append(1.0 if self.source == "ORIGINAL" else 0.0)
            elif name == "source_manga":
                vector.append(1.0 if self.source == "MANGA" else 0.0)
            elif name == "source_light_novel":
                vector.append(1.0 if self.source == "LIGHT_NOVEL" else 0.0)
            elif name == "popularity":
                vector.append(float(self.popularity or 0) / 1000000.0)
            elif name == "community_score":
                vector.append((self.community_score or 5.0) / 10.0)
            else:
                vector.append(0.0)
        return np.array(vector, dtype=np.float32)


class TasteDriver(BaseModel):
    """A learned taste driver (feature weight)."""
    feature: str
    weight: float
    feature_type: str  # genre, tag, studio, format, metadata
    description: str
    is_positive: bool

    @property
    def abs_weight(self) -> float:
        return abs(self.weight)


class RecommendationResult(BaseModel):
    """A recommended anime with prediction details."""
    media: CachedMedia
    predicted_score: float
    community_score: float | None
    score_difference: float  # predicted - community
    is_planning: bool = False
    is_sequel_without_prequel: bool = False
    prequel_title: str | None = None
    similar_to: list[dict] = Field(default_factory=list)  # [{watched_title, watched_score, shared_genres, shared_tags, shared_studios, similarity}]

    @property
    def display_title(self) -> str:
        return self.media.display_title