"""AniList GraphQL API client with rate limiting and caching."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx
import orjson

from miraiq.anilist.models import (
    CachedMedia,
    CachedRelation,
    CachedUserEntry,
    Media,
    MediaListCollection,
    MediaListEntry,
    Page,
    PageInfo,
    User,
)


@dataclass
class RateLimiter:
    """Token bucket rate limiter for AniList API (30 req/min = 1 req/2s)."""

    max_requests: int = 30
    time_window: float = 60.0
    _requests: list[float] = field(default_factory=list)

    def __post_init__(self):
        self._requests = []

    async def acquire(self):
        now = time.monotonic()
        # Remove old requests outside the window
        self._requests = [t for t in self._requests if now - t < self.time_window]

        if len(self._requests) >= self.max_requests:
            # Wait until the oldest request expires
            oldest = self._requests[0]
            wait_time = self.time_window - (now - oldest) + 0.1
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            now = time.monotonic()
            self._requests = [t for t in self._requests if now - t < self.time_window]

        self._requests.append(now)


class AniListClient:
    """Client for AniList GraphQL API."""

    BASE_URL = "https://graphql.anilist.co"
    USER_AGENT = "MiraiQ/1.0 (Personal Anime Recommendation CLI)"

    # GraphQL Queries
    USER_QUERY = """
    query ($userName: String) {
        User(name: $userName) {
            id
            name
            avatar { large medium }
            about
            siteUrl
            mediaListOptions { scoreFormat rowOrder }
        }
    }
    """

    USER_LIST_QUERY = """
    query ($userName: String, $type: MediaType, $status: MediaListStatus) {
        MediaListCollection(userName: $userName, type: $type, status: $status) {
            user { id name }
            lists {
                status
                entries {
                    id
                    mediaId
                    status
                    score
                    progress
                    progressVolumes
                    repeat
                    priority
                    private
                    notes
                    startedAt { year month day }
                    completedAt { year month day }
                    updatedAt
                    media {
                        id
                        idMal
                        title { romaji english native userPreferred }
                        format
                        status
                        description
                        season
                        seasonYear
                        episodes
                        duration
                        source
                        genres
                        tags { id name description category rank
                               isGeneralSpoiler isMediaSpoiler isAdult }
                        studios { edges { id isMain node { id name isAnimationStudio siteUrl } } }
                        averageScore
                        meanScore
                        popularity
                        trending
                        favourites
                        siteUrl
                        coverImage { extraLarge large medium color }
                        bannerImage
                        isAdult
                        synonyms
                        relations { edges { id relationType node { id title { romaji english } } } }
                    }
                }
            }
        }
    }
    """

    POPULAR_ANIME_QUERY = """
    query (
        $page: Int,
        $perPage: Int,
        $type: MediaType,
        $sort: [MediaSort],
        $genre_not_in: [String]
    ) {
        Page(page: $page, perPage: $perPage) {
            pageInfo { total perPage currentPage lastPage hasNextPage }
            media(type: $type, sort: $sort, genre_not_in: $genre_not_in) {
                id
                idMal
                title { romaji english native userPreferred }
                format
                status
                description
                season
                seasonYear
                episodes
                duration
                source
                genres
                tags { id name description category rank
                       isGeneralSpoiler isMediaSpoiler isAdult }
                studios { edges { id isMain node { id name isAnimationStudio siteUrl } } }
                averageScore
                meanScore
                popularity
                trending
                favourites
                siteUrl
                coverImage { extraLarge large medium color }
                bannerImage
                isAdult
                synonyms
                relations { edges { id relationType node { id title { romaji english } } } }
            }
        }
    }
    """

    def __init__(self, rate_limiter: RateLimiter | None = None):
        self.rate_limiter = rate_limiter or RateLimiter()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"User-Agent": self.USER_AGENT, "Content-Type": "application/json"},
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _execute(self, query: str, variables: dict) -> dict:
        await self.rate_limiter.acquire()
        assert self._client is not None
        response = await self._client.post("", json={"query": query, "variables": variables})
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise AniListError(data["errors"])
        return data.get("data", {})

    async def get_user(self, username: str) -> User | None:
        data = await self._execute(self.USER_QUERY, {"userName": username})
        if data.get("User"):
            return User(**data["User"])
        return None

    async def get_score_format(self, username: str) -> str:
        """Get user's score format (POINT_5 or POINT_100)."""
        data = await self._execute(self.USER_QUERY, {"userName": username})
        if data.get("User") and data["User"].get("mediaListOptions"):
            return data["User"]["mediaListOptions"].get("scoreFormat", "POINT_100")
        return "POINT_100"

    async def get_user_list(self, username: str) -> MediaListCollection:
        # Don't pass status to get all statuses
        data = await self._execute(
            self.USER_LIST_QUERY,
            {"userName": username, "type": "ANIME"},
        )
        if data.get("MediaListCollection"):
            return MediaListCollection(**data["MediaListCollection"])
        return MediaListCollection(user=User(id=0, name=username), lists=[])

    async def get_popular_anime(
        self,
        page: int = 1,
        per_page: int = 50,
        sort: list[str] | None = None,
        exclude_genres: list[str] | None = None,
    ) -> Page:
        variables = {
            "page": page,
            "perPage": per_page,
            "type": "ANIME",
            "sort": sort or ["POPULARITY_DESC", "TRENDING_DESC"],
        }
        if exclude_genres:
            variables["genre_not_in"] = exclude_genres

        data = await self._execute(self.POPULAR_ANIME_QUERY, variables)
        if data.get("Page"):
            return Page(**data["Page"])
        return Page(
            pageInfo=PageInfo(
                total=0,
                perPage=per_page,
                currentPage=page,
                lastPage=0,
                hasNextPage=False,
            ),
            media=[],
        )


class AniListError(Exception):
    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__(f"AniList API errors: {orjson.dumps(errors).decode()}")


def media_to_cached(media: Media) -> CachedMedia:
    """Convert full Media to cached version."""
    relations = []
    if media.relations:
        for edge in media.relations.edges:
            relations.append(
                CachedRelation(
                    id=edge.node.id,
                    relation_type=edge.relationType.value,
                    title_romaji=edge.node.title.romaji,
                    title_english=edge.node.title.english,
                )
            )

    return CachedMedia(
        id=media.id,
        title_romaji=media.title.romaji or "",
        title_english=media.title.english,
        title_native=media.title.native,
        format=media.format.value if media.format else None,
        status=media.status.value if media.status else None,
        season_year=media.seasonYear,
        season=media.season.value if media.season else None,
        episodes=media.episodes,
        duration=media.duration,
        genres=media.genres,
        tag_ranks=media.tag_ranks,
        main_studios=media.main_studios,
        all_studios=media.all_studios,
        average_score=media.averageScore,
        popularity=media.popularity,
        favourites=media.favourites,
        is_adult=media.isAdult,
        source=media.source.value if media.source else None,
        description=media.description,
        cover_image=media.coverImage.extraLarge if media.coverImage else None,
        banner_image=media.bannerImage,
        site_url=media.siteUrl,
        synonyms=media.synonyms,
        relations=relations,
    )


def entry_to_cached(entry: MediaListEntry) -> CachedUserEntry:
    """Convert MediaListEntry to cached version."""
    return CachedUserEntry(
        id=entry.id,
        media_id=entry.mediaId,
        status=entry.status if entry.status is not None else "",
        score=entry.personal_rating,
        progress=entry.progress,
        started_at=entry.startedAt.model_dump() if entry.startedAt else None,
        completed_at=entry.completedAt.model_dump() if entry.completedAt else None,
        updated_at=entry.updatedAt,
    )
