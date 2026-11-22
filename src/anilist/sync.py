"""Synchronize user data and anime catalog from AniList."""

from __future__ import annotations

from datetime import datetime

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.anilist.client import AniListClient, RateLimiter, entry_to_cached, media_to_cached
from src.anilist.models import CachedMedia, CacheMetadata
from src.cache.cache import CacheManager

console = Console()


class SyncEngine:
    """Orchestrates fetching and caching data from AniList."""

    # Genres to exclude from candidate pool (typically non-anime or problematic)
    EXCLUDE_GENRES = ["Hentai", "Erotica"]

    # Target number of anime in candidate pool
    TARGET_CATALOG_SIZE = 2000

    def __init__(
        self,
        cache_manager: CacheManager,
        rate_limiter: RateLimiter | None = None,
    ):
        self.cache_manager = cache_manager
        self.rate_limiter = rate_limiter or RateLimiter()

    async def sync_user(self, username: str, force: bool = False) -> CacheMetadata:
        """
        Sync user profile and watch list only (does NOT fetch global catalog).

        Use `fetch_catalog()` separately to fetch the global candidate catalog.

        Args:
            username: AniList username
            force: Force re-sync even if cache exists

        Returns:
            CacheMetadata with sync statistics
        """
        console.print(f"[bold cyan]Syncing data for [yellow]{username}[/yellow]...[/bold cyan]")

        async with AniListClient(self.rate_limiter) as client:
            # 1. Get user profile and score format
            with console.status("[bold green]Fetching user profile..."):
                user = await client.get_user(username)
                if not user:
                    raise ValueError(f"User '{username}' not found on AniList")

                # Get score format
                score_format = "POINT_100"
                if user.mediaListOptions and user.mediaListOptions.scoreFormat:
                    score_format = user.mediaListOptions.scoreFormat

            console.print(f"  Found user: [bold]{user.name}[/bold] (ID: {user.id})")
            console.print(f"  Score format: [bold]{score_format}[/bold]")

            # 2. Get user's anime list
            with console.status("[bold green]Fetching anime list..."):
                collection = await client.get_user_list(username)

            # Flatten all entries
            all_entries = []
            for list_group in collection.lists:
                all_entries.extend(list_group.entries)

            console.print(f"  Found [bold]{len(all_entries)}[/bold] anime entries")

            # Convert to cached format with proper score conversion
            cached_entries = []
            for entry in all_entries:
                cached = entry_to_cached(entry)
                # Convert score based on user's score format
                if cached.score is not None and cached.score > 0:
                    if score_format == "POINT_5":
                        # POINT_5 is 1-5 scale, convert to 0-10
                        cached.score = (cached.score / 5.0) * 10.0
                    elif score_format == "POINT_100":
                        # POINT_100 is 1-100 scale, convert to 0-10
                        cached.score = cached.score / 10.0
                    elif score_format == "POINT_10_DECIMAL":
                        # POINT_10_DECIMAL is 0-10 scale with decimals, no conversion needed
                        pass
                cached_entries.append(cached)

            # 3. Save to cache
            with console.status("[bold green]Saving to cache..."):
                self.cache_manager.save_user_entries(username, cached_entries)

                # Report existing catalog count for metadata (0 if none exists)
                existing_catalog = self.cache_manager.load_media_catalog(username)

                metadata = CacheMetadata(
                    username=username,
                    user_id=user.id,
                    synced_at=datetime.now().isoformat(),
                    total_user_entries=len(cached_entries),
                    total_cached_media=len(existing_catalog),
                )
                self.cache_manager.save_metadata(metadata)

        if not self.cache_manager.global_catalog_exists():
            console.print(
                "[yellow]Note: No global catalog found. Run 'miraiq fetch-catalog' "
                "to fetch candidate anime for recommendations.[/yellow]"
            )

        console.print("[bold green]Sync complete![/bold green]")
        return metadata

    async def _fetch_candidate_catalog(
        self, client: AniListClient, limit: int = TARGET_CATALOG_SIZE
    ) -> list[CachedMedia]:
        """Fetch popular anime for the candidate pool.

        Args:
            client: AniList API client
            limit: Maximum number of anime to fetch
        """
        all_media: list[CachedMedia] = []
        seen_ids: set[int] = set()

        # Calculate pages needed (50 per page)
        per_page = 50
        estimated_pages = (limit // per_page) + 2

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching anime pages...", total=estimated_pages)

            for page_num in range(1, estimated_pages + 1):
                progress.update(task, description=f"Fetching page {page_num}/{estimated_pages}")

                page = await client.get_popular_anime(
                    page=page_num,
                    per_page=per_page,
                    exclude_genres=self.EXCLUDE_GENRES,
                )

                if not page.media:
                    break

                new_media = []
                for media in page.media:
                    if media.id not in seen_ids and not media.isAdult:
                        seen_ids.add(media.id)
                        new_media.append(media_to_cached(media))

                all_media.extend(new_media)
                progress.advance(task)

                if not page.pageInfo.hasNextPage:
                    break

                # Stop if we have enough
                if len(all_media) >= limit:
                    break

        return all_media[:limit]

    async def fetch_catalog(self, limit: int = TARGET_CATALOG_SIZE) -> CacheMetadata:
        """Fetch the global candidate catalog (shared across all users).

        Args:
            limit: Maximum number of anime to fetch

        Returns:
            CacheMetadata with catalog statistics
        """
        console.print(
            f"[bold cyan]Fetching global candidate catalog (target: {limit} anime)...[/bold cyan]"
        )

        async with AniListClient(self.rate_limiter) as client:
            media_catalog = await self._fetch_candidate_catalog(client, limit=limit)

        self.cache_manager.save_media_catalog("__global__", media_catalog)

        # Update metadata for all existing users
        if self.cache_manager.user_dir.exists():
            for user_dir in self.cache_manager.user_dir.iterdir():
                if user_dir.is_dir():
                    user_name = user_dir.name
                    metadata = self.cache_manager.load_metadata(user_name)
                    if metadata:
                        metadata.synced_at = datetime.now().isoformat()
                        metadata.total_cached_media = len(media_catalog)
                        self.cache_manager.save_metadata(metadata)

        console.print(
            f"[bold green]Global catalog fetched: {len(media_catalog)} anime[/bold green]"
        )
        return CacheMetadata(
            username="__global__",
            user_id=0,
            synced_at=datetime.now().isoformat(),
            total_user_entries=0,
            total_cached_media=len(media_catalog),
        )


async def run_sync(
    username: str,
    cache_dir: str = "~/.miraiq/cache",
    force: bool = False,
) -> CacheMetadata:
    """Convenience function to run user sync."""
    cache_manager = CacheManager(cache_dir)
    rate_limiter = RateLimiter()
    engine = SyncEngine(cache_manager, rate_limiter)
    return await engine.sync_user(username, force=force)


async def run_fetch_catalog(
    cache_dir: str = "~/.miraiq/cache",
    limit: int = SyncEngine.TARGET_CATALOG_SIZE,
) -> CacheMetadata:
    """Convenience function to fetch global candidate catalog."""
    cache_manager = CacheManager(cache_dir)
    rate_limiter = RateLimiter()
    engine = SyncEngine(cache_manager, rate_limiter)
    return await engine.fetch_catalog(limit=limit)
