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
        Full sync: fetch user profile, watch list, and candidate catalog (global).

        Args:
            username: AniList username
            force: Force re-sync even if cache exists

        Returns:
            CacheMetadata with sync statistics
        """
        console.print(f"[bold cyan]Syncing data for [yellow]{username}[/yellow]...[/bold cyan]")

        # Check if global catalog exists (unless forced)
        catalog_exists = self.cache_manager.global_catalog_exists()
        if catalog_exists and not force:
            console.print("  [green]Using existing global catalog[/green]")
        else:
            console.print("  [yellow]Fetching global catalog (first user or forced)...[/yellow]")

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

            # 3. Fetch candidate catalog (popular/acclaimed anime) ONLY if not exists
            if not catalog_exists or force:
                console.print(
                    f"  Fetching candidate catalog (target: {self.TARGET_CATALOG_SIZE} anime)..."
                )
                media_catalog = await self._fetch_candidate_catalog(client)
                console.print(
                    f"  Fetched [bold]{len(media_catalog)}[/bold] anime for global catalog"
                )
            else:
                media_catalog = self.cache_manager.load_media_catalog(username)
                console.print(
                    f"  Loaded [bold]{len(media_catalog)}[/bold] anime from global catalog"
                )

            # 4. Save to cache
            with console.status("[bold green]Saving to cache..."):
                self.cache_manager.save_user_entries(username, cached_entries)
                if not catalog_exists or force:
                    self.cache_manager.save_media_catalog(username, media_catalog)

                metadata = CacheMetadata(
                    username=username,
                    user_id=user.id,
                    synced_at=datetime.now().isoformat(),
                    total_user_entries=len(cached_entries),
                    total_cached_media=len(media_catalog),
                )
                self.cache_manager.save_metadata(metadata)

        console.print("[bold green]Sync complete![/bold green]")
        return metadata

    async def _fetch_candidate_catalog(self, client: AniListClient) -> list[CachedMedia]:
        """Fetch popular anime for the candidate pool."""
        all_media: list[CachedMedia] = []
        seen_ids: set[int] = set()

        # Calculate pages needed (50 per page)
        per_page = 50
        estimated_pages = (self.TARGET_CATALOG_SIZE // per_page) + 2

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
                if len(all_media) >= self.TARGET_CATALOG_SIZE:
                    break

        # Also ensure user's watched anime are in catalog (for training)
        # This is handled by the training phase which merges user entries with catalog

        return all_media[: self.TARGET_CATALOG_SIZE]

    async def sync_catalog_only(self, username: str) -> CacheMetadata:
        """Re-fetch the global candidate catalog (affects all users)."""
        console.print("[bold cyan]Updating global catalog...[/bold cyan]")

        async with AniListClient(self.rate_limiter) as client:
            media_catalog = await self._fetch_candidate_catalog(client)

            self.cache_manager.save_media_catalog(username, media_catalog)

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
            f"[bold green]Global catalog updated: {len(media_catalog)} anime[/bold green]"
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
    """Convenience function to run full sync."""
    cache_manager = CacheManager(cache_dir)
    rate_limiter = RateLimiter()
    engine = SyncEngine(cache_manager, rate_limiter)
    return await engine.sync_user(username, force=force)


async def run_catalog_update(cache_dir: str = "~/.miraiq/cache") -> CacheMetadata:
    """Convenience function to update global catalog only."""
    cache_manager = CacheManager(cache_dir)
    rate_limiter = RateLimiter()
    engine = SyncEngine(cache_manager, rate_limiter)
    # Use a dummy username since sync_catalog_only still needs one for metadata update
    # but the catalog itself is global
    return await engine.sync_catalog_only("__global__")
