"""MiraiQ CLI - Personal Anime Recommendation Engine."""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.anilist.sync import run_catalog_update, run_sync
from src.cache.cache import CacheManager
from src.model.recommender import (
    RecommendationFilters,
    load_engine,
)
from src.model.trainer import train_model

app = typer.Typer(
    name="miraiq",
    help="Personal Anime Recommendation Engine powered by your AniList taste",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

DEFAULT_CACHE_DIR = ".cache"


def get_cache_manager(cache_dir: str = DEFAULT_CACHE_DIR) -> CacheManager:
    return CacheManager(cache_dir)


@app.command()
def sync(
    username: Annotated[str, typer.Argument(help="AniList username")],
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force re-sync even if cache exists")
    ] = False,
) -> None:
    """Sync your AniList profile and anime catalog (fetches ~2000 popular anime)."""
    cache_manager = get_cache_manager(cache_dir)

    if cache_manager.cache_exists(username) and not force:
        console.print(
            f"[yellow]Cache already exists for {username}. Use --force to re-sync.[/yellow]"
        )
        return

    import asyncio

    try:
        metadata = asyncio.run(run_sync(username, cache_dir, force))
        console.print(
            Panel(
                f"[bold green]Sync Complete![/bold green]\n"
                f"User: {metadata.username} (ID: {metadata.user_id})\n"
                f"Entries: {metadata.total_user_entries}\n"
                f"Catalog: {metadata.total_cached_media} anime\n"
                f"Synced: {metadata.synced_at}",
                title="Sync Results",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Sync failed:[/bold red] {e}")
        sys.exit(1)


@app.command()
def update_catalog(
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
) -> None:
    """Update the global anime catalog (shared across all users)."""
    cache_manager = get_cache_manager(cache_dir)

    if not cache_manager.global_catalog_exists():
        console.print("[red]No global catalog found. Run 'miraiq sync <username>' first.[/red]")
        sys.exit(1)

    import asyncio

    try:
        asyncio.run(run_catalog_update(cache_dir))
    except Exception as e:
        console.print(f"[bold red]Catalog update failed:[/bold red] {e}")
        sys.exit(1)


@app.command()
def train(
    username: Annotated[str, typer.Argument(help="AniList username")],
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
    min_ratings: Annotated[
        int, typer.Option("--min-ratings", "-m", help="Minimum rated anime required")
    ] = 5,
    tag_threshold: Annotated[
        int, typer.Option("--tag-threshold", help="Minimum tag rank % to include as feature")
    ] = 50,
    min_studio_count: Annotated[
        int, typer.Option("--min-studio-count", help="Minimum anime count for studio feature")
    ] = 3,
    include_format: Annotated[
        bool, typer.Option("--include-format", help="Include format (TV/MOVIE/OVA) as features")
    ] = False,
    include_source: Annotated[
        bool,
        typer.Option("--include-source", help="Include source (manga/LN/original) as features"),
    ] = False,
    include_metadata: Annotated[
        bool,
        typer.Option("--include-metadata", help="Include year/episodes/popularity as features"),
    ] = True,
) -> None:
    """Train your personal taste model from cached data."""
    cache_manager = get_cache_manager(cache_dir)

    if not cache_manager.cache_exists(username):
        console.print(f"[red]No cache found for {username}. Run 'miraiq sync' first.[/red]")
        sys.exit(1)

    if cache_manager.model_exists(username):
        console.print("[yellow]Model already exists. Re-training...[/yellow]")

    try:
        with console.status("[bold green]Training taste model..."):
            model, metrics = train_model(
                username,
                cache_manager,
                min_ratings=min_ratings,
                tag_threshold=tag_threshold,
                min_studio_count=min_studio_count,
                include_format=include_format,
                include_source=include_source,
                include_metadata=include_metadata,
            )

        console.print(
            Panel(
                f"[bold green]Training Complete![/bold green]\n"
                f"Training samples: {metrics['n_samples']} (rated: {metrics['n_rated']}, "
                f"dropped: {metrics['n_dropped']})\n"
                f"Features: {metrics['n_features']}\n"
                f"Train RMSE: {metrics['train_rmse']:.3f}\n"
                f"CV RMSE: {metrics['cv_rmse']:.3f}",
                title="Model Training Results",
                border_style="green",
            )
        )
    except ValueError as e:
        console.print(f"[bold red]Training failed:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Training failed:[/bold red] {e}")
        sys.exit(1)


@app.command()
def taste(
    username: Annotated[str, typer.Argument(help="AniList username")],
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
    top: Annotated[int, typer.Option("--top", "-t", help="Number of top drivers to show")] = 15,
) -> None:
    """Display your learned taste drivers (what you like/dislike)."""
    cache_manager = get_cache_manager(cache_dir)

    if not cache_manager.model_exists(username):
        console.print(f"[red]No trained model for {username}. Run 'miraiq train' first.[/red]")
        sys.exit(1)

    drivers = cache_manager.load_taste_drivers(username)

    if not drivers:
        console.print("[yellow]No taste drivers found.[/yellow]")
        return

    positive = [d for d in drivers if d.is_positive][:top]
    negative = [d for d in drivers if not d.is_positive][:top]

    console.print(f"\n[bold cyan]Your Taste Profile for [yellow]{username}[/yellow][/bold cyan]\n")

    if positive:
        table = Table(
            title="Positive Drivers (Things You Like)",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Feature", style="cyan", width=40)
        table.add_column("Type", style="dim", width=12)
        table.add_column("Weight", justify="right", style="green")
        for d in positive:
            table.add_row(d.description, d.feature_type, f"+{d.weight:.3f}")
        console.print(table)
        console.print()

    if negative:
        table = Table(
            title="Negative Drivers (Things You Dislike)",
            show_header=True,
            header_style="bold red",
        )
        table.add_column("Feature", style="cyan", width=40)
        table.add_column("Type", style="dim", width=12)
        table.add_column("Weight", justify="right", style="red")
        for d in negative:
            table.add_row(d.description, d.feature_type, f"{d.weight:.3f}")
        console.print(table)
        console.print()


@app.command()
def recommend(
    username: Annotated[str, typer.Argument(help="AniList username")],
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of recommendations")] = 20,
    sort_by: Annotated[
        str, typer.Option("--sort", "-s", help="Sort: predicted, community, difference")
    ] = "predicted",
    include_genres: Annotated[
        list[str] | None,
        typer.Option("--include-genre", "-g", help="Include genres (can repeat)"),
    ] = None,
    exclude_genres: Annotated[
        list[str] | None,
        typer.Option("--exclude-genre", "-G", help="Exclude genres (can repeat)"),
    ] = None,
    include_tags: Annotated[
        list[str] | None,
        typer.Option("--include-tag", "-t", help="Include tags (can repeat)"),
    ] = None,
    exclude_tags: Annotated[
        list[str] | None,
        typer.Option("--exclude-tag", "-T", help="Exclude tags (can repeat)"),
    ] = None,
    min_tag_rank: Annotated[
        int, typer.Option("--min-tag-rank", help="Minimum tag relevance %")
    ] = 50,
    formats: Annotated[
        list[str] | None,
        typer.Option("--format", "-f", help="Formats: TV, MOVIE, OVA, ONA"),
    ] = None,
    year_min: Annotated[int | None, typer.Option("--year-min", help="Minimum release year")] = None,
    year_max: Annotated[int | None, typer.Option("--year-max", help="Maximum release year")] = None,
    show_sequels: Annotated[
        bool, typer.Option("--show-sequels", help="Show sequels without prequel watched")
    ] = False,
    include_planning: Annotated[
        bool,
        typer.Option(
            "--include-planning/--no-planning",
            help="Include Planning list anime in recommendations (default: on)",
        ),
    ] = True,
    unreleased: Annotated[
        bool, typer.Option("--unreleased", help="Include not-yet-released anime")
    ] = False,
    include_chinese_korean: Annotated[
        bool,
        typer.Option(
            "--include-chinese-korean",
            help="Include Chinese/Korean anime (excluded by default)",
        ),
    ] = False,
) -> None:
    """Get personalized anime recommendations."""
    try:
        engine = load_engine(username, cache_dir)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)

    filters = RecommendationFilters(
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        min_tag_rank=min_tag_rank,
        formats=formats,
        year_min=year_min,
        year_max=year_max,
        hide_sequels_without_prequel=not show_sequels,
        include_planning=include_planning,
        released_only=not unreleased,
        exclude_chinese_korean=not include_chinese_korean,
        sort_by=sort_by,
        limit=limit,
    )

    with console.status("[bold green]Generating recommendations..."):
        results = engine.recommend(filters)

    if not results:
        console.print("[yellow]No recommendations match your filters.[/yellow]")
        return

    sort_labels = {
        "predicted": "Predicted Score",
        "community": "Community Score",
        "difference": "Personal vs Community",
    }

    console.print(
        f"\n[bold cyan]Top {len(results)} Recommendations for "
        f"[yellow]{username}[/yellow][/bold cyan]"
    )
    console.print(f"[dim]Sorted by: {sort_labels.get(sort_by, sort_by)}[/dim]\n")

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("ID", style="dim", width=8, justify="right")
    table.add_column("Anime", style="cyan", width=38)
    table.add_column("Format", style="dim", width=8)
    table.add_column("Year", style="dim", width=6, justify="right")
    table.add_column("Eps", style="dim", width=5, justify="right")
    table.add_column("Pred", style="bold green", width=6, justify="right")
    table.add_column("Comm", style="blue", width=6, justify="right")
    table.add_column("Diff", style="yellow", width=7, justify="right")
    table.add_column("Tags", style="dim", width=26)
    table.add_column("Similar To", style="dim", width=34)

    for i, rec in enumerate(results, 1):
        top_tags = sorted(rec.media.tag_ranks.items(), key=lambda x: -x[1])[:3]
        tags_str = ", ".join(f"{tag} ({rank}%)" for tag, rank in top_tags)

        diff_str = f"{rec.score_difference:+.1f}"
        if rec.score_difference > 1:
            diff_str = f"[bold green]{diff_str}[/bold green]"
        elif rec.score_difference < -1:
            diff_str = f"[bold red]{diff_str}[/bold red]"

        title = rec.display_title
        if rec.is_planning:
            title += " [yellow]*[/yellow]"
        if rec.is_sequel_without_prequel:
            title += " [red]->[/red]"

        similar_parts = []
        for sim in rec.similar_to[:2]:
            title_short = sim["watched_title"][:25]
            score = sim["watched_score"] or 0
            shared = ", ".join(sim["shared_genres"][:2]) if sim["shared_genres"] else ""
            if sim["shared_tags"]:
                shared += (", " if shared else "") + ", ".join(sim["shared_tags"][:2])
            similar_parts.append(f"{title_short}({score:.0f}): {shared}")
        similar_str = "; ".join(similar_parts)

        table.add_row(
            str(i),
            str(rec.media.id),
            title,
            rec.media.format or "?",
            str(rec.media.season_year or "?"),
            str(rec.media.episodes or "?"),
            f"{rec.predicted_score:.1f}",
            f"{rec.community_score:.1f}" if rec.community_score else "N/A",
            diff_str,
            tags_str,
            similar_str,
        )

    console.print(table)

    console.print(
        "[dim]Legend: * = in your Planning list | -> = sequel (prequel not completed)[/dim]"
    )


@app.command()
def explain(
    username: Annotated[str, typer.Argument(help="AniList username")],
    anime_id: Annotated[int, typer.Argument(help="AniList anime ID")],
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
) -> None:
    """Explain why an anime was recommended (shows feature contributions)."""
    try:
        engine = load_engine(username, cache_dir)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)

    media = engine._media_by_id.get(anime_id)
    if not media:
        console.print(f"[red]Anime ID {anime_id} not found in catalog.[/red]")
        sys.exit(1)

    if not engine.model:
        console.print("[red]Model not loaded.[/red]")
        sys.exit(1)

    contributions = engine.model.explain_prediction(media, top_k=10)

    raw_score = engine.model.predict(media)

    similar_watched = engine._get_similar_watched(media, top_k=5)
    boost = 0.0
    for sim in similar_watched:
        if sim["watched_score"] and sim["watched_score"] >= 7.0:
            boost += sim["similarity"] * 0.5 * (sim["watched_score"] / 10.0)
    boost = min(boost, 0.5)
    boosted_score = raw_score + boost

    score_display = f"{boosted_score:.1f}"
    if boost > 0:
        score_display += f" (+{boost:.1f} boost)"

    console.print(
        f"\n[bold cyan]Why [yellow]{media.display_title}[/yellow] scored "
        f"{score_display}/10[/bold cyan]\n"
    )

    if not contributions:
        console.print("[dim]No significant feature contributions.[/dim]")
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Feature", style="cyan", width=45)
        table.add_column("Contribution", justify="right", width=15)
        table.add_column("Direction", width=10)

        for desc, contrib, direction in contributions:
            style = "green" if direction == "positive" else "red"
            table.add_row(desc, f"{contrib:+.3f}", f"[{style}]{direction}[/{style}]")

        console.print(table)

    if similar_watched:
        console.print("\n[bold cyan]Based on what you've watched:[/bold cyan]\n")
        for sim in similar_watched:
            score_str = (
                f"rated {sim['watched_score']:.1f}/10" if sim["watched_score"] else "not rated"
            )
            parts = []
            if sim["shared_genres"]:
                parts.append(f"genres: {', '.join(sim['shared_genres'])}")
            if sim["shared_tags"]:
                parts.append(f"tags: {', '.join(sim['shared_tags'])}")
            if sim["shared_studios"]:
                parts.append(f"studio: {', '.join(sim['shared_studios'])}")
            detail = "; ".join(parts) if parts else "no shared features"
            console.print(
                f"• [dim]{sim['watched_title']}[/dim] "
                f"({score_str}, similarity {sim['similarity']:.2f}) - {detail}"
            )
    else:
        console.print("\n[dim]No similar watched anime found.[/dim]")


@app.command()
def tags(
    username: Annotated[str, typer.Argument(help="AniList username")],
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
    search: Annotated[
        str | None,
        typer.Option("--search", "-s", help="Search tags by name"),
    ] = None,
    min_rank: Annotated[int, typer.Option("--min-rank", help="Minimum tag relevance %")] = 50,
    list_genres: Annotated[
        bool, typer.Option("--genres", help="List genres instead of tags")
    ] = False,
    list_studios: Annotated[bool, typer.Option("--studios", help="List studios")] = False,
    list_formats: Annotated[bool, typer.Option("--formats", help="List formats")] = False,
) -> None:
    """List available genres, tags, studios, or formats from the catalog."""
    try:
        engine = load_engine(username, cache_dir)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)

    if list_genres:
        genres = engine.get_available_genres()
        console.print(f"\n[bold cyan]Available Genres ({len(genres)})[/bold cyan]\n")
        for i, genre in enumerate(genres, 1):
            console.print(f"  {i:3d}. {genre}")
        return

    if list_studios:
        studios = engine.get_available_studios()
        console.print(f"\n[bold cyan]Available Studios ({len(studios)})[/bold cyan]\n")
        for i, studio in enumerate(studios, 1):
            console.print(f"  {i:3d}. {studio}")
        return

    if list_formats:
        formats = engine.get_available_formats()
        console.print(f"\n[bold cyan]Available Formats ({len(formats)})[/bold cyan]\n")
        for fmt in formats:
            console.print(f"  • {fmt}")
        return

    if search:
        tags = engine.search_tags(search, min_rank)
        console.print(f"\n[bold cyan]Tags matching '{search}' ({len(tags)})[/bold cyan]\n")
    else:
        tags = engine.get_available_tags(min_rank)
        console.print(f"\n[bold cyan]All Tags (min rank {min_rank}%) ({len(tags)})[/bold cyan]\n")

    if not tags:
        console.print("[dim]No tags found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=5, justify="right")
    table.add_column("Tag", style="cyan", width=50)
    table.add_column("Max Rank", justify="right", style="green")

    for i, (tag, rank) in enumerate(tags, 1):
        table.add_row(str(i), tag, f"{rank}%")

    console.print(table)


@app.command()
def info(
    username: Annotated[str, typer.Argument(help="AniList username")],
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
) -> None:
    """Show cache and model information."""
    cache_manager = get_cache_manager(cache_dir)
    info = cache_manager.get_cache_info(username)

    if not info.get("exists"):
        console.print(f"[red]No cache found for {username}[/red]")
        return

    console.print(f"\n[bold cyan]Cache Info for [yellow]{username}[/yellow][/bold cyan]\n")

    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Cache Directory", info["cache_dir"])
    table.add_row("Media Catalog", "[OK]" if info["media_catalog"] else "[X]")
    table.add_row("User Entries", "[OK]" if info["user_entries"] else "[X]")
    table.add_row("Metadata", "[OK]" if info["metadata"] else "[X]")
    table.add_row("Trained Model", "[OK]" if info["model"] else "[X]")

    if "synced_at" in info:
        table.add_row("Last Synced", info["synced_at"])
    if "total_user_entries" in info:
        table.add_row("User Entries", str(info["total_user_entries"]))
    if "total_cached_media" in info:
        table.add_row("Cached Anime", str(info["total_cached_media"]))
    if "media_catalog_size_mb" in info:
        table.add_row("Catalog Size", f"{info['media_catalog_size_mb']:.1f} MB")
    if "user_entries_size_mb" in info:
        table.add_row("Entries Size", f"{info['user_entries_size_mb']:.1f} MB")

    console.print(table)


@app.command()
def clear(
    username: Annotated[str, typer.Argument(help="AniList username")],
    cache_dir: Annotated[
        str, typer.Option("--cache-dir", "-c", help="Cache directory")
    ] = DEFAULT_CACHE_DIR,
    confirm: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Clear all cached data for a user."""
    cache_manager = get_cache_manager(cache_dir)

    if not cache_manager.cache_exists(username) and not cache_manager.model_exists(username):
        console.print(f"[yellow]No data found for {username}[/yellow]")
        return

    if not confirm:
        confirm = typer.confirm(f"Delete all cache and model data for {username}?")

    if confirm:
        cache_manager.clear_user_cache(username)
        console.print(f"[green]Cleared all data for {username}[/green]")
    else:
        console.print("[yellow]Cancelled[/yellow]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
