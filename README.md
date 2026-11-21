# MiraiQ

Personal anime recommendation engine powered by your AniList taste. Learns what you like from your watch history, trains a local ML model, and recommends unwatched anime with predicted personal ratings.

## Install

```bash
# From source
git clone https://github.com/minkxx/MiraiQ
cd MiraiQ
pip install -e .
```

Requires Python 3.10+. Dependencies: `rich`, `httpx`, `pydantic`, `numpy`, `scikit-learn`, `joblib`, `typer`, `orjson`, `tqdm`.

## Quick Start

```bash
# 1. Sync your AniList data
miraiq sync your_username

# 2. Train your taste model
miraiq train your_username

# 3. View learned taste drivers
miraiq taste your_username

# 4. Get recommendations
miraiq recommend your_username
```

## Commands

| Command | Description |
|---------|-------------|
| `sync <user>` | Fetch AniList profile + ~2000 anime catalog |
| `update-catalog <user>` | Refresh global anime catalog only |
| `train <user>` | Train personalized model (min 5 rated) |
| `taste <user>` | Show positive/negative taste drivers |
| `recommend <user>` | Get recommendations with filtering |
| `explain <user> <anime_id>` | Explain prediction with feature contributions |
| `tags <user>` | Browse genres, tags, studios, formats |
| `info <user>` | Show cache/model status |
| `clear <user>` | Delete all cached data |

## Filtering Options

```bash
# Genres (AND logic - all must match)
miraiq recommend user --include-genre Romance --include-genre Comedy

# Tags (with min relevance %)
miraiq recommend user --include-tag "Time Travel" --min-tag-rank 70

# Formats & years
miraiq recommend user --format TV --format MOVIE --year-min 2010

# Sorting
miraiq recommend user --sort predicted    # Your predicted score (default)
miraiq recommend user --sort community    # AniList average
miraiq recommend user --sort difference   # Hidden gems
```

Other options: `--show-sequels`, `--include-planning`, `--exclude-chinese-korean`, `-n 50`

## How It Works

1. **Sync** — Pulls your AniList list + popular anime catalog (rate-limited to 30 req/min)
2. **Cache** — Stores JSON in `~/.miraiq/cache/<user>/`
3. **Features** — Genres (binary), tags (weighted by rank %), studios (binary), metadata (year, episodes, popularity, community score)
4. **Train** — GradientBoostingRegressor on rated anime (+ dropped as negative signal)
5. **Predict** — Scores unwatched anime 0–10, boosted by similarity to highly-rated watched shows

## License

MIT — see [LICENSE](LICENSE)