# MiraiQ

Personal Anime Recommendation Engine powered by your AniList taste.

Miraiq is a standalone Python package with a command-line interface that syncs your AniList watch list, trains a personal taste model from your ratings, and generates personalized anime recommendations — all running locally from your own cached data.

## Installation

```bash
pip install miraiq
```

Or install from source:

```bash
git clone https://github.com/minkxx/MiraiQ.git
cd MiraiQ/package
pip install -e .
```

This installs the `miraiq` CLI command.

## Quick start

```bash
# 1. Sync your AniList profile and anime list
miraiq sync minkxx

# 2. Fetch the global anime candidate catalog (shared across all users)
miraiq fetch-catalog

# 3. Train your personal taste model
miraiq train minkxx

# 4. Get personalized recommendations
miraiq recommend minkxx
```

## CLI reference

### `miraiq sync`

Sync your AniList profile and anime list into the local cache.

```bash
miraiq sync <username> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |
| `--force`, `-f` | Force re-sync even if cache exists | `False` |

### `miraiq fetch-catalog`

Fetch the global anime candidate catalog used for recommendations. This is shared across all users.

```bash
miraiq fetch-catalog [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |
| `--limit`, `-n` | Number of anime to fetch | `2000` |
| `--force`, `-f` | Re-fetch even if catalog exists | `False` |

### `miraiq train`

Train your personal taste model from your cached ratings.

```bash
miraiq train <username> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |
| `--min-ratings`, `-m` | Minimum rated anime required | `5` |
| `--tag-threshold` | Minimum tag rank % to include as feature | `50` |
| `--min-studio-count` | Minimum anime count for studio feature | `3` |
| `--include-format` | Include format (TV/MOVIE/OVA) as features | `False` |
| `--include-source` | Include source (manga/LN/original) as features | `False` |
| `--include-metadata` | Include year/episodes/popularity as features | `True` |
| `--model` | Regressor to train | `gbr` |

### `miraiq evaluate`

Evaluate prediction accuracy with hold-out validation (offline, no API calls).

```bash
miraiq evaluate <username> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |
| `--test-size` | Fraction of rated anime held out for validation | `0.2` |
| `--n-splits` | Number of CV folds | `5` |
| `--temporal` | Train on older anime, predict newer ones | `False` |

### `miraiq taste`

Display your learned taste drivers (what you like and dislike).

```bash
miraiq taste <username> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |
| `--top`, `-t` | Number of top drivers to show | `15` |
| `--model` | Trained model to show drivers for | `gbr` |

### `miraiq recommend`

Get personalized anime recommendations.

```bash
miraiq recommend <username> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |
| `--limit`, `-n` | Number of recommendations | `20` |
| `--sort`, `-s` | Sort: predicted, community, difference | `predicted` |
| `--include-genre`, `-g` | Include genres (can repeat) | — |
| `--exclude-genre`, `-G` | Exclude genres (can repeat) | — |
| `--include-tag`, `-t` | Include tags (can repeat) | — |
| `--exclude-tag`, `-T` | Exclude tags (can repeat) | — |
| `--min-tag-rank` | Minimum tag relevance % | `50` |
| `--format`, `-f` | Formats: TV, MOVIE, OVA, ONA | — |
| `--year-min` | Minimum release year | — |
| `--year-max` | Maximum release year | — |
| `--show-sequels` | Show sequels without prequel watched | `False` |
| `--include-planning` / `--no-planning` | Include Planning list anime | `True` |
| `--unreleased` | Include not-yet-released anime | `False` |
| `--include-chinese-korean` | Include Chinese/Korean anime | `False` |
| `--model` | Trained model to use | `gbr` |

### `miraiq explain`

Explain why an anime was recommended, showing feature contributions.

```bash
miraiq explain <username> <anime_id> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |

### `miraiq tags`

List available genres, tags, studios, or formats from the catalog.

```bash
miraiq tags <username> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |
| `--search`, `-s` | Search tags by name | — |
| `--min-rank` | Minimum tag relevance % | `50` |
| `--genres` | List genres instead of tags | `False` |
| `--studios` | List studios | `False` |
| `--formats` | List formats | `False` |

### `miraiq info`

Show cache and model information for a user.

```bash
miraiq info <username> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |

### `miraiq clear`

Clear all cached data for a user.

```bash
miraiq clear <username> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-dir`, `-c` | Directory to store cached data | `.cache` |
| `--yes`, `-y` | Skip confirmation | `False` |

## License

MIT License. See [LICENSE](LICENSE) for details.
