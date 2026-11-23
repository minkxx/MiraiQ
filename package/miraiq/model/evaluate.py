"""Offline accuracy evaluation for the taste model.

Holds out a portion of the user's rated anime, trains on the rest, and
measures how well held-out scores are predicted (RMSE/MAE) and how often
held-out anime surface in top-N recommendations (hit rate).

Because everything runs on cached data, no AniList API calls are made.
Use this before and after any model change to verify accuracy actually
improves:

    miraiq evaluate <username>

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from miraiq.anilist.models import CachedMedia, CachedUserEntry
from miraiq.cache.cache import CacheManager
from miraiq.model.trainer import REGRESSORS, FeatureExtractor, TasteModel

HIT_AT = (10, 20, 50)


@dataclass
class FoldScore:
    """Result for one model on one validation fold."""

    model: str
    fold: int
    n_train: int
    n_test: int
    rmse: float
    mae: float


@dataclass
class EvaluationReport:
    """Aggregated offline evaluation results."""

    username: str
    n_rated: int
    n_dropped: int
    n_catalog: int
    split: str  # "temporal" or "random"
    fold_scores: list[FoldScore] = field(default_factory=list)
    baseline_rmse: float = float("nan")
    baseline_mae: float = float("nan")
    hit_rate_at_n: dict[int, float] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for model in sorted({s.model for s in self.fold_scores}):
            scores = [s for s in self.fold_scores if s.model == model]
            out[model] = {
                "rmse": float(np.mean([s.rmse for s in scores])),
                "mae": float(np.mean([s.mae for s in scores])),
                "folds": len(scores),
            }
        return out

    @property
    def best_model(self) -> str | None:
        if not self.summary:
            return None
        return min(self.summary, key=lambda m: self.summary[m]["rmse"])  # type: ignore[arg-type]


def _prepare_xy(
    entries: list[CachedUserEntry],
    media_by_id: dict[int, CachedMedia],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature/target/weight arrays from entries (rated + dropped)."""
    X: list[np.ndarray] = []
    y: list[float] = []
    w: list[float] = []
    for entry in entries:
        media = media_by_id.get(entry.media_id)
        if media is None:
            continue
        if not (entry.is_rated or entry.is_negative_signal):
            continue
        X.append(extractor.get_feature_vector(extractor.transform(media)))
        if entry.is_rated:
            score = entry.score
            if score is None:
                continue
            y.append(score)
            w.append(1.0)
        else:
            # Dropped anime: negative signal at reduced weight
            y.append(2.0)
            w.append(0.3)

    n_features = extractor.n_features
    return (
        np.array(X, dtype=np.float32) if X else np.empty((0, n_features), dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(w, dtype=np.float32),
    )


def _train_on(
    entries: list[CachedUserEntry],
    catalog: list[CachedMedia],
    make_regressor: Any,
    min_ratings: int = 1,
) -> TasteModel | None:
    """Train a TasteModel on `entries` using `make_regressor`; None if too little data."""
    model = TasteModel(regressor_factory=make_regressor)
    try:
        model.train(entries, catalog, min_ratings=min_ratings)
    except ValueError:
        return None
    return model


def _split_indices(
    n: int,
    *,
    temporal: bool,
    test_size: float,
    n_splits: int,
    random_state: int,
    order: list[int],
) -> list[tuple[list[int], list[int]]]:
    """Return (train_idx, test_idx) pairs over the ordered rated list."""
    splits: list[tuple[list[int], list[int]]] = []
    if temporal:
        n_test = max(1, int(round(n * test_size)))
        train_idx = order[:-n_test]
        test_idx = order[-n_test:]
        splits.append((train_idx, test_idx))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_pos, test_pos in kf.split(np.asarray(order)):
            splits.append(([order[i] for i in train_pos], [order[i] for i in test_pos]))
    return splits


def evaluate(
    username: str,
    cache_manager: CacheManager,
    *,
    test_size: float = 0.2,
    n_splits: int = 5,
    temporal: bool = False,
    models: list[str] | None = None,
    random_state: int = 42,
) -> EvaluationReport:
    """Evaluate prediction accuracy with hold-out validation.

    Args:
        username: AniList username.
        cache_manager: Cache manager with synced entries + catalog.
        test_size: Fraction of rated anime held out (temporal split only).
        n_splits: Number of CV folds (random split only).
        temporal: If True, train on older anime and predict newer ones
            (single split ordered by release year). Otherwise random KFold.
        models: Regressor names to compare (default: all of REGRESSORS).
        random_state: Seed for the random split.

    Returns:
        EvaluationReport with per-model RMSE/MAE, a mean-rating baseline,
        and hit-rate @N for the best model.
    """
    entries = cache_manager.load_user_entries(username)
    catalog = cache_manager.load_media_catalog(username)
    media_by_id = {m.id: m for m in catalog}

    rated = [e for e in entries if e.is_rated]
    dropped = [e for e in entries if e.is_negative_signal]
    if len(rated) < 3:
        raise ValueError(
            f"Not enough rated anime to evaluate ({len(rated)} rated). "
            "Rate more anime and re-train."
        )
    if not catalog:
        raise ValueError("No media catalog found. Run 'miraiq fetch-catalog' first.")

    model_names = models or list(REGRESSORS)
    unknown = [m for m in model_names if m not in REGRESSORS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {list(REGRESSORS)}.")

    # Feature space is fixed from the full catalog (no target leakage here).
    extractor = FeatureExtractor()
    extractor.fit(catalog)

    # Order rated anime by release year (missing years last).
    def year_key(entry: CachedUserEntry) -> float:
        media = media_by_id.get(entry.media_id)
        return float(media.season_year) if media and media.season_year else float("inf")

    order = sorted(range(len(rated)), key=lambda i: year_key(rated[i]))

    splits = _split_indices(
        len(rated),
        temporal=temporal,
        test_size=test_size,
        n_splits=n_splits,
        random_state=random_state,
        order=order,
    )

    report = EvaluationReport(
        username=username,
        n_rated=len(rated),
        n_dropped=len(dropped),
        n_catalog=len(catalog),
        split="temporal" if temporal else "random",
    )

    # Watched ids that should never be candidates (mirrors recommend()).
    all_watched_ids = {e.media_id for e in entries}

    baseline_rmses: list[float] = []
    baseline_maes: list[float] = []
    hit_rates: dict[int, list[float]] = {}

    for fold, (train_pos, test_pos) in enumerate(splits):
        train_entries = [rated[i] for i in train_pos] + dropped
        test_entries = [rated[i] for i in test_pos]
        test_ids = {e.media_id for e in test_entries}

        # Baseline: predict the mean training rating for everything.
        train_scores = [e.score for e in train_entries if e.is_rated and e.score is not None]
        baseline_pred = float(np.mean(train_scores)) if train_scores else 5.0
        test_scores = [e.score for e in test_entries if e.score is not None]
        if test_scores:
            baseline_rmses.append(
                float(np.sqrt(np.mean((np.array(test_scores) - baseline_pred) ** 2)))
            )
            baseline_maes.append(float(np.mean(np.abs(np.array(test_scores) - baseline_pred))))

        candidate_media = [m for m in catalog if m.id not in (all_watched_ids - test_ids)]

        for model_name in model_names:
            model = _train_on(train_entries, catalog, REGRESSORS[model_name])
            if model is None:
                continue

            # Accuracy on held-out scores.
            preds: list[float] = []
            trues: list[float] = []
            for entry in test_entries:
                if entry.score is None:
                    continue
                media = media_by_id.get(entry.media_id)
                if media is None:
                    # Held-out anime not in the catalog: cannot score it.
                    continue
                preds.append(model.predict(media))
                trues.append(entry.score)

            if preds:
                rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(trues)) ** 2)))
                mae = float(np.mean(np.abs(np.array(preds) - np.array(trues))))
                report.fold_scores.append(
                    FoldScore(
                        model=model_name,
                        fold=fold,
                        n_train=len(train_entries),
                        n_test=len(test_entries),
                        rmse=rmse,
                        mae=mae,
                    )
                )

            # Ranking: would held-out anime appear in top-N recommendations?
            if candidate_media and model.pipeline is not None:
                candidate_preds = model.predict_batch(candidate_media)
                ranked = sorted(
                    zip(candidate_media, candidate_preds, strict=False),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                for n in HIT_AT:
                    top_n_ids = {m.id for m, _ in ranked[:n]}
                    hits = len(test_ids & top_n_ids)
                    hit_rates.setdefault(n, []).append(hits / max(len(test_ids), 1))

    # Average hit rates across folds.
    report.hit_rate_at_n = {n: float(np.mean(vals)) for n, vals in hit_rates.items()}
    if baseline_rmses:
        report.baseline_rmse = float(np.mean(baseline_rmses))
        report.baseline_mae = float(np.mean(baseline_maes))
    return report


def format_report(report: EvaluationReport) -> str:
    """Render the report as a human-readable string."""
    lines: list[str] = []
    lines.append(f"Offline evaluation for {report.username}")
    lines.append(
        f"  rated={report.n_rated}  dropped={report.n_dropped}  "
        f"catalog={report.n_catalog}  split={report.split}"
    )
    if report.fold_scores:
        lines.append(
            f"  baseline (mean rating) RMSE={report.baseline_rmse:.3f}  "
            f"MAE={report.baseline_mae:.3f}"
        )
        lines.append("  model            RMSE   MAE   folds")
        for model, stats in report.summary.items():
            marker = " *" if model == report.best_model else ""
            lines.append(
                f"  {model:<16} {stats['rmse']:.3f} {stats['mae']:.3f} {stats['folds']}{marker}"
            )
        lines.append(f"  best model: {report.best_model}")
    if report.hit_rate_at_n:
        hits = "  ".join(
            f"@{n}={report.hit_rate_at_n[n]:.2f}" for n in sorted(report.hit_rate_at_n)
        )
        lines.append(f"  hit rate: {hits}  (fraction of held-out anime in top-N predictions)")
    lines.append("  * = lowest RMSE")
    return "\n".join(lines)
