"""Taste modeling and score prediction using gradient boosting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.anilist.models import (
    AnimeFeatures,
    CachedMedia,
    CachedUserEntry,
    TasteDriver,
)
from src.cache.cache import CacheManager


class FeatureExtractor:
    """Extract feature vectors from anime data."""

    KNOWN_GENRES = [
        "Action",
        "Adventure",
        "Comedy",
        "Drama",
        "Ecchi",
        "Fantasy",
        "Horror",
        "Mahou Shoujo",
        "Mecha",
        "Music",
        "Mystery",
        "Psychological",
        "Romance",
        "Sci-Fi",
        "Slice of Life",
        "Sports",
        "Supernatural",
        "Thriller",
    ]

    def __init__(
        self,
        tag_threshold: int = 50,
        min_studio_count: int = 3,
        include_format: bool = False,
        include_source: bool = False,
        include_metadata: bool = True,
    ):
        self._all_tags: set[str] = set()
        self._all_studios: set[str] = set()
        self._feature_names: list[str] = []
        self._tag_threshold = tag_threshold
        self._min_studio_count = min_studio_count
        self._include_format = include_format
        self._include_source = include_source
        self._include_metadata = include_metadata

    def fit(self, media_list: list[CachedMedia]) -> None:
        """Learn all unique tags and studios from the catalog."""
        studio_counts: dict[str, int] = {}

        for media in media_list:
            for tag, rank in media.tag_ranks.items():
                if rank >= self._tag_threshold:
                    self._all_tags.add(tag)
            for studio in media.main_studios:
                studio_counts[studio] = studio_counts.get(studio, 0) + 1

        self._all_studios = {s for s, c in studio_counts.items() if c >= self._min_studio_count}

        self._feature_names = []

        for genre in sorted(self.KNOWN_GENRES):
            self._feature_names.append(f"genre_{genre}")

        # Genre interaction features (capture combinations like Rom-Com)
        self._feature_names.append("genre_Romance_x_Comedy")
        self._feature_names.append("genre_Action_x_Adventure")
        self._feature_names.append("genre_Fantasy_x_Adventure")
        self._feature_names.append("genre_Sci_Fi_x_Mystery")
        self._feature_names.append("genre_Drama_x_Romance")
        self._feature_names.append("genre_Drama_x_Psychological")

        for tag in sorted(self._all_tags):
            self._feature_names.append(f"tag_{tag}")

        for studio in sorted(self._all_studios):
            self._feature_names.append(f"studio_{studio}")

        if self._include_format:
            self._feature_names.extend(["format_TV", "format_MOVIE", "format_OVA", "format_ONA"])

        if self._include_metadata:
            self._feature_names.extend(
                [
                    "season_year",
                    "episodes",
                    "duration",
                    "popularity",
                    "community_score",
                ]
            )

        if self._include_source:
            self._feature_names.extend(
                [
                    "source_original",
                    "source_manga",
                    "source_light_novel",
                ]
            )

    def transform(self, media: CachedMedia) -> AnimeFeatures:
        """Convert CachedMedia to AnimeFeatures."""
        return AnimeFeatures(
            media_id=media.id,
            genres=media.genres,
            tag_ranks=media.tag_ranks,
            main_studios=media.main_studios,
            format=media.format,
            season_year=media.season_year,
            episodes=media.episodes,
            duration=media.duration,
            source=media.source,
            popularity=media.popularity,
            community_score=media.community_score,
        )

    def get_feature_vector(self, features: AnimeFeatures) -> np.ndarray:
        """Convert AnimeFeatures to numpy feature vector."""
        return np.array(features.to_vector(self._feature_names), dtype=np.float32)

    def get_feature_names(self) -> list[str]:
        return self._feature_names.copy()

    @property
    def n_features(self) -> int:
        return len(self._feature_names)


class TasteModel:
    """Personalized anime rating prediction model."""

    def __init__(
        self,
        tag_threshold: int = 50,
        min_studio_count: int = 3,
        include_format: bool = False,
        include_source: bool = False,
        include_metadata: bool = True,
    ):
        self.feature_extractor = FeatureExtractor(
            tag_threshold=tag_threshold,
            min_studio_count=min_studio_count,
            include_format=include_format,
            include_source=include_source,
            include_metadata=include_metadata,
        )
        self.pipeline: Pipeline | None = None
        self._is_trained = False
        # Calibration: track prediction statistics
        self._train_preds: list[float] = []
        self._train_targets: list[float] = []

    def train(
        self,
        user_entries: list[CachedUserEntry],
        media_catalog: list[CachedMedia],
        min_ratings: int = 5,
    ) -> dict[str, Any]:
        """
        Train the model on user's rated anime.

        Args:
            user_entries: User's anime list entries
            media_catalog: Full anime catalog with features
            min_ratings: Minimum rated entries required

        Returns:
            Training metrics and diagnostics
        """
        media_by_id = {m.id: m for m in media_catalog}

        X_train = []
        y_train = []
        sample_weights = []

        rated_count = 0
        dropped_count = 0

        for entry in user_entries:
            media = media_by_id.get(entry.media_id)
            if not media:
                continue

            features = self.feature_extractor.transform(media)
            feature_vector = self.get_feature_vector(features)

            if entry.is_rated:
                X_train.append(feature_vector)
                y_train.append(entry.score)
                sample_weights.append(1.0)
                rated_count += 1
            elif entry.is_negative_signal:
                # Negative signal: user dropped this anime
                # Treat as low rating (2.0/10) with lower weight
                X_train.append(feature_vector)
                y_train.append(2.0)
                sample_weights.append(0.3)
                dropped_count += 1

        if rated_count < min_ratings:
            raise ValueError(
                f"Need at least {min_ratings} rated anime to train. "
                f"Found {rated_count} rated + {dropped_count} dropped."
            )

        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.float32)
        weights = np.array(sample_weights, dtype=np.float32)

        # Fit feature extractor on full catalog first
        self.feature_extractor.fit(media_catalog)

        # Re-extract features with fitted extractor (only for training samples)
        X_train = []
        for entry in user_entries:
            media = media_by_id.get(entry.media_id)
            if not media:
                continue
            if entry.is_rated or entry.is_negative_signal:
                features = self.feature_extractor.transform(media)
                X_train.append(self.get_feature_vector(features))

        X = np.array(X_train, dtype=np.float32)

        gbr = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        self.pipeline = Pipeline(
            [
                ("gbr", gbr),
            ]
        )
        self.pipeline.fit(X, y, gbr__sample_weight=weights)

        self._is_trained = True

        y_pred = self.pipeline.predict(X)
        self._train_preds = np.asarray(y_pred).tolist()
        self._train_targets = y.tolist()

        n_splits = min(5, len(X))
        if n_splits >= 3 and len(X) >= 10:
            cv_scores = cross_val_score(
                self.pipeline,
                X,
                y,
                cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
                scoring="neg_mean_squared_error",
            )
            cv_rmse = float(np.sqrt(-cv_scores.mean()))
        else:
            cv_rmse = float("nan")

        train_rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        gbr_model = self.pipeline.named_steps["gbr"]
        feature_weights = gbr_model.feature_importances_

        # Calibration: compute prediction intervals
        residuals = y - y_pred
        residual_std = float(np.std(residuals))

        return {
            "n_samples": len(X),
            "n_rated": rated_count,
            "n_dropped": dropped_count,
            "n_features": len(self.feature_extractor.get_feature_names()),
            "train_rmse": float(train_rmse),
            "cv_rmse": float(cv_rmse),
            "feature_weights": feature_weights,
            "residual_std": residual_std,
        }

    def get_feature_vector(self, features: AnimeFeatures) -> np.ndarray:
        return features.to_vector(self.feature_extractor.get_feature_names())

    def predict(self, media: CachedMedia) -> float:
        """Predict personal rating for an anime."""
        return self.predict_with_confidence(media)[0]

    def predict_with_confidence(self, media: CachedMedia) -> tuple[float, float]:
        """
        Predict rating with confidence interval.

        Returns:
            (prediction, confidence_half_width) where confidence_half_width
            is approximately 1 std of residuals (68% confidence interval)
        """
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        features = self.feature_extractor.transform(media)
        feature_vector = self.get_feature_vector(features).reshape(1, -1)
        prediction = self.pipeline.predict(feature_vector)[0]
        # Don't clamp - let it go above 10 for better ranking
        # prediction = float(np.clip(prediction, 0.0, 10.0))

        # Confidence based on training residual std (if available)
        confidence = 1.0  # Default
        if hasattr(self, "_train_preds") and self._train_preds:
            residuals = np.array(self._train_targets) - np.array(self._train_preds)
            confidence = float(np.std(residuals))

        return float(prediction), confidence

    def predict_batch(self, media_list: list[CachedMedia]) -> list[float]:
        """Predict ratings for multiple anime."""
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        feature_vectors = []
        for media in media_list:
            features = self.feature_extractor.transform(media)
            feature_vectors.append(self.get_feature_vector(features))

        X = np.array(feature_vectors, dtype=np.float32)
        predictions = self.pipeline.predict(X)
        return [float(p) for p in predictions]  # No clamp - let ranking be natural

    def predict_batch_with_confidence(
        self, media_list: list[CachedMedia]
    ) -> list[tuple[float, float]]:
        """Predict ratings with confidence for multiple anime."""
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        results = []
        for media in media_list:
            pred, conf = self.predict_with_confidence(media)
            results.append((pred, conf))
        return results

    def get_taste_drivers(self, top_k: int = 20) -> list[TasteDriver]:
        """Extract top positive and negative taste drivers."""
        if not self._is_trained or self.pipeline is None:
            return []

        feature_names = self.feature_extractor.get_feature_names()

        gbr_model = self.pipeline.named_steps["gbr"]
        weights = gbr_model.feature_importances_

        drivers = []
        for name, weight in zip(feature_names, weights, strict=False):
            if abs(weight) < 0.01:  # Skip negligible weights
                continue

            feature_type = "metadata"
            description = name

            if name.startswith("genre_"):
                feature_type = "genre"
                description = f"Genre: {name[6:]}"
            elif name.startswith("tag_"):
                feature_type = "tag"
                description = f"Tag: {name[4:]}"
            elif name.startswith("studio_"):
                feature_type = "studio"
                description = f"Studio: {name[7:]}"
            elif name.startswith("format_"):
                feature_type = "format"
                description = f"Format: {name[7:]}"
            elif name in ("season_year", "episodes", "duration", "popularity", "community_score"):
                feature_type = "metadata"
                description = f"Metadata: {name}"
            elif name.startswith("source_"):
                feature_type = "metadata"
                description = f"Source: {name[7:]}"

            drivers.append(
                TasteDriver(
                    feature=name,
                    weight=float(weight),
                    feature_type=feature_type,
                    description=description,
                    is_positive=weight > 0,
                )
            )

        drivers.sort(key=lambda d: d.abs_weight, reverse=True)
        return drivers[:top_k]

    def explain_prediction(
        self, media: CachedMedia, top_k: int = 5
    ) -> list[tuple[str, float, str]]:
        """Explain why an anime got its predicted score."""
        if not self._is_trained or self.pipeline is None:
            return []

        features = self.feature_extractor.transform(media)
        feature_vector = self.get_feature_vector(features)
        feature_names = self.feature_extractor.get_feature_names()

        gbr_model = self.pipeline.named_steps["gbr"]
        weights = gbr_model.feature_importances_

        contributions = []
        for i, (name, value) in enumerate(zip(feature_names, feature_vector, strict=False)):
            if value == 0:
                continue
            contribution = value * weights[i]
            if abs(contribution) < 0.01:
                continue

            if name.startswith("genre_"):
                desc = f"Genre: {name[6:]}"
            elif name.startswith("tag_"):
                desc = f"Tag: {name[4:]} (relevance: {int(features.tag_ranks.get(name[4:], 0))}%)"
            elif name.startswith("studio_"):
                desc = f"Studio: {name[7:]}"
            elif name.startswith("format_"):
                desc = f"Format: {name[7:]}"
            elif name == "season_year":
                desc = f"Year: {features.season_year}"
            elif name == "episodes":
                desc = f"Episodes: {features.episodes}"
            elif name == "duration":
                desc = f"Duration: {features.duration}min"
            elif name.startswith("source_"):
                desc = f"Source: {name[7:]}"
            elif name == "popularity":
                desc = f"Popularity: {features.popularity}"
            elif name == "community_score":
                desc = f"Community Score: {features.community_score:.1f}"
            else:
                desc = name

            contributions.append(
                (desc, float(contribution), "positive" if contribution > 0 else "negative")
            )

        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:top_k]


def train_model(
    username: str,
    cache_manager: CacheManager,
    min_ratings: int = 5,
    tag_threshold: int = 50,
    min_studio_count: int = 3,
    include_format: bool = False,
    include_source: bool = False,
    include_metadata: bool = True,
) -> tuple[TasteModel, dict[str, Any]]:
    """Train a taste model for a user from cached data."""
    user_entries = cache_manager.load_user_entries(username)
    media_catalog = cache_manager.load_media_catalog(username)

    if not user_entries:
        raise ValueError(f"No user entries found for {username}. Run sync first.")
    if not media_catalog:
        raise ValueError(f"No media catalog found for {username}. Run sync first.")

    model = TasteModel(
        tag_threshold=tag_threshold,
        min_studio_count=min_studio_count,
        include_format=include_format,
        include_source=include_source,
        include_metadata=include_metadata,
    )
    metrics = model.train(user_entries, media_catalog, min_ratings=min_ratings)

    cache_manager.save_model(username, model.pipeline, model.feature_extractor.get_feature_names())
    taste_drivers = model.get_taste_drivers(top_k=50)
    cache_manager.save_taste_drivers(username, taste_drivers)

    return model, metrics
