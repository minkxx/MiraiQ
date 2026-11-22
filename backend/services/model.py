from __future__ import annotations

from typing import Any

from miraiq.model.evaluate import evaluate
from miraiq.model.trainer import REGRESSORS, train_model

from .base import BaseService


class ModelService(BaseService):
    def train(self, username: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_name = payload.get("model_name", "gbr")
        if model_name not in REGRESSORS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from {list(REGRESSORS)}."
            )

        model_obj, metrics = train_model(
            username,
            self.cm(username),
            min_ratings=int(payload.get("min_ratings", 5)),
            tag_threshold=int(payload.get("tag_threshold", 50)),
            min_studio_count=int(payload.get("min_studio_count", 3)),
            include_format=bool(payload.get("include_format", False)),
            include_source=bool(payload.get("include_source", False)),
            include_metadata=bool(payload.get("include_metadata", True)),
            regressor_factory=REGRESSORS[model_name],
            model_name=model_name,
        )
        return {
            "model": model_name,
            "train_rmse": metrics["train_rmse"],
            "cv_rmse": metrics.get("cv_rmse"),
            "n_samples": metrics["n_samples"],
            "n_features": metrics["n_features"],
            "taste_drivers": [
                d.model_dump() for d in model_obj.get_taste_drivers(top_k=12)
            ],
        }

    def evaluate(self, username: str, payload: dict[str, Any]) -> dict[str, Any]:
        models = payload.get("models") or list(REGRESSORS)
        unknown = [m for m in models if m not in REGRESSORS]
        if unknown:
            raise ValueError(
                f"Unknown models: {unknown}. Choose from {list(REGRESSORS)}."
            )

        report = evaluate(
            username,
            self.cm(username),
            test_size=float(payload.get("test_size", 0.2)),
            n_splits=int(payload.get("n_splits", 5)),
            temporal=bool(payload.get("temporal", False)),
            models=models,
        )
        return {
            "best_model": report.best_model,
            "split": report.split,
            "baseline_rmse": report.baseline_rmse,
            "baseline_mae": report.baseline_mae,
            "summary": report.summary,
            "hit_rate_at_n": report.hit_rate_at_n,
        }
