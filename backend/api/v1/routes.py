from __future__ import annotations

from flask import jsonify, request

from ...config import CACHE_DIR
from ...services import (
    ModelService,
    RecommendService,
    TasteService,
    UserService,
)
from .bp import v1_bp

user_service = UserService(CACHE_DIR)
model_service = ModelService(CACHE_DIR)
recommend_service = RecommendService(CACHE_DIR)
taste_service = TasteService(CACHE_DIR)


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@v1_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@v1_bp.get("/users/<username>/info")
def cache_info(username: str):
    try:
        return jsonify(user_service.info(username))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@v1_bp.get("/users/<username>/models")
def list_models(username: str):
    try:
        return jsonify(user_service.models(username))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@v1_bp.post("/users/<username>/sync")
def sync_user(username: str):
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(user_service.sync(username, payload))
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@v1_bp.post("/catalog")
def fetch_catalog():
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(user_service.catalog(payload))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@v1_bp.post("/users/<username>/train")
def train_model_endpoint(username: str):
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(model_service.train(username, payload))
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@v1_bp.post("/users/<username>/evaluate")
def evaluate_models(username: str):
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(model_service.evaluate(username, payload))
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@v1_bp.post("/users/<username>/recommend")
def recommend(username: str):
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(recommend_service.recommend(username, payload))
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@v1_bp.get("/users/<username>/taste/<model_name>")
def taste_drivers(username: str, model_name: str):
    try:
        return jsonify(taste_service.drivers(username, model_name))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
