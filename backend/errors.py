from __future__ import annotations

from flask import Flask, jsonify


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(err):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(err):
        return jsonify({"error": "method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(err):
        return jsonify({"error": "internal server error"}), 500
