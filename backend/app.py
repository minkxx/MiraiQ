from __future__ import annotations

from flask import Flask

from .api.v1 import routes  # noqa: F401
from .api.v1.bp import v1_bp  # noqa: E402
from .config import DEBUG, HOST, PORT  # noqa: E402
from .errors import register_error_handlers  # noqa: E402
from .extensions import init_extensions  # noqa: E402


def create_app() -> Flask:
    app = Flask(__name__)

    init_extensions(app)
    register_error_handlers(app)
    app.register_blueprint(v1_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
