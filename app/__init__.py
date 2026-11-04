"""Flask application factory for the SpamShield Mail demo."""
import sys
import threading
from pathlib import Path

from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from .config import Config  # noqa: E402
from .services.classifier import get_classifier  # noqa: E402
from .services.generator import EmailGenerator  # noqa: E402
from .services.mailbox import MailboxStore  # noqa: E402


def create_app(config: type[Config] | dict | None = None) -> Flask:
    app = Flask(__name__, instance_path=str(ROOT / "instance"))
    app.config.from_object(Config)
    if isinstance(config, dict):
        app.config.update(config)
    elif config is not None:
        app.config.from_object(config)

    app.extensions["mailbox"] = MailboxStore(app.config["MAILBOX_PATH"])
    app.extensions["generator"] = EmailGenerator(app.config["EMAIL_DB_PATH"], seed=app.config.get("SEED"))
    app.extensions["classifier"] = get_classifier()

    from .routes import api_bp, web_bp
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    if app.config.get("PRELOAD_MODEL"):
        # Warm the transformer up in the background so the first request is fast.
        threading.Thread(target=app.extensions["classifier"].load, daemon=True).start()
    return app
