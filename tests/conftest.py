import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from spam_detection.model import Prediction  # noqa: E402


class FakeClassifier:
    """Keyword-based stand-in so API tests don't download / load the transformer."""
    model_name = "fake"
    is_loaded = True
    is_finetuned = False
    threshold = 0.5

    def load(self):
        return self

    def predict(self, texts):
        out = []
        for t in texts:
            p = 0.97 if any(w in t.lower() for w in ("win", "prize", "free", "claim", "urgent")) else 0.03
            out.append(Prediction("spam" if p >= 0.5 else "ham", p, max(p, 1 - p)))
        return out

    def predict_one(self, text):
        return self.predict([text])[0]


@pytest.fixture
def app(tmp_path):
    from app import create_app
    application = create_app({"TESTING": True, "PRELOAD_MODEL": False, "SEED": 1,
                              "MAILBOX_PATH": tmp_path / "mailbox.json"})
    application.extensions["classifier"] = FakeClassifier()
    return application


@pytest.fixture
def client(app):
    return app.test_client()
