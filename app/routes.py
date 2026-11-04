"""HTTP routes: the single-page mail client and its JSON API."""
from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from .services.classifier import text_for_classification
from .services.mailbox import FOLDERS

web_bp = Blueprint("web", __name__)
api_bp = Blueprint("api", __name__)


def _mailbox():
    return current_app.extensions["mailbox"]


def _classifier():
    return current_app.extensions["classifier"]


def _classify_and_deliver(emails: list[dict]) -> list[dict]:
    texts = [text_for_classification(e.get("subject", ""), e.get("body", "")) for e in emails]
    predictions = _classifier().predict(texts)
    return [_mailbox().deliver(e, p.to_dict()) for e, p in zip(emails, predictions)]


# --------------------------------------------------------------------------- web
@web_bp.get("/")
def index():
    return render_template("index.html", owner=current_app.config["MAILBOX_OWNER"])


# --------------------------------------------------------------------------- api
@api_bp.get("/health")
def health():
    clf = _classifier()
    return jsonify(status="ok", model=clf.model_name, model_loaded=clf.is_loaded,
                   fine_tuned=clf.is_finetuned, threshold=clf.threshold,
                   pool_size=current_app.extensions["generator"].pool_size)


@api_bp.get("/stats")
def stats():
    return jsonify(_mailbox().stats())


@api_bp.get("/emails")
def list_emails():
    folder = request.args.get("folder", "inbox")
    if folder not in (*FOLDERS, "all"):
        abort(400, description="folder must be inbox, spam or all")
    return jsonify(_mailbox().list(folder, request.args.get("q", "")))


@api_bp.get("/emails/<uid>")
def get_email(uid):
    email = _mailbox().get(uid) or abort(404)
    if not email["read"]:
        email = _mailbox().update(uid, read=True)
    return jsonify(email)


@api_bp.patch("/emails/<uid>")
def patch_email(uid):
    data = request.get_json(silent=True) or {}
    allowed = {k: data[k] for k in ("folder", "read", "starred") if k in data}
    try:
        email = _mailbox().update(uid, **allowed)
    except ValueError as exc:
        abort(400, description=str(exc))
    return jsonify(email or abort(404))


@api_bp.delete("/emails/<uid>")
def delete_email(uid):
    if not _mailbox().delete(uid):
        abort(404)
    return "", 204


@api_bp.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    try:
        n = max(1, min(int(data.get("count", 1)), current_app.config["MAX_GENERATE"]))
    except (TypeError, ValueError):
        abort(400, description="count must be an integer")
    ratio = data.get("spam_ratio")
    ratio = None if ratio in (None, "") else max(0.0, min(float(ratio), 1.0))
    emails = current_app.extensions["generator"].sample(n, spam_ratio=ratio)
    if not emails:
        abort(503, description="E-mail pool is empty. Run scripts/build_email_db.py first.")
    return jsonify(delivered=_classify_and_deliver(emails), stats=_mailbox().stats())


@api_bp.post("/compose")
def compose():
    data = request.get_json(silent=True) or {}
    body, subject = (data.get("body") or "").strip(), (data.get("subject") or "").strip()
    if not body and not subject:
        abort(400, description="subject or body is required")
    email = {"sender_name": (data.get("sender_name") or "External sender").strip()[:80],
             "sender_email": (data.get("sender_email") or "sender@external.com").strip()[:120],
             "subject": subject[:200], "body": body[:5000], "true_label": None}
    return jsonify(delivered=_classify_and_deliver([email]), stats=_mailbox().stats())


@api_bp.post("/classify")
def classify():
    data = request.get_json(silent=True) or {}
    text = text_for_classification(data.get("subject", ""), data.get("body", data.get("text", "")))
    if not text:
        abort(400, description="text is required")
    return jsonify(_classifier().predict_one(text).to_dict())


@api_bp.post("/reset")
def reset():
    _mailbox().reset()
    return jsonify(_mailbox().stats())


@api_bp.errorhandler(400)
@api_bp.errorhandler(404)
@api_bp.errorhandler(503)
def json_error(err):
    return jsonify(error=getattr(err, "description", str(err))), err.code
