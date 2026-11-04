"""Process-wide singleton around :class:`spam_detection.model.SpamClassifier`."""
from functools import lru_cache

from spam_detection.model import SpamClassifier


@lru_cache(maxsize=1)
def get_classifier() -> SpamClassifier:
    return SpamClassifier()


def text_for_classification(subject: str, body: str) -> str:
    """Combine subject and body; skip the subject when it is just a prefix of the body."""
    subject, body = (subject or "").strip(), (body or "").strip()
    if not subject or body.startswith(subject.rstrip("…")):
        return body or subject
    return f"{subject}. {body}" if body else subject
