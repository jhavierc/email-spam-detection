"""Thread-safe, JSON-persisted dummy mailbox with two folders: ``inbox`` and ``spam``."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

FOLDERS = ("inbox", "spam")


class MailboxStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._emails: dict[str, dict] = {}
        self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._emails = {e["uid"]: e for e in data.get("emails", [])}
            except (json.JSONDecodeError, KeyError):
                self._emails = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"emails": list(self._emails.values())}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # ---------------------------------------------------------------- commands
    def deliver(self, email: dict, prediction: dict) -> dict:
        """Store an e-mail in the folder chosen by the classifier."""
        folder = "spam" if prediction["label"] == "spam" else "inbox"
        record = {
            "uid": uuid.uuid4().hex[:12],
            "source_id": email.get("id"),
            "sender_name": email.get("sender_name", "Unknown"),
            "sender_email": email.get("sender_email", "unknown@example.com"),
            "subject": email.get("subject") or "(no subject)",
            "body": email.get("body", ""),
            "true_label": email.get("true_label"),          # ground truth (None for composed mail)
            "prediction": prediction,
            "folder": folder,
            "classified_folder": folder,
            "user_moved": False,
            "read": False,
            "starred": False,
            "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            self._emails[record["uid"]] = record
            self._save()
        return record

    def get(self, uid: str) -> dict | None:
        with self._lock:
            e = self._emails.get(uid)
            return dict(e) if e else None

    def update(self, uid: str, **fields) -> dict | None:
        with self._lock:
            e = self._emails.get(uid)
            if e is None:
                return None
            if "folder" in fields:
                if fields["folder"] not in FOLDERS:
                    raise ValueError(f"unknown folder {fields['folder']!r}")
                fields["user_moved"] = fields["folder"] != e["classified_folder"]
            e.update(fields)
            self._save()
            return dict(e)

    def delete(self, uid: str) -> bool:
        with self._lock:
            removed = self._emails.pop(uid, None) is not None
            if removed:
                self._save()
            return removed

    def reset(self) -> None:
        with self._lock:
            self._emails.clear()
            self._save()

    # ---------------------------------------------------------------- queries
    def list(self, folder: str | None = None, query: str = "") -> list[dict]:
        q = query.lower().strip()
        with self._lock:
            items = [dict(e) for e in self._emails.values() if folder in (None, "all") or e["folder"] == folder]
        if q:
            items = [e for e in items if q in f"{e['sender_name']} {e['sender_email']} {e['subject']} {e['body']}".lower()]
        return sorted(items, key=lambda e: e["received_at"] + e["uid"], reverse=True)

    def stats(self) -> dict:
        with self._lock:
            emails = list(self._emails.values())
        out = {f: {"total": 0, "unread": 0} for f in FOLDERS}
        tp = tn = fp = fn = 0
        for e in emails:
            out[e["folder"]]["total"] += 1
            out[e["folder"]]["unread"] += int(not e["read"])
            truth = e.get("true_label")
            if truth is None:
                continue
            pred = e["prediction"]["label"]
            tp += truth == "spam" and pred == "spam"
            tn += truth == "ham" and pred == "ham"
            fp += truth == "ham" and pred == "spam"
            fn += truth == "spam" and pred == "ham"
        labelled = tp + tn + fp + fn
        return {
            "folders": out,
            "total": len(emails),
            "evaluation": {
                "labelled": labelled, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "accuracy": (tp + tn) / labelled if labelled else None,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
            },
        }
