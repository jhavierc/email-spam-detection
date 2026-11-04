"""Turn SMS messages into realistic-looking dummy e-mails (sender, address, subject)."""
from __future__ import annotations

import random
import re

FIRST_NAMES = ["Laura", "Carlos", "Sofía", "Andrés", "Valentina", "Mateo", "Camila", "Daniel", "Isabella",
               "Santiago", "Mariana", "Julián", "Emma", "Oliver", "Lucas", "Ana", "David", "Sarah", "James",
               "Priya", "Kenji", "Chloe", "Diego", "Paula", "Tomás", "Natalia", "Felipe", "Grace", "Liam", "Mia"]
LAST_NAMES = ["García", "Rodríguez", "Martínez", "López", "Gómez", "Smith", "Johnson", "Brown", "Taylor",
              "Wilson", "Ramírez", "Torres", "Castro", "Moreno", "Chen", "Patel", "Kim", "Silva", "Rossi", "Müller"]
PERSONAL_DOMAINS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com", "unal.edu.co"]

SPAM_SENDERS = [
    ("Prize Claims Dept.", "claims@winner-rewards.co.uk"), ("Mobile Offers", "offers@mobile-deals.biz"),
    ("Customer Service", "noreply@account-update.info"), ("Lucky Draw", "draw@weekly-lotto.net"),
    ("Ringtone Club", "club@tones4u.mobi"), ("Dating Alerts", "alerts@meet-singles.xyz"),
    ("Bonus Team", "bonus@cash-awards.top"), ("Network Rewards", "rewards@network-loyalty.co"),
    ("Holiday Promotions", "travel@free-holidays.club"), ("Urgent Notice", "urgent@secure-verify.online"),
]


def _person(rng: random.Random) -> tuple[str, str]:
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    slug = lambda s: (s.lower().replace("á", "a").replace("é", "e").replace("í", "i")
                      .replace("ó", "o").replace("ú", "u").replace("ü", "u"))
    sep = rng.choice([".", "_", ""])
    num = rng.choice(["", "", str(rng.randint(1, 99))])
    return f"{first} {last}", f"{slug(first)}{sep}{slug(last)}{num}@{rng.choice(PERSONAL_DOMAINS)}"


def make_subject(text: str, max_words: int = 7) -> str:
    """Use the first sentence (or first words) of the message as the subject line."""
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    words = first.split()
    subject = " ".join(words[:max_words])
    if len(words) > max_words:
        subject = subject.rstrip(",.;:") + "…"
    return subject or "(no subject)"


def make_email(idx: int, text: str, label: str, rng: random.Random) -> dict:
    name, address = rng.choice(SPAM_SENDERS) if label == "spam" else _person(rng)
    return {"id": f"msg-{idx:05d}", "sender_name": name, "sender_email": address,
            "subject": make_subject(text), "body": text, "true_label": label}
