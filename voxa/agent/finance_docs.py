from __future__ import annotations

KINDS: dict[str, str] = {
    "invoice": "Invoices",
    "receipt": "Receipts",
    "warranty": "Warranties",
    "tax": "Tax",
    "statement": "Statements",
}

KIND_WORDS: dict[str, tuple[str, ...]] = {
    "invoice": ("invoice",),
    "receipt": ("receipt",),
    "warranty": ("warranty",),
    "tax": ("tax", "w2", "1099"),
    "statement": ("statement",),
}

PLURAL: dict[str, tuple[str, str]] = {
    "invoice": ("invoice", "invoices"),
    "receipt": ("receipt", "receipts"),
    "warranty": ("warranty", "warranties"),
    "tax": ("tax document", "tax documents"),
    "statement": ("statement", "statements"),
}


def kind_of(name: str) -> str | None:
    low = name.lower()
    for kind, words in KIND_WORDS.items():
        for w in words:
            if w in low:
                return kind
    return None


def folder_name(kind: str) -> str:
    return KINDS[kind]


def normalize_kind(spoken: str) -> str:
    norm = " ".join(spoken.lower().split())
    if norm in ("invoice", "invoices"):
        return "invoice"
    if norm in ("receipt", "receipts"):
        return "receipt"
    if norm in ("warranty", "warranties"):
        return "warranty"
    if norm in ("tax",):
        return "tax"
    if norm in ("statement", "statements", "bank statement", "bank statements"):
        return "statement"
    return norm


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def spoken_sort(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if total == 0:
        return "Nothing to file."
    parts = []
    for kind in sorted(counts):
        n = counts[kind]
        word = PLURAL[kind][0] if n == 1 else PLURAL[kind][1]
        parts.append(f"{n} {word}")
    return f"I filed {total} documents: {_join(parts)}."


def spoken_count(kind: str, n: int) -> str:
    if n == 0:
        return f"You have no {PLURAL[kind][1]}."
    word = PLURAL[kind][0] if n == 1 else PLURAL[kind][1]
    return f"You have {n} {word}."


def spoken_undo(n: int) -> str:
    if n == 0:
        return "Nothing to undo."
    return f"I moved {n} documents back."
