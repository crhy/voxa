"""Background review of the little brain's answers (crhy/voxa#116 §6).

Pure Python: no GTK, no host access, and no model of its own. The caller injects
``review_fn(question, answer)``, so this module never loads a model, opens a
connection or touches the GPU. Review runs on its own worker thread: the answer
the user just got is never delayed by it, and a real disagreement comes back later
as a "Correction: ..." payload plus a logged case.

#116 §7 keeps the big-brain model choice, the GPU split and the sampling rate
open; this module touches none of them — the rate is a constructor value with a
documented default.
"""

from __future__ import annotations

import logging
import re
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "AGREE",
    "DISAGREE",
    "SKIPPED",
    "UNSURE",
    "ERROR",
    "DEFAULT_SAMPLE_RATE",
    "ESCALATION_WORTHY_KINDS",
    "ReviewCase",
    "Reviewer",
    "escalation_worthy",
    "should_review",
]

AGREE = "agree"
DISAGREE = "disagree"
UNSURE = "unsure"
ERROR = "error"
SKIPPED = "skipped"

# §7 leaves the rate open; 1 in 5 is a documented placeholder until it is measured.
DEFAULT_SAMPLE_RATE = 0.2
# Request kinds whose answers the little brain can plausibly get wrong, so they
# are reviewed even when it stayed on the fast path.
ESCALATION_WORTHY_KINDS = frozenset({"model", "chat"})

CORRECTION_PREFIX = "Correction: "
_MAX_NOTE = 300

# What the reviewer says first decides the verdict; the rest of its sentence is
# the note that becomes the correction.
_VERDICT_WORD = re.compile(
    r"^(agree|agreed|disagree|disagreed|wrong|incorrect|false"
    r"|correct|ok|okay|yes|fine|looks right|no problem"
    r"|error|failed|fail|unsure|unclear|maybe|can'?t (?:say|tell)|no verdict|none|skipped)\b"
    r"[:.,;!\-–—]*\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_AGREE_WORDS = frozenset({"agree", "agreed", "correct", "ok", "okay", "yes", "fine", "looks right", "no problem"})
_DISAGREE_WORDS = frozenset({"disagree", "disagreed", "wrong", "incorrect", "false"})
_ERROR_WORDS = frozenset({"error", "failed", "fail"})
_SKIPPED_WORDS = frozenset({"no verdict", "none", "skipped"})


def escalation_worthy(kind: str, escalated: bool) -> bool:
    """A turn the big brain should re-check whatever the little brain did."""
    return bool(escalated) or str(kind) in ESCALATION_WORTHY_KINDS


def _unit(key: str) -> float:
    """A stable position in [0, 1) for a turn key: the sample is reproducible."""
    return zlib.crc32(str(key).encode("utf-8")) / (0xFFFFFFFF + 1)


def should_review(kind: str, escalated: bool, sample_rate: float = DEFAULT_SAMPLE_RATE, key: str = "") -> bool:
    """#116 §6: every escalation-worthy answer, a fixed fraction of the rest.

    ``key`` is a stable id for the turn (its text, or a counter): the same key
    always lands on the same side of the sample, so the fraction is reproducible
    instead of a fresh coin toss per turn. With no key the kind decides, which
    keeps the answer stable per request kind.
    """
    if escalation_worthy(kind, escalated):
        return True
    rate = max(0.0, min(1.0, float(sample_rate)))
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return _unit(key or kind) < rate


def parse_verdict(value: Any) -> tuple[str, str]:
    """Turn whatever the reviewer said into a verdict and a note.

    ``None`` or an empty answer means no verdict at all; anything that does not
    read as agreement or disagreement is treated as unsure, never as a correction.
    """
    if value is None:
        return SKIPPED, ""
    if isinstance(value, (tuple, list)):
        parts = [str(part) for part in value]
        said, note = parts[0], " ".join(parts[1:])
    else:
        said, note = str(value), ""
    said = " ".join(said.split())
    if not said:
        return SKIPPED, " ".join(str(note).split())[:_MAX_NOTE]
    match = _VERDICT_WORD.match(said)
    if match:
        word = match.group(1).lower()
        rest = " ".join(match.group(2).split())
        note = f"{note} {rest}".strip()[:_MAX_NOTE] if note else rest[:_MAX_NOTE]
        said = word
    if said in _DISAGREE_WORDS:
        return DISAGREE, note
    if said in _AGREE_WORDS:
        return AGREE, note
    if said in _ERROR_WORDS:
        return ERROR, note
    if said in _SKIPPED_WORDS:
        return SKIPPED, note
    if not note:
        note = said[:_MAX_NOTE]
    return UNSURE, note


def correction_payload(note: str) -> str:
    """What Voxa says when the reviewer really disagreed."""
    detail = " ".join(str(note).split())[:_MAX_NOTE].strip(" .;")
    if not detail:
        detail = "the reviewer disagreed with that answer."
    return CORRECTION_PREFIX + detail + "."


@dataclass(frozen=True, slots=True)
class ReviewCase:
    """One reviewed answer: the verdict, the reviewer's note, any correction."""

    kind: str
    escalated: bool
    verdict: str
    note: str = ""
    correction: str = ""


class Reviewer:
    """Runs the injected reviewer off the reply path and reports disagreements.

    ``request`` never blocks: it decides from the sample, starts a worker thread
    and returns. Finished cases are picked up later with ``collect`` (or ``join``
    in the tests and the improvement loop). Only a real disagreement carries a
    correction and reaches ``case_logger``.
    """

    def __init__(
        self,
        review_fn: Callable[[str, str], Any] | None,
        *,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        case_logger: Callable[[ReviewCase], None] | None = None,
        max_workers: int = 2,
    ) -> None:
        self.review_fn = review_fn
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self.case_logger = case_logger
        self.max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        self._running = 0
        self._threads: list[threading.Thread] = []
        self._done: list[ReviewCase] = []

    def request(
        self,
        *,
        question: str,
        answer: str,
        kind: str = "",
        escalated: bool = False,
        key: str = "",
    ) -> bool:
        """Schedule a review of an answer that was just spoken. Returns at once."""
        if self.review_fn is None:
            return False
        if not should_review(kind, escalated, self.sample_rate, key):
            return False
        with self._lock:
            if self._running >= self.max_workers:
                return False  # §7: the rate is capped by the load while the user talks
            self._running += 1
        thread = threading.Thread(
            target=self._run,
            kwargs={"question": question, "answer": answer, "kind": kind, "escalated": escalated},
            name="voxa-background-review",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            with self._lock:
                self._running -= 1
            log.warning("could not start a background review")
            return False
        with self._lock:
            self._threads = [t for t in self._threads if t is not thread and t.is_alive()]
            self._threads.append(thread)
        return True

    def _run(self, *, question: str, answer: str, kind: str, escalated: bool) -> None:
        case = self.judge(question, answer, kind, escalated)
        with self._lock:
            self._running -= 1
            self._done.append(case)
        if case.correction and self.case_logger is not None:
            try:
                self.case_logger(case)
            except Exception as exc:
                log.warning("could not log the review case: %s", exc)

    def judge(self, question: str, answer: str, kind: str = "", escalated: bool = False) -> ReviewCase:
        """Ask the injected reviewer and build the case. Blocks: call it off the reply path."""
        try:
            value = self.review_fn(question, answer) if self.review_fn is not None else None
        except Exception as exc:
            return ReviewCase(kind=kind, escalated=escalated, verdict=ERROR, note=str(exc))
        verdict, note = parse_verdict(value)
        correction = correction_payload(note) if verdict == DISAGREE else ""
        return ReviewCase(kind=kind, escalated=escalated, verdict=verdict, note=note, correction=correction)

    def pending(self) -> int:
        with self._lock:
            return self._running

    def collect(self) -> list[ReviewCase]:
        """Take the cases that have finished; never waits."""
        with self._lock:
            cases, self._done = self._done, []
        return cases

    def join(self, timeout: float | None = 5.0) -> list[ReviewCase]:
        """Wait for the reviews already scheduled, then hand back every case.

        The reply path never calls this; the tests and the VoxaTest loop do.
        """
        with self._lock:
            threads, self._threads = self._threads, []
        for thread in threads:
            thread.join(timeout)
        return self.collect()
