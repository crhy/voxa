"""Background review of the little brain's answers (crhy/voxa#116 §6).

A fake reviewer stands in for the big brain: no model, no GPU, no audio, no
network. What is under test is the sampling rule, the fact that review never
delays the reply, and that only a real disagreement produces a correction.
"""

from __future__ import annotations

import json
import threading
import time

from voxa.agent.brainreview import (
    AGREE,
    DEFAULT_SAMPLE_RATE,
    DISAGREE,
    ERROR,
    SKIPPED,
    UNSURE,
    ReviewCase,
    Reviewer,
    escalation_worthy,
    should_review,
)


def test_every_escalation_worthy_answer_is_reviewed() -> None:
    assert escalation_worthy("model", True) is True
    assert should_review("command", True, 0.0) is True
    assert should_review("command", True, 0.0, key="anything") is True
    # the little brain stayed on the fast path for a question it could get wrong
    assert should_review("model", False, 0.0) is True
    assert should_review("chat", False, 0.0, key="whatever") is True


def test_only_a_fixed_fraction_of_the_rest_is_reviewed() -> None:
    assert should_review("command", False, 0.0) is False
    assert should_review("command", False, 0.0, key="open gmail") is False
    assert should_review("command", False, 1.0, key="open gmail") is True
    assert should_review("command", False, DEFAULT_SAMPLE_RATE, key="open gmail") in {True, False}

    picked = [key for key in (f"turn {index}" for index in range(400)) if should_review("command", False, 0.2, key)]
    assert 60 <= len(picked) <= 140  # a fifth of the rest, not all of it and not none


def test_the_sample_is_deterministic() -> None:
    first = [should_review("command", False, 0.2, key=f"turn {index}") for index in range(50)]
    second = [should_review("command", False, 0.2, key=f"turn {index}") for index in range(50)]
    assert first == second
    assert any(first) and not all(first)


def test_the_reviewer_never_blocks_the_caller() -> None:
    released = threading.Event()
    calls: list[tuple[str, str]] = []

    def slow_review(question: str, answer: str) -> str:
        calls.append((question, answer))
        released.wait(5)
        return "disagree: the year was wrong"

    reviewer = Reviewer(slow_review, sample_rate=1.0)
    started = time.monotonic()
    assert reviewer.request(question="what year is this", answer="2025", kind="model", key="t1") is True
    assert time.monotonic() - started < 0.2
    assert reviewer.pending() == 1
    assert reviewer.collect() == []  # nothing has landed yet

    released.set()
    cases = reviewer.join(timeout=5)
    assert len(cases) == 1
    assert calls == [("what year is this", "2025")]
    assert cases[0].verdict == DISAGREE
    assert cases[0].correction == "Correction: the year was wrong."
    assert reviewer.pending() == 0


def test_a_correction_and_a_logged_case_need_a_real_disagreement() -> None:
    logged: list[ReviewCase] = []
    verdicts = iter(
        [
            "agree: the answer checks out",
            None,
            "unsure, could not tell",
            "",
            "wrong: the station closed in 1974, not 1979",
        ]
    )
    reviewer = Reviewer(
        lambda question, answer: next(verdicts),
        sample_rate=1.0,
        case_logger=logged.append,
    )
    for _ in range(5):
        reviewer.request(question="when did it close", answer="1979", kind="model", key="t")
    cases = reviewer.join(timeout=5)
    assert [case.verdict for case in cases] == [AGREE, SKIPPED, UNSURE, SKIPPED, DISAGREE]
    assert [case.correction for case in cases[:4]] == ["", "", "", ""]
    assert cases[4].correction == "Correction: the station closed in 1974, not 1979."
    assert len(logged) == 1
    assert logged[0].correction.startswith("Correction:")
    assert logged[0].kind == "model"


def test_a_reviewer_that_raises_is_an_error_not_a_correction() -> None:
    def broken(question: str, answer: str) -> str:
        raise RuntimeError("the big brain is busy")

    logged: list[ReviewCase] = []
    reviewer = Reviewer(broken, sample_rate=1.0, case_logger=logged.append)
    assert reviewer.request(question="q", answer="a", kind="model", key="t") is True
    cases = reviewer.join(timeout=5)
    assert len(cases) == 1
    assert cases[0].verdict == ERROR
    assert cases[0].correction == ""
    assert logged == []


def test_a_reviewer_can_be_switched_off_or_stay_unsampled() -> None:
    reviewer = Reviewer(None, sample_rate=1.0)
    assert reviewer.request(question="q", answer="a", kind="model", key="t") is False
    assert reviewer.join(timeout=1) == []

    calls: list[str] = []
    off = Reviewer(lambda question, answer: calls.append("ran") or "agree", sample_rate=0.0)
    assert off.request(question="q", answer="a", kind="command", key="open gmail") is False
    assert off.request(question="q", answer="a", kind="command", key="open gmail") is False
    assert calls == []


def test_reviews_are_capped_while_the_user_is_talking() -> None:
    held = threading.Event()
    reviewer = Reviewer(
        lambda question, answer: held.wait(5) or "agree",
        sample_rate=1.0,
        max_workers=2,
    )
    assert reviewer.request(question="q1", answer="a", kind="model", key="k1") is True
    assert reviewer.request(question="q2", answer="a", kind="model", key="k2") is True
    assert reviewer.pending() == 2
    assert reviewer.request(question="q3", answer="a", kind="model", key="k3") is False
    held.set()
    assert len(reviewer.join(timeout=5)) == 2


def test_a_verdict_fills_the_turn_record_the_issue_asks_for() -> None:
    from voxa.agent.turnlog import TurnRecord, summarize

    reviewer = Reviewer(lambda question, answer: "disagree: the price is 3, not 4", sample_rate=1.0)
    reviewer.request(question="what is the price", answer="4", kind="model", key="t")
    case = reviewer.join(timeout=5)[0]
    record = TurnRecord(
        time="2026-11-02T09:15:00",
        kind=case.kind,
        brain="little",
        first_spoken_ms=640,
        escalated=False,
        escalate_reason="",
        review_verdict=case.verdict,
        review_note=case.note,
        correction=case.correction,
        barge_in=False,
    )
    summary = summarize([json.loads(record.to_json())])
    assert summary["review"]["disagreed"] == 1
    assert summary["escalation"]["missed"] == 1
    assert summary["corrections"] == 1
    assert summary["first_spoken_ms"]["little"] == 640
