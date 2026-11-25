#!/usr/bin/env python3
"""Issue #72, task v08 step 4: replay the rc4 live log through the real decision path.

The section of ``barge.log`` from 18:20:00 of the v07 live run (the 0.8B brain answering
itself) is copied to ``tests/fixtures/selfhearing-rc4-live.log``: every ``speak-start``,
``words heard=`` and ``INTERRUPT`` line. This tool rebuilds the utterance state and the
per-turn timeline from the speak-start lines (turn = one reply: a new turn begins at a
``whole-reply`` start or after a 12 s silence) and replays every heard line through the
REAL ``BargeInController.observe()`` - attribution, the turn-wide text veto, everything.

The evidence is what the FIXED pipeline has at that moment (step 2): the reference is
live across the whole turn (the ``speaking_now`` signal rebuilt from the log's own
speak-start lines), the reconstructed timeline is aligned, and the transcript proves the
microphone held speech. The old log's audio numbers ("reference silent") are NOT fed in -
that is the failure this task fixed.

Prints exactly two lines:

    self-interrupts: N of M echo lines      (required: N = 0)
    user lines heard: K of U               (required: K = U)

Lines labelled ``unclear`` (empty transcripts, room noise) join neither count: they are
held, never interrupted with, by ``_has_interrupt_words``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxa.attribution import SourceEvidence
from voxa.bargectl import BargeInController

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "selfhearing-rc4-live.log"
LABELS = ROOT / "tests" / "fixtures" / "selfhearing-rc4-live.labels.json"

# A reply is silent for more than this between two of its own speak-starts only when the
# NEXT reply began (the live controller saw the gap; so does the replay).
TURN_GAP_SECONDS = 12.0

SPEAK_RE = re.compile(
    r"^(?P<t>\d\d:\d\d:\d\d) speak-start path='(?P<path>[a-z-]+)'(?P<rest>.*)"
    r"text=(['\"])(?P<text>.*)\4\s*$")
WORDS_RE = re.compile(
    r"^(?P<t>\d\d:\d\d:\d\d) words heard=(['\"])(?P<heard>.*?)"
    r"\2 raw=(['\"])(?P<raw>.*?)\4(?:\s|$)")


def _seconds(stamp: str) -> float:
    h, m, s = (int(part) for part in stamp.split(":"))
    return float(h * 3600 + m * 60 + s)


def _events() -> list[tuple]:
    events: list[tuple] = []
    for line in FIXTURE.read_text().splitlines():
        m = SPEAK_RE.match(line)
        if m:
            events.append((_seconds(m["t"]), "speak", m["path"], m["text"]))
            continue
        m = WORDS_RE.match(line)
        if m:
            events.append((_seconds(m["t"]), "words", m["heard"], m["raw"]))
    return events


def replay() -> tuple[int, int, int, int]:
    """Drive the real controller; returns (self_interrupts, echo_lines, user_heard, user_lines)."""
    labels: dict[str, str] = json.loads(LABELS.read_text())["labels"]
    by_time: dict[float, str] = {}
    for stamp, label in labels.items():
        by_time[_seconds(stamp)] = label

    now = {"t": 0.0}
    control = BargeInController(clock=lambda: now["t"])
    turn_open = False
    last_speak = None
    self_interrupts = echo_lines = user_heard = user_lines = 0

    def quiet_until(t: float) -> None:
        """Teach the level gate what her quiet reply sounds like between events."""
        if last_speak is None:
            return
        for _ in range(min(20, int((t - max(last_speak, now["t"])) // 0.1) + 1)):
            control.observe(0.05, speaking=True, muted=True, dt_ms=100.0)
        now["t"] = t

    for t, kind, a, b in _events():
        now["t"] = t
        if kind == "speak":
            path, text = a, b
            new_turn = (not turn_open or path == "whole-reply"
                        or (last_speak is not None and t - last_speak > TURN_GAP_SECONDS))
            if new_turn:
                if turn_open:
                    control.reply_ended()
                control.note_turn(text)
                turn_open = True
            else:
                control.note_piece(text)
            control.reply_started(text)
            last_speak = t
            continue
        heard, raw = a, b
        quiet_until(t)
        reference_live = control.speaking_now()  # the step-2 signal, from the log's own lines

        def evidence() -> SourceEvidence:
            """What the FIXED capture path holds while these words are judged: the reference
            is live across the turn (never the old silent tap), alignment healthy, and the
            microphone towers over her level by the ratio the gate itself measured (the
            log's line only exists because the candidate opened). No VAD number is invented:
            the log has none, and the cleaned-signal VAD is not part of the decision path
            this replay tests."""
            ratio = (control.pending_ratio if control.pending_ratio is not None
                     else control.gate.peak / max(control.gate.baseline, control.gate.min_level))
            return SourceEvidence(reference_active=reference_live, speaking=True,
                                  alignment_healthy=True, mic_level=0.9, level_ratio=ratio)

        interrupt_seen = own_seen = False
        for _ in range(12):  # the loud run the words arrived in (the gate needs its samples)
            outcome = control.observe(0.9, speaking=True, muted=True, dt_ms=100.0,
                                      candidate=lambda: (heard, raw), evidence=evidence())
            if outcome.action == "interrupt":
                interrupt_seen = True
            elif outcome.action == "own-voice":
                own_seen = True
        label = by_time.get(t)
        if label == "echo":
            echo_lines += 1
            if interrupt_seen:
                self_interrupts += 1
        elif label == "user":
            user_lines += 1
            if not own_seen:
                user_heard += 1
    if turn_open:
        control.reply_ended()
    return self_interrupts, echo_lines, user_heard, user_lines


def main() -> int:
    self_interrupts, echo_lines, user_heard, user_lines = replay()
    print(f"self-interrupts: {self_interrupts} of {echo_lines} echo lines")
    print(f"user lines heard: {user_heard} of {user_lines}")
    return 0 if self_interrupts == 0 and user_heard == user_lines else 1


if __name__ == "__main__":
    raise SystemExit(main())
