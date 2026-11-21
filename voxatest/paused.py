"""Paused-mode VoxaTest: while Voxa is PAUSED nothing may reach her but the wake word.

Issue 136. A paused Voxa is "switched on but ignoring everything except the wake word"
(:class:`voxa.controller.AssistantController`), yet several paths made her act anyway:
the fuzzy wake-word matcher waking her on "voxel"/"Roxa", an exit phrase ("never mind",
"done") ending a conversation she had paused, a late tool reply or reminder speaking
over the pause, and an echo of her own last reply (which keeps its protected "Voxa")
waking her up. The decisions live in :mod:`voxa.pausepolicy`; this module measures them
through the SAME hearing pipeline the hearing bench runs (``voxa.bench.runner.ClipRunner``),
so the gate scores the product's real behaviour, not a simulation of it.

How a case is spoken (``--offline-audio``, no speakers, no microphone):

* a paused case with ``"wake": true`` addresses her in one breath ("Voxa, what time is
  it?"): the render is the whole sentence, and she must respond - a wake miss is a
  recall failure, not a false positive;
* every other paused case is background speech: commands, near-misses of the wake word,
  TV/podcast speech, exit phrases, two people talking, music with vocals - she must
  ignore all of it;
* an echo case names her last reply (``reply``): the same words come back through the
  microphone as her own voice, and :class:`voxa.ownspeech.OwnSpeech` is seeded with that
  reply so the product's echo filter sees what the live filter would see.

What the pipeline delivered is then judged by the product's own pause policy
(:func:`voxa.pausepolicy.wake_allowed` / :func:`voxa.pausepolicy.exit_action`): did that
utterance wake her, act, or get ignored? The run-level metrics -
``paused_false_trigger_rate`` and ``paused_wake_recall`` - join the Comprehension block
so ``voxatest check`` reads them like every other comprehension key. The fuzzy-delivery
rate of the UNFIXED pipeline is reported beside it (``paused_fuzzy_false_trigger_rate``)
so the before/after difference is visible in one run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from voxa.ownspeech import OwnSpeech
from voxa.pausepolicy import exit_action as paused_exit_action
from voxa.pausepolicy import wake_allowed

from .cases import TestCase
from .commands import CommandCase
from .commands import check as route_check
from .grader import GradeResult
from .runner import RunResult

# The clock OwnSpeech is seeded with: a constant, so evidence stays "recent" while the
# clip runs faster than real time (the bench's own simulated clock never reaches this
# instance - the harness recomputes the share with the same inputs the live filter had).
_SEED_NOW = [0.0]


def _seed_clock() -> float:
    return _SEED_NOW[0]


def is_paused_case(case: TestCase) -> bool:
    """A case of the paused suite (issue 136): spoken while Voxa is paused."""
    return bool(case.paused)


class PausedClipRunner:
    """ClipRunner with Voxa's last reply remembered (echo cases), and nothing playing.

    The live app remembers a reply in :class:`voxa.ownspeech.OwnSpeech` when she starts
    speaking it; paused, she IS NOT speaking, but the reply she was cut off saying stays
    remembered (voxa/window.py:_pause_now marks it finished, its evidence ages out after
    six seconds). Seeding the same memory here is what lets the echo of her last reply
    be measured as what it is: her own voice, not a fresh sentence addressed to her.
    """

    def __init__(self, *, asr: Any, echo_mode: str, wake_word: str, settings: Any,
                 seed_reply: str = "") -> None:
        from voxa.bench.runner import ClipRunner

        self._clip_runner_cls = ClipRunner
        self.asr = asr
        self.echo_mode = echo_mode
        self.wake_word = wake_word
        self.settings = settings
        self.seed_reply = seed_reply
        self.runner: Any = None

    def run(self, mic: np.ndarray, ref: np.ndarray, truth: dict) -> dict:
        self.runner = self._clip_runner_cls(asr=self.asr, echo_mode=self.echo_mode,
                                            wake_word=self.wake_word, settings=self.settings)
        if self.seed_reply:
            _SEED_NOW[0] = time.monotonic()
            self.runner.own_speech = OwnSpeech(clock=_seed_clock)
            self.runner.own_speech.started(self.seed_reply)
            self.runner.own_speech.finished()
        return self.runner.run(mic, ref, truth)


def own_echo_share(transcript: str, reply: str, wake_word: str) -> float:
    """How much of this transcript was her own last reply (the live filter's feature)."""
    if not reply:
        return 0.0
    _SEED_NOW[0] = time.monotonic()
    own = OwnSpeech(clock=_seed_clock)
    own.started(reply)
    own.finished()
    return float(own.features(transcript, protect=[wake_word]).get("share") or 0.0)


def paused_stimulus(case: TestCase, settings: Any, variant: Any, data_root: Path,
                    wake_word: str) -> dict[str, Any]:
    """A paused wake case is addressed the way the live harness speaks it (offline V2 default):
    "Voxa." in its own breath, then the request. A background-speech case is just the sentence,
    with no wake clip anywhere: what she must ignore, she must ignore."""
    from .offline import build_stimulus, render_prompt

    if case.wake:
        samples, info = build_stimulus(case, settings, variant, Path(data_root), wake_word, "separate")
        return {"samples": samples, "engine": info.get("engine"), "voice": info.get("voice"),
                "rate": info.get("rate"), "spoken": info.get("spoken"),
                "wake_engine": info.get("wake_engine")}
    voice = variant.voice or settings.tts_voice
    rate = max(80, int(round(settings.tts_rate * variant.rate_scale)))
    rendered = render_prompt(case.prompt, voice, rate, Path(data_root))
    return {"samples": rendered["samples"], "engine": rendered["engine"],
            "voice": rendered["voice"], "rate": rendered["rate"],
            "spoken": case.prompt, "wake_engine": rendered["engine"]}


def judge(transcripts: list[str], reply: str, wake_word: str) -> list[dict[str, Any]]:
    """One row per finished utterance: what the pipeline heard, and what the pause policy allows.

    ``acted`` is the FIXED product's decision (voxa/pausepolicy.py, the code window.py
    now runs while paused); ``acted_before`` is what the UNFIXED pipeline did with the
    same utterance: the fuzzy wake-word match (strip_wake_word's 0.65 threshold) or a
    bare exit phrase got through to the assistant.
    """
    from voxa.conversation import detect_exit_phrase, strip_wake_word

    rows: list[dict[str, Any]] = []
    for heard in transcripts:
        if not heard.strip():
            continue
        share = own_echo_share(heard, reply, wake_word)
        allowed, rest, why = wake_allowed(heard, wake_word, own_share=share)
        exit_kind = paused_exit_action(heard, wake_word)
        rows.append({
            "heard": heard,
            "acted": bool(allowed or exit_kind is not None),
            "resumed": bool(allowed),
            "exit": exit_kind or "",
            "why": why,
            "own_share": round(share, 3),
            "acted_before": bool(strip_wake_word(heard, wake_word) is not None
                                 or detect_exit_phrase(heard, wake_word) is not None),
        })
    return rows


def run_paused_offline_case(case: TestCase, settings: Any, asr: Any, registry: Any,
                            variant: Any, seed: int, data_root: Path, *, index: int = 0,
                            variant_index: int = 0, wake_word: str | None = None,
                            log: Callable[[str], None] = lambda _m: None,
                            echo_mode: str = "off") -> RunResult:
    """One paused case: render -> the real hearing pipeline -> the product's pause policy."""
    from .offline import build_clip, pipeline_settings

    started = time.perf_counter()
    wake = str(wake_word if wake_word is not None else settings.wake_word)
    stimulus = paused_stimulus(case, settings, variant, Path(data_root), wake)
    mic, ref = build_clip(stimulus["samples"])
    truth = {"id": case.id, "scenario": "paused", "playback": {"text": ""}, "expect": "keep",
             "overlap_position": None, "pair": None}
    runner = PausedClipRunner(asr=asr, echo_mode=echo_mode, wake_word=wake,
                              settings=pipeline_settings(settings), seed_reply=case.reply)
    summary = runner.run(mic, ref, truth)
    rows = judge(list(summary["transcript_texts"]), case.reply, wake)
    acted = any(row["acted"] for row in rows)
    resumed = any(row["resumed"] for row in rows)
    echo_rejected = any(row["why"] == "echo" for row in rows)
    rest_text = _rest_of(rows, wake)

    routed = route_check(CommandCase(case.id, case.category, " ".join(summary["requests"]).strip(),
                                     case.expected_tool or None,
                                     dict(case.expected_args) if case.expected_args else None), registry)
    heard = " ".join(summary["requests"]).strip()

    if case.wake:
        ok = resumed
        notes = (f"Paused (wake): {heard or 'nothing'} -> "
                 f"{'she resumed and acted' if ok else 'she stayed paused'}")
    else:
        ok = not acted
        notes = (f"Paused (speech): {'ignored' if ok else 'FALSE TRIGGER'} - "
                 f"heard {summary['transcript_texts'][:2]}")
        if echo_rejected:
            notes += " (echo of her own reply rejected)"
    notes += f"; before the fix: {'acted' if any(row['acted_before'] for row in rows) else 'ignored'}."
    grade = GradeResult(100 if ok else 0, ok, notes, "paused")

    wall = time.perf_counter() - started
    paused_block = {
        "wake": bool(case.wake),
        "echo_case": bool(case.reply),
        "acted": acted,
        "resumed": resumed,
        "echo_rejected": echo_rejected,
        "rest": rest_text,
        "rows": rows,
        "acted_before": any(row["acted_before"] for row in rows),
    }
    action_record = {"heard": heard, "route": "paused-audio", "tool": routed.got_tool or "",
                     "args": dict(routed.got_args), "ok": ok, "variant": variant.name}
    offline = {
        "mode": "offline-audio",
        "variant": variant.name,
        "engine": stimulus["engine"],
        "voice": stimulus["voice"],
        "rate": stimulus["rate"],
        "wake_mode": "paused",
        "spoken": stimulus["spoken"],
        "rate_scale": variant.rate_scale,
        "level_db": 0.0,
        "noise": "none",
        "noise_kind": "none",
        "snr_db": None,
        "peak": None,
        "peak_guard": None,
        "audio_seconds": round(float(summary["seconds"]), 3),
        "wall_seconds": round(wall, 3),
        "asr_calls": int(summary["asr_calls"]),
        "asr_processing_seconds": float(summary["asr_processing_seconds"]),
        "requests": list(summary["requests"]),
        "turns": list(summary["turns"]),
        "counts": dict(summary["counts"]),
        "routed_tool": routed.got_tool,
        "routed_args": dict(routed.got_args),
        "routing_problem": "",
        "hearing_word_accuracy": None,
        "echo_mode": echo_mode,
        "paused": paused_block,
    }
    log(f"  {variant.name}: heard {summary['transcript_texts'][:2]} -> "
        f"{'acted' if acted else 'ignored'} ({wall:.2f}s wall)")
    return RunResult(
        case.id, case.category, case.prompt, "", grade, "", wall,
        action_record, case.expected_tool, heard, case.capability,
        [routed.got_tool] if routed.got_tool else [], list(summary["transcript_texts"]),
        offline,
    )


def _rest_of(rows: list[dict[str, Any]], wake_word: str) -> str:
    """The request that would follow the wake word (what the router would run)."""
    from voxa.pausewords import resume_request

    for row in rows:
        if not row["resumed"]:
            continue
        _ok, rest = resume_request(str(row["heard"]), wake_word)
        return rest
    return ""


def score_paused(measures: list[dict[str, Any]]) -> dict[str, Any]:
    """The paused block of the Comprehension report: the two keys the gate reads, plus the detail.

    ``paused_false_trigger_rate`` - of the background-speech cases, how many reached her
    while paused (0.0 is the goal); ``paused_wake_recall`` - of the cases that addressed
    her by name, how many woke her back up (1.0 is the goal). ``paused_fuzzy_false_trigger_rate``
    is the same false-trigger rate under the UNFIXED pipeline: the number the task fixes.
    """
    if not measures:
        return {}
    wake = [row for row in measures if row.get("wake")]
    nonwake = [row for row in measures if not row.get("wake")]
    acted = sum(1 for row in nonwake if row.get("acted"))
    acted_before = sum(1 for row in nonwake if row.get("acted_before"))
    resumed = sum(1 for row in wake if row.get("resumed"))
    block: dict[str, Any] = {
        "paused_cases": len(measures),
        "paused_wake_cases": len(wake),
        "paused_nonwake_cases": len(nonwake),
        "paused_false_triggers": acted,
        "paused_false_trigger_rate": round(acted / len(nonwake), 4) if nonwake else None,
        "paused_wake_recall": round(resumed / len(wake), 4) if wake else None,
        "paused_fuzzy_false_triggers": acted_before,
        "paused_fuzzy_false_trigger_rate": (round(acted_before / len(nonwake), 4)
                                            if nonwake else None),
        "paused_echo_cases": sum(1 for row in measures if row.get("echo_case")),
        "paused_echo_rejected": sum(1 for row in measures if row.get("echo_rejected")),
        "paused_details": measures,
    }
    return block


def format_paused(comprehension: dict[str, Any]) -> list[str]:
    """The paused section of the text report: the false-trigger rate and the wake recall."""
    if not comprehension.get("paused_cases"):
        return []
    lines = [
        "  paused (issue 136): only the wake word may reach her",
        f"    false_trigger_rate {_fmt(comprehension.get('paused_false_trigger_rate'))}"
        f"  ({comprehension.get('paused_false_triggers', 0)} of "
        f"{comprehension.get('paused_nonwake_cases', 0)} background-speech cases)",
        f"    wake_recall        {_fmt(comprehension.get('paused_wake_recall'))}"
        f"  ({comprehension.get('paused_wake_cases', 0)} cases addressed to her)",
        f"    before the fix:    {_fmt(comprehension.get('paused_fuzzy_false_trigger_rate'))}"
        f"  ({comprehension.get('paused_fuzzy_false_triggers', 0)} false triggers)",
        f"    echo of her reply: {comprehension.get('paused_echo_rejected', 0)} of "
        f"{comprehension.get('paused_echo_cases', 0)} echo cases rejected",
    ]
    return lines


def _fmt(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"
