"""V3: the voxatest hearing suite - timed interruptions, short answers, repetitions, pauses, quiet speech.

Handoff 8.2 (the test matrix) and 8.3 (the metrics) name the cells a hearing test must cover;
V2 (``voxatest/offline.py``) already measures comprehension with nobody talking. This module adds
the part V2 cannot measure: what happens **while Voxa is speaking**.

Every hearing case names what Voxa is reading out (``reply``), what the test user says (``prompt``),
where the user starts relative to that reply (``barge_at``), how loud at the microphone
(``level_db``, a ratio, not a volume slider), a deliberate mid-sentence pause (``pause_at`` /
``pause_s``) and what the case expects (``expect``: ``yield`` | ``keep`` | ``answer``).

LIVE (rhY, speakers and a microphone): the harness makes Voxa read the reply, watches her speech
start on the microphone, and speaks the user's words at the scheduled moment
(:class:`LiveConductor`, :func:`speak_timed`).

OFFLINE (Strata, the gate): the overlap is built with the hearing bench's own renderer
(:mod:`voxa.bench.render`: Voxa's TTS voice, the test user's voice, the room impulse response, the
bulk delay, the nonlinearity) and pushed through the same pipeline the bench runs
(:class:`voxa.bench.runner.ClipRunner`). No sound device is touched.

What is measured per case (:func:`measure_case`), and the run keys they aggregate into
(:func:`score_hearing`, the seven keys the task fixes):

``yielded``              did her own playback stop           -> ``interrupt_recall``
``stop_latency_ms``      user onset -> her audio stops       -> ``stop_latency_p50_ms`` / ``_p95_ms``
``request_complete``     the user's words reached her whole  -> ``request_word_retention``
``premature``            she answered inside a deliberate pause -> ``premature_rate``
``self_interrupt``       she stopped with no user speech     -> ``self_interrupt_count``
                                                              -> ``keep_accuracy`` (she kept going)

The difficult cells (quiet, far-field) are reported separately, never averaged away: handoff 8.4
says to report them apart rather than hide them in an average, so every key comes with a
``by_cell`` and a ``by_overlap`` breakdown next to the headline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voxa.bench import render as bench_render
from voxa.bench.metrics import wer, word_list
from voxa.bench.runner import ClipRunner

from .cases import TestCase
from .comprehension import hearing_accuracy
from .offline import (
    LEAD_SILENCE_SECONDS,
    TAIL_SILENCE_SECONDS,
    WAKE_GAP_SECONDS,
    render_prompt,
    wake_clip_text,
)

SR = bench_render.SR

# The three turn decisions a hearing case can expect.
EXPECTS = ("yield", "keep", "answer")
# A case expects the user's words to become a request when it says "answer", and a barge-in
# ("yield") is a request too: the whole point of Milestone A is that those words survive.
REQUEST_EXPECTS = ("yield", "answer")

# Named overlap positions (handoff 8.2 "Overlap position"). Fractions of the reply's length;
# ``after_cancel`` and ``after_reply`` are measured from the END of her playback.
POSITION_FRACTIONS = {
    "onset": 0.0,
    "first_syllable": 0.06,
    "middle": 0.5,
    "last_word": 0.85,
    "after_cancel": 1.0,
    "after_reply": 1.0,
}
# Where the user starts for the two positions measured from the end of the reply, in seconds.
AFTER_CANCEL_SECONDS = 0.15   # "immediately after cancellation": inside the echo tail
AFTER_REPLY_GAP_SECONDS = 0.45  # a normal answer to a question she has just finished asking
# A case with no reply audio and no barge-in starts this far into the clip (the bench's own jitter).
USER_ONLY_ONSET_SECONDS = 0.30
# The declared normal-use range (handoff 8.4: report the difficult cells separately).
NORMAL_CELLS = ("normal",)
DEFAULT_CELL = "normal"
DEFAULT_LEVEL_DB = 5.0
# The echo path of a hearing clip: a desk-distance room and a laptop-speaker delay, seeded.
RT60_RANGE = (0.25, 0.55)
DELAY_RANGE_MS = (60.0, 180.0)
NONLINEARITY = {"kind": "tanh", "param": 1.6}
# A reply's tail dips between words; only a start this far after her declared end is a NEW reply.
NEW_REPLY_MARGIN_SECONDS = 0.35
# The live conductor: how a level proves she has started speaking, and how long it will wait.
LIVE_SPEECH_LEVEL = 0.05
LIVE_SPEECH_MS = 200.0
LIVE_START_TIMEOUT_SECONDS = 20.0
# How long the microphone has to be quiet before the live harness calls her playback stopped.
LIVE_QUIET_MS = 300.0
LIVE_FLOOR_LEVEL = 0.02
# The reply length assumed when the live harness schedules a fraction before it knows her length.
NOMINAL_REPLY_SECONDS = 3.0


# --------------------------------------------------------------------------- schema


def is_hearing_case(case: TestCase) -> bool:
    """A case belongs to the hearing suite when it names a turn decision."""
    return bool(str(getattr(case, "expect", "") or "").strip())


def validate_hearing(case: TestCase) -> str:
    """Why a hearing case is not a valid hearing case, or "" when it is.

    The rules the suite depends on: a decision, a sane level, a pause that is a fraction and a
    duration, and a barge-in position that is a fraction, a number of seconds, or a named cell.
    """
    expect = str(case.expect or "").strip()
    if expect and expect not in EXPECTS:
        return f"expect must be one of {'/'.join(EXPECTS)}, got {expect!r}"
    if not expect:
        return ""
    if expect in ("yield", "answer") and not str(case.prompt or "").strip():
        return f"a {expect} case must say what the user says"
    level = case.level_db
    if level is not None:
        try:
            value = float(level)
        except (TypeError, ValueError):
            return f"level_db must be a number, got {level!r}"
        if value < -40.0 or value > 40.0:
            return f"level_db out of the -40..40 dB range: {value}"
    if case.pause_s is not None:
        try:
            seconds = float(case.pause_s)
        except (TypeError, ValueError):
            return f"pause_s must be a number, got {case.pause_s!r}"
        if seconds < 0.2 or seconds > 3.0:
            return f"pause_s out of the 0.2..3.0 s range: {seconds}"
    if case.pause_at is not None:
        try:
            at = float(case.pause_at)
        except (TypeError, ValueError):
            return f"pause_at must be a number, got {case.pause_at!r}"
        if at <= 0.0 or at >= 1.0:
            return f"pause_at must be a fraction of the utterance in 0..1, got {at}"
    barge_at = case.barge_at
    if isinstance(barge_at, str) and barge_at.strip() and barge_at.strip() not in POSITION_FRACTIONS:
        return f"barge_at must be a number or one of {'/'.join(sorted(POSITION_FRACTIONS))}, got {barge_at!r}"
    if isinstance(barge_at, (int, float)) and not isinstance(barge_at, bool) and float(barge_at) < 0.0:
        return f"barge_at cannot be negative: {barge_at}"
    return ""


# --------------------------------------------------------------------------- scheduling


@dataclass(frozen=True, slots=True)
class Schedule:
    """Where everything happens in one clip, in seconds from the clip's first sample."""

    onset_s: float
    reply_end_s: float
    user_end_s: float = 0.0
    pause_start_s: float | None = None
    pause_end_s: float | None = None
    position: str = ""
    wake: bool = False
    wake_end_s: float = 0.0


def resolve_barge_at(barge_at: Any, reply_seconds: float) -> tuple[float, str]:
    """``barge_at`` -> (seconds from the clip start, the position name).

    A number below 1 is a FRACTION of Voxa's reply, a number of 1 or more is ABSOLUTE SECONDS
    (the task's "seconds or fraction of Voxa's reply"). A name is one of the handoff 8.2 cells.
    """
    if barge_at is None or (isinstance(barge_at, str) and not barge_at.strip()):
        return USER_ONLY_ONSET_SECONDS, "user_only"
    if isinstance(barge_at, str):
        name = barge_at.strip()
        if name == "after_cancel":
            return reply_seconds + AFTER_CANCEL_SECONDS, name
        if name == "after_reply":
            return reply_seconds + AFTER_REPLY_GAP_SECONDS, name
        return POSITION_FRACTIONS[name] * reply_seconds, name
    value = float(barge_at)
    if 0.0 < value < 1.0:
        return value * reply_seconds, f"fraction:{value}"
    return value, f"seconds:{value}"


def schedule_case(case: TestCase, reply_seconds: float, user_seconds: float = 0.0,
                  wake_seconds: float = 0.0) -> Schedule:
    """The clip timeline of one hearing case (pure arithmetic: no audio, no clock, no randomness).

    ``reply_seconds`` is how long Voxa's reply is, ``user_seconds`` how long the test user's
    utterance is WITHOUT the deliberate pause, ``wake_seconds`` how long the "Voxa." breath is.
    A case with no reply audio is spoken in two breaths, exactly as the live harness speaks it:
    "Voxa." first, then the request, so the timeline starts with the wake word.
    """
    onset, position = resolve_barge_at(case.barge_at, reply_seconds)
    wake = not str(case.reply or "").strip() and case.expect != "yield"
    shift = (wake_seconds + WAKE_GAP_SECONDS) if wake else 0.0
    onset_s = shift + onset
    pause_start = pause_end = None
    if case.pause_s is not None and case.pause_at is not None and user_seconds > 0.0:
        pause_start = onset_s + float(case.pause_at) * user_seconds
        pause_end = pause_start + float(case.pause_s)
    user_end_s = onset_s + user_seconds + (float(case.pause_s) if pause_end is not None else 0.0)
    return Schedule(onset_s=round(onset_s, 4), reply_end_s=round(reply_seconds, 4),
                    user_end_s=round(user_end_s, 4), pause_start_s=pause_start,
                    pause_end_s=pause_end, position=position, wake=wake,
                    wake_end_s=round(wake_seconds, 4) if wake else 0.0)


# --------------------------------------------------------------------------- the clip


def place_user(source: bench_render.Source, onset_s: float, pause_at: float | None,
               pause_s: float | None) -> tuple[np.ndarray, list[dict], float]:
    """The test user's audio in clip time, with a deliberate pause inserted at ``pause_at``.

    The pause falls at a FRACTION of the utterance (the case says where), never at a random word:
    a pause cell must be reproducible, and "complete command, then an added clause" means the
    pause sits at the clause boundary the case names.
    """
    words = list(source.words)
    head, tail = source.samples, np.zeros(0, np.float32)
    split = len(words)
    if pause_s is not None and pause_at is not None and len(words) >= 2:
        split = max(1, min(len(words) - 1, int(round(float(pause_at) * len(words)))))
        cut = int(round(words[split]["onset_s"] * SR))
        head, tail = source.samples[:cut], source.samples[cut:]
    parts = [head]
    if tail.size:
        parts.append(np.zeros(int(round(float(pause_s) * SR)), np.float32))
        parts.append(tail)
    body = np.concatenate(parts)
    start = int(round(onset_s * SR))
    out = np.zeros(start + len(body), np.float32)
    out[start:start + len(body)] = body
    shifted = []
    for index, word in enumerate(words):
        shift = float(pause_s) if (tail.size and index >= split) else 0.0
        shifted.append({"word": word["word"], "onset_s": onset_s + word["onset_s"] + shift,
                        "offset_s": onset_s + word["offset_s"] + shift})
    return out, shifted, (start + len(body)) / SR


@dataclass(frozen=True, slots=True)
class HearingClip:
    """One hearing clip: the microphone mix, Voxa's exact render, and the bench truth record."""

    mic: np.ndarray
    ref: np.ndarray
    truth: dict
    meta: dict


def build_case_clip(case: TestCase, data_root: Path, seed: int, index: int = 0,
                    wake_word: str = "voxa", voice: str = "") -> HearingClip:
    """Build the overlap with the bench renderer: Voxa's reply, the user's words, the room.

    The echo path is the bench's own model (a seeded room, a seeded bulk delay, the tanh
    nonlinearity) and the levels are ratios at the microphone: the echo sits at the bench's
    nominal level and the user at ``level_db`` against it, so a "quiet user" case is quiet AT THE
    MICROPHONE, which is what the handoff matrix asks for.
    """
    rng = np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=(int(index), 7)))
    rt60_s = float(rng.uniform(*RT60_RANGE))
    delay_ms = float(rng.uniform(*DELAY_RANGE_MS))

    reply_text = str(case.reply or "").strip()
    user_text = str(case.prompt or "").strip()
    user_voice = str(voice or case.voice or bench_render.USER_TTS_VOICE)

    tts = (bench_render.voice_source(reply_text, bench_render.VOXA_TTS_VOICE,
                                     bench_render.VOXA_TTS_RATE, Path(data_root), "tts")
           if reply_text else None)
    user_source = (bench_render.voice_source(user_text, user_voice, bench_render.VOXA_TTS_RATE,
                                             Path(data_root), "user_phrases")
                   if user_text else None)
    reply_seconds = len(tts.samples) / SR if tts is not None else 0.0

    wake_seconds = 0.0
    wake_samples = np.zeros(0, np.float32)
    if not reply_text and case.expect != "yield":
        wake = render_prompt(wake_clip_text(wake_word), user_voice,
                             bench_render.VOXA_TTS_RATE, Path(data_root))
        wake_samples = wake["samples"]
        wake_seconds = len(wake_samples) / SR

    user_seconds = len(user_source.samples) / SR if user_source is not None else 0.0
    plan = schedule_case(case, reply_seconds, user_seconds, wake_seconds)

    user_arr: np.ndarray = np.zeros(0, np.float32)
    user_words: list[dict] = []
    user_end = 0.0
    if user_source is not None:
        user_arr, user_words, user_end = place_user(user_source, plan.onset_s,
                                                    case.pause_at, case.pause_s)

    length = int(round((max(reply_seconds, user_end, plan.wake_end_s) + rt60_s
                        + TAIL_SILENCE_SECONDS + LEAD_SILENCE_SECONDS) * SR))
    ref = np.zeros(length, np.float32)
    if tts is not None:
        body = tts.samples[:length]
        ref[: len(body)] = body
    played = bench_render.apply_nonlinearity(ref, NONLINEARITY)
    echo = bench_render.fft_convolve(played, bench_render.make_rir(rng, rt60_s))
    d = bench_render.resample_clock(echo, length, int(round(delay_ms / 1000.0 * SR)), None, 0, 0.0)

    s = np.zeros(length, np.float32)
    if wake_samples.size:
        s[: min(len(wake_samples), length)] = wake_samples[:length]
    limit = min(len(user_arr), length)
    s[:limit] = np.maximum(s[:limit], user_arr[:limit])

    d = d * (bench_render.ECHO_RMS / bench_render.rms(d)) if bench_render.rms(d) > 1e-9 else d
    level_db = float(case.level_db if case.level_db is not None else DEFAULT_LEVEL_DB)
    if bench_render.rms(s) > 1e-9:
        s = s * ((bench_render.ECHO_RMS * 10.0 ** (level_db / 20.0)) / bench_render.rms(s))
    mix = s + d
    peak = max(float(np.max(np.abs(x))) for x in (s, d, mix))
    gain = min(1.0, bench_render.MIX_PEAK / peak) if peak > 0 else 1.0

    truth = {
        "id": case.id,
        "scenario": "voxatest-hearing",
        "expect": case.expect,
        "overlap_position": plan.position,
        "pair": None,
        "sr": SR,
        "frames": length,
        "seconds": length / SR,
        "playback": None if tts is None else {
            "text": tts.text, "voice": tts.voice, "engine": tts.engine,
            "onset_s": 0.0, "offset_s": reply_seconds,
            "echo_tail_end_s": reply_seconds + rt60_s,
            "words": bench_render._shift_words(tts, 0.0),
        },
        "user": None if user_source is None else {
            "text": user_source.text, "source": user_source.key, "voice": user_source.voice,
            "engine": user_source.engine, "onset_s": plan.onset_s, "offset_s": user_end,
            "pause_s": case.pause_s, "pause_start_s": plan.pause_start_s,
            "pause_end_s": plan.pause_end_s, "words": user_words,
        },
        "echo": {"rt60_s": rt60_s, "delay_ms": delay_ms, "delay_samples": int(round(delay_ms / 1000.0 * SR)),
                 "delay_step_ms": None, "drift_ppm": 0.0, "nonlinearity": NONLINEARITY},
        "levels": {"ser_db_target": level_db, "snr_db_target": None, "noise": False,
                   "echo_rms": bench_render.ECHO_RMS,
                   "mix_gain_db": 20.0 * float(np.log10(gain)) if gain != 1.0 else 0.0},
    }
    meta = {"rt60_s": round(rt60_s, 4), "delay_ms": round(delay_ms, 2), "level_db": level_db,
            "reply_engine": (tts.engine if tts is not None else ""),
            "user_engine": (user_source.engine if user_source is not None else ""),
            "user_voice": user_voice, "position": plan.position,
            "onset_s": plan.onset_s, "reply_seconds": round(reply_seconds, 3),
            "user_seconds": round(user_seconds, 3), "wake": plan.wake,
            "pause_start_s": plan.pause_start_s, "pause_end_s": plan.pause_end_s,
            "clip_seconds": round(length / SR, 3), "peak_guard": round(gain, 4)}
    return HearingClip(mic=np.asarray(mix * gain, dtype=np.float32),
                       ref=np.asarray(ref * gain, dtype=np.float32), truth=truth, meta=meta)


# --------------------------------------------------------------------------- measuring


def expected_request(case: TestCase, wake_word: str) -> str:
    """The words the case expects to reach the assistant, with the wake word taken out."""
    words = word_list(str(case.prompt or ""))
    wake = word_list(wake_word)
    kept = [word for word in words if word not in wake]
    return " ".join(kept)


def word_retention(expected: str, delivered: str) -> float | None:
    """The share of the expected request's words that reached her, 1.0 = the request arrived whole.

    The bench's own normalisation and word alignment (``voxa/bench/metrics``), the same one V1
    scores comprehension with: lower-case, punctuation dropped, numbers spelled out. Nothing is
    deleted from the expected side, so a short answer is measured as a short answer.
    """
    wanted = word_list(expected)
    if not wanted:
        return None
    rate = wer(wanted, word_list(delivered))
    if rate is None:
        return None
    return max(0.0, 1.0 - rate)


def _accepted_own_stream_duck(event: dict) -> bool:
    """True when a ``tts-stop`` is the pipeline's OWN accepted duck of Voxa's stream.

    The shipped chain (f173 joint double-talk suspect / f174 turn-state / f176-f181 promote
    path) pauses her own stream 12 dB under an overlap suspect - a reversible duck, never a
    hard self-interruption (handoff 6.5: "a reversible pause once independent user speech is
    strongly supported"). The runner carries that verdict onto the stop event: the joint gate's
    ``joint_suspect`` (f173) or the own-stream-duck / turn-state marker (``self_duck``, f174).
    A stop that carries either was her own stream being ducked, not a self-interruption.
    """
    if event.get("self_duck"):
        return True
    joint = event.get("joint_suspect")
    return joint is True


def measure_case(case: TestCase, truth: dict, events: list[dict], requests: list[str],
                 wake_word: str) -> dict[str, Any]:
    """One hearing case's five measures, read off the pipeline's own events.

    ``yielded``            a ``tts-stop`` event: her playback stopped.
    ``stop_latency_ms``    the user's onset (the truth file) to that stop.
    ``request_complete``   the delivered request holds every expected word.
    ``premature``          a request delivered, or a NEW reply started, inside the deliberate pause.
    ``self_interrupt``     a stop with no user speech at or before it (her own voice stopped her).
    """
    user = truth.get("user") or {}
    onset_s = user.get("onset_s")
    playback = truth.get("playback") or {}
    reply_end_s = float(playback.get("offset_s") or 0.0)
    stops = [event for event in events if event.get("event") == "tts-stop"]
    starts = [event for event in events if event.get("event") == "tts-start"]
    request_events = [event for event in events if event.get("event") == "request"]

    self_interrupts = 0
    for event in stops:
        stopped_at = float(event.get("t_s") or 0.0)
        if onset_s is None:
            # Nobody talked at all: any stop of her own reply is her own voice stopping her,
            # which is exactly what a self-interruption is (handoff 5.2 / the live measure).
            self_interrupts += 1
        elif stopped_at < float(onset_s):
            # She stopped her OWN stream before the user's words. When that stop is the
            # ACCEPTED duck of her own stream under an open overlap suspect - the f173 joint
            # verdict / f174 turn-state recorded on the event (joint_suspect / self_duck) -
            # it is a reversible pause on the shipped chain (self_pause is an accepted floor),
            # NOT a self-interruption: a user turn is arriving and she has only paused. Only a
            # bare stop the pipeline does not mark as an own-stream duck is her voice stopping
            # her with no user speech at all.
            if not _accepted_own_stream_duck(event):
                self_interrupts += 1
    yielded = any(float(event.get("t_s") or 0.0) >= float(onset_s or 0.0)
                  for event in stops) if onset_s is not None else False

    latencies = [round((float(event.get("t_s") or 0.0) - float(onset_s)) * 1000.0, 1)
                 for event in stops if onset_s is not None
                 and float(event.get("t_s") or 0.0) >= float(onset_s)]

    request_text = " ".join(str(text) for text in requests if str(text).strip()).strip()
    retention = word_retention(expected_request(case, wake_word), request_text)

    premature = False
    pause_start, pause_end = user.get("pause_start_s"), user.get("pause_end_s")
    if pause_start is not None and pause_end is not None:
        for event in request_events:
            at = float(event.get("t_s") or 0.0)
            if float(pause_start) <= at <= float(pause_end):
                premature = True
        for event in starts:
            at = float(event.get("t_s") or 0.0)
            if float(pause_start) <= at <= float(pause_end) and at > reply_end_s + NEW_REPLY_MARGIN_SECONDS:
                premature = True

    kept = True
    if case.expect == "keep":
        # With a reply playing, "keep" is about the floor: she must go on speaking. With no reply
        # (a deliberate pause in the user's own turn), "keep" is about the turn: she must not take
        # it before the user has finished.
        kept = not yielded and self_interrupts == 0 and not (premature and not playback)
    return {
        "expect": case.expect,
        "cell": str(case.cell or DEFAULT_CELL),
        "position": truth.get("overlap_position"),
        "overlap": bool(playback and onset_s is not None and float(onset_s) < reply_end_s),
        "pause_s": user.get("pause_s"),
        "user_onset_s": None if onset_s is None else round(float(onset_s), 3),
        "reply_end_s": round(reply_end_s, 3),
        "yielded": bool(yielded),
        "stop_latency_ms": min(latencies) if latencies else None,
        "request_text": request_text,
        "expected_request": expected_request(case, wake_word),
        "request_complete": bool(retention == 1.0),
        "word_retention": None if retention is None else round(retention, 4),
        "premature": bool(premature),
        "self_interrupt_count": int(self_interrupts),
        "kept_turn": bool(kept),
        "stop_events": [round(float(event.get("t_s") or 0.0), 3) for event in stops],
    }


# --------------------------------------------------------------------------- the run keys


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def percentile(values: list[float], percent: float) -> float | None:
    """The p-th percentile of a list of numbers (numpy's linear interpolation), rounded to 0.1 ms."""
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), float(percent))), 1)


def score_hearing(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """The seven run keys the task fixes, with the counts and the per-cell breakdown beside them.

    Nothing is averaged away: every key is reported for the whole suite AND split by cell and by
    overlap, so a quiet-cell failure is visible as itself, exactly as handoff 8.4 asks.
    """
    scored = [case for case in cases if case.get("expect") in EXPECTS]
    if not scored:
        return {}

    def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        wants_yield = [row for row in rows if row.get("expect") == "yield"]
        wants_keep = [row for row in rows if row.get("expect") == "keep"]
        wants_request = [row for row in rows if row.get("expect") in REQUEST_EXPECTS]
        pauses = [row for row in rows if row.get("pause_s") is not None]
        latents = [float(row["stop_latency_ms"]) for row in rows
                   if row.get("yielded") and row.get("stop_latency_ms") is not None]
        retentions = [float(row["word_retention"]) for row in wants_request
                      if row.get("word_retention") is not None]
        return {
            "cases": len(rows),
            "yield_cases": len(wants_yield),
            "yielded_cases": sum(1 for row in wants_yield if row.get("yielded")),
            "keep_cases": len(wants_keep),
            "kept_cases": sum(1 for row in wants_keep if row.get("kept_turn")),
            "request_cases": len(wants_request),
            "complete_requests": sum(1 for row in wants_request if row.get("request_complete")),
            "pause_cases": len(pauses),
            "premature_cases": sum(1 for row in pauses if row.get("premature")),
            "self_interrupt_count": sum(int(row.get("self_interrupt_count") or 0) for row in rows),
            "interrupt_recall": (None if not wants_yield
                                 else _mean([1.0 if row.get("yielded") else 0.0 for row in wants_yield])),
            "keep_accuracy": (None if not wants_keep
                              else _mean([1.0 if row.get("kept_turn") else 0.0 for row in wants_keep])),
            "stop_latency_p50_ms": percentile(latents, 50),
            "stop_latency_p95_ms": percentile(latents, 95),
            "premature_rate": (None if not pauses
                               else _mean([1.0 if row.get("premature") else 0.0 for row in pauses])),
            "request_word_retention": _mean(retentions),
        }

    overall = group(scored)
    by_cell: dict[str, Any] = {}
    by_overlap: dict[str, Any] = {}
    by_expect: dict[str, Any] = {}
    for row in scored:
        by_cell.setdefault(str(row.get("cell") or DEFAULT_CELL), []).append(row)
        by_overlap.setdefault("overlap" if row.get("overlap") else "no_overlap", []).append(row)
        by_expect.setdefault(str(row.get("expect")), []).append(row)
    overall["by_cell"] = {name: group(rows) for name, rows in sorted(by_cell.items())}
    overall["by_overlap"] = {name: group(rows) for name, rows in sorted(by_overlap.items())}
    overall["by_expect"] = {name: group(rows) for name, rows in sorted(by_expect.items())}
    overall["cases"] = len(scored)
    return overall


# --------------------------------------------------------------------------- running one case


def run_hearing_case(case: TestCase, settings: Any, asr: Any, registry: Any, seed: int,
                     data_root: Path, *, index: int = 0, echo_mode: str = "voxa",
                     wake_word: str | None = None, voice: str = "",
                     log: Callable[[str], None] = print) -> dict[str, Any]:
    """One hearing case offline: build the overlap, run the real pipeline, measure the five things.

    Returns the per-case hearing block (``measure_case``) plus what the router made of the words
    and what the run cost, ready to drop into the case's report row.
    """
    from .commands import CommandCase
    from .commands import check as route_check

    started = time.perf_counter()
    wake = str(wake_word if wake_word is not None else settings.wake_word)
    clip = build_case_clip(case, Path(data_root), seed, index=index, wake_word=wake, voice=voice)
    runner = ClipRunner(asr=asr, echo_mode=echo_mode, wake_word=wake,
                        settings=_pipeline_settings(settings))
    summary = runner.run(np.asarray(np.rint(np.asarray(clip.mic, dtype=np.float64) * 32767.0)
                                    .astype("<i2")), clip.ref, clip.truth)
    requests = list(summary["requests"])
    measures = measure_case(case, clip.truth, runner.events, requests, wake)
    routed = route_check(CommandCase(case.id, case.category, measures["request_text"],
                                     case.expected_tool or None,
                                     dict(case.expected_args) if case.expected_args else None),
                         registry)
    measures.update({
        "echo_mode": echo_mode,
        "routed_tool": routed.got_tool,
        "routed_args": dict(routed.got_args),
        "hearing_word_accuracy": hearing_accuracy(case.prompt, measures["request_text"], wake),
        "turns": list(summary["turns"]),
        "counts": dict(summary["counts"]),
        "audio_seconds": round(float(summary["seconds"]), 3),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "clip": clip.meta,
    })
    measures["hearing_word_accuracy"] = (None if measures["hearing_word_accuracy"] is None
                                         else round(float(measures["hearing_word_accuracy"]), 4))
    log(f"  onset {measures['user_onset_s']} s yielded {measures['yielded']} "
        f"stop {measures['stop_latency_ms']} ms retention {measures['word_retention']} "
        f"requests {requests} ({measures['wall_seconds']}s wall)")
    return measures


def _pipeline_settings(settings: Any):
    """Voxa's own pipeline settings, taken from the harness (V2's helper, imported, not copied)."""
    from .offline import pipeline_settings

    return pipeline_settings(settings)


# --------------------------------------------------------------------------- the live conductor


@dataclass(frozen=True, slots=True)
class LivePlan:
    """What the live harness does with a hearing case, in seconds from the start of her reply."""

    lead_prompt: str
    speak_at_s: float
    words: str
    tail_words: str = ""
    tail_at_s: float | None = None
    wake_first: bool = False
    position: str = ""


def plan_live(case: TestCase, wake_word: str = "voxa", reply_seconds: float = 0.0) -> LivePlan:
    """The live schedule of a hearing case, before any audio exists.

    The lead prompt makes Voxa read the reply the case names (her own ``say`` intent), so the
    barge-in lands on HER voice, not on a recording of it. ``speak_at_s`` is measured from the
    moment her speech is seen on the microphone; a case with a deliberate pause is spoken in two
    parts with exactly ``pause_s`` of quiet between them. A fraction of the reply is turned into
    seconds with ``reply_seconds`` when the harness knows the reply's length (from her own word
    timings), and with the reply's nominal length when it does not.
    """
    reply = str(case.reply or "").strip()
    lead = f"say {reply}" if reply else ""
    wake_first = not reply and case.expect != "yield"
    nominal = reply_seconds if reply_seconds > 0.0 else NOMINAL_REPLY_SECONDS
    position, speak_at = "", 0.0
    barge_at = case.barge_at
    if barge_at is None or (isinstance(barge_at, str) and not str(barge_at).strip()):
        position = "user_only"
    elif isinstance(barge_at, str):
        position = str(barge_at).strip()
        if position in {"onset", "first_syllable"}:
            speak_at = POSITION_FRACTIONS[position] * nominal
        elif position == "middle":
            speak_at = POSITION_FRACTIONS["middle"] * nominal
        elif position == "last_word":
            speak_at = POSITION_FRACTIONS["last_word"] * nominal
        elif position == "after_cancel":
            speak_at = nominal + AFTER_CANCEL_SECONDS
        else:
            speak_at = nominal + AFTER_REPLY_GAP_SECONDS
    else:
        value = float(barge_at)
        if 0.0 < value < 1.0:
            position, speak_at = f"fraction:{value}", value * nominal
        else:
            position, speak_at = f"seconds:{value}", value
    words, tail = str(case.prompt or "").strip(), ""
    if case.pause_s is not None and case.pause_at is not None:
        parts = words.split(" ")
        cut = max(1, min(len(parts) - 1, int(round(float(case.pause_at) * len(parts)))))
        words, tail = " ".join(parts[:cut]), " ".join(parts[cut:])
    return LivePlan(lead_prompt=lead, speak_at_s=round(speak_at, 3), words=words,
                    tail_words=tail, tail_at_s=(round(speak_at + float(case.pause_s), 3)
                                                if tail else None),
                    wake_first=wake_first, position=position)


def speak_timed(speak: Callable[[str], None], plan: LivePlan, reply_start_s: float,
                clock: Callable[[], float], sleep: Callable[[float], None]) -> list[float]:
    """Speak the case at the scheduled moments, on a clock the caller owns.

    ``reply_start_s`` is when her reply was seen to start; ``clock`` and ``sleep`` are injected,
    which is what makes the scheduling testable with a fake clock and no audio. Returns the times
    the words were spoken.
    """
    spoken: list[float] = []
    first = reply_start_s + plan.speak_at_s
    while clock() < first:
        sleep(min(0.05, first - clock()))
    speak(plan.words)
    spoken.append(clock())
    if plan.tail_words and plan.tail_at_s is not None:
        target = reply_start_s + float(plan.tail_at_s)
        while clock() < target:
            sleep(min(0.05, target - clock()))
        speak(plan.tail_words)
        spoken.append(clock())
    return spoken


@dataclass(slots=True)
class LiveConductor:
    """Watches the microphone levels for the start of Voxa's reply, then speaks on schedule.

    Fed one ``(level, t_s)`` per capture callback. It decides "she has started" from
    ``LIVE_SPEECH_MS`` of audio at or above ``LIVE_SPEECH_LEVEL`` (the harness's own level scale),
    and stamps ``started_s`` on that frame: that is the zero every ``speak_at_s`` is measured from.
    If she never starts within ``timeout_s`` the schedule runs from the timeout anyway and
    ``missed_start`` says so, because a harness that never speaks tells rhY nothing.
    """

    plan: LivePlan
    clock: Callable[[], float] = time.monotonic
    timeout_s: float = LIVE_START_TIMEOUT_SECONDS
    started_s: float | None = None
    missed_start: bool = False
    due: list[float] = field(default_factory=list)
    _loud_ms: float = 0.0
    _last_t: float | None = None
    _base: float | None = None

    def note(self, level: float, t_s: float, dt_ms: float = 20.0) -> bool:
        """One microphone frame: True on the frame where her reply is judged to have started."""
        if self.started_s is not None:
            return False
        if self._base is None:
            self._base = float(t_s)
        if self._last_t is None:
            self._last_t = float(t_s)
        gap_ms = (float(t_s) - self._last_t) * 1000.0
        dt = gap_ms if gap_ms > 0.0 else float(dt_ms)
        self._last_t = float(t_s)
        self._loud_ms = max(0.0, self._loud_ms + (dt if float(level) >= LIVE_SPEECH_LEVEL else -self._loud_ms))
        timed_out = (float(t_s) - self._base) >= self.timeout_s
        if self._loud_ms >= LIVE_SPEECH_MS or timed_out:
            self.started_s = float(t_s)
            self.missed_start = bool(timed_out and self._loud_ms < LIVE_SPEECH_MS)
            self.due = self.speak_times()
            return True
        return False

    def speak_times(self) -> list[float]:
        """The absolute times the harness must speak at, once her reply has started."""
        if self.started_s is None:
            return []
        times = [self.started_s + self.plan.speak_at_s]
        if self.plan.tail_words and self.plan.tail_at_s is not None:
            times.append(self.started_s + float(self.plan.tail_at_s))
        return times


def measure_live_stop_latency(levels: list[tuple[float, float]], onset_s: float,
                              floor_level: float = 0.02, quiet_ms: float = 300.0) -> float | None:
    """Live stop latency from the recording: user onset to the microphone falling quiet.

    ``levels`` is ``(t_s, level)`` over the case's recording. After the user's onset, the first
    stretch of ``quiet_ms`` at or below ``floor_level`` is where her playback ceased. This is an
    acoustic estimate from the recording, not a tap on her output stream: the live harness has no
    tap of its own (that is task 1), and the notes say so.
    """
    quiet_ms = float(quiet_ms)
    run_ms = 0.0
    for t_s, level in levels:
        if float(t_s) < onset_s:
            continue
        if float(level) <= floor_level:
            run_ms += 20.0
            if run_ms >= quiet_ms:
                return round((float(t_s) - onset_s) * 1000.0, 1)
        else:
            run_ms = 0.0
    return None


def measure_live(case: TestCase, *, spoken_at_s: float | None, tail_at_s: float | None,
                 levels: list[tuple[float, float]], transcript: str, wake_word: str,
                 reply_seconds: float = 0.0) -> dict[str, Any]:
    """One LIVE hearing case, measured from the recording (handoff 8.3: onset to playback cessation).

    Times are seconds from the start of Voxa's reply (``t = 0`` is where :class:`LiveConductor`
    judged her speech to begin), which is also where the harness's own words were scheduled. The
    harness has no tap of Voxa's output (that is roadmap task 1), so "her audio stopped" is read
    acoustically: the microphone going quiet for ``LIVE_QUIET_MS`` after the user's onset. The
    numbers this yields are labelled ``live`` in the report, and the notes say what is estimated.
    """
    stop_latency = (measure_live_stop_latency(levels, spoken_at_s)
                    if spoken_at_s is not None else None)
    self_interrupts = 0
    if not str(case.prompt or "").strip() and reply_seconds > 0.0:
        # Nobody talked while she was reading: the microphone going quiet INSIDE her reply is her
        # own voice stopping her, which is exactly what a self-interruption is.
        probe_s = max(0.3, min(0.6, reply_seconds * 0.5))
        ceased = measure_live_stop_latency(levels, probe_s)
        if ceased is not None and probe_s + ceased / 1000.0 < reply_seconds:
            self_interrupts = 1
    premature = False
    if tail_at_s is not None:
        # She answered during the deliberate pause if the microphone carries her voice between the
        # two parts of the user's turn (the harness is silent there by construction).
        speech_ms = 0.0
        for t_s, level in levels:
            if spoken_at_s is not None and t_s < spoken_at_s:
                continue
            if t_s >= tail_at_s:
                break
            speech_ms = speech_ms + 20.0 if float(level) >= LIVE_SPEECH_LEVEL else 0.0
            if speech_ms >= LIVE_SPEECH_MS:
                premature = True
                break
    retention = word_retention(expected_request(case, wake_word), transcript)
    # "Yielded" only means something when she was playing something: with no reply on the speakers
    # (a user-only case), the microphone going quiet after the user speaks is not a yield.
    yielded = bool(stop_latency is not None) if (spoken_at_s is not None and reply_seconds > 0.0) else False
    kept = not yielded and self_interrupts == 0 and not (premature and reply_seconds <= 0.0)
    return {
        "expect": case.expect,
        "cell": str(case.cell or DEFAULT_CELL),
        "position": f"live:{(spoken_at_s if spoken_at_s is not None else 0.0):.3f}",
        "measured": "live",
        "overlap": bool(spoken_at_s is not None and spoken_at_s < reply_seconds),
        "pause_s": case.pause_s,
        "user_onset_s": None if spoken_at_s is None else round(float(spoken_at_s), 3),
        "reply_end_s": round(float(reply_seconds), 3),
        "yielded": yielded,
        "stop_latency_ms": stop_latency,
        "request_text": transcript,
        "expected_request": expected_request(case, wake_word),
        "request_complete": bool(retention == 1.0),
        "word_retention": None if retention is None else round(retention, 4),
        "premature": bool(premature),
        "self_interrupt_count": int(self_interrupts),
        "kept_turn": bool(kept),
        "stop_events": [round(stop_latency, 1)] if stop_latency is not None else [],
    }


# --------------------------------------------------------------------------- reporting


KEYS = ("interrupt_recall", "keep_accuracy", "stop_latency_p50_ms", "stop_latency_p95_ms",
        "premature_rate", "self_interrupt_count", "request_word_retention")


def format_hearing(block: dict[str, Any]) -> list[str]:
    """The Hearing section of the text report: the seven keys, their counts, the cells apart."""
    if not block:
        return []
    lines = ["Hearing suite (timed interruptions, short answers, repetitions, pauses, quiet speech)"]
    for key in KEYS:
        value = block.get(key)
        lines.append(f"  {key:26s} {'null' if value is None else (f'{value:.4f}' if isinstance(value, float) else value)}")
    lines.append(f"  cases {block.get('cases')}  yielded {block.get('yielded_cases')}/{block.get('yield_cases')}"
                 f"  kept {block.get('kept_cases')}/{block.get('keep_cases')}"
                 f"  complete requests {block.get('complete_requests')}/{block.get('request_cases')}"
                 f"  premature {block.get('premature_cases')}/{block.get('pause_cases')}")
    for group_name, title in (("by_cell", "cell"), ("by_overlap", "overlap"), ("by_expect", "expect")):
        lines.append(f"  per {title}:")
        for name, group in (block.get(group_name) or {}).items():
            lines.append(
                f"    {name}: cases {group.get('cases')} recall {_num(group.get('interrupt_recall'))}"
                f" keep {_num(group.get('keep_accuracy'))}"
                f" retention {_num(group.get('request_word_retention'))}"
                f" p50 {_num(group.get('stop_latency_p50_ms'))} ms"
                f" self-interrupts {group.get('self_interrupt_count')}"
                f" premature {_num(group.get('premature_rate'))}"
            )
    return lines


def _num(value: Any) -> str:
    return "null" if value is None else (f"{float(value):.4f}" if not isinstance(value, int) else str(value))


__all__ = ["KEYS", "LiveConductor", "LivePlan", "EXPECTS", "POSITION_FRACTIONS", "REQUEST_EXPECTS",
           "build_case_clip", "expected_request", "format_hearing", "is_hearing_case",
           "measure_case", "measure_live_stop_latency", "percentile", "plan_live", "place_user",
           "resolve_barge_at", "run_hearing_case", "schedule_case", "score_hearing",
           "speak_timed", "validate_hearing", "word_retention"]
