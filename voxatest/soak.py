"""The soak: Voxa speaks for hours with nobody talking, and every self-trigger is counted.

Roadmap V4 (issue #72, handoff 8.4): "Long-duration playback-only soak testing on the real desktop
route. Zero events in 100 hours implies an approximate one-sided 95% Poisson upper bound of 0.03
events/hour, not proof of zero risk forever."

What a soak does, live (rhY runs it; Strata builds and unit-tests it offline):

* VoxaTest speaks a long, varied stream of replies through the speakers - including her wake word,
  "stop", "cancel" and offline phrases, the words that most often become a self-command - and nobody
  says anything;
* every command Voxa creates out of her own voice is counted from her action log, and every pause,
  duck, yield or stop of her own playback from her barge-in events;
* the run reports events per hour with the one-sided 95% Poisson upper bound, so a clean run states
  what it proves and what it does not.

The counting rules match the hearing bench's own (``voxa/bench/metrics.py``): a playback cessation is
``tts-stop``/``yield``/``INTERRUPT``/``offline-interrupt``/``duck``, a self-command is a request her
own playback created. A ``pause`` event is NOT a self-pause: in the runner it marks a segment boundary
in the microphone, and counting those would measure the segmentation, not the self-interruption.

Nothing here opens a microphone, changes a volume, or plays a sound by itself: the live run needs
``--yes``, and ``--plan`` answers every question about it without touching the speakers.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- the corpus

#: Every family the plan draws from. ``wake`` is replaced by the user's own wake word.
REPLY_FAMILIES: dict[str, list[str]] = {
    "plain": [
        "The weather in San Diego is sunny today with a high of seventy two degrees and a light "
        "breeze from the west. Tomorrow looks cooler, with a chance of showers after four o'clock.",
        "Your calendar has three meetings tomorrow: the stand up at nine, a review at half past "
        "eleven, and a call with the studio at four. The afternoon is otherwise free until six.",
        "I found the invoice you asked for. It is dated the twelfth of November, for two hundred "
        "and forty pounds, and it is already paid. The file is in your Documents folder.",
        "The nearest pharmacy is open for another twenty minutes, about six hundred feet away on "
        "the corner. Walking there takes about seven minutes at a normal pace.",
        "Here is what changed in the file: two paragraphs were added at the top, the table lost a "
        "column, and the closing paragraph was shortened by one sentence.",
        "Your battery is at sixty one percent with about two hours of work left. The last charge "
        "cycle finished yesterday evening, and the health of the battery reads as good.",
    ],
    "wake": [
        "Yes, I am here. Say {wake} whenever you want me to listen, and I will answer as soon as "
        "you stop talking. You never have to repeat your name for me.",
        "I heard the word {wake} in something I said just now. That is my own voice coming back to "
        "me through the speakers, so I am not going to answer it.",
        "When you say {wake} I start listening. When I say {wake} myself, nothing should happen, "
        "and this sentence is here exactly to check that it does not.",
    ],
    "stop": [
        "If you want me to stop, say stop. I will stop at once, and you can carry on without me. "
        "Saying the word stop myself, in the middle of this reply, must not stop me at all.",
        "The command stop ends what I am saying. I am saying the word stop now, on purpose, in the "
        "middle of a sentence, to see whether I interrupt myself. Stop.",
        "Stopping a long piece of music is one word: stop. Saying it to myself is not a command, "
        "so I will carry on reading this reply to the end, and stop is only a word here.",
    ],
    "cancel": [
        "You can cancel the timer, cancel the reminder, or cancel the download. I am cancelling "
        "nothing now, because the word cancel in my own mouth is not a request from you.",
        "Cancel means the same thing to me from you as from me, and it should not: this sentence "
        "says cancel twice to check that my own voice never becomes a command.",
    ],
    "offline": [
        "The model is offline right now, so I will answer from what I already know. Going offline "
        "is something you can ask for; me saying the word offline in a reply is not that request.",
        "I am not offline, I am only saying the word offline. If my own voice made me go offline, "
        "this reply would have ended two sentences ago, and it has not.",
    ],
    "pause": [
        "Let me think about that for a moment. Pause is a word you can say to me, and I am saying "
        "it myself now, which should leave me exactly where I am: reading this reply.",
        "Wait a second while I look that up. Saying wait to myself is not a reason to wait, so I "
        "will finish this sentence, and then this one, without pausing.",
    ],
    "mixed": [
        "You said: stop dictation. I heard that from you earlier, and now I am repeating it. "
        "Repeating your words is my voice, not a new command, so nothing should happen. Stop dictation.",
        "Here is the summary: the timer is running, the download is at sixty percent, and you asked "
        "me to cancel the reminder and to stop the music when the download finishes. Cancel, stop.",
        "Playing the radio now. Say Voxa, stop, or say enough, and I will be quiet. Say offline and "
        "I will stop using the network. Those are your words to say, not mine.",
    ],
    "numbers": [
        "The total is one thousand two hundred and thirty four point five six, the tax is two "
        "hundred and six pounds, and the date is the twenty third of November, two thousand and "
        "twenty six. The reference number is seven seven four one.",
        "Your account balance is four hundred and twelve pounds, the standing order is forty five, "
        "and the interest rate is zero point seven three percent. The next statement is due on the "
        "first of December.",
    ],
    "question": [
        "Do you want me to open the browser and go to your music page, or should I open the music "
        "player on its own? Say yes or no and I will do the one you chose.",
        "I found three files with that name. The first is in Documents, the second is on the "
        "Desktop, and the third is in the archive. Which one do you want me to open?",
    ],
}

#: Families whose text carries a word that can turn her own voice into a command.
TRIGGER_FAMILIES = ("wake", "stop", "cancel", "offline", "pause", "mixed")

CHARS_PER_SECOND = 14.0  # a rough speaking rate for the plan's time estimate


@dataclass(slots=True)
class SoakReply:
    """One reply the soak will speak, and when it belongs in the plan."""

    index: int
    text: str
    family: str
    voice: str
    start_s: float
    est_seconds: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "family": self.family,
            "voice": self.voice,
            "start_s": round(self.start_s, 2),
            "est_seconds": round(self.est_seconds, 2),
            "text": self.text,
        }


def fill_wake_word(text: str, wake_word: str) -> str:
    """Put the user's own wake word into a template."""
    return text.replace("{wake}", wake_word or "voxa")


def estimate_seconds(text: str) -> float:
    """Rough spoken length of one reply, for the plan only (never for a measurement)."""
    return round(max(1.0, len(text) / CHARS_PER_SECOND), 2)


def soak_plan(
    hours: float,
    seed: int = 0,
    gap_seconds: float = 2.5,
    voices: Sequence[str] = ("en-US-GuyNeural", "en-US-JennyNeural"),
    wake_word: str = "voxa",
) -> list[SoakReply]:
    """A deterministic stream of replies that fills ``hours`` hours of playback.

    Seeded: the same seed, hours and gap give byte-identical text, so a soak can be repeated and its
    counts compared. Families are drawn with the trigger-word families weighted heavily, because
    those are the ones that make Voxa interrupt and command herself.
    """
    if hours <= 0:
        raise ValueError("--hours must be a positive number of hours")
    if gap_seconds < 0:
        raise ValueError("--gap cannot be negative")
    voices = tuple(voices) or ("en-US-GuyNeural",)
    rng = random.Random(seed)
    families = list(REPLY_FAMILIES)
    weights = [3.0 if name in TRIGGER_FAMILIES else 1.0 for name in families]
    plan: list[SoakReply] = []
    total_seconds = hours * 3600.0
    at = 0.0
    index = 0
    while at < total_seconds:
        family = rng.choices(families, weights=weights, k=1)[0]
        text = fill_wake_word(rng.choice(REPLY_FAMILIES[family]), wake_word)
        voice = voices[index % len(voices)]
        est = estimate_seconds(text)
        plan.append(SoakReply(index=index + 1, text=text, family=family, voice=voice,
                              start_s=round(at, 2), est_seconds=est))
        at += est + gap_seconds
        index += 1
    return plan


def plan_summary(plan: Sequence[SoakReply], gap_seconds: float = 2.5) -> dict[str, Any]:
    """What a plan contains: counts per family, spoken seconds, voices."""
    families: dict[str, int] = {}
    for reply in plan:
        families[reply.family] = families.get(reply.family, 0) + 1
    spoken = sum(reply.est_seconds for reply in plan)
    return {
        "replies": len(plan),
        "families": families,
        "trigger_replies": sum(1 for reply in plan if reply.family in TRIGGER_FAMILIES),
        "spoken_seconds": round(spoken, 1),
        "gap_seconds": gap_seconds,
        "playback_hours": round(spoken / 3600.0, 4),
        "wall_hours": round((spoken + gap_seconds * len(plan)) / 3600.0, 4),
        "voices": sorted({reply.voice for reply in plan}),
    }


# ---------------------------------------------------------------- the counting

#: Events that mean "Voxa's own playback was cut short" (handoff 8.3, self-interruption rate).
#: ``pause`` is deliberately absent: it marks a microphone segment boundary, not an output pause.
SELF_PAUSE_EVENT_NAMES = frozenset({
    "tts-stop", "yield", "INTERRUPT", "offline-interrupt", "duck", "output-pause", "self-pause",
})

#: Events that mean "her own voice created a request" (the barge-in side of a self-command).
SELF_COMMAND_EVENT_NAMES = frozenset({"request", "interrupt", "promote"})


def parse_barge_line(line: str) -> dict[str, Any] | None:
    """One line of Voxa's ``barge.log`` as an event, or None for a line we cannot read.

    The format Voxa writes is ``HH:MM:SS event key=value key=value``; the bench's own events
    (``{"event": "tts-stop", ...}``) are accepted as JSON too.
    """
    line = (line or "").strip()
    if not line:
        return None
    if line.startswith("{"):
        try:
            payload = json.loads(line)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) and payload.get("event") else None
    parts = line.split()
    if len(parts) < 2:
        return None
    event: dict[str, Any] = {"event": parts[1], "clock": parts[0]}
    for item in parts[2:]:
        key, _, value = item.partition("=")
        if not key:
            continue
        text = value.strip("'\"")
        try:
            event[key] = int(text)
        except ValueError:
            try:
                event[key] = float(text)
            except ValueError:
                event[key] = text
    return event


def is_self_pause(event: dict[str, Any]) -> bool:
    """Does this event cut Voxa's own playback short? (provisional yields included)"""
    name = str(event.get("event", ""))
    return name in SELF_PAUSE_EVENT_NAMES


def is_self_command_record(record: dict[str, Any]) -> bool:
    """Is this action-log line a command her own playback created?

    During a soak nobody speaks, so any line where Voxa heard words is a user turn she made out of
    her own voice. A line naming a tool counts even when the transcript came out empty: the command
    ran, which is the more serious failure of the two in handoff 8.3.
    """
    heard = str(record.get("heard", "")).strip()
    tool = str(record.get("tool", "")).strip()
    return bool(heard) or bool(tool)


def count_self_events(
    action_records: Iterable[dict[str, Any]],
    events: Iterable[dict[str, Any]],
    hours: float,
    replies_spoken: int = 0,
) -> dict[str, Any]:
    """The soak's numbers: every self-command and self-pause, per hour, with the Poisson bound.

    ``action_records`` are the action-log lines written during the run (one self-command each),
    ``events`` are the barge-in events (one self-pause each). The two sources are counted separately
    and added: an action-log line and a barge-in event describe different acts (a command ran, her
    playback was cut), so adding them is the honest total of self-triggers.
    """
    if hours <= 0:
        raise ValueError("the soak needs a positive number of hours")
    records = [record for record in action_records if isinstance(record, dict)]
    parsed = [event for event in events if isinstance(event, dict) and event.get("event")]
    commands = [record for record in records if is_self_command_record(record)]
    pauses = [event for event in parsed if is_self_pause(event)]
    command_events = [event for event in parsed if str(event.get("event")) in SELF_COMMAND_EVENT_NAMES]
    total = len(commands) + len(pauses)
    rate = total / hours
    return {
        "hours": round(hours, 4),
        "replies_spoken": int(replies_spoken),
        "self_command_count": len(commands),
        "self_pause_count": len(pauses),
        "self_event_count": total,
        "events_per_hour": round(rate, 6),
        "poisson_upper_95_per_hour": round(poisson_upper_bound(total, hours), 6),
        "self_pause_per_hour": round(len(pauses) / hours, 6),
        "self_command_rate": round(len(commands) / replies_spoken, 6) if replies_spoken else None,
        "command_events": len(command_events),
        "self_commands": [
            {"heard": str(record.get("heard", "")), "tool": str(record.get("tool", "")),
             "route": str(record.get("route", "")), "time": str(record.get("time", ""))}
            for record in commands
        ],
        "self_pauses": [dict(event) for event in pauses],
    }


# ---------------------------------------------------------------- the bound

def poisson_cdf(count: int, mean: float) -> float:
    """P(Poisson(mean) <= count), computed by accumulating the terms (no scipy)."""
    if mean <= 0.0:
        return 1.0 if count >= 0 else 0.0
    term = math.exp(-mean)
    total = term
    for k in range(1, count + 1):
        term *= mean / k
        total += term
        if total >= 1.0:
            return 1.0
    return min(1.0, total)


def poisson_upper_bound(count: int, hours: float, confidence: float = 0.95) -> float:
    """One-sided upper bound of a Poisson rate: the largest rate still able to produce ``count``.

    The bound is the ``confidence`` quantile of the chi-square with ``2(count+1)`` degrees of freedom
    divided by the observed time, which is exactly the statement the handoff makes: zero events in 100
    hours gives 2.996/100 = 0.03 events per hour, not proof of zero risk forever. Solved by bisection
    on ``P(Poisson(rate*hours) <= count) = 1 - confidence``, so it needs no scipy and no tables.
    """
    if hours <= 0:
        raise ValueError("the bound needs a positive number of hours")
    if count < 0:
        raise ValueError("a count cannot be negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    target = 1.0 - confidence
    lo = 0.0
    hi = 1.0
    while poisson_cdf(count, hi) > target:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if poisson_cdf(count, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0 / hours


def format_bound(result: dict[str, Any]) -> str:
    """The soak's headline line, in the words the handoff uses."""
    total = int(result.get("self_event_count", 0))
    hours = float(result.get("hours", 0.0))
    bound = float(result.get("poisson_upper_95_per_hour", 0.0))
    rate = float(result.get("events_per_hour", 0.0))
    if total == 0:
        return (
            f"{total} self-triggers in {hours:.2f} h: {rate:.4f} events/hour, one-sided 95% "
            f"Poisson upper bound {bound:.4f} events/hour. Zero observed is not proof of zero risk."
        )
    return (
        f"{total} self-triggers in {hours:.2f} h: {rate:.4f} events/hour, one-sided 95% "
        f"Poisson upper bound {bound:.4f} events/hour "
        f"({result.get('self_command_count', 0)} self-commands, {result.get('self_pause_count', 0)} self-pauses)."
    )


# ---------------------------------------------------------------- the live run

@dataclass
class SoakLogs:
    """Where the two sources of truth are, and how far we have read them."""

    action_log: Path
    barge_log: Path
    action_offset: int = 0
    barge_offset: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def start(self) -> None:
        """Remember where the run begins, so only what happens during it is counted."""
        self.action_offset = _size(self.action_log)
        self.barge_offset = _size(self.barge_log)

    def poll(self) -> list[dict[str, Any]]:
        """Read what the logs gained since the last poll; returns the new self-triggers."""
        fresh: list[dict[str, Any]] = []
        lines, self.action_offset = _tail(self.action_log, self.action_offset)
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and is_self_command_record(record):
                self.records.append(record)
                fresh.append({"kind": "self-command", "heard": record.get("heard", ""),
                              "tool": record.get("tool", "")})
        lines, self.barge_offset = _tail(self.barge_log, self.barge_offset)
        for line in lines:
            event = parse_barge_line(line)
            if event is not None and is_self_pause(event):
                self.events.append(event)
                fresh.append({"kind": "self-pause", "event": event.get("event", "")})
        return fresh

    def result(self, hours: float, replies_spoken: int) -> dict[str, Any]:
        return count_self_events(self.records, self.events, hours, replies_spoken)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _tail(path: Path, offset: int) -> tuple[list[str], int]:
    """Whole lines written after ``offset``, and the new offset (partial lines stay unread)."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return [], offset
    if not data:
        return [], offset
    complete = data.rfind("\n")
    if complete < 0:
        return [], offset
    return data[: complete].splitlines(), offset + len(data[: complete].encode("utf-8"))


def barge_log_path(action_log: Path) -> Path:
    """Voxa writes her barge-in trace beside the action log (``voxa/window.py``)."""
    return action_log.with_name("barge.log")


def run_soak(
    plan: Sequence[SoakReply],
    logs: SoakLogs,
    *,
    hours: float,
    speak: Callable[[SoakReply], None],
    gap_seconds: float = 2.5,
    on_event: Callable[[dict[str, Any]], None] = lambda _payload: None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 10.0,
) -> dict[str, Any]:
    """Speak the plan, watch the logs, and return the soak's numbers.

    ``speak`` is the only thing that touches the speakers, and it is injected: the unit tests hand in
    a recorder, the live command hands in Voxa's own TTS. The microphone is never opened here - Voxa
    owns it, and this run is about what she does with nobody talking.
    """
    logs.start()
    started = clock()
    deadline = started + hours * 3600.0
    spoken = 0
    last_poll = started
    last = plan[-1] if len(plan) else None
    interrupted = False
    try:
        for reply in plan:
            if clock() >= deadline:
                break
            spoken += 1  # the reply counts as spoken the moment we ask for it: an interrupted run is honest
            speak(reply)
            on_event({"event": "soak_reply", "index": reply.index, "family": reply.family,
                      "voice": reply.voice, "seconds": reply.est_seconds})
            sleep(gap_seconds)
            now = clock()
            if now - last_poll >= poll_seconds or reply is last:
                for fresh in logs.poll():
                    on_event({"event": "soak_self_trigger", "index": reply.index, **fresh})
                last_poll = now
    except KeyboardInterrupt:
        # Ctrl-C ends the soak; what was observed up to now is still a measurement.
        interrupted = True
    for fresh in logs.poll():
        on_event({"event": "soak_self_trigger", "index": spoken, **fresh})
    elapsed = max(1e-9, clock() - started)
    result = logs.result(elapsed / 3600.0, spoken)
    result["wall_seconds"] = round(elapsed, 2)
    result["interrupted"] = interrupted
    result["plan"] = plan_summary(plan, gap_seconds)
    return result


def format_plan(plan: Sequence[SoakReply], summary: dict[str, Any]) -> str:
    """The plan in words, for ``voxatest soak --plan`` and the live run's heads-up."""
    lines = [
        f"Soak plan: {summary['replies']} replies, {summary['spoken_seconds']:.0f} s of speech, "
        f"{summary['playback_hours']:.2f} playback hours, gap {summary['gap_seconds']} s.",
        f"Trigger-word replies: {summary['trigger_replies']} "
        f"({', '.join(f'{name} {count}' for name, count in sorted(summary['families'].items()))}).",
        f"Voices: {', '.join(summary['voices'])}.",
    ]
    for reply in plan[:5]:
        lines.append(f"  [{reply.index}] {reply.family} +{reply.start_s:.0f}s: {reply.text[:96]}")
    if len(plan) > 5:
        lines.append(f"  ... {len(plan) - 5} more replies.")
    return "\n".join(lines)
