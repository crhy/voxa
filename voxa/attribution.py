"""Decide WHO made a sound: the user, or Voxa's own voice coming back through the speakers.

Handoff 6.3 ("separate echo removal from user attribution") asks for three evidence streams
and this module is where they are combined:

1. **Echo consistency** - is the microphone explained by the reference that was playing at that
   moment (delayed reference power, residual left over after the echo-path fit, alignment health)?
2. **Independent speech** - is there speech of its own in the CLEANED signal (VAD probability,
   double-talk evidence in the residual)?
3. **Conversation evidence** - the words: overlap with what she is saying, letter similarity,
   stop/wake cues. This is a WEIGHTED FEATURE. It can shift a confidence a little; on its own it
   must never decide, and a SHORT utterance is never evidence of echo (handoff 5.2: "yes",
   "blue", "that one" were being thrown away for having fewer than three words).

The result is one of three things, and the third one matters as much as the first two:

* ``user``    - a second source is supported: yield the floor, keep the words.
* ``echo``    - her own voice is supported: keep speaking, and only now may the words be stripped.
* ``uncertain`` - nothing is decided. The audio and the transcript are kept, nothing is deleted,
  no automatic command runs from it, and the span stays available for the next observation.

Everything here is pure: numbers in, a decision out. No audio devices, no GTK, no threads, no
allocation that grows with the session (the echo-path baseline is a bounded ring).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

USER = "user"
ECHO = "echo"
UNCERTAIN = "uncertain"

# --- calibration ------------------------------------------------------------------------------
# These weights are a DECLARED prior, not a fit: the frozen clip set cannot calibrate them today,
# because the bench feeds the reference mask an int16 microphone and a float32 reference, and the
# dB numbers that come out are degenerate (tasks/notes-f139-*.md records the measured distributions:
# Voxa-only frames report "explained -120 dB / residual -120 dB", double-talk frames report residuals
# up to +127 dB, which is more energy than the microphone delivered). The numbers below say what
# would count as evidence if the capture path measured itself honestly; the task that fixes the
# instrument (roadmap task 1's reference tap, task 2's canceller) re-measures them.
# Residual energy left after the echo path is fitted, this far ABOVE the echo path's own recent
# baseline, counts as a second source. Below it, the microphone is still her voice.
USER_RESIDUAL_DB = 6.0
# The microphone is (almost) fully accounted for by the delayed reference: positive echo evidence.
ECHO_EXPLAINED_DB = -1.5
# A VAD probability in the cleaned signal at or above this is speech of its own.
VAD_SPEECH = 0.5
# Text features may move the score by at most this much, never decide by themselves.
TEXT_WEIGHT = 0.3
# ... unless the capture path has NO echo model at all (no reference sink on this machine):
# then the words are the only evidence there is, they get more weight, and the decision is
# reported as degraded (the app says so; roadmap task 1 gives the reference tap back).
TEXT_ONLY_WEIGHT = 0.7
# Roadmap task 7: the word-timed overlap's own weight, never above the cap the words are allowed
# (a time-matched word is the single strongest word feature, but still one supporter, not a vote).
WORD_TIMED_WEIGHT = TEXT_WEIGHT
# A stop word is a cue about what to DO once a second source is supported.
STOP_CUE = 0.15
STOP_CUE_WITHOUT_AUDIO = 0.5
# The wake word is a weaker cue than a command: someone is addressing her, that is all.
WAKE_CUE = 0.1
WAKE_CUE_WITHOUT_AUDIO = 0.25
# The microphone standing well above her own voice is audio evidence, scaled by how unlike her
# sentence the words are (a misheard echo towers too).
LEVEL_LIFT = 0.45
# The joint double-talk gate (voxa.voicedetect.DoubleTalkGate) has ALREADY combined the three
# audio streams and held them together for a span before firing; when it fires its word is worth
# more than any single stream, but it is still one supporter: the words keep their cap.
JOINT_LIFT = 0.8
GATE_RATIO = 2.0
# A score inside this band is not a decision.
UNCERTAIN_BAND = 0.25
# How much the words count as "not hers" when there are too few letters to compare at all: neither
# evidence for nor against.
UNKNOWN_DISSIMILARITY = 0.5
# How many recent frames the echo-path baseline remembers (0.02 s each -> 2 s).
ECHO_BASELINE_FRAMES = 100
# A reference is "active" above this power. The same number lives in voxa/anc.py (the echo path
# decides whether to fit itself with it); attribution is model-free and keeps its own copy so it
# imports no audio code - the two are kept equal by tests/test_attribution.py.
MIN_REFERENCE_POWER = 1e-5  # about -50 dB: below this the computer is effectively silent


@dataclass(slots=True)
class SourceEvidence:
    """What the capture path knows about one moment, measured rather than inferred from words.

    Every audio field is optional: ``None`` means "this stage cannot measure it yet", which is
    different from 0.0 and pushes the decision towards ``uncertain``.
    """

    reference_active: bool = False
    reference_power: float = 0.0
    echo_explained_db: float | None = None   # dB of the microphone the delayed reference accounts for
    residual_db: float | None = None         # dB of the microphone left over after the echo-path fit
    residual_margin_db: float | None = None  # that residual above the echo path's recent baseline
    # The residual mask's OWN output against its OWN predicted residual (roadmap task 5): the
    # honest margin, measured where the prediction lives instead of by EchoPathBaseline.
    residual_over_prediction_db: float | None = None
    # The joint double-talk gate's voice-only verdict for this moment
    # (voxa.voicedetect.DoubleTalkGate): True only when all three audio streams held together.
    joint_suspect: bool | None = None
    # The gate's MIRROR verdict: the microphone held the delayed reference (high coherence,
    # nothing of its own) for the same window. Only this, not "not suspect", may suppress the
    # caller's fallback gate or vote for echo.
    joint_echo: bool | None = None
    vad_prob: float | None = None    # speech probability in the CLEANED signal (None = not measured)
    alignment_healthy: bool = True           # delay known, fresh, within tolerance
    mic_level: float = 0.0                   # 0..1, the scale BargeInGate uses
    own_level: float = 0.0                   # the level her own voice produces in the microphone
    # How far the microphone stands above her own level (the level gate's own ratio): crude, but
    # it IS audio evidence, and on a machine with no reference sink it is the only audio evidence.
    level_ratio: float | None = None
    # False when the heard text has too few letters for a letter match to mean anything: unknown
    # evidence must not be read as "these words are nothing like her sentence".
    text_letters_measurable: bool = True
    # Conversation features: weighted, capped, never the whole decision.
    text_overlap: float = 0.0
    text_sound: float = 0.0
    word_count: int = 0
    stop_word: bool = False
    wake_word: bool = False
    # Roadmap task 7: the WORD-TIMED overlap - the share of heard words that fuzzy-match a TTS
    # word her speakers were audible reading within +-W ms (``voxa/ownspeech``'s ``timed_overlap``).
    # A time-matched word is stronger echo evidence than a bare out-of-order overlap (it matches
    # at the moment the echo could actually land), but it is still ONE weighted feature: it may
    # nudge the confidence, never decide, and never on a confirmed near-end utterance.
    word_timed_overlap: float = 0.0


@dataclass(slots=True)
class SourceDecision:
    """A source, the confidence in it, and the evidence that produced it (traced, never discarded)."""

    source: str
    confidence: float
    reasons: tuple[str, ...] = ()
    degraded: bool = False
    evidence: SourceEvidence | None = None


class EchoPathBaseline:
    """The echo path's own recent residual level, as a bounded ring.

    A residual is only meaningful against what THIS path normally leaves behind: a canceller that
    removes nothing leaves a big residual on her own voice too. The margin, not the raw number,
    is what says "someone else added sound".
    """

    def __init__(self, frames: int = ECHO_BASELINE_FRAMES) -> None:
        self._values: deque[float] = deque(maxlen=max(4, int(frames)))

    def note(self, residual_db: float | None) -> float | None:
        """Feed one frame's residual; return the margin of the SAME frame over the baseline before it."""
        if residual_db is None:
            return None
        if not self._values:
            self._values.append(residual_db)
            return None
        previous = sum(self._values) / len(self._values)
        self._values.append(residual_db)
        return residual_db - previous

    def reset(self) -> None:
        self._values.clear()

    @property
    def level_db(self) -> float | None:
        return sum(self._values) / len(self._values) if self._values else None


def text_score(overlap: float, sound: float, weight: float = TEXT_WEIGHT) -> float:
    """How much the WORDS look like her own sentence, capped at the weight the evidence allows.

    Overlap and letter similarity are combined with the larger of the two winning, so a
    mishearing that keeps the letters ("BLC doesn't seem to be open" -> "Bmc, done, themes")
    still counts, and a single shared word does not. The NUMBER of words is not a feature:
    handoff 5.2 shows "yes", "blue" and "that one" are valid brief answers.
    """
    return weight * max(overlap, sound)


def attribute(ev: SourceEvidence, baseline_db: float | None = None) -> SourceDecision:
    """Combine the evidence streams into a source, a confidence and the reasons.

    Each stream contributes one signed number. Deciding that a SECOND SOURCE is present needs
    two streams to agree, because that is the decision whose failure the task exists to fix:
    one crude signal (a loud mic, a stop word, a level tower) used to be enough and produced
    self-interruption. Deciding ECHO needs no such majority: it only ever holds a span back, and
    on a path with no echo model the words are the only stream there is.
    """
    reasons: list[str] = []
    streams: list[tuple[str, float]] = []  # (name, signed weight); > 0 supports the user

    def add(name: str, value: float) -> None:
        if value:
            streams.append((name, value))

    if not ev.reference_active:
        # Nothing is coming out of the speakers: there is no echo to explain the microphone, so
        # any speech in it is the user's. (The near-end-only case, and the case where Voxa is
        # silent.) A reference that is merely quiet is still a reference: see the power check below.
        reasons.append("reference silent: no echo possible")
        speech = ev.vad_prob if ev.vad_prob is not None else (1.0 if ev.mic_level > 0.0 else 0.0)
        if speech >= VAD_SPEECH or ev.mic_level > 0.0:
            return SourceDecision(USER, min(1.0, 0.6 + 0.4 * speech), tuple(reasons), evidence=ev)
        return SourceDecision(UNCERTAIN, 0.0, tuple(reasons + ["no sound measured"]),
                              degraded=False, evidence=ev)

    if not ev.alignment_healthy and ev.echo_explained_db is not None:
        # The echo path cannot be fitted right now, so its audio numbers are unqualified and no
        # decision may be taken from them (handoff 6.8) - a visible degraded state. With no echo
        # model on this path at all there is nothing to align either: that is the degraded case
        # below, not an unhealthy one.
        reasons.append("alignment unhealthy: audio unqualified, no decision")
        return SourceDecision(UNCERTAIN, 0.0, tuple(reasons), degraded=True, evidence=ev)

    # 1. Echo consistency: how much of the microphone the delayed reference already explains.
    explained = ev.echo_explained_db
    if explained is not None:
        if explained >= ECHO_EXPLAINED_DB:
            add("explained", -0.5)
            reasons.append(f"microphone explained by reference {explained:+.1f} dB")
        else:
            value = min(0.5, (ECHO_EXPLAINED_DB - explained) / 12.0)
            add("explained", value)
            reasons.append(f"microphone holds {explained:+.1f} dB beyond the reference")

    # 2. Independent speech: the residual over what this echo path usually leaves, and the VAD.
    # Roadmap task 5: the mask's own margin (output over ITS predicted residual) outranks the
    # EchoPathBaseline margin - the baseline rings a floor it can never measure honestly.
    margin = (ev.residual_over_prediction_db if ev.residual_over_prediction_db is not None
              else ev.residual_margin_db if ev.residual_margin_db is not None else baseline_db)
    if margin is not None:
        if margin >= USER_RESIDUAL_DB:
            add("residual", 0.5)
            reasons.append(f"residual +{margin:.1f} dB over the echo path's baseline")
        elif margin <= 0.0:
            add("residual", -0.3)
            reasons.append(f"residual {margin:+.1f} dB at or under the echo path's baseline")
    if ev.joint_suspect:
        add("joint", JOINT_LIFT)
        reasons.append("joint double-talk gate: speech, margin and echo-inconsistency held together")
    elif ev.joint_echo:
        # The MIRROR verdict of the same gate: this span was the delayed reference the whole time.
        add("joint", -JOINT_LIFT)
        reasons.append("joint double-talk gate: microphone held the delayed reference, nothing of its own")
    if ev.vad_prob is not None:
        if ev.vad_prob >= VAD_SPEECH:
            add("vad", 0.3)
            reasons.append(f"speech in the cleaned signal p={ev.vad_prob:.2f}")
        else:
            add("vad", -0.1)
            reasons.append(f"cleaned signal not speech-like p={ev.vad_prob:.2f}")
    else:
        reasons.append("no VAD on the cleaned signal for this span")

    # 2b. The level gate's own evidence, read together with the words: the microphone towers over
    #     the level her voice produces, and the words are NOT her sentence. A tower alone is crude
    #     (her own voice misheard also towers), so it is scaled by how unlike her sentence the words
    #     are; it is audio evidence, not a length rule.
    if ev.level_ratio is not None:
        tower = min(1.0, max(0.0, ev.level_ratio - 1.0) / (GATE_RATIO - 1.0))
        similarity = max(ev.text_overlap, ev.text_sound)
        dissimilarity = (1.0 - similarity) if ev.text_letters_measurable else UNKNOWN_DISSIMILARITY
        lift = LEVEL_LIFT * tower * dissimilarity
        if lift:
            add("level", lift)
            reasons.append(f"microphone {ev.level_ratio:.1f}x her own level, words {dissimilarity:.0%} "
                           f"unlike her sentence (+{lift:.2f})")

    # A run of sound is NOT a stream: while Voxa reads a reply the microphone is above the floor
    # continuously, so "the sound lasted 3 s" says nothing about who made it. Only what the echo
    # path cannot explain, and what the words are not, count as evidence of a second source.
    # 3. Conversation evidence: weighted, capped, and never a length rule. Without an echo model
    #    the words carry more, and the decision says so.
    audio_streams = sum(1 for value in (ev.echo_explained_db, ev.residual_margin_db,
                                        ev.residual_over_prediction_db)
                        if value is not None)
    weight = TEXT_WEIGHT if audio_streams else TEXT_ONLY_WEIGHT
    text_value = -text_score(ev.text_overlap, ev.text_sound, weight)
    # Roadmap task 7: the word-timed overlap is a heavier word feature than a bare overlap - a
    # heard word that lands inside the audible span of the very word she was reading is an echo
    # candidate at the moment the echo could actually arrive. Cap it at the same weight the words
    # are allowed; it never decides alone and never on a confirmed near-end utterance.
    if ev.word_timed_overlap:
        timed = -min(weight, WORD_TIMED_WEIGHT * ev.word_timed_overlap)
        text_value += timed
        reasons.append(f"words time-matched to her playback {ev.word_timed_overlap:.2f} "
                       f"(+{abs(timed):.2f} toward echo)")
    add("text", text_value)
    if ev.text_overlap >= 0.6 or ev.text_sound >= 0.6:
        reasons.append(f"words resemble her sentence (overlap {ev.text_overlap:.2f}, "
                       f"letters {ev.text_sound:.2f}, weight {weight:.1f})")
    if not audio_streams:
        reasons.append("no echo model on this path: words carry the decision (degraded)")
    if ev.stop_word:
        # A stop word is a cue about what to DO once a second source is supported; it is not
        # evidence that a second source exists, and it must not stop her when the audio says the
        # word is her own (task A2 step 4).
        cue = STOP_CUE if audio_streams else STOP_CUE_WITHOUT_AUDIO
        add("cue", cue)
        reasons.append(f"stop word present (cue {cue:+.2f}, not evidence)")
    if ev.wake_word:
        # Her name is a weaker cue than a command: someone is addressing her, that is all.
        add("cue", WAKE_CUE if audio_streams else WAKE_CUE_WITHOUT_AUDIO)
        reasons.append("wake word heard")

    score = sum(value for _, value in streams)
    for name, value in streams:
        reasons.append(f"{name} {value:+.2f}")
    supporters = [name for name, value in streams if value > 0.0]
    confidence = abs(score)
    if confidence < UNCERTAIN_BAND:
        reasons.append(f"evidence too weak to decide ({score:+.2f})")
        return SourceDecision(UNCERTAIN, confidence, tuple(reasons), degraded=not audio_streams,
                              evidence=ev)
    if score > 0.0 and audio_streams and len(supporters) < 2:
        # With an echo model on the path, a second source needs two of its streams to agree: that
        # is the decision whose failure produced the self-interruptions this task exists to fix.
        # Without one, the level tower and the words are all there is, and the decision says it is
        # degraded rather than pretending otherwise.
        reasons.append(f"only one stream supports a second source ({', '.join(supporters)})")
        return SourceDecision(UNCERTAIN, confidence, tuple(reasons), degraded=not audio_streams,
                              evidence=ev)
    return SourceDecision(USER if score > 0.0 else ECHO, min(1.0, confidence), tuple(reasons),
                          degraded=not audio_streams, evidence=ev)


def text_only_decision(ev: SourceEvidence) -> SourceDecision:
    """Degraded attribution for a capture path with NO reference at all.

    Without a reference there is nothing to compare the microphone with, so the words are all
    there is. This is a visible degraded state: it may only ever answer ``echo`` or
    ``uncertain`` - it must never claim a second source strongly enough to delete or rewrite
    anything, and it never uses the length of an utterance as evidence of echo.
    """
    score = text_score(ev.text_overlap, ev.text_sound) - 0.5
    if ev.stop_word or ev.wake_word:
        score -= 0.4
    if score >= UNCERTAIN_BAND:
        return SourceDecision(ECHO, min(1.0, score), ("no reference: text-only attribution",),
                              degraded=True, evidence=ev)
    return SourceDecision(UNCERTAIN, 0.0, ("no reference: cannot tell a second source from echo",),
                          degraded=True, evidence=ev)
