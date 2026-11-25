"""The barge-in decision, free of GTK and of the window (roadmap B2, step 1).

``voxa/window.py`` and the hearing bench (``voxa/bench/runner.py``) call this one class,
so what the bench measures is exactly what the app decides. The logic is the code that
used to live in ``MainWindow._maybe_barge_in`` (window.py:3152); behaviour is unchanged
by this move — the window keeps the effects (stopping speech, unmuting, arming the prompt)
and this module keeps the decision.

Everything here is pure: a clock the caller supplies, a level the caller measured, and a
callable that returns the words heard. No GTK, no threads, no audio devices. In the live app
``observe`` runs on the hearing worker's thread (``voxa/hearing.py``), which is what keeps the
candidate ASR off the GTK thread; the window only receives the decisions.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .attribution import ECHO, ECHO_EXPLAINED_DB, USER, SourceEvidence
from .attribution import UNCERTAIN as SOURCE_UNCERTAIN
from .bargein import FRAME_MS, BargeInGate, STOP_WORDS, _words, overlap, sounds_like, source_decision
from .onset import OnsetDetector
from .conversation import detect_addressed_exit_phrase, strip_wake_word

# Barge-in: loud sustained speech while a reply is being read interrupts it.
# There is NO grace period any more (handoff 5.4, task A3): ``BARGE_IN_GRACE_SECONDS`` was a
# blind fold over the microphone for the first 0.4 s of every reply, which is exactly where an
# interruption that starts together with her speech lives. What replaces it is the evidence rule
# in ``voxa/onset.py``: the microphone holding sound the speaker path cannot explain, measured
# in milliseconds of audio, live from the first rendered frame.
# After the onset, how much audio to keep collecting before the WORDS are judged. This no longer
# blocks yielding: the decision to stop her stream is acoustic and fast (task A3); the transcript
# completes later and decides what happens to her words, not whether the floor was yielded.
BARGE_IN_LISTEN_MS = 900.0
BARGE_IN_LISTEN_SECONDS = BARGE_IN_LISTEN_MS / 1000.0  # kept for callers that still speak in seconds

# Issue #72, step 4: the second guard behind DTLN. The canceller removes most of her voice but a
# few words leak through the mask ("5 times more." from "The answer is: 5 times 4 is 20."); a leak
# that matches what she is SAYING right now is not an interruption. Deliberately narrow: a stop or
# wake word always beats the text (task A2: a person saying "stop" over her "stop" must interrupt).
TEXT_ECHO_OVERLAP = 0.5
TEXT_ECHO_SOUNDS = 0.55

# Issue #72, task v08: the echo she hears back is often the piece spoken BEFORE the current one
# (the live log's heard lines match the PREVIOUS piece's words), so the veto compares against the
# whole turn, and the guard must not swallow a real question that shares her function words: the
# heard line only dies to the veto when most of its words are hers.
VETO_PREVIOUS_SECONDS = 10.0

# Issue #72, task v11 step 3: an abstain must not be permanent. When the reference is UNUSABLE
# (the live tap reads silent while she speaks, or the alignment is unhealthy) attribution can only
# ever hold the span back, so nothing said over her could ever become ``user`` and real barge-ins
# were dead (the rc5 live run recorded 0 INTERRUPTs). While she speaks and the reference is
# unusable, decide by the TURN-WIDE text veto instead: heard CONTENT words that are NOT in her
# whole turn's text are the user's. The threshold is deliberately high and the wake word opens it:
# >= 3 foreign content words, or the wake word plus >= 2. Her own words (misheard echoes land a
# handful of foreign words at most) never reach it, so they still never interrupt her.
DEGRADED_FOREIGN_MIN = 3
DEGRADED_FOREIGN_WITH_WAKE = 2
# Room + microphone latency of the live run: her own tail is still in the microphone this long
# after the last piece ended, so the reference is still hers, and she must not answer it.
ECHO_TAIL_SECONDS = 1.5


# Whisper writes her own content words as near-misses ("leveling" for "lemonade", "Bans" for
# "Paris", "five" for "5", "online" for "only"); a pure echo line therefore carries AT MOST three
# content words the turn never said. A real question over her reply carries more ("capital",
# "France", "life", "thank" in the logged France line), and a real maths prompt addresses her by
# name - the wake word exempts it from the veto outright. Function words carry no signal.
_VETO_FOREIGN_CONTENT_MAX = 3

# Closed class: never decide narrowness on these. Numbers are NOT in it - "at 15 to 20" and
# "100x4" are words to the veto (voxa.bargein._words drops digits, which made the logged maths
# prompts look like a handful of her own function words).
_FUNCTION_WORDS = frozenset("""
    i me my mine we us our you your yours he him his she her it its they them their this that
    these those a an the and or but nor to of in on at for with by from into over under again
    then is am are was were be been being do does did done have has had will would can could
    shall should not no so too very just now here there all any both each few other some such
    same what which who whom whose where when why how s t d ll re ve m
""".split())

# Number words are digits to the veto: "five times more." leaks from "5 times 4 is 20." and must
# not be counted as foreign content.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000",
}


def _heard_tokens(text: str) -> list[str]:
    """Words of a transcript, numbers included: "15 to 20" and "100x4" are words to the veto.

    ``voxa.bargein._words`` drops digits (it scores letter features), which would make a maths
    prompt look like a handful of her own function words.
    """
    return re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())


def _foreign_core(heard: str, turn_text: str, wake_word: str) -> tuple[int, int]:
    """How many heard CONTENT words are hers, and how many are not (the veto's narrowness check).

    A content word is hers when it stands in what she said this turn (or fuzzy-matches a word of
    it, which is what Whisper's mishearing of her own voice does: "leveling" for "lemonade",
    "five" for "5", "100" inside "100x4"). Each word counts ONCE - a doubled mishearing is still
    one mishearing. Wake and stop words never count as foreign: they are cues, and the veto
    already steps aside for them.
    """
    tokens = _heard_tokens(turn_text)
    spoken_raw = {_NUMBER_WORDS.get(token, token) for token in tokens} | set(tokens)
    spoken = {token for token in spoken_raw if token not in _FUNCTION_WORDS}
    # Task v11c step 5: ASR writes her own words with different letters all the time
    # ("weather" for "whether", "receit" for "receipt"). A word that SOUNDS like a word of her
    # turn is hers too, before it is counted foreign and promotes the text veto.
    from .apps import phonetic_key
    from difflib import SequenceMatcher
    spoken_keys = {k for k in (phonetic_key(word) for word in spoken) if len(k) >= 3}
    wake = wake_word.casefold()
    hers: set[str] = set()
    foreign: set[str] = set()
    for token in _heard_tokens(heard):
        token = _NUMBER_WORDS.get(token, token)
        if token == wake or token in STOP_WORDS:
            continue
        if token in _FUNCTION_WORDS:
            continue
        if token in spoken or any(len(token) > 2 and (token in word or word in token)
                                  for word in spoken):
            hers.add(token)
            continue
        key = phonetic_key(token)
        if len(key) >= 3 and any(
                key == word_key or SequenceMatcher(None, key, word_key).ratio() >= 0.8
                for word_key in spoken_keys):
            hers.add(token)
        else:
            foreign.add(token)
    return len(hers), len(foreign)


# Outcomes of one observation; the caller performs the effects.
IGNORED = "ignored"
OFFLINE = "offline"
OWN_VOICE = "own-voice"
INTERRUPT = "interrupt"
HOLD = "hold"  # the evidence did not decide: keep the audio, strip nothing, run no command
# The fast, acoustic one: stop Voxa's playback NOW. The words are still being collected, and the
# later INTERRUPT / OWN_VOICE / HOLD outcome decides what happens to them (promoted, rejected,
# kept) - never whether the floor was yielded.
YIELD = "yield"

# The presenter of a station Voxa has just started is not the user, unless the words
# are an interrupt word or address her by name.
_MEDIA_EXCEPTION = re.compile(r"\b(?:stop|pause|cancel|quiet|enough|wait)\b")

# The capture path reports levels in int16 units; the gate works on the 0..1 scale the
# window builds by dividing by this (voxa/window.py: ``self._latest_level / 4000.0``).
LEVEL_SCALE = 4000.0

# Sounds Whisper writes for hesitation, breathing and room noise rather than a word. A transcript
# made only of these is not a person talking over Voxa (issue #130 self-interruption).
_FILLER_TRANSCRIPT = {
    "uh", "uhh", "um", "ummm", "er", "err", "eh", "hmm", "hmmm", "huh", "ah", "aha", "ahem",
    "mm", "mmm", "oh", "ooh", "oooh", "hm", "hmph", "psst", "tsk", "laugh", "sigh",
}


def _has_interrupt_words(heard: str, raw: str, wake_word: str = "voxa") -> bool:
    """Does the transcript hold a real word to interrupt Voxa with?

    Empty, a transcript that is only the wake word, or one made only of hesitation/noise markers
    (``[noise]``, ``(...)``, ``uh``, ``hmm``) is not a person talking over her reply, so it must
    never be the reason to interrupt. Real words - including a bare "stop" - always are.
    """
    combined = f"{heard} {raw}".strip()
    if not combined:
        return False
    # A bracketed noise/silence marker ("[noise]", "(inaudible)") carries no word.
    stripped = re.sub(r"[\[\(\{][^\]\)\}]*[\]\)\}]", " ", combined)
    wake = wake_word.casefold().strip()
    # The wake word alone is a cue, not an interruption.
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", stripped) if w.casefold() != wake]
    return any(w.lower() not in _FILLER_TRANSCRIPT for w in words)


def _text_echo_veto(heard: str, raw: str, wake_word: str, speaking: str) -> bool:
    """Second guard behind echo cancellation (issue #72): a leak through the canceller that
    matches what she is SAYING right now is her own echo, not a barge-in.

    DTLN removes almost all of her voice; a few words leak through the mask ("5 times more."
    from "The answer is: 5 times 4 is 20."). This veto is deliberately narrow: the words must
    overlap heavily with the current utterance AND sound like it letter for letter. A stop word
    or a bare wake word always beats the text (task A2: a person saying "stop" over her saying
    "stop" must always interrupt), and the check only runs after audio failed to prove own
    voice - it is the second guard, never the first.
    """
    if not speaking or not (heard or "").strip():
        return False
    combined = f"{heard} {raw}"
    if strip_wake_word(combined, wake_word) is not None:
        return False
    if any(word in STOP_WORDS for word in _words(combined)):
        return False
    return (overlap(heard, speaking) >= TEXT_ECHO_OVERLAP
            and sounds_like(heard, speaking) >= TEXT_ECHO_SOUNDS)


@dataclass(slots=True)
class BargeOutcome:
    """One barge-in observation: what to do, and the evidence that decided it."""

    action: str
    reason: str = ""
    heard: str = ""
    raw: str = ""
    level: float = 0.0
    gate_opened: bool = False
    listening_seconds: float = 0.0
    # Which source the evidence supported, how strongly, and whether the path was degraded
    # (traced to barge.log and recorded by the bench; the reason a span was kept or dropped).
    source: str = ""
    confidence: float = 0.0
    degraded: bool = False
    # True on the observation where speech onset is confirmed: independent speech onset.
    # The caller pins the captured span from here (voxa/conversation.py:note_onset), which is
    # what keeps the beginning of an interruption alive while the words are still being judged.
    onset: bool = False
    # Where the run of unexplained sound started, how much of it there is (milliseconds of audio,
    # handoff 5.4), and how far the microphone stood over the speaker path.
    onset_audio_ms: float = 0.0
    evidence_ms: float = 0.0
    margin_db: float = 0.0
    # Roadmap task 8 (f174): the joint double-talk gate's suspect verdict for the frame this
    # decision was made on - True suspect, False measured echo, None could-not-measure. The window
    # uses it to DUCK her own stream on the suspect (never on a measured echo, never on a degraded
    # frame) before deciding to yield.
    joint_suspect: bool | None = None
    # Stamped by ``voxa/hearing.py`` with the generation this decision belongs to: a decision
    # that crossed a yield is not applied at all (see ``HearingDirector.accept``).
    generation: int = 0


class BargeInController:
    """State of the barge-in decision: the level gate, the grace and listening windows,
    the text Voxa is speaking (for the own-voice check) and the trace bookkeeping.

    ``observe`` is called once per level sample (the window does it every 50 ms, the bench
    once per 20 ms frame) and returns what should happen; the caller carries out the
    effects, because only the caller can stop speech or arm a prompt.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        listen_ms: float = BARGE_IN_LISTEN_MS,
        gate: BargeInGate | None = None,
        onset: OnsetDetector | None = None,
        trace: Callable[..., None] | None = None,
    ) -> None:
        self.clock = clock
        self.listen_ms = listen_ms
        self.gate = gate if gate is not None else BargeInGate()
        # Speech onset, in milliseconds of audio (voxa/onset.py). Live from the first frame of
        # every reply: no grace period hides the beginning of her speech any more.
        self.onset = onset if onset is not None else OnsetDetector()
        self.trace = trace
        # When the current reply started being read out (0.0 = not speaking).
        self.speaking_since = 0.0
        # When the candidate started being collected (0.0 = not pending), and how much AUDIO
        # has been collected since - the listening window is counted in milliseconds of sound,
        # not in callbacks.
        self.pending_since = 0.0
        self.pending_audio_ms = 0.0
        # How far the microphone stood above her own level at the moment the gate opened. The gate
        # watches 160 ms of samples and the words are judged about a second later, so the level of
        # the judging frame is not the level of the sound that made us listen.
        self.pending_ratio: float | None = None
        # Per-reply counters, traced to barge.log so a failed interruption can be explained.
        self.stats: dict = {}
        # What Voxa is saying now and the reply before it: the own-voice check's evidence.
        self.speaking_text = ""
        self.recent_spoken = ""
        # Issue #72, task v08: the reply she is reading out right now is a TURN - every piece of
        # it is one thing she said. The veto reads the turn's words (echo lands ~1 s late, against
        # the PREVIOUS piece), and the ``speaking`` signal spans the turn until its echo tail.
        self.turn_started_at = 0.0
        self.turn_ended_at = 0.0
        self._turn_log: list[tuple[float, str]] = []
        self._previous_turn_log: list[tuple[float, str]] = []
        # Has the floor already been yielded for this reply? Yielding happens once: a second
        # run of unexplained sound inside the same reply is the same interruption, not a new one.
        self._yielded = False

    def _clear_candidate(self) -> None:
        """Drop the pending candidate: the gate's streak and the level ratio it opened on.

        The ratio belongs to the sound that made us listen; once that sound is over, a new one
        has to earn it again.
        """
        self.pending_since = 0.0
        self.pending_ratio = None
        self.pending_audio_ms = 0.0

    def onset_level_floor(self) -> float:
        """The microphone level that separates speech from silence for THIS reply, in the
        caller's level units (the window and the bench divide the capture level by LEVEL_SCALE
        before feeding the gate, so the floor comes back scaled the same way).

        Used to walk the capture ring back over the run of speech that an onset belongs to,
        which is what recovers the first word of an interruption that started before the gate
        had time to open.
        """
        return max(self.gate.min_level, self.gate.ratio * self.gate.baseline) * LEVEL_SCALE

    def _open_candidate(self, now: float, level01: float, stats: dict) -> None:
        """Start collecting the candidate: the sound that made us listen is remembered."""
        self.pending_since = now
        self.pending_audio_ms = 0.0
        stats["gate_opened"] = stats.get("gate_opened", 0) + 1
        self.pending_ratio = self.gate.peak / max(self.gate.baseline, self.gate.min_level)
        self._trace("gate-open", level=level01, baseline=float(self.gate.baseline),
                    settled=self.gate.settled, ratio=round(self.pending_ratio, 2))

    def _trace(self, event: str, **fields) -> None:
        if self.trace is None:
            return
        try:
            self.trace(event, **fields)
        except Exception:  # noqa: BLE001 - tracing must never disturb the conversation
            pass

    def note_speaking(self) -> None:
        """Speech started being read out: the gate's settling window and the onset evidence restart.

        Clearing the onset detector is what makes it live from the FIRST frame of this reply
        instead of carrying a stale run of sound over from the previous one; clearing the gate's
        settling is what keeps her speaker's own onset from teaching the level gate that her
        voice is loud.
        """
        self.speaking_since = self.clock()
        self.gate.note_reply_start()
        self.onset.reset()
        self._yielded = False

    def reply_started(self, text: str) -> None:
        """A reply began being read out (window.py:3004-3010 did exactly this before)."""
        self.recent_spoken = self.speaking_text
        self.speaking_text = text
        self.note_speaking()

    def note_turn(self, text: str = "") -> None:
        """The reply's FIRST ``speak-start`` opens one turn (issue #72, task v08).

        Every piece of that reply belongs to it: the text veto reads the whole turn (the echo of
        piece 1 arrives while piece 2 plays), and the turn's words stay the evidence until the
        NEXT reply rolls them into ``_previous_turn_log`` for :data:`VETO_PREVIOUS_SECONDS`.
        """
        self.turn_started_at = self.clock()
        self.turn_ended_at = 0.0
        self._previous_turn_log = self._turn_log
        self._turn_log = [(self.turn_started_at, text)] if text else []

    def note_piece(self, text: str = "") -> None:
        """A later piece of the SAME reply began being read out."""
        if text:
            self._turn_log.append((self.clock(), text))

    def turn_veto_text(self) -> str:
        """Everything she said in the current turn, plus the last 10 s of the previous one."""
        now = self.clock()
        parts = [text for _started, text in self._turn_log]
        parts += [text for started, text in self._previous_turn_log
                  if now - started <= VETO_PREVIOUS_SECONDS]
        return " ".join(parts).strip()

    def speaking_now(self) -> bool:
        """True from the turn's first speak-start until its last audio end plus the echo tail.

        This is the ``speaking`` flag i02 marks the reference with: her voice is still arriving in
        the microphone for :data:`ECHO_TAIL_SECONDS` after the last piece left the speakers, and the
        gaps between pieces are hers too - the moments that read silent are exactly what made her
        answer herself.
        """
        if self.turn_started_at > 0.0 and self.turn_ended_at <= 0.0:
            return True
        return (self.turn_ended_at > 0.0
                and self.clock() < self.turn_ended_at + ECHO_TAIL_SECONDS)

    def _turn_text_echo_veto(self, heard: str, raw: str, wake_word: str) -> bool:
        """Issue #72, task v08: the second guard reads her WHOLE TURN, not just the current piece.

        The echo lands about a second late - the heard words are usually the PREVIOUS piece's
        text, which the per-piece veto never saw. But the heard line only dies to the veto while
        MOST of its words really are hers: a real question that shares her function words (the
        logged "What is the capital of France?" at overlap 0.68) survives the guard and interrupts.
        """
        turn_text = self.turn_veto_text() or self.speaking_text
        if not _text_echo_veto(heard, raw, wake_word, turn_text):
            return False
        hers, foreign = _foreign_core(f"{heard} {raw}", turn_text, wake_word)
        return foreign < hers and foreign <= _VETO_FOREIGN_CONTENT_MAX

    @staticmethod
    def _reference_unusable(decision) -> bool:
        """Did attribution abstain because the REFERENCE could not be trusted (not because the
        audio was weak)? Issue #72, task v11 step 3: the two degraded states that hold forever -
        a silent tap while she speaks, and an unhealthy alignment - are exactly these reasons."""
        return any("silent tap, abstain" in reason or "alignment unhealthy" in reason
                   for reason in decision.reasons)

    def _degraded_turn_promote(self, heard: str, raw: str, wake_word: str,
                               evidence: SourceEvidence | None = None) -> bool:
        """Issue #72, task v11 step 3: while she speaks and the reference is UNUSABLE, promote by
        the turn-wide text veto. Heard CONTENT words that are not in her whole turn are the user;
        >= 3 of them, or the wake word plus >= 2, is a barge-in. Her own words (few foreign) never
        clear it, so an echo of her reply still never interrupts her.

        Task v11c step 5: the OWN VOICE is decided by AUDIO. When the microphone energy this frame
        IS explained by the render envelope at the alignment's delay (the same number attribution
        used before it abstained), the text veto must not promote: text is the fallback for an
        unusable reference, never a way to interrupt audio that already proved itself echo.
        """
        if (evidence is not None and evidence.alignment_healthy
                and evidence.echo_explained_db is not None
                and evidence.echo_explained_db >= ECHO_EXPLAINED_DB):
            return False
        turn_text = self.turn_veto_text() or self.speaking_text
        if not turn_text or not (heard or raw):
            return False
        _hers, foreign = _foreign_core(f"{heard} {raw}", turn_text, wake_word)
        wake_present = strip_wake_word(f"{heard} {raw}", wake_word) is not None
        if wake_present:
            return foreign >= DEGRADED_FOREIGN_WITH_WAKE
        return foreign >= DEGRADED_FOREIGN_MIN

    def reply_ended(self) -> None:
        """The reply finished, was stopped or the mode closed (window.py:3060-3061 and friends)."""
        self.speaking_since = 0.0
        if self.turn_started_at > 0.0:
            # The reply's audio ended; her echo is still in the microphone for the tail.
            self.turn_ended_at = self.clock()
        self.gate.reset()
        self.onset.reset()
        self._yielded = False

    def invalidate(self) -> None:
        """The floor was yielded: the evidence starts over, the reply's words do not.

        The gate's baseline and the onset detector's run of unexplained sound are cleared, and
        the once-per-reply yield flag is released for the NEXT reply. ``speaking_text`` and
        ``recent_spoken`` stay: the words she was saying are still the evidence the own-voice
        check is judged against, and a rejected interruption has to be rejected against them.
        """
        self.gate.reset()
        self.onset.reset()
        self._yielded = False

    def observe(
        self,
        level01: float,
        *,
        speaking: bool,
        paused: bool = False,
        muted: bool = False,
        wake_word: str = "voxa",
        media_just_started: bool = False,
        candidate: Callable[[], tuple[str, str]] | None = None,
        evidence: Callable[[], SourceEvidence] | SourceEvidence | None = None,
        mic_db: float | None = None,
        reference_db: float | None = None,
        delay_ms: float | None = None,
        dt_ms: float = FRAME_MS,
    ) -> BargeOutcome:
        """Judge one level sample while a reply may be playing.

        ``level01`` is the microphone level on the 0..1 scale the gate uses (the window
        divides ``_latest_level`` by 4000). ``candidate`` returns ``(heard, raw)``: the
        transcript of the audio captured while muted, after and before the own-speech
        filter. ``speaking``, ``paused`` and ``muted`` are the caller's state, read the
        way ``_maybe_barge_in`` read them. ``evidence`` is the capture path's
        :class:`voxa.attribution.SourceEvidence` for this moment (a callable is evaluated
        only once the words are actually judged); ``None`` means the caller has no reference
        to compare the microphone with, which puts the decision in its degraded text-only
        form.

        ``mic_db`` and ``reference_db`` are the microphone level and the reference level of this
        same moment, BOTH in the same dB scale (see ``voxa/onset.py``: an int16 RMS has to be
        brought to dBFS with ``INT16_DB_OFFSET`` before it can be compared with a float reference),
        and they are the inputs of the onset rule - the reason onset detection needs no grace period
        is that the speaker path says what it could have put in the microphone, and sound above
        that ceiling is not her echo. ``delay_ms`` is that path's echo travel time, or ``None``
        when it was never estimated, which turns the onset claim off rather than on. ``dt_ms`` is
        how much audio this sample stands for; every window in this controller is counted in those
        milliseconds.
        """
        now = self.clock()
        dt = max(0.0, float(dt_ms))
        stats = self.stats
        onset_open = False
        if speaking != stats.get("speaking", False):
            if not speaking:
                self._trace("reply-end", **{key: value for key, value in stats.items() if key != "speaking"})
            stats.clear()
            stats["speaking"] = speaking
        if speaking:
            stats["calls"] = stats.get("calls", 0) + 1
        if paused:
            if speaking:
                stats["paused_ticks"] = stats.get("paused_ticks", 0) + 1
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(IGNORED, reason="paused", level=level01)
        if not muted:
            if speaking:
                stats["not_muted_ticks"] = stats.get("not_muted_ticks", 0) + 1
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(IGNORED, reason="not-muted", level=level01)
        if not speaking:
            return BargeOutcome(IGNORED, reason="not-speaking", level=level01)
        if self.pending_since > 0.0:
            # The listening window is a duration of AUDIO, not a number of callbacks.
            self.pending_audio_ms += dt
        stats["ticks"] = stats.get("ticks", 0) + 1
        stats["max_level"] = max(stats.get("max_level", 0.0), level01)
        stats["baseline"] = float(self.gate.baseline)
        self.gate.note_reply_audio(dt)
        opened = self.gate.update(level01, self.gate.baseline, dt)
        # Roadmap task 5: with a live reference, the JOINT double-talk gate is the opening rule -
        # the raw level ratio is what interrupted her on her own loudest syllables. The gate
        # still tracks levels (baseline teaching, the fallback below), but the candidate window
        # opens when the three audio streams agreed, not when the microphone towers.
        audio = evidence() if callable(evidence) else evidence
        joint = audio.joint_suspect if audio is not None else None
        if joint is not None and audio.reference_active:
            if joint:
                opened = True
            elif getattr(audio, "joint_echo", False):
                # Only the gate's MIRROR verdict may suppress the fallback: the microphone held
                # the delayed reference, so the tower is her own voice. "Not suspect" alone is
                # indeterminate, and barge-ins the joint window has not confirmed keep the floor.
                opened = False
                stats["joint_closed"] = stats.get("joint_closed", 0) + 1
                if self.pending_since > 0.0:
                    # f172: the mirror verdict closes a candidate that the fallback level gate
                    # already opened: the span it is collecting is her echo, and every frame
                    # of it stays in the capture ring (evidence, never deletion). If a real
                    # second source follows, the suspect or the fallback opens the window
                    # again while its audio is still there.
                    self._clear_candidate()
                    stats["joint_closed_candidate"] = stats.get("joint_closed_candidate", 0) + 1
        if self.gate.listening and self.pending_since <= 0.0 and not opened:
            # Still her voice (or noise): teach the gate what "normal" is.
            self.gate.note_speaking_level(level01, dt)
        # --- speech onset: milliseconds of audio the speaker path cannot explain -----------------
        self.onset.note_delay(delay_ms)
        onset = self.onset.observe(mic_db, reference_db, dt)
        if onset.onset and not self._yielded and joint is False and audio.reference_active:
            # f172 (roadmap task 5): the fast yield is reference-aware now. The onset rule says
            # the microphone towers over what the speaker path could have produced, but the
            # joint double-talk gate MEASURED this frame and did not find a second source -
            # the cleaned stream's speech, its margin over the predicted residual and the
            # reference at the locked delay never held together. The floor stays: the level
            # gate and the words carry on (the candidate path below), and the suspect may
            # still raise the yield on a later frame of the same onset.
            stats["onset_blocked"] = stats.get("onset_blocked", 0) + 1
            self._trace("onset-blocked", evidence_ms=round(onset.evidence_ms, 1),
                        margin_db=round(onset.margin_db, 2),
                        onset_audio_ms=round(onset.onset_audio_ms, 1),
                        level=round(level01, 4), speaking=self.speaking_text[:60])
        elif onset.onset and not self._yielded:
            # The fast half of the decision (task A3): stop her playback NOW, before a single
            # word has been transcribed. The words are still being collected, and the outcome
            # that decides what happens to them comes later.
            self._yielded = True
            if self.pending_since <= 0.0:
                self._open_candidate(now, level01, stats)
            stats["onset"] = stats.get("onset", 0) + 1
            self._trace("onset", evidence_ms=round(onset.evidence_ms, 1),
                        margin_db=round(onset.margin_db, 2),
                        onset_audio_ms=round(onset.onset_audio_ms, 1),
                        level=round(level01, 4), speaking=self.speaking_text[:60])
            return BargeOutcome(YIELD, reason="onset", level=level01, onset=True, gate_opened=True,
                                onset_audio_ms=onset.onset_audio_ms,
                                evidence_ms=onset.evidence_ms, margin_db=onset.margin_db,
                                joint_suspect=joint)
        if self.pending_since <= 0.0:
            if not opened:
                return BargeOutcome(IGNORED, reason="gate-closed", level=level01)
            self._open_candidate(now, level01, stats)
            onset_open = True
        # Always judge the words as well, on the path where no reference exists to compare the
        # microphone with (echo off, no reference sink). A loudness shortcut was tried and
        # interrupted her on her own loudest syllables; the user is given a word or two first.
        if self.pending_audio_ms < self.listen_ms:
            return BargeOutcome(IGNORED, reason="listening", level=level01, gate_opened=True,
                                onset=onset_open)
        started = self.clock()
        heard, raw = (candidate() if candidate is not None else ("", ""))
        # A spoken OFFLINE command is a kill switch even over Voxa's reply, and short
        # phrases such as "go offline" otherwise read as her own audio.
        if detect_addressed_exit_phrase(heard, wake_word) == "goodbye":
            self._trace("offline-interrupt", heard=heard, raw=raw)
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(OFFLINE, heard=heard, raw=raw, level=level01)
        decision = source_decision(heard, self.speaking_text, self.recent_spoken, raw=raw,
                                   evidence=audio,
                                   level_ratio=(self.pending_ratio
                                                if self.pending_ratio is not None else
                                                self.gate.peak / max(self.gate.baseline,
                                                                     self.gate.min_level)))
        self._trace("words", heard=heard, raw=raw, source=decision.source,
                    confidence=round(decision.confidence, 2), degraded=decision.degraded,
                    reasons="; ".join(decision.reasons)[:200],
                    seconds=self.clock() - started, speaking=self.speaking_text[:60])
        if decision.source == SOURCE_UNCERTAIN:
            # Issue #72, task v11 step 3: an abstain is not permanent. When the REFERENCE is
            # unusable (silent tap while she speaks, or an unhealthy alignment) attribution can
            # only ever hold the span back, so a real barge-in could never be credited to the
            # user. While she speaks, decide by the turn-wide text veto instead: heard words that
            # are not in her whole turn (>= 3, or the wake word plus >= 2) ARE the user ->
            # INTERRUPT. Her own words never clear the bar, so they fall through to the HOLD.
            if (self.turn_started_at > 0.0 and self.turn_ended_at <= 0.0
                    and self._reference_unusable(decision)
                    and _has_interrupt_words(heard, raw, wake_word)
                    and self._degraded_turn_promote(heard, raw, wake_word,
                                                    evidence=decision.evidence)):
                self._trace("words", heard=heard, raw=raw, source="user",
                            confidence=0.0, degraded=True,
                            reasons="degraded reference: turn-wide text veto promoted to user",
                            seconds=self.clock() - started, speaking=self.speaking_text[:60])
                self.gate.reset()
                self._clear_candidate()
                self.speaking_since = 0.0
                if self.turn_started_at > 0.0 and self.turn_ended_at <= 0.0:
                    self.turn_ended_at = self.clock()
                return BargeOutcome(INTERRUPT, heard=heard, raw=raw, level=level01,
                                    listening_seconds=self.clock() - started,
                                    source=USER, confidence=0.0, degraded=True,
                                    joint_suspect=joint)
            # Not her voice and not proven to be a person: hold. The captured span stays pinned
            # for its evidence window, nothing is stripped from the transcript, and no command
            # runs from audio this weak (handoff 6.5: an immediate output change for every raw
            # VAD event would reintroduce self-interruption). The gate is cleared so the candidate
            # is judged again on the next run of loud frames rather than staying stuck open.
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(HOLD, reason="uncertain", heard=heard, raw=raw, level=level01,
                                source=decision.source, confidence=decision.confidence,
                                degraded=decision.degraded, joint_suspect=joint)
        own = decision.source == ECHO
        if (
            not own
            and media_just_started
            and not _MEDIA_EXCEPTION.search(f"{heard} {raw}".casefold())
            and strip_wake_word(f"{heard} {raw}", wake_word) is None
        ):
            # The station she is announcing has already started: its presenter is not the user.
            own = True
        # Issue #72, step 4 + task v08: the canceller leaves a few leaked words of her own voice in
        # the microphone, and this second guard reads their text against her WHOLE TURN - the echo
        # of piece 1 arriving while piece 2 plays is her echo, never a barge-in - but a stop word
        # or a bare wake word always beats the text (task A2), and so do mostly-foreign words.
        if not own and self._turn_text_echo_veto(heard, raw, wake_word):
            own = True
        if own:
            # The loud sound was Voxa's own voice: carry on speaking (unless the floor was
            # already yielded by the onset, in which case the playback stopped and only the
            # words are rejected). This is the only branch that may strip words, and it is
            # reached on audio evidence - never because the transcript is short or happens to
            # contain a word she is saying herself (task A2: her saying "stop" alone must never
            # stop her, and a person saying "stop" over her saying "stop" must always stop her).
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(OWN_VOICE, heard=heard, raw=raw, level=level01,
                                source=decision.source, confidence=decision.confidence,
                                degraded=decision.degraded, joint_suspect=joint)
        if not _has_interrupt_words(heard, raw, wake_word):
            # A barge-in must never fire on an empty, wake-word-only or filler/noise transcript
            # (rhY, issue #130). The traced self-interruption was a level rise with heard='' that
            # a silent reference misread as the user: there were no words to interrupt with, so
            # this is a HOLD (keep speaking, keep the pinned span, judge again), never an interrupt.
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(HOLD, reason="no-words", heard=heard, raw=raw, level=level01,
                                source=decision.source, confidence=decision.confidence,
                                degraded=decision.degraded, joint_suspect=joint)
        self.gate.reset()
        self._clear_candidate()
        self.speaking_since = 0.0
        if self.turn_started_at > 0.0 and self.turn_ended_at <= 0.0:
            # Her playback stops now; its echo is still in the microphone for the tail.
            self.turn_ended_at = self.clock()
        return BargeOutcome(INTERRUPT, heard=heard, raw=raw, level=level01,
                            listening_seconds=self.clock() - started,
                            source=decision.source, confidence=decision.confidence,
                            degraded=decision.degraded, joint_suspect=joint)
