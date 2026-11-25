"""Decide whether a loud sound in the microphone is the user interrupting.

Pure and model-free: word overlap for the own-voice check, and a gate that
works from sound levels alone before any transcription exists.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

OWN_VOICE_OVERLAP = 0.6
BASELINE_ALPHA = 0.05
# Every window the gate keeps is a duration of AUDIO, not a number of UI callbacks (handoff 5.4:
# "express evidence accumulation in milliseconds rather than UI callback counts"). A 50 ms UI timer
# and a 20 ms bench frame therefore accumulate the same evidence for the same amount of sound.
FRAME_MS = 20.0        # what one sample stands for when the caller does not say (the bench frame)
SETTLED_MS = 400.0     # of her own voice the baseline must hear before it can be trusted
WINDOW_MS = 160.0      # the run of audio the gate watches: syllable dips live inside it
LOUD_MS = 80.0         # of that window must tower over the baseline for the gate to open
# Kept so callers and tests that still count samples keep working: both are the same audio.
SETTLED_SAMPLES = int(SETTLED_MS / FRAME_MS)
WINDOW = int(WINDOW_MS / FRAME_MS)



def _words(text: str) -> list[str]:
    return [word for word in re.findall(r"[^\W\d_]+", text.lower()) if len(word) >= 2]


def overlap(heard: str, speaking: str) -> float:
    """Fraction of the words in `heard` (lower-cased, punctuation stripped,
    words of 2+ letters) that also occur in `speaking`; 0.0 when `heard` has
    no such words."""
    heard_words = _words(heard)
    if not heard_words:
        return 0.0
    speaking_words = set(_words(speaking))
    hits = sum(1 for word in heard_words if word in speaking_words)
    return hits / len(heard_words)


# Words that ask her to DO something. "voxa" is the wake word, not a command, and is scored
# separately: hearing her name is a weaker cue than hearing "stop".
STOP_WORDS = frozenset({"stop", "wait", "cancel", "pause", "quiet", "enough", "no", "hey", "silence", "shush"})


def _letters(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def sounds_like(heard: str, spoken: str) -> float:
    """How closely `heard` matches some stretch of `spoken`, letter by letter (0..1).

    Whisper often mishears Voxa's own voice into different WORDS that keep most of the LETTERS
    ("BLC doesn't seem to be open" came back as "Bmc, done, themes"), so words alone cannot tell
    her voice from a person's.
    """
    h, s = _letters(heard), _letters(spoken)
    if not h or not s:
        return 0.0
    if len(s) <= len(h):
        return SequenceMatcher(None, h, s).ratio()
    best = 0.0
    step = max(1, len(h) // 4)
    for width in (len(h), int(len(h) * 1.3) + 1):
        for start in range(0, max(1, len(s) - width + 1), step):
            best = max(best, SequenceMatcher(None, h, s[start:start + width]).ratio())
    return best


OWN_VOICE_SOUND = 0.5
# Below this many letters, a letter match against her sentence is noise: any three letters occur
# inside any sentence ("yes" matched inside "quarterly report"), and a brief user answer is then
# scored as her echo - the exact failure handoff 5.2 says not to make.
LETTER_FEATURE_MIN = 5
# Words other than interrupt words that are mostly hers mark an echo; with fewer than four of them
# the letters must match closely too.
ECHO_REST_OVERLAP = 0.75
ECHO_REST_SOUND = 0.75


def text_features(heard: str, speaking_now: str, recently_spoken: str = "", raw: str = "") -> dict:
    """The WORDS as features: how much they look like her own sentence, plus the cues.

    These are inputs to a weighted score, never a decision on their own, and the NUMBER of words
    is not among them: handoff 5.2 shows that "yes", "blue" and "that one" are valid brief
    answers, so a short transcript is uncertainty, not echo.
    """
    from .attribution import SourceEvidence

    spoken_text = f"{speaking_now} {recently_spoken}"
    words = _words(heard)
    spoken = set(_words(spoken_text))
    # The LETTER feature needs enough letters to be specific: any three letters can be found
    # inside any sentence, so below this size the match is noise, not evidence. A reliability guard
    # on one feature - it decides nothing by itself, and the word count is not a feature.
    letters = _letters(raw or heard)
    sound_source = (raw or heard) if len(letters) >= LETTER_FEATURE_MIN else ""
    def hers(word: str) -> bool:
        # She says this word herself ("Stopping dictation" heard as "Stop", her own "Voxa" in the
        # prompt hint's prefix): a word she is saying cannot be a cue that someone else is talking.
        return word in spoken or any(s.startswith(word) for s in spoken)

    stop = [w for w in words if w in STOP_WORDS and not hers(w)]
    rest = [w for w in words if w not in STOP_WORDS]
    wake = [w for w in words if w.startswith("voxa") and not hers(w)]
    rest_overlap = overlap(" ".join(rest), spoken_text) if rest else 0.0
    rest_sound = sounds_like(" ".join(rest), spoken_text) if len(_letters(" ".join(rest))) >= LETTER_FEATURE_MIN else 0.0
    sentence_echo = bool(len(rest) >= 2 and rest_overlap >= ECHO_REST_OVERLAP and
                         (len(rest) >= 4 or rest_sound >= ECHO_REST_SOUND))
    return {
        "evidence": SourceEvidence(
            text_overlap=max(overlap(heard, spoken_text), rest_overlap if sentence_echo else 0.0),
            text_sound=max(sounds_like(sound_source, spoken_text), rest_sound if sentence_echo else 0.0),
            text_letters_measurable=bool(len(letters) >= LETTER_FEATURE_MIN),
            word_count=len(words),
            stop_word=bool(stop),
            wake_word=bool(wake),
        ),
        # "her own sentence with an interrupt word inside it" (traced 14 October): a text feature
        # with a heavier weight, still not a decision by itself.
        "sentence_echo": sentence_echo,
        "stop_in_own_speech": bool(any(w in spoken or any(s.startswith(w) for s in spoken)
                                        for w in words if w in STOP_WORDS)),
    }


def source_decision(heard: str, speaking_now: str, recently_spoken: str = "", raw: str = "",
                    evidence=None, level_ratio: float | None = None) -> "SourceDecision":
    """The three-way answer for one candidate: user, echo, or undecided.

    ``bargectl`` uses this directly, because "uncertain" is an outcome of its own: the audio and
    the transcript are kept, nothing is stripped, and no command runs from it.
    """
    from .attribution import attribute

    features = text_features(heard, speaking_now, recently_spoken, raw=raw)
    ev = features["evidence"]
    if features["sentence_echo"]:
        ev.text_overlap = max(ev.text_overlap, ECHO_REST_OVERLAP)
        ev.text_sound = max(ev.text_sound, ECHO_REST_SOUND)
    if evidence is None:
        # No capture-path evidence: a reply IS being read out (that is the only state in which
        # the words are judged), so the reference is live and the audio streams are simply
        # missing - the degraded path inside ``attribute``, reported as degraded.
        ev.reference_active = True
        ev.level_ratio = level_ratio
        return attribute(ev)
    merged = _merge(ev, evidence)
    if merged.level_ratio is None:
        merged.level_ratio = level_ratio
    return attribute(merged)


def is_own_voice(heard: str, speaking_now: str, recently_spoken: str = "", raw: str = "",
                 evidence=None) -> bool:
    """True when the evidence says the sound is (a mishearing of) what Voxa is saying.

    ``evidence`` is a :class:`voxa.attribution.SourceEvidence` filled in by the caller from the
    capture path. Without it there is no reference to compare the microphone with, so the words
    are all there is: the degraded text-only path may answer ``echo`` or ``uncertain``, and
    ``uncertain`` is NOT "her voice" - it means "do not decide", which is why the old
    ``len(words) < 3 -> own voice`` rule is gone (handoff 5.2: short length is uncertainty).
    """
    from .attribution import ECHO

    return source_decision(heard, speaking_now, recently_spoken, raw=raw,
                           evidence=evidence).source == ECHO


def _merge(text_ev, audio_ev):
    """Combine the caller's audio evidence with the words' features into one evidence record."""
    from .attribution import SourceEvidence

    fields = {name: getattr(audio_ev, name) for name in audio_ev.__slots__}
    for name in ("text_overlap", "text_sound", "word_count", "stop_word", "wake_word",
                 "text_letters_measurable"):
        value = getattr(text_ev, name)
        if name in ("text_overlap", "text_sound"):
            fields[name] = max(float(fields[name]), float(value))
        elif name == "text_letters_measurable":
            # The caller's evidence has no words of its own: this feature comes from the transcript.
            fields[name] = bool(value)
        else:
            fields[name] = bool(fields[name]) or bool(value)
    return SourceEvidence(**fields)



class BargeInGate:
    """Opens once `LOUD_MS` of the last `WINDOW_MS` of audio tower over the level heard
    while Voxa speaks; a quieter frame only lowers the count, it does not reset."""

    def __init__(self, ratio: float = 2.0, min_level: float = 0.02, needed_ms: float = LOUD_MS,
                 window_ms: float = WINDOW_MS, settled_ms: float = SETTLED_MS) -> None:
        self._ratio = ratio
        self._min_level = min_level
        self._needed_ms = needed_ms
        self._window_ms = window_ms
        self._settled_ms = settled_ms
        self._loud_ms = 0.0
        self._recent: list[tuple[bool, float]] = []  # (loud?, ms of audio), bounded by window_ms
        self._levels: list[tuple[float, float]] = []
        self._baseline = 0.0
        self._settled_audio_ms = 0.0
        # How much of THIS reply has been heard: the first SETTLED_MS of it teaches nothing.
        self._reply_audio_ms = 0.0

    def update(self, mic_level: float, baseline: float, dt_ms: float = FRAME_MS) -> bool:
        if not self.listening:
            # The speaker's own onset window: the gate hears nothing at all (see note_reply_audio).
            return False
        # Speech dips between syllables, so "six loud ticks in a row" almost never happened for a
        # short "Voxa, stop" (traced: 374 ticks at full level, gate never opened). Count the loud
        # audio inside the last window instead.
        dt = max(0.0, float(dt_ms))
        loud = mic_level > max(self._min_level, self._ratio * baseline)
        self._recent.append((loud, dt))
        self._levels.append((mic_level, dt))
        self._trim()
        self._loud_ms = sum(ms for flag, ms in self._recent if flag)
        # Never open before the baseline has heard enough of her own voice to be trusted:
        # at session start the baseline is 0.0, so any loud sound would open the gate.
        return self.settled and self._loud_ms >= self._needed_ms

    def note_speaking_level(self, mic_level: float, dt_ms: float = FRAME_MS) -> None:
        self._baseline += BASELINE_ALPHA * (mic_level - self._baseline)
        self._settled_audio_ms += max(0.0, float(dt_ms))

    def note_reply_audio(self, dt_ms: float = FRAME_MS) -> None:
        """One more frame of THIS reply, whatever it sounded like: the settling window advances.

        The first ``SETTLED_MS`` of a reply is her speaker's onset - the loudest moment of her
        own voice - so during it the gate hears nothing at all: it neither opens nor learns. The
        old code expressed the same window as ``BARGE_IN_GRACE_SECONDS`` of wall clock and applied
        it to the whole decision; here it is milliseconds of AUDIO and belongs to the LEVEL GATE
        only. The fast path (``voxa/onset.py``) has no window and judges the FIRST frame of the
        reply, which is where an interruption that starts together with her speech lives.
        """
        self._reply_audio_ms += max(0.0, float(dt_ms))

    @property
    def listening(self) -> bool:
        """True once the reply's speaker-onset window is over and the gate is hearing again."""
        return self._reply_audio_ms > self._settled_ms

    def note_reply_start(self) -> None:
        """A reply began being read out: the onset window, the loud run and the settling restart.

        The baseline survives - it describes what Voxa's own voice sounds like in this room, and
        a new reply does not change that - but it is not retrained from her speaker's onset.
        """
        self._reply_audio_ms = 0.0
        self._settled_audio_ms = 0.0
        self._loud_ms = 0.0
        self._recent = []
        self._levels = []

    def _trim(self) -> None:
        """Drop the oldest samples until the stored audio fits the window again."""
        total = sum(ms for _flag, ms in self._recent)
        while self._recent and total > self._window_ms:
            _flag, ms = self._recent.pop(0)
            total -= ms
        total = sum(ms for _level, ms in self._levels)
        while self._levels and total > self._window_ms:
            _level, ms = self._levels.pop(0)
            total -= ms

    @property
    def settled(self) -> bool:
        """True once the baseline has heard enough of her own voice to be trusted."""
        return self._settled_audio_ms >= self._settled_ms

    @property
    def baseline(self) -> float:
        return self._baseline

    @property
    def ratio(self) -> float:
        """How far above her own level the microphone must be to count as someone else speaking."""
        return self._ratio

    @property
    def peak(self) -> float:
        """The loudest of the last WINDOW samples.

        The gate stays open across a dip between syllables, so the level of the frame that happens
        to do the judging is not the level of the sound: "is the microphone towering" has to be
        asked about the run of frames, not the instant.
        """
        return max((level for level, _ms in self._levels), default=0.0)

    @property
    def min_level(self) -> float:
        """The floor below which the microphone is silence, whatever the baseline says."""
        return self._min_level

    def reset(self) -> None:
        # The baseline and the settled audio survive a reset on purpose: they describe what Voxa's
        # own voice sounds like in this room, which one interruption does not change.
        self._loud_ms = 0.0
        self._recent = []


# --------------------------------------------------------------------------- task 8
#
# Roadmap task 8 (report 76-83, handoff 6.5/218-236): an EXPLICIT turn state machine, a
# three-state barge-in decision (suspect -> duck, confirm -> yield, reject -> resume), and the
# conversation-context truncation a yield owes. Everything here is pure: a state, a duration of
# AUDIO, and the words heard. No GTK, no audio device, no system volume. The caller (the live
# app in voxa/window.py, the measurement in voxa/bench/runner.py) performs the effects.

# The application turn states (handoff 6.5:220). These are application states: the capture path
# and the echo canceller run continuously ACROSS them (brief rule: "the canceller runs
# continuously, never reset per sentence"). ``YIELDED`` is entered only by a CONFIRMED
# interruption; ``OVERLAP_PENDING`` is the ducked-while-suspect state the false-pause rate is
# calibrated on. ``PLANNING`` is Voxa preparing a response it has not yet started to speak.
LISTENING = "LISTENING"
USER_TURN = "USER_TURN"
PLANNING = "PLANNING"
SPEAKING = "SPEAKING"
OVERLAP_PENDING = "OVERLAP_PENDING"
YIELDED = "YIELDED"
RECOVERING_AUDIO = "RECOVERING_AUDIO"
TURN_STATES = (LISTENING, USER_TURN, PLANNING, SPEAKING,
               OVERLAP_PENDING, YIELDED, RECOVERING_AUDIO)

# Duck: lower ONLY Voxa's own stream by this many dB over a 20 ms ramp (handoff 6.5/roadmap 76,
# "duck TTS by 10-15 dB ... the ramp is 20 ms"), never the system volume (brief design rule +
# handoff 233: "Do not globally lower system volume as a way to pass tests").
DUCK_DB = 12.0
DUCK_RAMP_MS = 20.0
# A duck is LINEAR gain, not a mute: this is the 0..1 factor 12 dB maps to.
DUCK_GAIN = 10.0 ** (-DUCK_DB / 20.0)

# CONFIRM (report 76:83, task line 10): these decide a real interruption. A bare backchannel is
# never enough on its own; a protected command word is enough on its own.
CONFIRM_MIN_WORDS = 2            # non-backchannel words that, together, confirm
CONFIRM_GATED_SPEECH_MS = 500.0  # of continuous gated speech that confirms on its own
# REJECT (report 76:83, task line 10): these decide a false alarm and resume her.
REJECT_NO_WORDS_MS = 1500.0      # no words at all inside this window of AUDIO
REJECT_NO_SPEECH_PROB = 0.6      # faster-whisper no_speech_prob above this = bracketed non-speech

# Backchannels: brief acknowledgements that normally let her finish (handoff 6.5:227 "normally
# keep speaking"). The yes/no context rule below can turn one of these into an ANSWER.
BACKCHANNELS = frozenset({"yes", "no", "ok", "okay", "thanks", "thank you",
                          "yeah", "uh-huh", "mm-hm", "right", "mhm", "mm"})
# Protected command words: any one of these CONFIRMS on its own (roadmap task 7, task 8).
PROTECTED_COMMANDS = frozenset({"stop", "cancel", "pause", "quiet", "wait", "enough", "shush"})
# "no, the other one" -style corrections (handoff 6.5:235): yield quickly even though the words
# are not stop-word commands. A negation followed by an anaphor pointing elsewhere.
_CORRECTION = re.compile(r"\b(no|not|wait)\b.{0,24}\b(other|instead|wrong|meant|that one|this one)\b")


def _content_words(text: str) -> list[str]:
    return _words(text)


def is_backchannel_only(text: str) -> bool:
    """True when every word is a brief acknowledgement ('mm-hm', 'yeah', 'okay', 'right')."""
    words = _content_words(text)
    return bool(words) and all(w in BACKCHANNELS for w in words)


def is_correction(text: str) -> bool:
    """A 'no, the other one' correction: the user is pushing back, not acknowledging."""
    return bool(_CORRECTION.search((text or "").lower()))


# The three-state barge-in decision, as data (report 76-83). The caller turns an outcome into an
# effect: ``duck`` lowers her own stream, ``yield`` hard-stops it and opens the user turn, ``resume``
# restores the duck, ``hold`` keeps collecting without deciding.
DECISION_SUSPECT = "suspect"   # -> duck
DECISION_CONFIRM = "confirm"   # -> yield / stop
DECISION_REJECT = "reject"     # -> resume


def decide_bargein(heard: str, *, gated_speech_ms: float, no_speech_prob: float | None,
                   asked_yes_no: bool = False, just_spoken: str = "") -> str:
    """The three-state decision for one overlap candidate (report 76-83, task line 10-11).

    CONFIRM (yield) on: >= CONFIRM_MIN_WORDS non-backchannel words, any protected command word, a
    'no, the other one' correction, or >= CONFIRM_GATED_SPEECH_MS of continuous gated speech.
    REJECT (resume) on: backchannels only, a bracketed non-speech token (no_speech_prob above
    REJECT_NO_SPEECH_PROB), or no words at all for REJECT_NO_WORDS_MS of AUDIO. Context decides a
    short 'yes': after Voxa asks a yes/no question it is an ANSWER (CONFIRM), not a backchannel.
    Everything else holds: the audio is kept and the evidence window keeps running.
    """
    words = _content_words(heard)
    # A protected word confirms on its own, whatever else is present.
    if any(w in PROTECTED_COMMANDS for w in words):
        return DECISION_CONFIRM
    if is_correction(heard):
        return DECISION_CONFIRM
    # A short 'yes' is an answer to a question she just asked, not a backchannel (handoff 6.5:235).
    if asked_yes_no and is_backchannel_only(heard):
        return DECISION_CONFIRM
    if is_backchannel_only(heard):
        # A backchannel is a REJECT only once we are sure it is not speech she is mishearing as
        # words of her own; with no extra signal it still lets her finish.
        return DECISION_REJECT
    if words:
        non_back = [w for w in words if w not in BACKCHANNELS]
        if len(non_back) >= CONFIRM_MIN_WORDS:
            return DECISION_CONFIRM
    if no_speech_prob is not None and no_speech_prob > REJECT_NO_SPEECH_PROB:
        return DECISION_REJECT
    # No words at all for the reject window of AUDIO: the gated sound produced nothing to say
    # (bracketed non-speech), so it resumes her - checked BEFORE the gated-speech confirm so a full
    # 1.5 s of wordless gated audio rejects rather than confirms.
    if not words and gated_speech_ms >= REJECT_NO_WORDS_MS:
        return DECISION_REJECT
    if gated_speech_ms >= CONFIRM_GATED_SPEECH_MS:
        return DECISION_CONFIRM
    return DECISION_SUSPECT


# Issue #72, task v11d item 2: Whisper hallucinates fixed phrases on silence, music and room
# noise even when nobody spoke. The one INTERRUPT that survived the v11c no-speech guard was
# ``heard='Thanks for watching.'`` - her own closing phrase, invented out of the echo path's
# residual. These are the transcripts Whisper invents when there is no real speech; a transcript
# that is (essentially) one of them, or that Whisper itself brackets as non-speech with a high
# no_speech_prob, is NOT evidence of a second source UNLESS the cleaned signal really carries
# speech (VAD >= VAD_SPEECH). A genuine user turn with real speech is never dropped by this guard.
HALLUCINATION_PHRASES = frozenset({
    "", ".", "you", "music", "thank you", "thanks", "thanks for watching",
    "thanks for listening", "thank you for watching", "thank you very much",
    "like and subscribe", "subscribe", "like and subscribe and hit the bell",
    "don't forget to subscribe", "don't forget to like and subscribe",
    "okay", "ok", "the end", "to be continued", "bye", "goodbye",
})


def is_whisper_hallucination(heard: str, raw: str = "", *,
                             no_speech_prob: float | None = None,
                             vad_prob: float | None = None) -> bool:
    """True when a transcript is a Whisper hallucination on noise, not a real second source.

    High ``no_speech_prob`` (above :data:`REJECT_NO_SPEECH_PROB`, Whisper's own non-speech
    bracket) or a transcript that is one of Whisper's known silence/music phrases decides it -
    but only when the CLEANED signal carries no real speech (``vad_prob`` missing or below the
    attribution VAD threshold). With real speech in the cleaned signal the same words are kept:
    this is a noise filter, never a "the user is wrong" filter.
    """
    from .attribution import VAD_SPEECH

    if vad_prob is not None and vad_prob >= VAD_SPEECH:
        return False  # the cleaned signal really has speech: not a hallucination
    if no_speech_prob is not None and no_speech_prob > REJECT_NO_SPEECH_PROB:
        return True
    for text in (heard, raw):
        norm = (text or "").strip().casefold()
        # Whisper appends sentence punctuation to its inventions; "Thanks for watching." and
        # "thanks for watching" are the same hallucination.
        norm = norm.strip(" \t\"'“”‘’")
        while norm and not norm[-1].isalnum():
            norm = norm[:-1].strip()
        if norm in HALLUCINATION_PHRASES:
            return True
    return False


def truncate_to_heard(spoken: str, audible_upto: str) -> str:
    """Truncate Voxa's assistant turn in the conversation context to the words she audibly played.

    On a CONFIRMED yield the LLM context must describe only what was actually heard (report 76:76
    "truncate the assistant turn in the LLM context to what was actually heard", task line 11).
    ``spoken`` is the full reply text, ``audible_upto`` the words up to the stop position from the
    word timeline. The kept text is the longest prefix of ``spoken`` whose words are all within the
    audible set; trailing words she never reached are dropped.
    """
    spoken_words = _content_words(spoken)
    audible = _content_words(audible_upto)
    if not spoken_words:
        return spoken
    if not audible:
        return ""
    # Keep the longest leading run of spoken words that the audible playback covered.
    keep = 0
    for i, word in enumerate(spoken_words):
        if i < len(audible) and audible[i] == word:
            keep = i + 1
        else:
            break
    if keep >= len(spoken_words):
        return spoken
    return " ".join(spoken_words[:keep])


def rewind_to_last_word_boundary(spoken: str, heard_upto: str) -> str:
    """On a long pause-then-reject, rewind the assistant text to the last WORD boundary (task
    line 11): a mid-word pause must not leave a partial word in the context. The kept text is the
    same audible prefix, but never past the last word that actually began being heard."""
    return truncate_to_heard(spoken, heard_upto)


class TurnStateMachine:
    """The explicit playback/turn state machine (handoff 6.5:218-236), pure and GTK-free.

    One instance describes one reply. The caller drives it with the events the hearing pipeline
    already produces (an onset, a candidate transcript, a pause, an alignment failure) and
    performs the effect each transition asks for. AEC and capture keep running in every state.
    """

    def __init__(self) -> None:
        self.state: str = LISTENING
        self._asked_yes_no = False
        self._just_spoken = ""
        self._pinned = False

    # -- context the state machine reads (handoff 6.5:235) --------------------------------
    def note_question(self, asked_yes_no: bool) -> None:
        """Voxa just asked a yes/no question: a following 'yes' is an ANSWER, not a backchannel."""
        self._asked_yes_no = bool(asked_yes_no)

    def note_speaking(self, text: str) -> None:
        """A reply began being read out: ``SPEAKING``, and this is the assistant turn to truncate."""
        self.state = SPEAKING
        self._just_spoken = text or ""
        self._pinned = False

    def note_planning(self) -> None:
        """The LLM is preparing a response but has not started speaking it (PLANNING)."""
        self.state = PLANNING

    # -- the state-table rows (handoff 6.5:222-231) ----------------------------------------
    def playback_only_echo(self) -> str:
        """Row 1: an echo of her own playback never creates a user turn or a command. Stay put."""
        return self.state  # SPEAKING/LISTENING unchanged; no user turn, no command

    def user_onset(self) -> str:
        """Row 2: a possible independent user onset pins the audio at once and ducks her stream.

        ``OVERLAP_PENDING``: the audio is pinned and evidence accumulates WITHOUT deleting it
        (brief: a classifier reads a snapshot, it never consumes the only copy).
        """
        self._pinned = True
        self.state = OVERLAP_PENDING
        return DECISION_SUSPECT

    def confirm(self, heard: str, *, gated_speech_ms: float = 0.0,
                no_speech_prob: float | None = None) -> str:
        """Rows 3/4: decide the overlap candidate. CONFIRM yields, REJECT resumes, SUSPECT holds.

        The caller maps the returned decision: CONFIRM -> hard-stop her stream, cancel the stale
        response, continue the same user turn from the retained pre-roll; REJECT -> restore the
        duck (resume); SUSPECT -> keep ducked and keep collecting.
        """
        decision = decide_bargein(
            heard, gated_speech_ms=gated_speech_ms, no_speech_prob=no_speech_prob,
            asked_yes_no=self._asked_yes_no, just_spoken=self._just_spoken)
        if decision == DECISION_CONFIRM:
            self.state = YIELDED
        elif decision == DECISION_REJECT:
            # Resume: back to the state the duck interrupted, the reply continuing from where it
            # paused. The ducked-then-resumed reply still counts in self_pause_per_hour.
            self.state = SPEAKING
        # SUSPECT stays OVERLAP_PENDING.
        return decision

    def repeat_of_own_words(self) -> str:
        """Row 5: the user repeats words Voxa just spoke. Preserve the words; the source evidence
        and context decide (never delete). The state does not change on the word overlap alone."""
        return self.state

    def user_pause(self) -> str:
        """Row 6: a mid-request pause. Hold the floor when evidence indicates continuation; the
        state stays where it was (the pre-roll keeps the utterance alive)."""
        return self.state

    def resume_during_planning(self) -> str:
        """Row 7: the user resumes while Voxa is preparing a response. Invalidate the pending
        response before it starts and merge the continuation into the same user turn."""
        self.state = USER_TURN
        return USER_TURN

    def alignment_failed(self) -> str:
        """Row 8: audio alignment failed. Enter recovery, retain the capture, and prevent
        unqualified automatic commands from playback contamination (``RECOVERING_AUDIO``)."""
        self.state = RECOVERING_AUDIO
        return RECOVERING_AUDIO
