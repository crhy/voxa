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
