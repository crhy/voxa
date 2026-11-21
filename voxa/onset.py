"""Speech onset, measured in milliseconds of audio (handoff 5.4, task A3 step 1).

The old rule was a time window: ``BARGE_IN_GRACE_SECONDS`` folded the microphone for the first
0.4 s of every reply, and the level gate then needed a number of loud SAMPLES before it would
believe anyone else was talking. Both are blind: neither says anything about whether the sound
in the microphone could have come out of the speaker. The rule here is evidence:

* the reference stream says what the speaker path could have put in the microphone at this
  moment (its own level, carried through the echo's travel time, scaled by the coupling between
  the speaker output and what the microphone hears of it);
* the microphone holding sound above that ceiling is not her echo;
* that sound HOLDING for ``ONSET_EVIDENCE_MS`` of audio is someone starting to talk.

Milliseconds of AUDIO, not of wall clock: the same 320 ms of unexplained sound means the same
thing whether the frames arrive at 50 Hz or the bench replays a clip at 40x. Every window here
is advanced by the duration of the sample being fed, never by a clock.

``None`` is not zero, twice over. A reference this stage cannot measure (no reference sink, echo
off) leaves ``reference_db`` ``None``; an echo whose travel time was never estimated leaves
``delay_ms`` ``None``. Either way the detector says NOTHING and the decision falls back to the
level gate and the words - the degraded path ``voxa/attribution.py`` already describes. It never
becomes "the echo explains nothing", which would interrupt her on her own voice.

CALIBRATION (measured on the frozen smoke set, 48 clips, echo mode "voxa"; see the task notes):
the echo's travel time is only estimated on 11 of the 48 clips - ``estimate_delay`` needs 3 s of
reference audio and most clips are shorter - and neither of the 2 clips the set marks ``yield``
ever gets one in time. On the clips where the delay IS known, the microphone of a real
interruption sits 0-10 dB over the speaker path's ceiling while Voxa's own reverb tail reaches
the same 0-10 dB (RT60 0.2-0.8 s in this set). ``UNEXPLAINED_DB = 12.0`` with
``ONSET_EVIDENCE_MS = 320.0`` and a window of ``delay + 500 ms`` is the smallest setting on that
set that yields no onset on a Voxa-only clip or on a back-channel. It also yields no onset on
either ``yield`` clip, and the notes give the margin distribution that proves no window or
margin separates them (the two true ones peak at 27.4 and 28.8 dB; the clips that must NOT take
the floor reach 24-73 dB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# How far the microphone has to stand over the speaker path's ceiling. Calibrated on the frozen
# smoke set (see the task notes): below about 12 dB the reverb tail of Voxa's own reply (RT60
# 0.2-0.8 s there) reaches the same margin as a person talking over her, and the settings that
# avoid a false onset on a Voxa-only clip are also the settings that miss the interruption the
# clip set asks for - which is the finding, not a detail to tune away.
UNEXPLAINED_DB = 12.0
# How long the unexplained sound has to hold before the onset is confirmed. Shorter than this
# and a single loud frame - a door, her own plosive the echo path misjudged - would take the
# floor. Longer and the interruption is not fast any more.
ONSET_EVIDENCE_MS = 320.0
# The echo's travel time is measured (``estimate_delay``); the window covers it plus this much
# of the room's tail. A window shorter than the delay loses the echo and calls her own voice an
# interruption; a much longer one buries the interruption under the previous reply. The tail is
# the half-life of the room (RT60 0.2-0.8 s in the frozen set), not a fixed 150 ms.
REFERENCE_EXTRA_MS = 500.0
# The reference is only a reference while it is carrying sound: a silent reference means the
# speaker path is producing nothing, so the microphone hearing something is not evidence about
# an interruption either (it may be the room, it may be the tail of the reply before this one).
REFERENCE_FLOOR_DB = -60.0
# How fast the coupling between the speaker output and the microphone adapts (an EMA weight).
# The coupling is what makes the comparison absolute-dB meaningful; it is learned from the
# frames the echo path DOES explain, never from the ones claimed as an interruption.
COUPLING_ALPHA = 0.05
# The frame duration the caller feeds when it does not say otherwise (voxa/audio.py:CHUNK_MS).
FRAME_MS = 20.0
# An RMS measured on int16 samples sits this much above the same level in dBFS (20*log10(32768)).
# The microphone of the live app and of the bench is reported that way (``AudioCapture._rms``,
# ``runner.level_db``), while a reference stream held as float -1..1 is already dBFS: comparing
# them without this offset would put the microphone 90 dB over the speaker path, always.
INT16_DB_OFFSET = 20.0 * math.log10(32768.0)


def dbfs(int16_level_db: float) -> float:
    """Bring an int16 RMS level in dB down to dBFS, so it can meet a float reference's dB."""
    return float(int16_level_db) - INT16_DB_OFFSET


def level_dbfs(rms: float) -> float:
    """An int16 RMS level (what ``AudioCapture._rms`` reports) expressed in dBFS.

    A silent frame floors at -140 dB rather than going to minus infinity: the onset rule reads
    "nothing measurable", not "infinitely quiet".
    """
    value = float(rms)
    return 20.0 * math.log10(value) - INT16_DB_OFFSET if value > 1e-6 else -140.0


def power_dbfs(power: float) -> float:
    """The energy of a float -1..1 signal (the reference ring's chunks) in dBFS."""
    value = float(power)
    return 10.0 * math.log10(value) if value > 1e-14 else -140.0


@dataclass(slots=True)
class OnsetOutcome:
    """What one observation of the microphone against the reference decided."""

    # True the moment the onset is confirmed, and on no later frame of the same run: the caller
    # pins the captured span from here (voxa/conversation.py:note_onset), and the barge controller
    # yields the floor from here (bargectl.YIELD).
    onset: bool = False
    # The run of unexplained sound, in milliseconds of audio, and where it started.
    evidence_ms: float = 0.0
    onset_audio_ms: float = 0.0
    # How far the microphone stood over the speaker path's ceiling on this frame.
    margin_db: float = 0.0
    # Why nothing was decided: "no-reference" (nothing to compare the microphone with),
    # "unaligned" (the echo's travel time is unknown, so its moment is unknown), "quiet-reference"
    # (the speaker path is producing nothing in the window the echo travels through),
    # "unanchored" (no frame yet where both sides carry sound, so the two dB scales have no
    # common point), "building" (not held long enough), "explained" (the echo accounts for it).
    reason: str = ""


class OnsetDetector:
    """The microphone against the reference, in milliseconds of audio.

    Fed once per frame by whoever owns the frames - the hearing worker in the live app, the
    bench runner in the measurement. It holds no audio and no transcript: two numbers per frame
    and a counter.
    """

    def __init__(self, *, margin_db: float = UNEXPLAINED_DB, evidence_ms: float = ONSET_EVIDENCE_MS,
                 extra_ms: float = REFERENCE_EXTRA_MS, coupling_alpha: float = COUPLING_ALPHA,
                 reference_floor_db: float = REFERENCE_FLOOR_DB) -> None:
        self._margin_db = margin_db
        self._evidence_ms = evidence_ms
        self._extra_ms = extra_ms
        self._coupling_alpha = coupling_alpha
        self._reference_floor_db = reference_floor_db
        # Audio seen since the reply started, and the audio at which the current run of
        # unexplained sound began (0.0 = no run).
        self._audio_ms = 0.0
        self._run_ms = 0.0
        self._onset_since_ms = 0.0
        # The reference's recent dB, bounded by the echo's travel window.
        self._reference: list[tuple[float, float]] = []
        self._delay_ms: float | None = None
        # dB the microphone hears of the speaker output, relative to the reference's own level:
        # the room, the speaker's distance and its gain. Learned, never assumed.
        self._coupling_db: float | None = None

    def reset(self) -> None:
        """A new reply: the run of sound, the reference history and the coupling start over.

        Called by ``BargeInController.note_speaking`` - this is what replaces the grace period.
        The evidence restarts instead of the microphone being folded, so the first frame of the
        reply is already judged.
        """
        self._audio_ms = 0.0
        self._run_ms = 0.0
        self._onset_since_ms = 0.0
        self._reference = []
        self._coupling_db = None

    def note_delay(self, delay_ms: float | None) -> None:
        """The echo's travel time, as the caller's alignment stage last measured it.

        ``None`` (never estimated) is not 0: without it the moment the echo arrives is unknown,
        and a ceiling that could be anywhere is no ceiling at all.
        """
        self._delay_ms = float(delay_ms) if delay_ms is not None else None

    def observe(self, mic_db: float | None, reference_db: float | None,
                dt_ms: float = FRAME_MS) -> OnsetOutcome:
        """Judge one frame: the microphone's dB and the reference's dB for the SAME moment.

        ``mic_db`` is the level of the RAW microphone frame (before the echo path removes her
        echo: the question is what the speaker path could have produced, so the comparison has
        to be made before that path's work, not after it). ``reference_db`` is the reference's
        own level in the same dB scale, or ``None`` when this stage cannot measure it.
        """
        dt = max(0.0, float(dt_ms))
        self._audio_ms += dt
        if reference_db is None:
            self._run_ms = 0.0
            return OnsetOutcome(reason="no-reference")
        if self._delay_ms is None:
            # The echo's moment is unknown, so the ceiling it implies is unknown, and a ceiling
            # that could be anywhere is no ceiling at all. Keep the history so a later estimate
            # can use it, but claim nothing: the decision falls back to the level gate and the
            # words. Measured on the frozen set (see the task notes): replacing this silence with
            # the conservative guess - the loudest reference in the last ``REFERENCE_EXTRA_MS`` -
            # fires onsets on 6 clips whose user speech should NOT take the floor and on none of
            # the 2 clips that ask for it, because their margins (27.4, 28.8 dB) sit under the
            # false ones (24-73 dB). No window or margin separates them on this material.
            self._note_reference(reference_db, dt)
            self._run_ms = 0.0
            return OnsetOutcome(reason="unaligned")
        self._note_reference(reference_db, dt)
        ceiling = max((db for db, _ms in self._reference), default=-math.inf)
        if not self._reference or ceiling <= self._reference_floor_db:
            self._run_ms = 0.0
            return OnsetOutcome(reason="quiet-reference")
        if self._coupling_db is None:
            # The first frame where both sides carry sound sets the coupling; after that it
            # adapts, and never on a frame already counted as an interruption. A frame too quiet
            # to anchor it decides nothing either: without the coupling the two dB scales have
            # no common point, so there is no margin to compare.
            if float(mic_db) > self._reference_floor_db:
                self._coupling_db = float(mic_db) - ceiling
            else:
                self._run_ms = 0.0
                return OnsetOutcome(reason="unanchored")
        margin = (float(mic_db) if mic_db is not None else -math.inf) - (ceiling + self._coupling_db)
        if margin > self._margin_db:
            if self._run_ms == 0.0:
                self._onset_since_ms = self._audio_ms - dt
            self._run_ms += dt
        else:
            self._run_ms = 0.0
            if self._coupling_db is not None and float(mic_db) > self._reference_floor_db:
                sample = float(mic_db) - ceiling
                self._coupling_db += self._coupling_alpha * (sample - self._coupling_db)
        if self._run_ms >= self._evidence_ms:
            return OnsetOutcome(onset=True, evidence_ms=self._run_ms,
                                onset_audio_ms=self._audio_ms - self._onset_since_ms,
                                margin_db=round(margin, 2))
        return OnsetOutcome(evidence_ms=self._run_ms,
                            onset_audio_ms=self._audio_ms - self._onset_since_ms,
                            margin_db=round(margin, 2),
                            reason=("building" if self._run_ms > 0.0 else "explained"))

    def _note_reference(self, reference_db: float, dt_ms: float) -> None:
        """Keep the reference's recent dB, bounded by the window the echo travels through."""
        self._reference.append((float(reference_db), max(0.0, float(dt_ms))))
        total = sum(ms for _db, ms in self._reference)
        limit = max(0.0, (self._delay_ms or 0.0)) + self._extra_ms
        while self._reference and total > limit:
            _db, ms = self._reference.pop(0)
            total -= ms

    @property
    def coupling_db(self) -> float | None:
        """The learned offset between the speaker output and the microphone (None = unlearned)."""
        return self._coupling_db

    @property
    def evidence_ms(self) -> float:
        """The run of unexplained sound built so far (0.0 = none)."""
        return self._run_ms

    @property
    def audio_ms(self) -> float:
        """How much audio this reply has been judged, in milliseconds."""
        return self._audio_ms
