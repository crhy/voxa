"""Transport health: counted drops and the degraded echo state (handoff 5.7 and 6.7, roadmap A4).

The capture path makes several resilience choices that are only acceptable while they are VISIBLE:

* the capture appsinks run ``max-buffers=12, drop=True`` (``voxa/audio.py``): when the pipeline
  falls behind, GStreamer throws WHOLE BUFFERS away and says nothing;
* the conversation queue is bounded (``maxsize=80``) and drops the oldest item when full;
* the hearing queue is bounded and drops the oldest sample when its worker falls behind
  (``voxa/hearing.py``);
* the capture ring is bounded and trims the rolling window;
* if the echo canceller raises, the capture path hands on the RAW microphone.

None of those may be silent. Every one of them is counted here, and the raw fallback is a DEGRADED
state: while a reply is playing, unqualified audio (audio the echo path did not qualify) must not
become an automatic command, a duck or a playback stop - her own voice with its echo still in it
would be heard as a new request. Degraded does NOT mean "stop listening": capture keeps running,
the captured audio is retained, and the counters stay readable after the fact.

What is deliberately NOT counted: the rolling trim of audio that is not part of any accepted or
candidate span. That is the bounded-memory rule (``voxa/capture.py``), not a loss; audio that belongs
to a pinned or promoted span is never trimmed, and when even that bound is exceeded it is counted
separately as ``pinned_frames_dropped``.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("voxa.transport")

# A gap larger than this many buffers is treated as a stopped/restarted source rather than a
# transport loss: the timestamps of a source that was closed and reopened do not continue.
MAX_GAP_BUFFERS = 64


class TransportHealth:
    """Counters plus the degraded flag, safe to touch from the audio callback and the UI thread.

    One object is shared by the capture path (buffer gaps, canceller fallbacks) and the conversation
    (queue overruns); the frame clock adds the hearing queue's drops and the capture ring's pinned
    drops when it reads :meth:`snapshot`, so no module has to reach into another one.
    """

    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self.capture_buffers_dropped = 0
        self.queue_overruns = 0
        self.canceller_fallbacks = 0
        self.degraded = False
        self.degraded_reason = ""
        self.degraded_at: float | None = None
        self.recovered_at: float | None = None
        # Buffer timestamps of the microphone stream: the only witness of an appsink drop.
        self._next_capture_ts: float | None = None
        # Log each kind of trouble once (then at doubling counts), so a long session cannot flood
        # the log or the UI with the same sentence.
        self._logged: dict[str, int] = {}

    # ------------------------------------------------------------- capture buffers

    def note_capture_buffer(self, duration: float, timestamp: float | None) -> int:
        """Record one microphone buffer; return the number of buffers found missing before it.

        ``duration`` and ``timestamp`` are the buffer's own length and start in the same unit
        (GStreamer uses nanoseconds). With ``drop=True`` a dropped buffer leaves no event behind, so
        a gap between consecutive buffers larger than one buffer's length is the evidence, and the
        gap divided by the buffer length is how many went.
        """
        if duration is None or duration <= 0 or timestamp is None:
            return 0
        with self._lock:
            dropped = 0
            if self._next_capture_ts is not None:
                gap = timestamp - self._next_capture_ts
                if gap > duration:
                    missing = int(round(gap / duration)) - 1
                    if 0 < missing <= MAX_GAP_BUFFERS:
                        dropped = missing
                        self.capture_buffers_dropped += missing
            self._next_capture_ts = max(self._next_capture_ts or timestamp, timestamp) + duration
        if dropped:
            self._report("capture-buffers", f"{self.capture_buffers_dropped} capture buffers dropped")
        return dropped

    def reset_capture_clock(self) -> None:
        """A source that closed and reopened does not continue its timestamps."""
        with self._lock:
            self._next_capture_ts = None

    # ------------------------------------------------------------- queues

    def note_queue_overrun(self) -> None:
        """The conversation queue was full and an item had to be pushed out."""
        with self._lock:
            self.queue_overruns += 1
        self._report("queue-overruns", f"conversation queue overrun: {self.queue_overruns}")

    # ------------------------------------------------------------- echo fallback

    def note_canceller_fallback(self, reason: str) -> None:
        """The echo canceller failed: the microphone is now RAW, which is UNQUALIFIED audio."""
        with self._lock:
            self.canceller_fallbacks += 1
            self.degraded = True
            self.degraded_reason = reason
            self.degraded_at = self._clock()
        self._report("canceller-fallback",
                     f"echo canceller failed, listening on raw audio: {reason}")

    def clear_degraded(self, reason: str = "") -> None:
        """An echo path was (re)fitted: the audio is qualified again. The counters stay."""
        with self._lock:
            if not self.degraded:
                return
            self.degraded = False
            self.degraded_reason = ""
            self.recovered_at = self._clock()
        log.info("echo protection recovered%s", f" ({reason})" if reason else "")

    @property
    def echo_qualified(self) -> bool:
        """False while the capture path is handing on audio no echo model qualified.

        The caller still has to ask whether a reply is playing: raw audio with nothing playing
        cannot be her echo, so only playback-time decisions are blocked by this flag.
        """
        with self._lock:
            return not self.degraded

    # ------------------------------------------------------------- reporting

    def _report(self, kind: str, message: str) -> None:
        count = self._logged.get(kind, 0)
        # Once, then at 2, 4, 8, ... so a flood is visible without being noisy.
        if count == 0 or (count & (count - 1)) == 0:
            self._logged[kind] = count + 1
            log.warning("%s: %s", kind, message)

    def snapshot(self, *, pinned_frames_dropped: int = 0, hearing_drops: int = 0) -> dict:
        """The health state as one dict: what the app shows and what the bench reports.

        ``pinned_frames_dropped`` and ``hearing_drops`` come from their owners (the capture ring
        and the hearing director); they are added here rather than pushed there so those modules
        keep their own simple counters.
        """
        with self._lock:
            counters = {
                "capture_buffers_dropped": self.capture_buffers_dropped,
                "queue_overruns": self.queue_overruns,
                "canceller_fallbacks": self.canceller_fallbacks,
            }
            degraded = self.degraded
            reason = self.degraded_reason
        total = sum(counters.values()) + int(pinned_frames_dropped) + int(hearing_drops)
        return {**counters,
                "pinned_frames_dropped": int(pinned_frames_dropped),
                "hearing_drops": int(hearing_drops),
                "total": total,
                "degraded": degraded,
                "reason": reason}

    @property
    def drops(self) -> int:
        """Everything counted so far (the caller's own counters are not included)."""
        with self._lock:
            return (self.capture_buffers_dropped + self.queue_overruns
                    + self.canceller_fallbacks)
