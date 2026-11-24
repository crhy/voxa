from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator

from .transcription import WhisperService
from .voicedetect import SILERO_TRAILING_PAD_MS

PRE_ROLL_BYTES = int(0.6 * 16000) * 2  # 0.6 s of 16 kHz 16-bit audio kept before the first loud moment
DICTATION_CHUNK_SECONDS = 6.0
# Roadmap task 6: the shortest gap the caller treats as the end of a turn. Silero's off-threshold
# plus this 200 ms trailing pad keep a soft last syllable or a mid-word breath from cutting the
# utterance short and eating the user's last word (premature_cutoff_rate).
TRAILING_PAD_SECONDS = SILERO_TRAILING_PAD_MS / 1000.0



class SpeechSegmenter:
    """The per-chunk state of :func:`segment_stream`, split out so a file-driven caller
    can drive the same speech-pause rules with its own clock (roadmap B2: the hearing
    bench replays WAV files faster than real time and must segment exactly as the live
    app does, without sleeping).

    ``step(pcm, level, now)`` takes one chunk plus the clock reading the caller took for
    it and returns the segment completed by that chunk, or ``None``. The live queue,
    the stop event and the two wall-clock timeouts stay in :func:`segment_stream`.
    """

    def __init__(
        self,
        *,
        threshold: int,
        silence_seconds: float,
        max_segment_seconds: float | Callable[[], float],
        early_silence_seconds: float | None = None,
        early_check: Callable[[bytes], bool] | None = None,
        is_voice: Callable[[bytes, float], bool] | None = None,
        endpoint_probe: Callable[[bytes], bool | None] | None = None,
        endpoint_policy: object = None,
    ) -> None:
        self.threshold = threshold
        self.silence_seconds = silence_seconds
        self.max_segment_seconds = max_segment_seconds
        self.early_silence_seconds = early_silence_seconds
        self.early_check = early_check
        self.is_voice = is_voice
        # Roadmap task 9: the acoustic endpoint decision. ``endpoint_probe`` is
        # called once per pause, ``endpoint_policy.trigger_seconds`` into it, with
        # the whole turn heard so far; True means end the turn at
        # ``complete_seconds`` of silence, False keep the turn open up to
        # ``incomplete_wait_seconds`` (or the cap), None fall back to plain
        # silence timing. It may run on the capture side's clock (the bench feeds
        # it silence steps while the mic is muted), which is the point: endpoint
        # decisions must not need the microphone to be unmuted.
        self.endpoint_probe = endpoint_probe
        self.endpoint_policy = endpoint_policy
        self.segment = bytearray()
        self.heard_voice = False
        self.heard_any_voice = False
        self.last_voice = 0.0
        self.last_any_voice = 0.0
        self.early_checked = False
        self._probe_checked = False
        self._probe_floor: float | None = None

    def step(self, pcm: bytes | None, level: float, now: float) -> bytes | None:
        """Consume one chunk (``pcm is None`` means nothing arrived) and return the
        segment it completed, or None. ``now`` is the caller's clock for this chunk."""
        if not self.last_voice:
            self.last_voice = self.last_any_voice = now
        if pcm is not None:
            self.segment.extend(pcm)
            if self.is_voice is not None:
                voiced = level >= self.threshold / 2 and self.is_voice(pcm, level)
            else:
                voiced = level >= self.threshold
            if voiced:
                self.heard_voice = True
                self.heard_any_voice = True
                self.last_voice = now
                self.last_any_voice = now
                self.early_checked = False
                # The pause ended: the endpoint probe runs again at the next one.
                self._probe_checked = False
                self._probe_floor = None

        duration = len(self.segment) / (16000 * 2)
        # Require a little recorded content before treating a pause as a
        # boundary; scale it with the silence setting so short pauses cut
        # sooner instead of accumulating toward the max-segment cutoff.
        min_content = min(0.8, max(0.3, self.silence_seconds / 2.0))
        # Never end the turn sooner than the 200 ms trailing pad after the last voiced frame.
        early_floor = max(self.early_silence_seconds or 0.0, TRAILING_PAD_SECONDS)
        pause_floor = max(self.silence_seconds, TRAILING_PAD_SECONDS)
        silence = now - self.last_voice
        completed: bytes | None = None
        if (
            self.early_check is not None
            and self.early_silence_seconds is not None
            and self.heard_voice
            and duration >= min_content
            and not self.early_checked
            and silence >= early_floor
        ):
            self.early_checked = True
            if self.early_check(bytes(self.segment)):
                completed = bytes(self.segment)
                self.segment = bytearray()
                self.heard_voice = False
                self.last_voice = now
                self.early_checked = False
        if (
            self.endpoint_probe is not None
            and self.heard_voice
            and duration >= min_content
            and not self._probe_checked
            and silence >= getattr(self.endpoint_policy, "trigger_seconds", 0.25)
        ):
            # Roadmap task 9: one probe per pause, on the WHOLE turn heard so
            # far. True ends it at the minimum silence floor; False keeps the
            # turn open for the speaker to resume, bounded by the cap; None
            # (no model, model failure) keeps the pause rule the app had.
            self._probe_checked = True
            try:
                verdict = self.endpoint_probe(bytes(self.segment))
            except Exception as exc:  # noqa: BLE001 - an endpoint probe must not lose the turn
                logging.debug("endpoint probe failed: %s", exc)
                verdict = None
            if verdict is True:
                self._probe_floor = max(getattr(self.endpoint_policy, "complete_seconds", 0.5),
                                        TRAILING_PAD_SECONDS)
            elif verdict is False:
                self._probe_floor = min(getattr(self.endpoint_policy, "incomplete_wait_seconds", 2.0),
                                        getattr(self.endpoint_policy, "cap_seconds", 3.0))
            else:
                self._probe_floor = None
        if self._probe_floor is not None:
            pause_floor = self._probe_floor
        pause_ready = (
            self.heard_voice
            and duration >= min_content
            and silence >= pause_floor
        )
        limit = (
            self.max_segment_seconds()
            if callable(self.max_segment_seconds)
            else self.max_segment_seconds
        )
        max_ready = self.heard_voice and duration >= limit
        if pause_ready or max_ready:
            completed = bytes(self.segment)
            self.segment = bytearray()
            self.heard_voice = False
            self.last_voice = now
            self._probe_checked = False
            self._probe_floor = None

        if not self.heard_voice and duration > 2.0:
            # Keep the last moments instead of emptying the buffer: a soft first syllable ("Vo…")
            # that arrived just before the buffer was emptied was lost with it.
            del self.segment[:-PRE_ROLL_BYTES]
        return completed

    def pending(self) -> bytes:
        """The audio collected since the last boundary (empty when no voice was heard)."""
        return bytes(self.segment) if self.heard_voice else b""


def segment_stream(
    input_queue: queue.Queue[tuple[bytes, float] | None],
    stop_event: threading.Event,
    *,
    threshold: int,
    silence_seconds: float,
    max_segment_seconds: float | Callable[[], float],
    idle_timeout_seconds: float | None = None,
    on_idle_timeout: Callable[[], None] | None = None,
    trailing_silence_seconds: float | None = None,
    early_silence_seconds: float | None = None,
    early_check: Callable[[bytes], bool] | None = None,
    is_voice: Callable[[bytes, float], bool] | None = None,
    endpoint_probe: Callable[[bytes], bool | None] | None = None,
    endpoint_policy: object = None,
) -> Iterator[bytes]:
    """Yield PCM segments split at speech pauses from a live audio queue.

    Pulls ``(pcm, level)`` pairs fed by an audio capture callback and yields
    a complete segment once a pause (or the max segment length) is reached.
    If ``idle_timeout_seconds`` elapses with no speech at all, ``on_idle_timeout``
    fires and the generator ends (without touching ``stop_event``, so callers
    decide what ending idle means for them).

    If ``trailing_silence_seconds`` elapses after the first voice in the stream, the
    pending segment (if it contains voice) is yielded and the generator ends
    on its own; ``on_idle_timeout`` is not called for this case.

    If ``early_silence_seconds`` and ``early_check`` are both given, then once a
    pause reaches ``early_silence_seconds`` (before the full ``silence_seconds``
    boundary) ``early_check`` is called once for that pause with the audio heard
    so far; if it returns True the segment is yielded immediately. The check
    runs at most once per pause, and a new burst of voice arms it again.

    If ``is_voice`` is given it replaces the ``level >= threshold`` loudness rule:
    a chunk counts as voice only when ``level >= threshold / 2`` (so pure silence
    stays cheap) and ``is_voice(pcm, level)`` returns True.

    ``max_segment_seconds`` may be a callable, read on every chunk, so a mode
    switched on mid-utterance (dictation) applies to the segment already in progress.

    The per-chunk rules live in :class:`SpeechSegmenter`; this function only supplies
    the queue, the stop event and the two wall-clock timeouts.
    """
    segmenter = SpeechSegmenter(
        threshold=threshold,
        silence_seconds=silence_seconds,
        max_segment_seconds=max_segment_seconds,
        early_silence_seconds=early_silence_seconds,
        early_check=early_check,
        is_voice=is_voice,
        endpoint_probe=endpoint_probe,
        endpoint_policy=endpoint_policy,
    )

    while not stop_event.is_set():
        try:
            item = input_queue.get(timeout=0.25)
        except queue.Empty:
            item = None
        now = time.monotonic()

        completed = segmenter.step(
            item[0] if item is not None else None,
            item[1] if item is not None else 0.0,
            now,
        )
        if completed is not None:
            yield completed

        if (
            trailing_silence_seconds is not None
            and segmenter.heard_any_voice
            and now - segmenter.last_any_voice > trailing_silence_seconds
        ):
            if segmenter.pending():
                yield segmenter.pending()
            return

        if idle_timeout_seconds is not None and now - segmenter.last_any_voice > idle_timeout_seconds:
            if on_idle_timeout is not None:
                on_idle_timeout()
            return

    if segmenter.pending():
        yield segmenter.pending()


class DictationController:
    """Segments live PCM around speech pauses and transcribes off the UI thread."""

    IDLE_TIMEOUT_SECONDS = 15.0

    def __init__(
        self,
        whisper: WhisperService,
        *,
        language: str,
        threshold: int,
        silence_ms: int,
        max_segment_seconds: float,
        on_text: Callable[[str], None],
        on_status: Callable[[str], None],
        on_auto_stop: Callable[[], None],
        on_error: Callable[[str], None],
        endpoint_probe: Callable[[bytes], bool | None] | None = None,
        endpoint_policy: object = None,
    ) -> None:
        self.whisper = whisper
        self.language = language
        self.threshold = threshold
        self.silence_seconds = silence_ms / 1000.0
        # Dictation should put text on screen while somebody is still speaking.
        # The general command limit can be much longer, so cap only this mode.
        self.max_segment_seconds = min(max_segment_seconds, DICTATION_CHUNK_SECONDS)
        self.on_text = on_text
        self.on_status = on_status
        self.on_auto_stop = on_auto_stop
        self.on_error = on_error
        # Roadmap task 9: dictation gets the same endpoint probe with its wider
        # silence cap (5 s), and the patience setting scales the waits.
        self.endpoint_probe = endpoint_probe
        self.endpoint_policy = endpoint_policy
        if endpoint_probe is not None and endpoint_policy is None:
            from .endpoint import EndpointPolicy

            self.endpoint_policy = EndpointPolicy(dictation=True)
        self.queue: queue.Queue[tuple[bytes, float] | None] = queue.Queue(maxsize=80)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="dictation-worker", daemon=True)
        self.thread.start()

    def feed(self, pcm: bytes, level: float) -> None:
        if self.stop_event.is_set():
            return
        try:
            self.queue.put_nowait((pcm, level))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((pcm, level))
            except queue.Full:
                pass

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def _on_idle_timeout(self) -> None:
        if self.stop_event.is_set():
            return
        self.on_status("No speech detected; dictation stopped.")
        self.on_auto_stop()

    def _run(self) -> None:
        for segment in segment_stream(
            self.queue,
            self.stop_event,
            threshold=self.threshold,
            silence_seconds=self.silence_seconds,
            max_segment_seconds=self.max_segment_seconds,
            idle_timeout_seconds=self.IDLE_TIMEOUT_SECONDS,
            on_idle_timeout=self._on_idle_timeout,
            endpoint_probe=self.endpoint_probe,
            endpoint_policy=self.endpoint_policy,
        ):
            self._flush(segment)

    def _flush(self, segment: bytes) -> None:
        if not segment or self.stop_event.is_set():
            return
        self.on_status("Transcribing…")
        try:
            text = self.whisper.transcribe(segment, self.language)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            if not self.stop_event.is_set():
                self.on_error(str(exc))
            return
        if text and not self.stop_event.is_set():
            self.on_text(text)
