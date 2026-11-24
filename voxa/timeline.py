"""One authoritative render timeline: the samples Voxa's speakers play, and exactly when.

Roadmap task 1 (handoff 6.1). Voxa's canceller can only remove her own voice if it knows what her
speakers were playing at the moment the microphone sampled it. Until now that knowledge came from a
loopback monitor of the whole output sink (``voxa/audio.py`` ``_start_reference`` / ``_start_webrtc``),
which is late, mis-aligned (the probe assumes its buffers are about to be played when they have
already been played) and full of every OTHER application's audio - including VoxaTest's test voice,
which must never be cancelled as if it were her echo.

This module is the reference instead of the guess:

* the PCM is tapped AFTER Voxa's own mixing, gain and ducking, immediately BEFORE the sink, on every
  output path (``voxa/speech.py`` streams and prepared files, the eSpeak fallback included;
  ``voxa/agent/player.py`` media Voxa starts; ``PATH_NOTIFICATION`` is reserved for earcons, which the
  app does not play yet). MP3 bytes, the source text, an animation clock and queued-but-cancelled
  speech are not the render reference - the submitted frames are.
* every submitted chunk carries a generation id, and generated / submitted / cancelled /
  estimated-audible samples are counted separately, so "she stopped" and "the sink finished" are
  different facts.
* the ring is keyed by the absolute index in Voxa's output stream, and the sink's latency query maps
  that index to a monotonic play time. A microphone chunk is stamped with its capture time minus the
  capture latency, and the reference window for that chunk is served from the map.
* stereo output keeps both reference channels; a mono canceller asks for a mono view explicitly.
* a route or rate change is an explicit epoch: the map and the ring are flushed together.

Nothing here touches a sound device: the sink is whatever the caller hands to :meth:`submit`, so the
bench can drive it with a simulated sink of known latency and jitter (``voxa/bench/sinksim.py``) and
the live app drives it with a GStreamer pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("voxa.timeline")

#: The rate the canceller and the recogniser work at (``voxa/audio.py`` REFERENCE_RATE).
REFERENCE_RATE = 16000
#: How far back the render timeline remembers (``voxa/audio.py`` REFERENCE_SECONDS).
TIMELINE_SECONDS = 3.0

# Every output path the tap knows about (handoff 6.1: "cover all Voxa output paths").
PATH_TTS_STREAM = "tts-stream"      # streamed Edge TTS, appsrc -> autoaudiosink
PATH_TTS_FILE = "tts-file"          # a reply fetched to a file, played with playbin
PATH_TTS_PREPARED = "tts-prepared"  # prepared ahead of time (the face needs the whole sentence)
PATH_TTS_OFFLINE = "tts-offline"    # the eSpeak NG offline fallback
PATH_NOTIFICATION = "notification"  # earcons and notification sounds
PATH_MEDIA = "media"                # music/video Voxa started herself (voxa/agent/player.py)

#: What the capture path is taking as its echo reference.
MODE_OWN_RENDER = "own-render"
MODE_MONITOR = "monitor"
MODE_NONE = "none"


class SampleRing:
    """Bounded ring of rendered samples, addressed by the absolute index in the output stream.

    ``total`` is the number of samples ever appended, so a sample's position in the output stream is
    its index; :meth:`window` returns the samples covering ``[start, start + length)`` and zero-pads
    positions the ring no longer holds (or has not produced yet). Mono streams store 1-D samples,
    stereo streams 2-D ``(samples, channels)`` - the channel count is fixed when the ring is created
    and a route change builds a new one.
    """

    def __init__(self, capacity: int, channels: int = 1) -> None:
        self._capacity = max(1, int(capacity))
        self._channels = max(1, int(channels))
        self._buf = np.zeros((0, self._channels), dtype=np.float32)
        self._total = 0
        self._lock = threading.Lock()

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, samples) -> tuple[int, int]:
        """Append one chunk; returns the ``(first_index, end_index)`` it landed at."""
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != self._channels:
            raise ValueError(f"render stream is {self._channels} channel, got {arr.shape[1]}")
        if arr.shape[0] == 0:
            with self._lock:
                first = self._total
                return first, first
        with self._lock:
            first = self._total
            self._total += arr.shape[0]
            merged = np.concatenate([self._buf, arr])[-self._capacity:]
            self._buf = merged
            return first, self._total

    def window(self, start: int, length: int) -> np.ndarray:
        """Samples for stream positions ``[start, start + length)``, zero-padded where absent."""
        length = max(0, int(length))
        out = np.zeros((length, self._channels), dtype=np.float32)
        if length == 0:
            return out
        with self._lock:
            oldest = self._total - self._buf.shape[0]
            lo = max(int(start), oldest, 0)
            hi = min(int(start) + length, self._total)
            if hi > lo:
                out[lo - int(start): hi - int(start)] = self._buf[lo - oldest: hi - oldest]
            return out


@dataclass(slots=True)
class Chunk:
    """One submission to the sink: what was handed over, by whom, and when it should be heard."""

    generation: str
    path: str
    first_index: int
    samples: int
    submitted_at: float
    audible_at: float
    epoch: int
    channels: int = 1

    @property
    def end_index(self) -> int:
        return self.first_index + self.samples

    def as_dict(self) -> dict:
        return {"generation": self.generation, "path": self.path, "first_index": self.first_index,
                "samples": self.samples, "submitted_at": self.submitted_at,
                "audible_at": self.audible_at, "epoch": self.epoch, "channels": self.channels}


@dataclass(slots=True)
class Generation:
    """One utterance / notification / media item, counted four different ways.

    ``generated`` is what the engine produced, ``submitted`` what actually reached the sink,
    ``cancelled`` what was thrown away instead (a stopped reply, a discarded prepared file, a queued
    tail the sink never played), and ``audible`` what the timeline believes the speakers really
    played. They are deliberately separate: "Voxa stopped talking" is not the same fact as "the sink
    drained", and only the last one may be used as an echo reference.
    """

    id: str
    path: str
    epoch: int
    opened_at: float
    first_index: int | None = None
    last_index: int = 0
    generated: int = 0
    submitted: int = 0
    cancelled: int = 0
    audible: int = 0
    state: str = "open"
    closed_at: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "path": self.path, "epoch": self.epoch, "opened_at": self.opened_at,
                "first_index": self.first_index, "generated": self.generated,
                "submitted": self.submitted, "cancelled": self.cancelled, "audible": self.audible,
                "state": self.state, "notes": list(self.notes)}


class FixedSinkClock:
    """A sink whose latency is a constant - the simplest thing a caller can hand to :meth:`submit`."""

    def __init__(self, latency: float = 0.0) -> None:
        self._latency = float(latency)

    def latency(self) -> float:
        return self._latency


class GstSinkClock:
    """Latency of a live GStreamer playback pipeline: how long until a sample submitted now is heard.

    ``pipeline`` is the element the tap is attached to (a ``playbin`` or the appsrc pipeline). The
    answer is the pipeline's own latency plus whatever is already queued in front of the sink. Every
    part of that query can fail on a pipeline that is still going to PAUSED, or on a sink that does
    not answer latency, so a failure falls back to ``fallback`` (the caller's measured/learned value)
    rather than to 0: a 0 here would tell the timeline the sound is audible the instant it is written.
    """

    def __init__(self, pipeline: Any, sample_rate: int = REFERENCE_RATE, fallback: float = 0.05,
                 queued: Callable[[], int] | None = None) -> None:
        self.pipeline = pipeline
        self.sample_rate = max(1, int(sample_rate))
        self.fallback = float(fallback)
        self._queued = queued or (lambda: 0)

    def latency(self) -> float:
        queued = max(0, int(self._queued())) / self.sample_rate
        try:
            success, minimum, maximum, _current = self.pipeline.query_latency()
        except Exception:  # noqa: BLE001 - a pipeline that will not answer is not an error here
            return self.fallback + queued
        if not success:
            return self.fallback + queued
        value = minimum if minimum and minimum > 0 else maximum
        if not value or value <= 0:
            return self.fallback + queued
        # GStreamer reports nanoseconds.
        return value / 1_000_000_000.0 + queued


@dataclass(slots=True)
class CaptureClock:
    """Stamps microphone chunks with the moment their samples left the microphone.

    A capture buffer delivered at monotonic time ``arrival`` holds ``samples`` samples that were
    captured *before* that: the driver's own latency sits between the diaphragm and the callback.
    Without that subtraction the reference window is served for the wrong span by exactly the capture
    latency, which is the same mistake the monitor path makes at the other end.
    """

    rate: int = REFERENCE_RATE
    latency: float | Callable[[], float] = 0.0

    def capture_latency(self) -> float:
        value = self.latency() if callable(self.latency) else self.latency
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def stamp(self, arrival: float, samples: int) -> tuple[float, float]:
        """``(start, end)`` monotonic times covered by a buffer of ``samples`` delivered at ``arrival``."""
        end = float(arrival) - self.capture_latency()
        return end - max(0, int(samples)) / self.rate, end


class RenderTimeline:
    """The one timeline of what Voxa renders: the tap writes it, the capture path reads it."""

    def __init__(self, sample_rate: int = REFERENCE_RATE, seconds: float = TIMELINE_SECONDS,
                 channels: int = 1, clock: Callable[[], float] = time.monotonic) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.seconds = max(0.05, float(seconds))
        self.epoch = 0
        self.ring = SampleRing(int(self.sample_rate * self.seconds), channels)
        self._clock = clock
        self._lock = threading.Lock()
        self._anchors: deque[Chunk] = deque(maxlen=512)
        self._generations: dict[str, Generation] = {}
        self._order: deque[str] = deque(maxlen=64)
        self._next_playable: float | None = None
        self._foreign: set[str] = set()
        # Roadmap task 2: when set, every render tap also registers a GStreamer webrtcechoprobe under
        # this name on Voxa's own playback branch, so a GStreamer webrtcdsp can use her own render as
        # its reference without ever listening to the speakers. None = no probe (the default).
        self.echo_probe_name: str | None = None

    # ------------------------------------------------------------------ identity

    @property
    def channels(self) -> int:
        return self.ring.channels

    @property
    def total(self) -> int:
        """Samples of Voxa's own render ever submitted to a sink."""
        return self.ring.total

    def new_epoch(self, reason: str = "", channels: int | None = None) -> None:
        """A route, device or rate change: the map and the ring are flushed together.

        Handoff 6.1 records the chosen sink, channel layout and rate with the timeline; when they
        change, the old positions no longer mean anything, so both sides restart at index 0 under a
        new epoch number instead of mixing two clocks into one reference.
        """
        with self._lock:
            self.epoch += 1
            self.ring = SampleRing(int(self.sample_rate * self.seconds),
                                   self.channels if channels is None else max(1, int(channels)))
            self._anchors.clear()
            self._next_playable = None
        log.info("render timeline epoch %d: %s", self.epoch, reason or "route change")

    def set_sample_rate(self, rate: int) -> None:
        if int(rate) != self.sample_rate:
            self.new_epoch(f"render rate {self.sample_rate} -> {rate}")

    # ------------------------------------------------------------------ generations

    def open(self, generation: str, path: str, *, now: float | None = None) -> Generation:
        """Start counting one utterance / notification / media item."""
        stamp = self._clock() if now is None else float(now)
        item = Generation(id=str(generation), path=str(path), epoch=self.epoch, opened_at=stamp)
        with self._lock:
            self._generations[item.id] = item
            self._order.append(item.id)
            while len(self._order) > 64:
                oldest = self._order.popleft()
                if oldest != item.id:
                    self._generations.pop(oldest, None)
        return item

    def generation(self, generation: str) -> Generation | None:
        with self._lock:
            return self._generations.get(str(generation))

    def note_generated(self, generation: str, samples: int) -> None:
        """The engine produced ``samples`` more samples for this generation (decoded, not played)."""
        with self._lock:
            item = self._generations.get(str(generation))
            if item is not None:
                item.generated += max(0, int(samples))

    def mark_foreign(self, generation: str, reason: str = "") -> None:
        """Declare a generation NOT Voxa's own voice, so it can never become her echo reference.

        VoxaTest speaks the test "user" through the same :class:`voxa.speech.SpeechService`; that
        playback represents the human and must stay out of the cancellation policy (handoff 8.2).
        """
        with self._lock:
            self._foreign.add(str(generation))
        if reason:
            log.info("render generation %s is not Voxa's own voice: %s", generation, reason)

    def is_foreign(self, generation: str) -> bool:
        with self._lock:
            return str(generation) in self._foreign

    # ------------------------------------------------------------------ the tap

    def submit(self, samples, *, generation: str, path: str, sink: Any = None,
               now: float | None = None) -> Chunk | None:
        """Record what was handed to the sink, immediately before the sink.

        ``samples`` is the decoded PCM after Voxa's own mixing, gain and ducking: 1-D for mono,
        2-D ``(samples, channels)`` for stereo. ``sink`` is anything with a ``latency()`` method
        reporting the seconds until a sample submitted right now leaves the speakers (a live
        :class:`GstSinkClock`, or the bench's simulated sink); ``None`` means "no query available"
        and the timeline then chains from the previous chunk instead of guessing 0.
        """
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] == 0:
            return None
        if arr.shape[1] != self.ring.channels:
            # A stereo reply on a mono timeline (or the other way round) is a route change, not a
            # reason to lose the reference: start a new epoch with the layout the sink really got.
            self.new_epoch(f"output layout {self.ring.channels} -> {arr.shape[1]} channel",
                           channels=arr.shape[1])
        stamp = self._clock() if now is None else float(now)
        latency = self._query_latency(sink)
        with self._lock:
            if str(generation) in self._foreign:
                # A foreign stream (VoxaTest's test voice) reached the tap: it is not Voxa's echo,
                # so it never enters the ring, whatever the mixer did with it.
                return None
            first, _end = self.ring.append(arr)
            playable = stamp + latency if latency is not None else self._next_playable
            if playable is None:
                playable = stamp
            if self._next_playable is not None:
                # The sink plays one stream at a time: a sample can never be heard before the sample
                # submitted before it.
                playable = max(playable, self._next_playable)
            chunk = Chunk(generation=str(generation), path=str(path), first_index=first,
                          samples=arr.shape[0], submitted_at=stamp, audible_at=playable,
                          epoch=self.epoch, channels=arr.shape[1])
            self._anchors.append(chunk)
            self._next_playable = playable + arr.shape[0] / self.sample_rate
            item = self._generations.get(chunk.generation)
            if item is not None:
                item.submitted += arr.shape[0]
                if item.first_index is None:
                    item.first_index = first
                item.last_index = chunk.end_index
                if item.state in ("open", "generated"):
                    item.state = "playing"
            return chunk

    def _query_latency(self, sink: Any) -> float | None:
        query = getattr(sink, "latency", None)
        if query is None:
            return None
        try:
            value = float(query())
        except Exception:  # noqa: BLE001 - a sink that cannot answer must not stop playback
            return None
        if value < 0.0:
            return None
        return value

    def cancel(self, generation: str, *, now: float | None = None, scope: str = "voxa",
               reason: str = "") -> Generation | None:
        """This generation will play no more.

        What the sink has already queued is still going to be heard, so the samples between "the
        playback position right now" and "the last sample submitted" are counted as audible, not
        thrown away; everything after the queue is cancelled inside the sink. ``scope`` records whose
        playback stopped: only Voxa's own generations are ever in this timeline, so cancelling other
        applications' playback is a different, deliberate act (handoff 6.1) and is never implied here.
        """
        stamp = self._clock() if now is None else float(now)
        with self._lock:
            item = self._generations.get(str(generation))
            if item is None or item.state == "cancelled":
                return item
            played = self._index_at_time(stamp)
            if item.first_index is not None:
                heard = min(max(played - item.first_index, 0), item.submitted)
                item.audible = max(item.audible, heard)
                item.cancelled += max(0, item.submitted - heard)
            if item.generated > item.submitted:
                item.cancelled += item.generated - item.submitted
            item.state = "cancelled"
            item.closed_at = stamp
            if reason:
                item.notes.append(reason)
            return item

    def finish(self, generation: str, *, now: float | None = None) -> Generation | None:
        """This generation played to its end."""
        stamp = self._clock() if now is None else float(now)
        with self._lock:
            item = self._generations.get(str(generation))
            if item is None or item.state in ("finished", "cancelled"):
                return item
            played = self._index_at_time(stamp)
            if item.first_index is not None:
                item.audible = max(item.audible, min(max(played - item.first_index, 0), item.submitted))
            if item.generated > item.submitted:
                item.cancelled += item.generated - item.submitted
            item.state = "finished"
            item.closed_at = stamp
            return item

    # ------------------------------------------------------------------ the clock

    def audible_time(self, index: int, *, now: float | None = None) -> float | None:
        """Monotonic time the sample at output-stream ``index`` is estimated to leave the speakers."""
        index = int(index)
        with self._lock:
            anchor = self._anchor_for_index(index)
            if anchor is None:
                return None
            return anchor.audible_at + (index - anchor.first_index) / self.sample_rate

    def index_at_audible_time(self, audible_time: float) -> int:
        """Output-stream index that is estimated to be leaving the speakers at ``audible_time``."""
        with self._lock:
            return self._index_at_time(float(audible_time))

    def played_through(self, now: float | None = None) -> int:
        """How far the speakers have reached at ``now`` (estimated from the map)."""
        return self.index_at_audible_time(self._clock() if now is None else float(now))

    def _index_at_time(self, moment: float) -> int:
        anchor = self._anchor_for_time(moment)
        if anchor is None:
            return 0
        return int(round(anchor.first_index + (moment - anchor.audible_at) * self.sample_rate))

    def reference_window(self, *, mic_start: float, length: int, echo_travel: float = 0.0,
                         mono: bool = True) -> np.ndarray:
        """The rendered samples the microphone could be hearing for a chunk starting at ``mic_start``.

        ``mic_start`` is the monotonic time of the chunk's FIRST sample, already corrected for the
        capture latency (:class:`CaptureClock.stamp`); ``echo_travel`` is the acoustic time from the
        speakers to the microphone the canceller has fitted. The window is served from the render
        ring, so it contains exactly Voxa's own voice and nothing any other application played.
        """
        index = self.index_at_audible_time(float(mic_start) - float(echo_travel))
        window = self.ring.window(index, length)
        if mono and window.shape[1] == 1:
            return window[:, 0]
        if mono:
            return window.mean(axis=1).astype(np.float32)
        return window

    def counters(self) -> dict:
        """generated / submitted / cancelled / estimated-audible, over every tracked generation."""
        with self._lock:
            items = list(self._generations.values())
        out = {"generated": 0, "submitted": 0, "cancelled": 0, "audible": 0, "generations": len(items)}
        for item in items:
            out["generated"] += item.generated
            out["submitted"] += item.submitted
            out["cancelled"] += item.cancelled
            out["audible"] += item.audible
        return out

    def snapshot(self) -> dict:
        return {"epoch": self.epoch, "sample_rate": self.sample_rate, "channels": self.channels,
                "total_samples": self.total, "anchors": len(self._anchors),
                **self.counters()}

    # -- internals (callers hold the lock) --------------------------------------

    def _anchor_for_index(self, index: int) -> Chunk | None:
        found = None
        for chunk in self._anchors:
            if chunk.first_index <= index:
                found = chunk
            else:
                break
        if found is None and self._anchors:
            found = self._anchors[0]
        return found

    def _anchor_for_time(self, moment: float) -> Chunk | None:
        found = None
        for chunk in self._anchors:
            if chunk.audible_at <= moment:
                found = chunk
            else:
                break
        if found is None and self._anchors:
            found = self._anchors[0]
        return found


# --------------------------------------------------------------------------- the GStreamer tap

TAP_CAPS = "audio/x-raw,format=F32LE,layout=interleaved,rate={rate}"


def attach_render_tap(pipeline: Any, timeline: RenderTimeline, generation: str, path: str,
                      gst: Any = None, sink_element: Any = None,
                      sink_clock: Any = None) -> dict[str, Any]:
    """Put a tee immediately in front of the sink and record exactly the frames the sink is handed.

    ``sink_element`` is the element to tap; by default the pipeline's element named ``sink`` (what
    ``autoaudiosink name=sink`` leaves behind, and what a playbin's ``audio-sink`` property holds).
    The link that fed the sink is cut, fed through the tee instead, and the tee's other source pad is
    handed back to the sink, so the audio the user hears is untouched and the tap sees the same
    buffers, at the same moment, in a branch of its own.

    The tap branch converts and resamples to the canceller's rate, so what the timeline stores is the
    PCM after mixing, gain and ducking, immediately before the sink - not the encoded bytes, not the
    text, not the animation clock. The tap branch's own latency would seem to make every recorded
    moment late, but the sink clock is queried at that same late moment and its queue is shorter by
    exactly that much, so ``arrival + queue depth`` - the audible time the timeline computes - is
    unaffected by how long the tap branch took.

    Returns what it managed to attach; every failure is reported, never raised, because a reference
    tap must never be a reason for speech not to play.
    """
    made: dict[str, Any] = {"attached": False, "elements": {}, "generation": generation, "path": path}
    if pipeline is None or gst is None:
        made["error"] = "no pipeline or no GStreamer"
        return made
    try:
        sink = sink_element
        if sink is None:
            for name in ("sink", "audio-sink"):
                try:
                    sink = pipeline.get_by_name(name)
                except Exception:  # noqa: BLE001 - a pipeline that will not answer is not fatal
                    sink = None
                if sink is not None:
                    break
        if sink is None:
            # A playbin holds its sink in a property, not by name.
            try:
                sink = pipeline.get_property("audio-sink")
            except Exception:  # noqa: BLE001 - not every pipeline has one
                sink = None
        if sink is None:
            made["error"] = "no sink element to tap in front of"
            return made
        sink_pad = sink.get_static_pad("sink")
        upstream = sink_pad.get_peer() if sink_pad is not None else None
        if upstream is None or sink_pad is None:
            made["error"] = "nothing feeds the sink to tap"
            return made

        tee = gst.ElementFactory.make("tee")
        tap_sink = gst.ElementFactory.make("appsink")
        convert = gst.ElementFactory.make("audioconvert")
        resample = gst.ElementFactory.make("audioresample")
        capsfilter = gst.ElementFactory.make("capsfilter")
        queue = gst.ElementFactory.make("queue")
        if not all((tee, tap_sink, convert, resample, capsfilter, queue)):
            made["error"] = "GStreamer elements for the tap are unavailable"
            return made
        # GStreamer dropped ``async`` from the queue element in 1.28; set it only when this
        # build has it, the way the leaky/drop properties below are guarded. Unconditional
        # set_property("async", ...) raised on a queue without the property, which took the
        # whole reference tap with it (the tap failed to attach, the reference stayed silent
        # and Voxa interrupted her own reply). The audio itself was never the failure.
        if queue.find_property("async") is not None:
            queue.set_property("async", False)
        queue.set_property("max-size-buffers", 64)
        # Never leaky: a dropped buffer is a hole in the reference, and a hole in the reference is a
        # canceller that fits the wrong echo. The callback below drains the branch as it arrives.
        if queue.find_property("leaky") is not None:
            queue.set_property("leaky", 0)
        capsfilter.set_property("caps", gst.Caps.from_string(
            TAP_CAPS.format(rate=timeline.sample_rate) + ",channels=" + str(timeline.channels)))
        tap_sink.set_property("emit-signals", True)
        tap_sink.set_property("sync", False)
        tap_sink.set_property("max-buffers", 64)
        if tap_sink.find_property("drop") is not None:
            tap_sink.set_property("drop", False)
        tap_sink.connect("new-sample", _tap_callback(timeline, generation, path, gst, sink_clock))

        branch = (queue, convert, resample, capsfilter, tap_sink)
        for element in (tee,) + branch:
            pipeline.add(element)
        if not tee.link(queue) or not all(a.link(b) for a, b in zip(branch, branch[1:], strict=False)):
            made["error"] = "could not connect the tap branch"
            return made

        probe_branch = _attach_echo_probe(pipeline, tee, timeline, gst)
        if probe_branch:
            made["echo_probe"] = probe_branch

        # ``upstream`` and ``to_sink`` are PADS, not elements: GStreamer's Pad.link returns a
        # Gst.PadLinkReturn whose OK is the integer 0. ``not upstream.link(...)`` therefore read
        # a successful link as a failure and cut the audio path, leaving the reference silent.
        # Compare against PadLinkReturn.OK where this build has the enum, and against True for a
        # plain boolean link (the fake pipeline in the tests).
        upstream.unlink(sink_pad)
        if not _pad_link_ok(upstream.link(tee.get_static_pad("sink")), gst):
            _relink(upstream, sink_pad)
            made["error"] = "could not feed the tee from the playback path"
            return made
        to_sink = tee.request_pad_simple("src_%u")
        if to_sink is None or not _pad_link_ok(to_sink.link(sink_pad), gst):
            made["error"] = "could not hand the sink back from the tee"
            return made
        made["elements"] = {"tee": tee, "queue": queue, "audioconvert": convert,
                            "audioresample": resample, "capsfilter": capsfilter, "appsink": tap_sink,
                            "sink": sink}
        made["attached"] = True
        made["sink_clock"] = sink_clock
        return made
    except Exception as exc:  # noqa: BLE001 - multimedia boundary
        made["error"] = str(exc)
        return made


def _attach_echo_probe(pipeline: Any, tee: Any, timeline: RenderTimeline, gst: Any) -> dict[str, Any] | None:
    """Feed a ``webrtcechoprobe`` from the same tee, on Voxa's own playback branch (roadmap task 2).

    GStreamer documents the probe as belonging on the PLAYBACK path; Voxa's tap sits immediately in
    front of the sink, which is exactly that path, and it carries only her own render - never another
    application's sound. The probe registers itself under ``timeline.echo_probe_name`` and a
    ``webrtcdsp`` in the capture pipeline names it, so the GStreamer canceller's reference is the
    samples her sink was handed. Nothing here may break the sound: when the probe branch cannot be
    connected the tap keeps working and the failure is only reported.
    """
    name = getattr(timeline, "echo_probe_name", None)
    if not name:
        return None
    if gst.ElementFactory.find("webrtcechoprobe") is None:
        # The plugin is not installed here: report it, and let the tap itself work on.
        return {"attached": False, "name": name, "error": "webrtcechoprobe element unavailable"}
    make = gst.ElementFactory.make
    probe = make("webrtcechoprobe")
    probe_sink = make("fakesink")
    convert = make("audioconvert")
    resample = make("audioresample")
    capsfilter = make("capsfilter")
    queue = make("queue")
    if not all((probe, probe_sink, convert, resample, capsfilter, queue)):
        return {"attached": False, "error": "webrtcechoprobe elements unavailable"}
    if queue.find_property("async") is not None:
        queue.set_property("async", False)
    queue.set_property("max-size-buffers", 64)
    if queue.find_property("leaky") is not None:
        queue.set_property("leaky", 0)
    # The probe and the processor must agree on the format: 16 kHz mono S16LE, as verified live.
    capsfilter.set_property("caps", gst.Caps.from_string(
        "audio/x-raw,format=S16LE,layout=interleaved,channels=1,rate=16000"))
    probe.set_property("name", name)
    probe_sink.set_property("sync", False)
    branch = (queue, convert, resample, capsfilter, probe, probe_sink)
    for element in branch:
        pipeline.add(element)
    if not tee.link(queue) or not all(a.link(b) for a, b in zip(branch, branch[1:], strict=False)):
        return {"attached": False, "error": "could not connect the echo probe branch", "name": name}
    return {"attached": True, "name": name, "elements": {"tee": tee, "queue": queue,
                                                         "audioconvert": convert, "audioresample": resample,
                                                         "capsfilter": capsfilter,
                                                         "webrtcechoprobe": probe, "fakesink": probe_sink}}


def _relink(upstream: Any, sink_pad: Any) -> None:
    """Put back the link the tap cut, if anything can."""
    try:
        upstream.link(sink_pad)
    except Exception:  # noqa: BLE001 - best effort at not leaving the sound broken
        pass


def _pad_link_ok(result: Any, gst: Any) -> bool:
    """Did a Pad.link succeed?

    Real GStreamer :class:`Gst.Pad` returns a :class:`Gst.PadLinkReturn` whose ``OK`` is the
    integer 0, so ``if not link_result`` treats a good link as a failure. A plain fake pipeline
    returns ``True``. Accept both, and nothing else.
    """
    if result is True:
        return True
    pad_link_return = getattr(gst, "PadLinkReturn", None)
    if pad_link_return is not None:
        try:
            return result == pad_link_return.OK
        except Exception:  # noqa: BLE001 - an unexpected enum shape is not a reason to cut audio
            return False
    return bool(result)


def _tap_callback(timeline: RenderTimeline, generation: str, path: str, gst: Any,
                  sink_clock: Any):
    """The appsink callback: hand the decoded PCM to the timeline, never break playback."""

    def on_sample(sink):
        try:
            sample = sink.emit("pull-sample")
            if sample is None:
                return gst.FlowReturn.OK
            buffer = sample.get_buffer()
            success, mapped = buffer.map(gst.MapFlags.READ)
            if not success:
                return gst.FlowReturn.OK
            try:
                data = np.frombuffer(bytes(mapped.data), dtype=np.float32)
            finally:
                buffer.unmap(mapped)
            structure = sample.get_caps().get_structure(0) if sample.get_caps() else None
            channels = 1
            if structure is not None and structure.has_field("channels"):
                channels = max(1, int(structure.get_value("channels") or 1))
            if channels > 1 and data.size % channels == 0:
                data = data.reshape(-1, channels)
            timeline.submit(data, generation=generation, path=path, sink=sink_clock)
        except Exception as exc:  # noqa: BLE001 - the reference must never stop the sound
            log.info("render tap dropped a buffer: %s", exc)
        return gst.FlowReturn.OK

    return on_sample
