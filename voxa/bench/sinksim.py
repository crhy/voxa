"""Task 1's instrument: a simulated output sink with known latency and jitter, plus a second app.

The frozen B1 clip set models the room (delay, drift, a delay step, RT60, a nonlinearity, SER/SNR)
but not the machine: it has no output sink with a queue, no capture latency, and no other
application playing at the same time. Roadmap task 1 is exactly about those three things, so this
module puts them AROUND the frozen clips without touching them:

* :class:`SimulatedSink` plays Voxa's exact render (``ref.wav`` of the clip) through a queue with a
  known bulk latency, a per-buffer jitter the latency query cannot see coming, and a slow wander of
  the queue depth the query CAN see. It knows its own true play times; the product's
  :class:`voxa.timeline.RenderTimeline` only ever gets the latency query, which is what a real
  ``pulsesink``/``pipewiresink`` offers (``pa_stream_get_latency``, ``pw_stream_get_time_n``).
* the microphone is rebuilt from the clip's own components (``d`` shifted by the sink's true delay
  profile, plus ``s`` and ``n``), stamped with a capture latency the product must subtract.
* a second application's music plays through the same sink: it is in the sink's monitor and in the
  room, and it is never in Voxa's own render.

Two numbers come out, per clip:

``align_err_ms``
    |predicted play time - true play time| for every submitted chunk. The prediction is the
    timeline's map (submit time + the sink's latency query); the truth is the simulated sink's own
    schedule. This is the "reference-to-echo alignment error" of the report's task 1 gate.
``erle_music_loss_db``
    ERLE of Voxa's own canceller on Voxa's own echo, with no second app playing, minus the same
    ERLE while a second app's music plays through the same speakers. The reference is Voxa's own
    render in both cases, which is why the loss is small; the same measurement with the whole-sink
    MONITOR as the reference (what the product does today) is reported next to it and is much worse.

Nothing here opens a device. The clip files are read-only inputs and no frozen bench file is changed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..anc import EchoCanceller, estimate_delay
from ..audio import ReferenceRing, mix_chunk
from ..timeline import MODE_OWN_RENDER, CaptureClock, FixedSinkClock, RenderTimeline
from .render import SR

#: The output path the simulation models. A PipeWire/PulseAudio desktop sits near these numbers:
#: a 25-50 ms sink queue, buffer scheduling that moves by a millisecond or two, a clock a few tens of
#: parts per million off, and a microphone that reports its buffer a couple of tens of ms late.
SINK_BASE_LATENCY_MS = 40.0
SINK_JITTER_MS = 1.0
SINK_WANDER_MS = 6.0
SINK_DRIFT_PPM = 50.0
CAPTURE_LATENCY_MS = 20.0
#: Acoustic speakers-to-microphone travel time the canceller is allowed to fit (task 3's problem;
#: task 1 is the render side, so this part is given and held fixed).
ECHO_TRAVEL_MS = 20.0
#: The second application's music, relative to the clip's own echo level. 0 dB = as loud as Voxa's
#: echo at the microphone, which is what "the user is playing music while Voxa answers" means.
MUSIC_LEVEL_DB = 0.0

CHUNK_MS = 20
CHUNK_SAMPLES = SR * CHUNK_MS // 1000

#: The clock the simulation runs on. Any fixed origin works; a non-zero one keeps the arithmetic
#: honest about absolute monotonic times.
CLOCK_ORIGIN = 1000.0


@dataclass(slots=True)
class SinkProfile:
    """The machine being simulated: queue latency, jitter the query cannot see, clock drift."""

    base_ms: float = SINK_BASE_LATENCY_MS
    jitter_ms: float = SINK_JITTER_MS
    wander_ms: float = SINK_WANDER_MS
    drift_ppm: float = SINK_DRIFT_PPM
    seed: int = 0
    key: str = ""

    def _unit(self, buffer_index: int) -> float:
        """A deterministic pseudo-random value in [0, 1) for one submitted buffer."""
        digest = hashlib.sha256(f"{self.seed}:{self.key}:{buffer_index}".encode()).digest()
        return int.from_bytes(digest[:8], "little") / float(1 << 64)

    def wander(self, moment: float) -> float:
        """The queue depth's slow drift, in seconds: the latency query sees it, so may the timeline."""
        return (self.wander_ms / 1000.0) * float(np.sin(moment * 0.7))

    def query_latency(self, moment: float) -> float:
        """What the sink's latency query answers at ``moment``: honest about the queue, blind to jitter."""
        return (self.base_ms + self.wander_ms * 0.0) / 1000.0 + self.wander(moment)

    def jitter(self, buffer_index: int) -> float:
        """The extra scheduling delay of one buffer, unknown when its latency was queried."""
        return (self.jitter_ms / 1000.0) * self._unit(buffer_index)

    def rate(self, moment: float) -> float:
        """The sink's sample clock, ppm-locked to nothing."""
        return 1.0 + self.drift_ppm * 1e-6 * float(np.sin(moment * 0.11))


class SimulatedSink:
    """A sink that knows exactly when it plays each sample and answers a latency query honestly.

    The product code only ever sees :meth:`latency`, exactly as it sees ``pa_stream_get_latency`` or
    ``pw_stream_get_time_n``. The jitter it cannot see is what makes the alignment error non-zero:
    a timeline that ignored the query entirely would be wrong by the whole queue latency.
    """

    def __init__(self, profile: SinkProfile, clock: callable, rate: int = SR) -> None:
        self.profile = profile
        self.clock = clock
        self.rate = rate
        self._end: float | None = None      # true time the last queued sample will be played
        self._index = 0
        self._buffers = 0
        self.records: list[dict] = []       # one per submitted buffer: truth, not prediction

    def latency(self) -> float:
        """Seconds from now until a sample submitted right now would leave the speakers."""
        moment = float(self.clock())
        base = self.profile.query_latency(moment)
        if self._end is not None:
            queued = self._end - moment
            if queued > base:
                return queued
        return base

    def submit(self, samples: np.ndarray) -> dict:
        """Queue ``samples``; returns the sink's own record of when they will really be played."""
        arr = np.asarray(samples, dtype=np.float32)
        moment = float(self.clock())
        first = self._index
        start = moment + self.profile.query_latency(moment)
        if self._end is not None:
            # The sink plays one stream at a time: a buffer cannot start before the one before it.
            start = max(start, self._end)
        # The scheduling delay of THIS buffer is decided when the scheduler gets it, after any latency
        # query made at submission time: this is the part a render timeline cannot know in advance.
        start += self.profile.jitter(self._buffers)
        span = arr.shape[0] / (self.rate * self.profile.rate(moment))
        self._end = start + span
        record = {"first_index": first, "samples": arr.shape[0], "submitted_at": moment,
                  "true_audible_at": start, "true_end_at": self._end}
        self.records.append(record)
        self._index += arr.shape[0]
        self._buffers += 1
        return record

    def true_time(self, index: int) -> float:
        """The true monotonic play time of output-stream sample ``index``."""
        index = int(index)
        for record in reversed(self.records):
            if record["first_index"] <= index:
                return record["true_audible_at"] + (index - record["first_index"]) / (
                    self.rate * self.profile.rate(record["submitted_at"]))
        first = self.records[0] if self.records else {"true_audible_at": CLOCK_ORIGIN, "first_index": 0,
                                                      "submitted_at": CLOCK_ORIGIN}
        return first["true_audible_at"] + (index - first["first_index"]) / self.rate

    def index_at_time(self, moment: float) -> float:
        """Which output-stream sample the sink is playing at monotonic ``moment`` (fractional)."""
        moment = float(moment)
        for record in reversed(self.records):
            if record["true_audible_at"] <= moment:
                return record["first_index"] + (moment - record["true_audible_at"]) * (
                    self.rate * self.profile.rate(record["submitted_at"]))
        if self.records:
            first = self.records[0]
            return first["first_index"] + (moment - first["true_audible_at"]) * self.rate
        return 0.0


def music_stream(length: int, rng: np.random.Generator, level_rms: float) -> np.ndarray:
    """A second application's music: a beat, a chord, and a little of everything else."""
    t = np.arange(length, dtype=np.float64) / SR
    beat = 0.5 + 0.5 * np.exp(-8.0 * ((t % 0.5) - 0.02))
    chord = sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate((196.0, 247.0, 294.0, 392.0)))
    wash = rng.standard_normal(length)
    spectrum = np.fft.rfft(wash)
    freqs = np.fft.rfftfreq(length, 1 / SR)
    spectrum *= 1.0 / np.sqrt(1.0 + (freqs / 900.0) ** 2)
    wash = np.fft.irfft(spectrum, length)
    out = 0.75 * chord * beat + 0.4 * wash
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out = out / peak
    rms_value = float(np.sqrt(np.mean(out * out))) if out.size else 0.0
    if rms_value > 0:
        out = out * (level_rms / rms_value)
    return out.astype(np.float32)


def _shift_by_profile(x: np.ndarray, source: np.ndarray) -> np.ndarray:
    """``out[p] = x[source[p]]`` with linear interpolation and zeros outside the source."""
    grid = np.arange(x.shape[0], dtype=np.float64)
    return np.interp(source, grid, x, left=0.0, right=0.0).astype(np.float32)


def _erle_over(mic: np.ndarray, cleaned: np.ndarray, active: np.ndarray) -> float | None:
    """Median dB removed on the frames where the speakers were playing."""
    values: list[float] = []
    for start in range(0, min(mic.shape[0], cleaned.shape[0]), CHUNK_SAMPLES):
        if not active[start]:
            continue
        a = mic[start:start + CHUNK_SAMPLES]
        b = cleaned[start:start + CHUNK_SAMPLES]
        if a.size < CHUNK_SAMPLES or b.size < CHUNK_SAMPLES:
            continue
        power = float(np.mean(np.asarray(a, dtype=np.float64) ** 2))
        if power < 1e-12:
            continue
        left = max(float(np.mean(np.asarray(b, dtype=np.float64) ** 2)), 1e-20)
        values.append(10.0 * float(np.log10(power / left)))
    if not values:
        return None
    return float(np.median(np.asarray(values)))


def _pcm16(samples: np.ndarray) -> bytes:
    """Float -1..1 samples as the int16 little-endian PCM the capture path moves."""
    x = np.clip(np.rint(np.asarray(samples, dtype=np.float64) * 32767.0), -32768, 32767)
    return x.astype("<i2").tobytes()


def _run_canceller(mic: np.ndarray, reference: np.ndarray, delay: int) -> np.ndarray:
    """Voxa's own canceller over the whole clip, fed ``reference`` sample-for-sample."""
    canceller = EchoCanceller(delay=delay)
    ring = ReferenceRing(max(CHUNK_SAMPLES * 4, SR))
    out: list[np.ndarray] = []
    for start in range(0, mic.shape[0], CHUNK_SAMPLES):
        chunk = mic[start:start + CHUNK_SAMPLES]
        if chunk.size < CHUNK_SAMPLES:
            break
        ring.append(reference[start:start + CHUNK_SAMPLES])
        cleaned = mix_chunk(_pcm16(chunk), ring, start, canceller, 0)
        out.append(np.frombuffer(cleaned, dtype="<i2").astype(np.float32) / 32767.0)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def measure_clip(arrays: dict, truth: dict, *, seed: int = 0, music_db: float = MUSIC_LEVEL_DB,
                 chunk_samples: int = CHUNK_SAMPLES, profile: SinkProfile | None = None,
                 sink_query: bool = True) -> dict:
    """One clip through the simulated machine: alignment error and the music ERLE comparison."""
    ref = np.asarray(arrays["ref"], dtype=np.float32)
    d = np.asarray(arrays["d"], dtype=np.float32)
    s = np.asarray(arrays.get("s", np.zeros_like(ref)), dtype=np.float32)
    n = np.asarray(arrays.get("n", np.zeros_like(ref)), dtype=np.float32)
    length = ref.shape[0]
    if length == 0 or float(np.max(np.abs(ref))) < 1e-6:
        return {"measurable": False, "reason": "the clip has no Voxa playback to tap"}

    key = str(truth.get("id", ""))
    profile = profile or SinkProfile(seed=seed, key=key)
    state = {"t": CLOCK_ORIGIN}
    clock = lambda: state["t"]  # noqa: E731 - the simulation's only clock

    timeline = RenderTimeline(sample_rate=SR, clock=clock)
    generation = f"render:{key}"
    timeline.open(generation, "tts-file", now=CLOCK_ORIGIN)
    sink = SimulatedSink(profile, clock, SR)

    # 1. The tap and the reference, interleaved exactly as the live app does them: the render chunk
    #    goes to the sink at its real moment, and the microphone chunk of the same moment is served
    #    its reference from what the timeline holds THEN (a 3 s ring, never the whole clip).
    capture = CaptureClock(rate=SR, latency=CAPTURE_LATENCY_MS / 1000.0)
    errors: list[float] = []
    own_reference = np.zeros(length, dtype=np.float32)
    for start in range(0, length, chunk_samples):
        chunk = ref[start:start + chunk_samples]
        if chunk.size == 0:
            break
        take = chunk.size
        state["t"] = CLOCK_ORIGIN + (start + take) / SR
        # The tap asks the sink how deep its queue is BEFORE the buffer goes in: a probe on the sink
        # pad runs before the sink has taken the buffer, and a query after the write would report
        # this very buffer as extra latency.
        clock_reading = sink.latency()
        record = sink.submit(chunk)
        # ``sink_query=False`` is the pre-task-1 assumption, kept measurable on purpose: a design that
        # never asks the sink treats the buffer it is writing as audible right now. The alignment error
        # it produces is the size of the thing the render timeline removes.
        asked = FixedSinkClock(clock_reading) if sink_query else FixedSinkClock(0.0)
        chunk_record = timeline.submit(chunk, generation=generation, path="tts-file",
                                      sink=asked, now=state["t"])
        if chunk_record is not None:
            predicted = timeline.audible_time(record["first_index"])
            if predicted is not None:
                errors.append(abs(predicted - record["true_audible_at"]) * 1000.0)
        arrival = state["t"] + capture.capture_latency()
        mic_start, _end = capture.stamp(arrival, take)
        own_reference[start:start + take] = timeline.reference_window(
            mic_start=mic_start, length=take, echo_travel=0.0)
    timeline.finish(generation, now=CLOCK_ORIGIN + length / SR)

    # 2. The room. The clip's own echo ``d`` is aligned to the render grid: ``d[y]`` is the sound the
    #    room makes at grid time ``y`` from the render sample written ``bulk`` samples earlier. The
    #    sink plays render sample ``j`` at true time ``true_time(j)``, so at grid time ``g`` the room
    #    is answering the sample written ``bulk + travel`` samples before the one being played then.
    bulk = int((truth.get("echo") or {}).get("delay_samples") or 0)
    travel = int(round(ECHO_TRAVEL_MS / 1000.0 * SR))
    grid = np.arange(length, dtype=np.float64)
    lag = bulk + travel
    source = np.array([sink.index_at_time(CLOCK_ORIGIN + (p - lag) / SR) + bulk for p in grid])
    echo = _shift_by_profile(d, source)
    # What the speakers actually put in the air (the sink's own output, jitter and all): the sink's
    # monitor carries this, the room carries it delayed by the acoustic travel time.
    played_source = np.array([sink.index_at_time(CLOCK_ORIGIN + p / SR) for p in grid])
    played = _shift_by_profile(ref, played_source)
    mic_base = (echo + s + n).astype(np.float32)

    rng = np.random.default_rng([int(seed), int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],
                                                           "little")])
    echo_rms = float(np.sqrt(np.mean(d * d))) or 0.05
    music_out = music_stream(length, rng, echo_rms * 10.0 ** (-music_db / 20.0))
    # The second app shares the sink, so it inherits the sink's own delay profile; the room adds the
    # same acoustic travel time as Voxa's voice.
    music_sink = _shift_by_profile(music_out, played_source)
    music_mic = _shift_by_profile(music_sink, grid - lag)

    # 3. The reference the product serves came from the timeline in step 1: the microphone buffer
    #    arrives ``capture latency`` after it was sampled, and the timeline answers with the render
    #    samples whose predicted play time is that moment.

    # 4. The whole-sink monitor, which is what the product uses today: everything the sink played,
    #    Voxa's own voice and the other application's music together, aligned by cross-correlation.
    monitor = (played + music_sink).astype(np.float32)

    active = np.zeros(length, dtype=bool)
    for start in range(0, length, chunk_samples):
        window = ref[start:start + chunk_samples]
        active[start:start + chunk_samples] = bool(window.size and float(np.mean(window * window)) > 1e-6)

    delay_search = min(length - 1, lag + int(SINK_BASE_LATENCY_MS / 1000.0 * SR) + 1600)
    delay_own = int(estimate_delay(mic_base, own_reference, max_delay=delay_search))
    delay_monitor = int(estimate_delay(mic_base, monitor, max_delay=delay_search))

    # The report's own definition of the alignment error: the predicted echo time against the
    # cross-correlation peak of the microphone with the exact render. The peak says which render
    # sample the microphone is answering at the middle of the window; the timeline's map says which
    # sample it believes is being played then. The difference is the alignment error, measured
    # end-to-end (it also catches a capture latency that was never subtracted).
    xcorr: list[float] = []
    window = min(length, SR * 2)
    for start in range(0, max(1, length - window), window):
        mic_window = mic_base[start:start + window]
        ref_window = ref[start:start + window]
        if float(np.mean(mic_window * mic_window)) < 1e-8:
            continue
        if float(np.mean(ref_window * ref_window)) < 1e-6:
            continue  # nothing of Voxa's is playing in this window: no echo to line up
        peak = int(estimate_delay(mic_window, ref_window, max_delay=delay_search))
        if peak == 0:
            continue
        mid = start + window // 2
        # The peak says: the microphone at ``mid`` is answering render sample ``mid - peak``.
        heard_index = mid - peak
        # The timeline says: the sample being heard at ``mid`` (given the acoustic travel time the
        # canceller fits) is this one.
        predicted_index = timeline.index_at_audible_time(CLOCK_ORIGIN + (mid - lag) / SR)
        xcorr.append(abs(heard_index - predicted_index) / SR * 1000.0)

    def clean(mic: np.ndarray, reference: np.ndarray, delay: int) -> np.ndarray:
        return _run_canceller(mic, reference, delay)

    # Four runs of Voxa's own canceller, two references by two room conditions. The delay each path is
    # fitted with is the one that path's own cross-correlation finds, exactly as the live capture path
    # estimates it (voxa/audio.py:_maybe_estimate_delay).
    delay_monitor_quiet = int(estimate_delay(mic_base, played, max_delay=delay_search))
    erle_own = _erle_over(mic_base, clean(mic_base, own_reference, delay_own), active)
    mic_music = (mic_base + music_mic).astype(np.float32)
    erle_own_music = _erle_over(mic_music, clean(mic_music, own_reference, delay_own), active)
    erle_monitor = _erle_over(mic_base, clean(mic_base, played, delay_monitor_quiet), active)
    erle_monitor_music = _erle_over(mic_music, clean(mic_music, monitor, delay_monitor), active)

    def loss(before: float | None, after: float | None) -> float | None:
        """How far ERLE MOVES when a second application starts playing, in either direction.

        A reference that holds only Voxa's own render cannot be disturbed by what someone else is
        playing, so it moves by about nothing. A monitor reference moves a lot: the other app's audio
        enters the reference and the filter starts eating it, which changes the number without
        cancelling any more of Voxa's echo. Both directions are a failure of the measurement, so the
        metric is the size of the move, not its sign.
        """
        if before is None or after is None:
            return None
        return round(abs(before - after), 3)

    errors_array = np.asarray(errors, dtype=np.float64)
    out = {
        "measurable": True,
        "align_samples": int(errors_array.size),
        "align_err_median_ms": round(float(np.median(errors_array)), 4) if errors_array.size else None,
        "align_err_p90_ms": round(float(np.percentile(errors_array, 90.0)), 4) if errors_array.size else None,
        "align_err_max_ms": round(float(errors_array.max()), 4) if errors_array.size else None,
        "align_xcorr_p90_ms": (round(float(np.percentile(np.asarray(xcorr), 90.0)), 4)
                               if xcorr else None),
        "sink_latency_ms": round(profile.base_ms, 3),
        "sink_jitter_ms": round(profile.jitter_ms, 3),
        "capture_latency_ms": round(CAPTURE_LATENCY_MS, 3),
        "echo_travel_ms": round(ECHO_TRAVEL_MS, 3),
        "music_db": music_db,
        "reference_mode": MODE_OWN_RENDER if sink_query else MODE_OWN_RENDER + "-no-query",
        "sink_queried": bool(sink_query),
        "erle_own_db": None if erle_own is None else round(erle_own, 3),
        "erle_own_music_db": None if erle_own_music is None else round(erle_own_music, 3),
        "erle_monitor_db": None if erle_monitor is None else round(erle_monitor, 3),
        "erle_monitor_music_db": None if erle_monitor_music is None else round(erle_monitor_music, 3),
        "erle_music_loss_db": loss(erle_own, erle_own_music),
        "erle_monitor_music_loss_db": loss(erle_monitor, erle_monitor_music),
        "delay_own_samples": delay_own,
        "delay_monitor_samples": delay_monitor,
        "timeline": timeline.snapshot(),
    }
    return out
