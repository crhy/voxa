"""Tests for the B2 file-driven runner (voxa/bench/runner.py) and the factored barge-in class.

Everything is offline and fast: two tiny generated clips, a stubbed ASR and a stubbed voice
detector, so no faster-whisper model, no network, no sound device and no sleeping.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from voxa.bargectl import IGNORED, INTERRUPT, OFFLINE, OWN_VOICE
from voxa.bench.render import write_wav
from voxa.bench.runner import CHUNK_SAMPLES, ClipRunner, load_clip


class StubASR:
    """A fake faster-whisper: the words depend on how much audio it was handed."""

    model_name = "stub"

    def __init__(self) -> None:
        self.calls = 0
        self.audio_seconds = 0.0
        self.processing_seconds = 0.0
        self.hints: list[str] = []

    def transcribe_words(self, pcm: bytes, hint: str = "") -> dict:
        self.calls += 1
        self.hints.append(hint)
        seconds = len(pcm) / 32000.0
        self.audio_seconds += seconds
        # The barge-in candidate is transcribed without a hint (the controller never ran
        # ``start()``, so its prompt hint is empty); utterances carry the wake hint.
        text = "stop" if hint == "" else "Voxa what time is it"
        return {
            "text": text,
            "words": [{"word": word, "onset_s": 0.0, "offset_s": round(seconds, 3)}
                      for word in text.split()],
            "no_speech_prob": 0.02,
            "audio_seconds": round(seconds, 3),
            "asr_seconds": 0.0,
            "segments": 1,
        }

    def transcribe(self, pcm_s16le: bytes, language: str = "en", hint: str = "") -> str:
        return self.transcribe_words(pcm_s16le, hint)["text"]


class StubDetector:
    """A fake Silero detector: always voice, so the loudness rule alone decides."""

    threshold = 0.5

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.last_prob = 0.9

    def reset(self) -> None:
        self.last_prob = 0.9

    def feed(self, pcm: bytes) -> bool:
        self.last_prob = 0.9
        return True

    @property
    def voiced(self) -> bool:
        """Hysteresis state of an always-voiced stream (f183's widened arriving signal)."""
        return self.last_prob >= self.threshold


@pytest.fixture(autouse=True)
def _stub_detector(monkeypatch):
    monkeypatch.setattr("voxa.voicedetect.VoiceDetector", StubDetector)


def level_signal(seconds: float, level: int, rate: int = 16000) -> np.ndarray:
    """A constant-amplitude stretch: the microphone level the pipeline will report."""
    samples = int(round(seconds * rate))
    wave = np.tile([level, -level], samples // 2 + 1)
    return (wave[:samples] / 32768.0).astype(np.float32)


def write_clip(root: Path, suite: str, clip_id: str, mic: np.ndarray, ref: np.ndarray,
               truth: dict) -> Path:
    clip_dir = root / suite / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    write_wav(clip_dir / "mic.wav", mic)
    write_wav(clip_dir / "ref.wav", ref)
    (clip_dir / "truth.json").write_text(json.dumps(truth))
    return clip_dir


def make_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    """Two clips: one where only the user speaks, one where Voxa is speaking and the user cuts in."""
    root = tmp_path / "data"

    user_only = np.concatenate([
        level_signal(0.3, 0),
        level_signal(1.6, 2000),
        level_signal(1.0, 0),
    ])
    write_clip(root, "tiny", "tiny-0000-nst-00", user_only, np.zeros(len(user_only), np.float32), {
        "id": "tiny-0000-nst-00", "scenario": "nst", "expect": "keep", "playback": None,
    })

    reply = np.concatenate([
        level_signal(1.2, 600),      # her own voice: grace, then the gate learns her level
        level_signal(1.6, 2600),     # the user talks over it: the gate opens, words are judged
        level_signal(1.0, 600),
        level_signal(1.0, 0),
    ])
    # Voxa keeps playing for 3.4 s: long enough for the canceller's 3 s delay estimate too.
    played = np.concatenate([level_signal(3.4, 500),
                             np.zeros(len(reply) - int(3.4 * 16000), dtype=np.float32)])
    write_clip(root, "tiny", "tiny-0001-dt_middle-00", reply, played, {
        "id": "tiny-0001-dt_middle-00", "scenario": "dt_middle", "expect": "keep",
        "playback": {"text": "The weather is clear today.", "onset_s": 0.0, "offset_s": 1.8,
                     "words": [{"word": "The", "onset_s": 0.2, "offset_s": 0.5},
                               {"word": "weather", "onset_s": 0.5, "offset_s": 0.9},
                               {"word": "is", "onset_s": 0.9, "offset_s": 1.05},
                               {"word": "clear", "onset_s": 1.05, "offset_s": 1.35},
                               {"word": "today", "onset_s": 1.35, "offset_s": 1.8}]},
    })
    return root, {"id": "tiny-0000-nst-00"}, {"id": "tiny-0001-dt_middle-00"}


def events_of(runner: ClipRunner, name: str) -> list[dict]:
    return [event for event in runner.events if event["event"] == name]


def event_names(runner: ClipRunner) -> list[str]:
    return [event["event"] for event in runner.events]


def test_frames_are_20_ms_chunks_with_contiguous_ids(tmp_path: Path) -> None:
    root, _clip, _other = make_fixture(tmp_path)
    loaded = load_clip(root / "tiny" / "tiny-0000-nst-00")
    runner = ClipRunner(asr=StubASR(), echo_mode="off", wake_word="voxa")
    summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])

    frames = events_of(runner, "frame")
    expected = math.ceil(len(loaded["mic"]) / CHUNK_SAMPLES)
    assert len(frames) == expected == summary["frames"]
    assert [event["frame"] for event in frames] == list(range(expected))
    assert [event["samples"] for event in frames][:3] == [[0, 320], [320, 640], [640, 960]]
    # The simulated clock advances 20 ms per frame: no sleeping, no wall-clock drift.
    assert [event["t_s"] for event in frames[:4]] == [0.02, 0.04, 0.06, 0.08]
    assert summary["seconds"] == round(expected * CHUNK_SAMPLES / 16000.0, 3)


def test_a_finished_utterance_emits_pause_transcript_endpoint_request_in_order(tmp_path: Path) -> None:
    root, _clip, _other = make_fixture(tmp_path)
    loaded = load_clip(root / "tiny" / "tiny-0000-nst-00")
    asr = StubASR()
    runner = ClipRunner(asr=asr, echo_mode="off", wake_word="voxa")
    summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])

    names = event_names(runner)
    for event in ("pause", "transcript", "endpoint", "request"):
        assert event in names, names
    assert (names.index("pause") < names.index("transcript") < names.index("endpoint")
            < names.index("request"))
    request = events_of(runner, "request")[0]
    # The wake word is stripped before the text would reach the assistant.
    assert request["text"] == "what time is it"
    assert request["kind"] == "wake"
    assert request["no_speech_prob"] == 0.02
    assert [word["word"] for word in request["words"]] == ["Voxa", "what", "time", "is", "it"]
    # The request is reported at the frame where the pause boundary fired.
    assert request["frame"] == events_of(runner, "pause")[0]["frame"]
    assert summary["requests"] == ["what time is it"]


def test_talking_over_a_reply_confirms_the_barge_in_and_stops_the_reply(tmp_path: Path) -> None:
    root, _clip, _other = make_fixture(tmp_path)
    loaded = load_clip(root / "tiny" / "tiny-0001-dt_middle-00")
    runner = ClipRunner(asr=StubASR(), echo_mode="off", wake_word="voxa")
    summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])

    names = event_names(runner)
    assert names[0] == "tts-start"
    assert "suspect" in names and "barge-in-confirm" in names and "tts-stop" in names
    assert names.index("suspect") < names.index("barge-in-confirm") < names.index("tts-stop")
    assert summary["counts"]["barge_in"] == 1
    confirm = events_of(runner, "barge-in-confirm")[0]
    assert confirm["heard"] == "stop"
    # The muted microphone never segments her own playback: the first pause comes after the reply.
    assert names.index("tts-stop") < names.index("pause")


def test_echo_mode_off_leaves_the_mic_untouched_and_voxa_cleans_it(tmp_path: Path) -> None:
    root, _clip, _other = make_fixture(tmp_path)
    loaded = load_clip(root / "tiny" / "tiny-0001-dt_middle-00")

    raw = ClipRunner(asr=StubASR(), echo_mode="off", wake_word="voxa")
    raw_summary = raw.run(loaded["mic"], loaded["ref"], loaded["truth"])
    assert all(event["erle_db"] == 0.0 for event in events_of(raw, "frame"))
    assert all(event["delay_samples"] == 0 for event in events_of(raw, "frame"))
    assert raw_summary["echo_mode"] == "off"

    cleaned = ClipRunner(asr=StubASR(), echo_mode="voxa", wake_word="voxa")
    cleaned_summary = cleaned.run(loaded["mic"], loaded["ref"], loaded["truth"])
    assert cleaned_summary["echo_mode"] == "voxa"
    assert any(event["erle_db"] != 0.0 for event in events_of(cleaned, "frame"))
    # Roadmap task 3: this synthetic mic carries no echo at all, so the tracker's coherence
    # guard must not invent a delay; the smoke suite clips (and the real ones) do have echo,
    # and there the estimates fire (delay_est_err_ms is measured from them).
    assert not events_of(cleaned, "delay-estimate"), "no echo in the mic: no delay may be claimed"
    assert events_of(cleaned, "frame")[0]["muted"] is True   # the reply starts at frame 0
    assert events_of(cleaned, "frame")[-1]["muted"] is False  # playback ended before the clip did


def test_webrtc_mode_reports_a_degraded_run_instead_of_claiming_cancellation(tmp_path: Path) -> None:
    root, _clip, _other = make_fixture(tmp_path)
    loaded = load_clip(root / "tiny" / "tiny-0001-dt_middle-00")
    runner = ClipRunner(asr=StubASR(), echo_mode="webrtc", wake_word="voxa")
    summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])
    if summary["echo_mode"] == "webrtc":  # a machine with GStreamer really cancels
        return
    assert summary["degraded"].startswith("webrtc unavailable")
    assert events_of(runner, "echo-mode")[0]["available"] is False


def test_the_runner_reads_a_rendered_clip_from_disk(tmp_path: Path) -> None:
    root, _clip, _other = make_fixture(tmp_path)
    loaded = load_clip(root / "tiny" / "tiny-0000-nst-00")
    assert loaded["truth"]["id"] == "tiny-0000-nst-00"
    assert loaded["mic"].dtype == np.dtype("<i2")
    assert len(loaded["mic"]) == len(loaded["ref"])


def open_the_gate(control, clock, candidate: str, her_level: float = 0.15,
                  user_level: float = 0.65) -> float:
    """Her own steady level settles the gate's baseline; a louder user then opens it.

    A constant level never opens the gate (that is what "it is still her voice" means), so the
    helper feeds two phases and returns the simulated time the gate opened. Each sample says
    how much AUDIO it stands for (``dt_ms``): the windows are counted in milliseconds of sound,
    not in callbacks.
    """
    clock["now"] = 0.5
    for _ in range(40):
        clock["now"] += 0.05
        control.observe(her_level, speaking=True, muted=True, wake_word="voxa",
                        candidate=lambda: (candidate, candidate), dt_ms=50.0)
    for _ in range(40):
        clock["now"] += 0.05
        outcome = control.observe(user_level, speaking=True, muted=True, wake_word="voxa",
                                  candidate=lambda: (candidate, candidate), dt_ms=50.0)
        if outcome.reason == "listening" and outcome.gate_opened:
            return clock["now"]
    raise AssertionError("the level gate never opened")


def judge(control, clock, level: float, heard: str, **kwargs):
    """Keep feeding audio until the collected words are judged (the listening window is audio).

    Returns the outcome that decided, or None when the evidence never decided.
    """
    for _ in range(40):
        clock["now"] += 0.05
        outcome = control.observe(level, speaking=True, muted=True, wake_word="voxa",
                                  candidate=lambda: (heard, heard), dt_ms=50.0, **kwargs)
        if outcome.action != IGNORED:
            return outcome
    return None


def test_barge_in_decision_judges_the_first_frame_of_the_reply() -> None:
    """No grace period: the first frame of a reply is already judged (task A3).

    What restarts when she starts speaking is the EVIDENCE - the gate's baseline, the onset
    detector's run of unexplained sound - not a fold over the microphone. The first frame is
    therefore judged, and what it reports is the gate still learning her level.
    """
    from voxa.bargectl import BargeInController, IGNORED

    clock = {"now": 0.0}
    control = BargeInController(clock=lambda: clock["now"])
    control.reply_started("The weather is clear today.")

    clock["now"] = 0.1
    first = control.observe(0.15, speaking=True, muted=True, wake_word="voxa",
                            candidate=lambda: ("stop", "stop"), dt_ms=50.0)
    assert first.action == IGNORED and first.reason == "gate-closed", first.reason
    assert control.onset.audio_ms > 0.0, "the onset evidence did not start at the first frame"

    opened_at = open_the_gate(control, clock, "stop")
    listening = control.observe(0.65, speaking=True, muted=True, wake_word="voxa",
                                candidate=lambda: ("stop", "stop"), dt_ms=50.0)
    assert listening.reason == "listening"

    decided = judge(control, clock, 0.65, "stop")
    assert decided is not None and decided.action == INTERRUPT, decided.reason
    assert decided.heard == "stop"
    assert control.speaking_since == 0.0, "the interrupt branch ends the reply"


def test_barge_in_decision_rejects_her_own_words_and_takes_offline_words_seriously() -> None:
    from voxa.bargectl import BargeInController

    clock = {"now": 0.0}
    control = BargeInController(clock=lambda: clock["now"])
    control.reply_started("Do not steal the ring.")
    open_the_gate(control, clock, "do not steel the wrong")
    own = judge(control, clock, 0.65, "do not steel the wrong")
    assert own.action == OWN_VOICE
    assert control.pending_since == 0.0, "her own words must not leave the gate open"

    clock = {"now": 0.0}
    control = BargeInController(clock=lambda: clock["now"])
    control.reply_started("The weather is clear today.")
    open_the_gate(control, clock, "Voxa, go offline")
    offline = judge(control, clock, 0.65, "Voxa, go offline")
    assert offline.action == OFFLINE


# --------------------------------------------------------------------------- f168: the voxa mode
# measures the SHIPPING path (voxa.align.ExactDelayLock), and the f164 locked arms stay as
# the comparison. On a synthetic clip with a known bulk delay + clock drift the two must
# hand the canceller the SAME locked delay, frame for frame.

def _broadband(n: int, seed: int) -> np.ndarray:
    """A broadband reference: a delay cannot be measured from pure tones."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / 16000.0)
    spec *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    return (0.5 * ref / np.max(np.abs(ref))).astype(np.float64)


def _room(seed: int, tail_s: float = 0.02) -> np.ndarray:
    """Room impulse response: a direct spike then a decaying tail."""
    rng = np.random.default_rng(seed)
    n = int(tail_s * 16000)
    tail = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.008 * 16000))
    h = 0.3 * tail / np.sqrt(np.sum(tail ** 2))
    h[0] += 1.0
    return h.astype(np.float64)


def _echo_at(ref: np.ndarray, delay_fn) -> np.ndarray:
    """ref delayed by ``delay_fn(t_seconds)`` samples (linear interp -> fractional lag)."""
    n = len(ref)
    t = np.arange(n) / 16000.0
    idx = np.arange(n, dtype=np.float64) - np.asarray(delay_fn(t), dtype=np.float64)
    return np.interp(idx, np.arange(n, dtype=np.float64), ref, left=0.0, right=0.0)


def test_the_voxa_mode_runs_the_shipping_lock_and_matches_the_locked_arm(tmp_path: Path) -> None:
    """f168: mode ``voxa`` is production's path (lock drives the canceller while proven, the
    estimate is only the honest fallback), and it must land the known delay as precisely as
    the ``voxa-locked`` arm - same lock, same numbers, no private bench alignment."""
    root = tmp_path / "data"

    def delay(t):  # 100 ms bulk delay + 300 ppm clock drift: known ground truth
        return 1600.0 + 300.0 / 1e6 * 16000 * t

    ref_src = _broadband(int(3.0 * 16000), 8)
    mic = np.convolve(_echo_at(ref_src, delay), _room(9))[: len(ref_src)]
    rng = np.random.default_rng(10)
    user = np.zeros(len(ref_src))
    user[int(1.6 * 16000):] = 0.05 * rng.standard_normal(len(ref_src) - int(1.6 * 16000))
    clip_dir = write_clip(root, "lock", "lock-0000-echo-00",
                          (mic + user).astype(np.float32), ref_src.astype(np.float32), {
                              "id": "lock-0000-echo-00", "scenario": "echo", "expect": "keep",
                              "playback": {"text": "The weather is clear today.",
                                           "onset_s": 0.0, "offset_s": 3.0}})
    loaded = load_clip(clip_dir)

    errors = {}
    for mode in ("voxa", "voxa-locked"):
        runner = ClipRunner(asr=StubASR(), echo_mode=mode, wake_word="voxa")
        summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])
        assert summary["echo_mode"] == mode
        frames = [event for event in events_of(runner, "frame") if event.get("delay_confident")]
        assert frames, mode
        est = np.array([event["delay_samples"] for event in frames], dtype=np.float64)
        truth = np.array([delay(event["t_s"]) for event in frames], dtype=np.float64)
        errors[mode] = est - truth
        assert max(abs(errors[mode])) <= 5.0, (mode, max(abs(errors[mode])))  # <= 0.31 ms
        assert events_of(runner, "delay-estimate"), mode
    # the shipping path locks the delay exactly like the comparison arm: identical frames
    assert len(errors["voxa"]) == len(errors["voxa-locked"])
    assert np.array_equal(errors["voxa"], errors["voxa-locked"])


def test_the_word_timed_timeline_is_fed_from_the_truth_once_per_clip(tmp_path: Path) -> None:
    """f175: the clip's playback words (edge-tts boundaries) become the word-timed timeline,
    fed once per clip with the LOCKED delay shift; the joint gate's per-frame verdicts are kept
    so the drop rule can ask whether it was closed at a heard word's moment."""
    root, _clip, other = make_fixture(tmp_path)
    loaded = load_clip(root / "tiny" / "tiny-0001-dt_middle-00")
    runner = ClipRunner(asr=StubASR(), echo_mode="off", wake_word="voxa",
                        word_window_ms=400.0)
    runner.run(loaded["mic"], loaded["ref"], loaded["truth"])
    played = runner.own_speech.played_words()
    assert played and {w.word for w in played} >= {"the", "weather", "today"}
    assert runner._timeline_fed
    # echo_mode off: no lock, so the shift is 0.0 and the spans keep the truth's times.
    weather = next(w for w in played if w.word == "weather")
    assert weather.start == 0.5 and weather.end == 0.9
    # Every frame recorded a joint verdict for the drop rule to consult.
    assert len(runner._joint_history) == runner._frame + 1
    first_t, first_closed = runner._joint_history[0]
    assert runner._joint_closed_at(first_t) is bool(first_closed)


def test_the_word_window_flag_reaches_the_runner_and_the_report(tmp_path: Path) -> None:
    from voxa.bench import runner as runner_module
    parser = runner_module.arg_parser()
    args = parser.parse_args(["--suite", "smoke", "--word-window-ms", "200"])
    assert args.word_window_ms == 200.0
    r = ClipRunner(asr=StubASR(), echo_mode="off", word_window_ms=200.0)
    assert r._word_window_ms == 200.0
