"""B1: the seeded synthetic clip renderer (``python -m voxa.bench render``).

Tiny generated signals only: the sources are faked, so nothing here touches the network, the
LibriSpeech corpus, faster-whisper, ffmpeg or a sound device. What is checked is what the roadmap
gate asks for: the realised SER, the injected delay, the truth times inside the clip, byte-identical
re-renders with the same seed, and every scenario of the smoke matrix present.
"""

from __future__ import annotations

import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from voxa.bench import render

SR = render.SR


def _tiny_speech(text: str, seconds: float, seed: int) -> np.ndarray:
    """Broadband pseudo-speech: three bursts of filtered noise, silence between them."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SR))
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spectrum *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    x = np.fft.irfft(spectrum, n)
    envelope = np.zeros(n)
    for burst in range(3):
        start = min(n - 1, int((0.1 + 0.35 * burst) * SR))
        stop = min(n, start + int(0.18 * SR))
        envelope[start:stop] = np.abs(np.sin(np.linspace(0.0, 3.0 * np.pi, stop - start)))
    x = x * envelope
    peak = float(np.max(np.abs(x))) or 1.0
    return (0.4 * x / peak).astype(np.float32)


def _tiny_source(text: str, seconds: float, seed: int, voice: str) -> render.Source:
    words = text.split()
    step = seconds / (len(words) + 1)
    ticks = [
        {"word": word, "onset_s": step * (index + 0.1), "offset_s": step * (index + 0.9)}
        for index, word in enumerate(words)
    ]
    return render.Source(key=f"tiny-{abs(hash(text)) % 9999}", text=text, voice=voice, engine="tiny",
                         samples=_tiny_speech(text, seconds, seed), words=tuple(ticks))


@pytest.fixture
def rendered(tmp_path: Path, monkeypatch) -> tuple[Path, dict[str, bytes]]:
    """Render the smoke suite twice into a temporary data root from tiny fake sources."""
    monkeypatch.setattr(render, "voice_source",
                       lambda text, voice, rate, root, kind: _tiny_source(text, 0.9, 11, voice))
    fake_corpus = [tmp_path / f"book-{n}.flac" for n in range(64)]
    monkeypatch.setattr(render, "librispeech_corpus", lambda root: fake_corpus)
    monkeypatch.setattr(render, "pick_librispeech", lambda corpus, need, seed: corpus[:need])
    monkeypatch.setattr(render, "librispeech_source",
                       lambda path, root: _tiny_source("the long user request over her voice", 1.2, 23,
                                                       "librispeech"))
    data_root = tmp_path / "data"
    manifests: dict[str, bytes] = {}
    for name in ("m1.json", "m2.json"):
        manifest_path = tmp_path / ".voxa-bench" / name
        code = render.main(["--suite", "smoke", "--seed", "0", "--data-root", str(data_root),
                            "--manifest", str(manifest_path)])
        assert code == 0
        manifests[name] = manifest_path.read_bytes()
    return data_root, manifests


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), "<i2").astype(np.float32) / 32768.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _correlation_peak(target: np.ndarray, reference: np.ndarray, max_delay: int = 5000) -> int:
    window = min(len(target), len(reference), 5 * SR)
    size = 1
    while size < 2 * window:
        size <<= 1
    corr = np.fft.irfft(np.fft.rfft(target[:window], size) * np.conj(np.fft.rfft(reference[:window], size)))
    return int(np.argmax(np.abs(corr[: max_delay + 1])))


def _truth(data_root: Path, entry: dict) -> dict:
    return json.loads((data_root / "smoke" / entry["id"] / "truth.json").read_text())


def test_smoke_plan_holds_every_scenario_of_the_matrix() -> None:
    items = render.plan("smoke", 0)
    assert len(items) == 48
    assert {item["scenario"] for item in items} == set(render.SCENARIO_COUNTS)
    for scenario, count in render.SCENARIO_COUNTS.items():
        assert sum(1 for item in items if item["scenario"] == scenario) == count
    assert {item["expect"] for item in items} == {"keep", "yield", "backchannel", "none"}
    # Voxa alone includes her stop, cancel and wake-word wording.
    fst_texts = " ".join(item["tts_text"] for item in items if item["scenario"] == "fst").lower()
    assert "stop" in fst_texts and "cancel" in fst_texts and "voxa" in fst_texts
    # Every short answer of the handoff matrix is spoken by the user over Voxa.
    assert {item["user_text"] for item in items if item["scenario"] == "short"} == set(render.SHORT_ANSWERS)
    # Whole commands, taken from tools/anc_lab.py's COMMANDS list, are also user utterances.
    assert any(item["user_text"] in render.COMMAND_PHRASES for item in items)
    # Matched pairs: three pairs, Voxa alone in one clip and the user over Voxa in the other.
    pairs = [item for item in items if item["scenario"] == "pair"]
    assert len(pairs) == 6
    assert sorted(item["pair_id"] for item in pairs) == sorted(p["id"] for p in render.PAIRS * 2)
    assert all(item["expect"] == "none" for item in pairs if item["pair_side"] == "voxa")
    assert {item["expect"] for item in pairs if item["pair_side"] == "user"} == {"yield", "keep"}
    for pair in render.PAIRS:
        voxa_side = next(i for i in pairs if i["pair_id"] == pair["id"] and i["pair_side"] == "voxa")
        user_side = next(i for i in pairs if i["pair_id"] == pair["id"] and i["pair_side"] == "user")
        assert voxa_side["tts_text"].lower().strip(".?!") == user_side["user_text"].lower().strip(".?!")
    # The DT positions of the matrix are all present, the roadmap fractions are used, and the
    # pause is 0.4-1.5 s.
    assert {item["position"] for item in items if item["scenario"].startswith("dt_")} == {
        "onset", "first_syllable", "middle", "last_word", "echo_tail"}
    assert {item["dt_fraction"] for item in items if item["scenario"] == "dt_middle"} == {0.2, 0.5, 0.8}
    assert all(0.4 <= item["pause_s"] <= 1.5 for item in items if item["scenario"] == "pause")


def test_full_suite_is_ten_times_smoke() -> None:
    assert len(render.plan("full", 0)) == 480
    assert render.plan("smoke", 0) == render.plan("smoke", 0)  # the plan itself is deterministic


def test_realised_ser_and_snr_stay_within_half_a_db(rendered) -> None:
    data_root, manifests = rendered
    checked = 0
    for entry in json.loads(manifests["m1.json"])["clips"]:
        truth = _truth(data_root, entry)
        levels = truth["levels"]
        clip = data_root / "smoke" / entry["id"]
        s, d, n = (_read_wav(clip / f"{name}.wav") for name in ("s", "d", "n"))
        if levels["ser_db_target"] is not None and levels["ser_db_realised"] is not None:
            realised = 20.0 * np.log10(_rms(s) / _rms(d))
            assert abs(realised - levels["ser_db_target"]) <= 0.5, (entry["id"], realised)
            assert abs(realised - levels["ser_db_realised"]) < 0.01
            checked += 1
        if levels["snr_db_target"] is not None and levels["snr_db_realised"] is not None:
            reference = _rms(s) if _rms(s) > 1e-9 else _rms(d)
            realised = 20.0 * np.log10(reference / _rms(n))
            assert abs(realised - levels["snr_db_target"]) <= 0.5, (entry["id"], realised)
    assert checked >= 30


def test_quiet_user_clips_sit_at_minus_ten_db(rendered) -> None:
    data_root, manifests = rendered
    quiet = [entry for entry in json.loads(manifests["m1.json"])["clips"] if entry["scenario"] == "quiet"]
    assert len(quiet) == 3
    for entry in quiet:
        levels = _truth(data_root, entry)["levels"]
        assert levels["ser_db_target"] == -10.0
        assert abs(levels["ser_db_realised"] + 10.0) <= 0.5


def test_injected_delay_is_measurable_within_one_sample(rendered) -> None:
    data_root, manifests = rendered
    clean = []
    for entry in json.loads(manifests["m1.json"])["clips"]:
        truth = _truth(data_root, entry)
        echo = truth["echo"]
        if echo and echo["drift_ppm"] == 0.0 and echo["delay_step_ms"] is None and echo["nonlinearity"] is None:
            clean.append((entry, truth))
    assert clean, "the plan must contain a clip with no drift, no step and no nonlinearity"
    for entry, truth in clean:
        clip = data_root / "smoke" / entry["id"]
        echo, reference = _read_wav(clip / "d.wav"), _read_wav(clip / "ref.wav")
        peak = _correlation_peak(echo, reference)
        assert abs(peak - truth["echo"]["delay_samples"]) <= 1, (entry["id"], peak, truth["echo"])
        assert abs(truth["echo"]["delay_ms"] - peak * 1000.0 / SR) <= 1000.0 / SR


def test_truth_times_lie_inside_the_clip(rendered) -> None:
    data_root, manifests = rendered
    for entry in json.loads(manifests["m1.json"])["clips"]:
        truth = _truth(data_root, entry)
        seconds = truth["seconds"]
        assert abs(truth["frames"] / SR - seconds) < 1e-9
        for block in ("playback", "user"):
            section = truth[block]
            if section is None:
                continue
            assert 0.0 <= section["onset_s"] <= section["offset_s"] <= seconds, (entry["id"], block)
            previous = 0.0
            for word in section["words"]:
                assert 0.0 <= word["onset_s"] < word["offset_s"] <= seconds, (entry["id"], block, word)
                assert word["onset_s"] >= previous - 1e-9, (entry["id"], block, word)
                previous = word["offset_s"]
        if truth["user"] and truth["user"]["pause_s"]:
            gaps = [b["onset_s"] - a["offset_s"]
                    for a, b in zip(truth["user"]["words"], truth["user"]["words"][1:], strict=False)]
            assert max(gaps) >= truth["user"]["pause_s"] - 1e-9


def test_mix_is_the_sum_of_the_components(rendered) -> None:
    data_root, manifests = rendered
    for entry in json.loads(manifests["m1.json"])["clips"]:
        clip = data_root / "smoke" / entry["id"]
        s, d, n, mic = (_read_wav(clip / f"{name}.wav") for name in ("s", "d", "n", "mic"))
        assert len(mic) == len(s) == len(d) == len(n)
        assert float(np.max(np.abs(mic - (s + d + n)))) < 2.0 / 32768.0
        assert float(np.max(np.abs(mic))) <= 1.0


def test_a_source_built_this_run_matches_the_cached_source(tmp_path: Path, monkeypatch) -> None:
    """A source rendered now must be bit-identical to the same source read back from the cache."""
    monkeypatch.setattr(render, "edge_voice", lambda text, voice, rate, tmp: (
        np.linspace(-0.31, 0.31, 400, dtype=np.float64).astype(np.float32),
        [{"word": text, "onset_s": 0.0, "offset_s": 0.025}]))
    root = tmp_path / "data"
    first = render.voice_source("Blue.", render.USER_TTS_VOICE, render.VOXA_TTS_RATE, root, "user_phrases")
    second = render.voice_source("Blue.", render.USER_TTS_VOICE, render.VOXA_TTS_RATE, root, "user_phrases")
    assert first.engine == "edge_tts"
    assert np.array_equal(first.samples, second.samples)
    assert first.words == second.words


def test_same_seed_renders_identical_bytes(rendered) -> None:
    data_root, manifests = rendered
    assert manifests["m1.json"] == manifests["m2.json"]
    manifest = json.loads(manifests["m1.json"])
    assert manifest["suite"] == "smoke" and manifest["seed"] == 0 and manifest["count"] == 48
    for entry in manifest["clips"]:
        for name, digest in entry["files"].items():
            on_disk = hashlib.sha256((clip_dir(data_root, entry, name)).read_bytes()).hexdigest()
            assert digest == on_disk, (entry["id"], name)


def clip_dir(data_root: Path, entry: dict, name: str) -> Path:
    return data_root / "smoke" / entry["id"] / name


def test_outputs_hold_no_wall_clock_values(rendered) -> None:
    data_root, manifests = rendered
    manifest = json.loads(manifests["m1.json"])
    payload = manifests["m1.json"].decode() + json.dumps(_truth(data_root, manifest["clips"][0]))
    assert not re.search(r"generated|timestamp|created|modified|elapsed|wall|now|started", payload, re.I)


def test_renderer_writes_the_five_streams_and_the_truth_file(rendered) -> None:
    data_root, manifests = rendered
    entry = json.loads(manifests["m1.json"])["clips"][0]
    assert set(entry["files"]) == {"s.wav", "d.wav", "n.wav", "mic.wav", "ref.wav", "truth.json"}
    truth = _truth(data_root, entry)
    for key in ("id", "scenario", "expect", "sr", "frames", "seconds", "playback", "user", "echo", "levels"):
        assert key in truth
    assert truth["sr"] == SR
