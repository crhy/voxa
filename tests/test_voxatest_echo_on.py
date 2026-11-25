"""Issue #72, step 3: a live VoxaTest run with echo cancellation ON, and DTLN keeps the prompts.

The old guard (voxatest/__main__.py refused every non-off echo mode) was written for an echo
canceller that listened to the WHOLE system output. DTLN's reference is Voxa's OWN playback
samples, so VoxaTest's spoken prompts - played by another process, never in that reference -
are not what it cancels. The guard now refuses only the modes that would cancel the prompts
(whole-system-output modes like webrtc, unless VoxaTest plays to its own device).

The DTLN proof below is offline: bench room audio (tests/anc_signals.py) plus a real speech
clip as the test prompt. It must show her echo removed and the prompt surviving: the prompt is
never in the reference, and a canceller that cannot see it must not remove it.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

import anc_signals
from voxa import dtln
from voxatest.config import HarnessSettings, refuses_echo_mode

SR = dtln.SR


# --------------------------------------------------------------------------- the guard

def test_dtln_and_voxas_own_modes_are_allowed() -> None:
    """Modes whose reference is her own playback can never cancel the test prompts."""
    for mode in ("dtln", "voxa", "aec3"):
        assert not refuses_echo_mode(mode, ""), mode
    assert not refuses_echo_mode("off", "")
    assert not refuses_echo_mode("", "")


def test_whole_system_output_modes_are_still_refused() -> None:
    """webrtc (and anything unknown) cancels the whole output: the prompts would vanish."""
    assert refuses_echo_mode("webrtc", "")
    assert refuses_echo_mode("not-a-mode", "")


def test_a_separate_output_device_still_excuses_webrtc() -> None:
    """VoxaTest playing to its own device keeps the prompts out of the system monitor."""
    assert not refuses_echo_mode("webrtc", "alsasink")


def test_run_preflight_allows_dtln(tmp_path, monkeypatch, capsys) -> None:
    """The refusal that made v05 run echo OFF no longer fires for DTLN."""
    class UnreachableClient:
        def list_models(self):
            raise OSError("connection refused")

    settings = HarnessSettings(voxa_echo_mode="dtln")
    monkeypatch.setattr(HarnessSettings, "make_client", lambda _self: UnreachableClient())

    import voxatest.__main__ as voxatest_main

    code = voxatest_main._run(
        settings, [], None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md")
    )
    err = capsys.readouterr().err
    # The run proceeds past the echo guard (it then stops on the unreachable backend, exit 2
    # like the sibling test) - the refusal message must never mention echo cancellation.
    assert "Echo cancellation" not in err
    assert "backend is not reachable" in err
    assert code == 2


def test_run_preflight_still_refuses_webrtc_without_a_device(tmp_path, capsys) -> None:
    import voxatest.__main__ as voxatest_main

    settings = HarnessSettings(voxa_echo_mode="webrtc")
    code = voxatest_main._run(
        settings, [], None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md")
    )
    assert code == 2
    assert "Echo" in capsys.readouterr().err


# --------------------------------------------------------------------------- DTLN offline proof

@pytest.fixture(scope="module")
def prompt_clip() -> np.ndarray:
    """A real speech clip: what VoxaTest's prompts sound like through the microphone."""
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    with wave.open(str(repo / "voxa" / "ui" / "assets" / "benchmark-speech.wav")) as handle:
        assert handle.getframerate() == SR
        return np.frombuffer(handle.readframes(handle.getnframes()), "<i2").astype(np.float32) / 32768.0


def _feed(canceller, mic, ref, chunk):
    out = []
    for i in range(0, len(mic), chunk):
        out.append(canceller.process(mic[i:i + chunk], ref[i:i + chunk]))
    out.append(canceller.process(np.zeros(0, "float32"), np.zeros(0, "float32")))
    return np.concatenate(out)


def test_dtln_removes_her_voice_and_keeps_the_prompt(prompt_clip) -> None:
    """Bench room audio + a prompt played by another process: the echo goes, the prompt stays."""
    if not dtln.dtln_available()[0]:
        pytest.skip("DTLN stage unavailable: " + dtln.dtln_available()[1])
    ref, echo, _bench_speech, _delay = anc_signals.make(seconds=7, seed=0)
    n = len(ref)
    prompt = np.zeros(n, dtype=np.float32)
    for start in (0.5, 3.0, 5.5):  # three prompt bursts spaced over her playback
        s = int(start * SR)
        chunk = prompt_clip[: int(0.9 * SR)]
        if s + len(chunk) <= n:
            prompt[s:s + len(chunk)] = chunk
    mic = (echo + prompt).astype(np.float32)

    canceller = dtln.DtlnCanceller(SR)
    cleaned = _feed(canceller, mic, ref, 320)
    m = min(len(mic), len(cleaned))

    def db(power):
        return 10.0 * np.log10(power + 1e-20)

    # Where only her echo played (before her first prompt burst), the reference can cancel it.
    echo_only = np.arange(m) < int(0.4 * SR)
    removed = db(np.mean(mic[:m][echo_only] ** 2)) - db(np.mean(cleaned[:m][echo_only] ** 2))
    assert removed >= 15.0, f"her voice not removed: only {removed:.1f} dB"

    # During a prompt the user/test voice dominates: DTLN's reference does not hold it, so
    # window energy must survive (the whole point of running live tests with DTLN ON).
    win = 3200
    bursts = [(int(s * SR), int(s * SR) + int(0.9 * SR)) for s in (0.5, 3.0, 5.5)]
    worst = 0.0
    for start, end in bursts:
        for i in range(start, end - win + 1, win):
            if np.mean(prompt[i:i + win] ** 2) < 1e-4:
                continue
            attenuation = db(np.mean(prompt[i:i + win] ** 2)) - db(np.mean(cleaned[i:i + win] ** 2))
            worst = max(worst, attenuation)
    assert worst <= 6.0, f"prompt attenuated by {worst:.1f} dB - it is not in the reference"
