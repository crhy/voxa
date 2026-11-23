"""Hearing 2 (task h01): the availability probes and the mode plumbing.

Every candidate arm answers the same question in the same shape - can it run on THIS machine, and
if not, exactly why. The answers are what the frontier table is built from, so a probe that lies
or a mode that quietly substitutes another arm is a measurement error, not a style problem.
"""

from __future__ import annotations

import pytest

from voxa.bench import modes, runner


def test_every_candidate_reports_its_state_and_a_reason():
    probe = modes.availability()
    assert set(probe) >= {"aec3full", "speex", "dtln", "nkf", "hybrid"}
    for name, state in probe.items():
        assert isinstance(state["available"], bool), name
        assert state["reason"].strip(), f"{name} gives no reason"


def test_hybrid_always_runs_and_says_which_linear_stage_it_picked():
    state = modes.availability()["hybrid"]
    assert state["available"] is True
    assert "linear" in state["reason"]


@pytest.mark.parametrize("mode", ("aec3full", "dtln", "nkf"))
def test_an_arm_that_cannot_run_here_refuses_instead_of_substituting(mode, capsys):
    """``run`` exits 2 with the probe's reason; it never measures another arm and calls it this."""
    if modes.availability()[mode]["available"]:
        pytest.skip(f"{mode} is measurable on this machine")
    code = runner.run(["--suite", "smoke", "--seed", "0", "--echo-mode", mode, "--limit", "1"])
    assert code == 2
    err = capsys.readouterr().err
    assert f"echo mode {mode} cannot run here" in err
    assert modes.availability()[mode]["reason"] in err


@pytest.mark.parametrize("mode", ("aec3full", "dtln", "nkf"))
def test_the_linear_factory_raises_the_same_reason(mode):
    with pytest.raises(RuntimeError) as exc:
        modes.make_linear(mode)
    assert "unavailable" in str(exc.value) or "no " in str(exc.value)


def test_speex_is_the_candidate_this_machine_can_actually_measure():
    """libspeexdsp is installed here (Debian libspeexdsp1); the bench arm really uses it."""
    ok, why = modes.speex_available()
    assert ok, why
    assert "libspeexdsp" in why


def test_the_new_arms_are_offered_by_the_command_line():
    """The parser accepts every new arm (and still rejects a name Hearing 2 never asked for)."""
    parser = runner.arg_parser()
    for mode in modes.HEARING2_MODES:
        assert parser.parse_args(["--echo-mode", mode]).echo_mode == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--echo-mode", "spectral-subtraction"])


def test_aec3full_records_the_build_route_it_was_asked_for(monkeypatch):
    """The task asks for a shim compiled from the freedesktop source; the probe says what is
    missing and honours an explicit shim path when someone builds one."""
    monkeypatch.delenv("VOXA_AEC3FULL_SHIM", raising=False)
    ok, why = modes.aec3full_available()
    assert ok is False
    assert "shim" in why
    monkeypatch.setenv("VOXA_AEC3FULL_SHIM", "/nonexistent/libvoxa_aec3full.so")
    assert modes.aec3full_available()[0] is False
