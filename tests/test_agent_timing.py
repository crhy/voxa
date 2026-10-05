"""Tests for voxa/agent/timing.py."""

from voxa.agent.timing import STAGES, Trace, summary


def test_trace_mark_valid_stage():
    t = Trace()
    t.mark("speech_end")
    assert "speech_end" in t._marks


def test_trace_mark_invalid_stage():
    t = Trace()
    try:
        t.mark("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_trace_first_mark_wins():
    t = Trace()
    t.mark("speech_end")
    first = t._marks["speech_end"]
    t.mark("speech_end")
    assert t._marks["speech_end"] == first


def test_trace_since_returns_ms():
    t = Trace()
    t.mark("speech_end")
    t.mark("transcribed")
    result = t.since("speech_end", "transcribed")
    assert isinstance(result, int)
    assert result >= 0


def test_trace_since_missing_returns_none():
    t = Trace()
    t.mark("speech_end")
    assert t.since("speech_end", "transcribed") is None


def test_trace_to_dict_empty():
    t = Trace()
    assert t.to_dict() == {}


def test_trace_to_dict_partial():
    t = Trace()
    t.mark("speech_end")
    t.mark("transcribed")
    d = t.to_dict()
    assert "transcribe_ms" in d
    assert "route_ms" not in d


def test_trace_to_dict_full():
    t = Trace()
    for stage in STAGES:
        t.mark(stage)
    d = t.to_dict()
    assert "transcribe_ms" in d
    assert "route_ms" in d
    assert "act_ms" in d
    assert "first_token_ms" in d
    assert "to_speech_ms" in d
    assert "total_ms" in d


def test_summary_empty():
    assert summary({}) == ""


def test_summary_total_only():
    assert summary({"total_ms": 400}) == "0.4 s"


def test_summary_with_parts():
    result = summary({"total_ms": 400, "transcribe_ms": 210, "act_ms": 80})
    assert result == "0.4 s (heard 0.21, acted 0.08)"


def test_summary_thread_safety():
    import threading
    t = Trace()
    threads = []
    for _ in range(10):
        th = threading.Thread(target=lambda: t.mark("speech_end"))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    assert "speech_end" in t._marks
