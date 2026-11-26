"""vf04: speech cleanup regressions from vf02 (U+2217, underscores, empty replies)."""

from __future__ import annotations

from voxa.speech import SpeechService, _speakable_text

import voxa.speech as speech_mod


def _stream_text(monkeypatch, service: SpeechService, text: str) -> dict:
    """Run speech() with the streaming start stubbed; return what it collected."""
    calls: dict = {"spoken": [], "done": 0, "error": []}
    monkeypatch.setattr(speech_mod, "edge_tts", object())
    monkeypatch.setattr(
        service,
        "_start_natural_stream",
        lambda text, _rate, _voice, _cancel: calls["spoken"].append(text),
    )
    service.speak(
        text,
        180,
        "en-US-AriaNeural",
        on_started=lambda: None,
        on_done=lambda: calls.__setitem__("done", calls["done"] + 1),
        on_error=lambda error: calls["error"].append(error),
    )
    return calls


def test_asterisk_operator_u2217_is_stripped_again(monkeypatch) -> None:
    # vf02 dropped U+2217 (∗) from _ASTERISK_MARKS: a reply like "2 ∗ 3 = 6"
    # was read as "2 asterisk operator 3".
    service = SpeechService()
    calls = _stream_text(monkeypatch, service, "2 ∗ 3 = 6, and a ＊ mark too")
    assert calls["spoken"] == ["2  3 = 6, and a  mark too"]
    assert "∗" not in calls["spoken"][0]


def test_underscore_pairs_only_never_mangles_snake_case() -> None:
    # _MD_UNDERSCORE used to strip EVERY _ and ~: snake_case became snakecase
    # and "~5 minutes" lost its "about".  Emphasis PAIRS lose their markers and
    # keep the word; unpaired underscores in identifiers survive.
    assert _speakable_text("call snake_case please") == "call snake_case please"
    assert _speakable_text("file_name.txt stays too") == "file_name.txt stays too"
    assert _speakable_text("this is _emph_ and __strong__ text") == "this is emph and strong text"
    assert _speakable_text("strike ~~gone~~ kept") == "strike gone kept"


def test_leading_approx_tilde_says_about() -> None:
    assert _speakable_text("it takes ~5 minutes") == "it takes about 5 minutes"
    assert _speakable_text("~50 km away") == "about 50 km away"


def test_emoji_only_reply_finishes_at_once(monkeypatch) -> None:
    # Cleaning to "" must not hand "" to edge-tts (which raises "returned no
    # audio"): speak() finishes immediately with on_done, no TTS, no error.
    service = SpeechService()
    monkeypatch.setattr(speech_mod, "edge_tts", object())
    monkeypatch.setattr(
        service,
        "_start_natural_stream",
        lambda *args: (_ for _ in ()).throw(AssertionError("TTS must not be called")),
    )
    monkeypatch.setattr(
        service,
        "_start_natural_file",
        lambda *args: (_ for _ in ()).throw(AssertionError("TTS must not be called")),
    )
    monkeypatch.setattr(
        service,
        "_start_fallback_worker",
        lambda *args: (_ for _ in ()).throw(AssertionError("fallback must not be called")),
    )
    done: list[int] = []
    errors: list[str] = []
    service.speak(
        "😊 🎉 ** ~~ 🙈",
        180,
        "en-US-AriaNeural",
        on_started=lambda: (_ for _ in ()).throw(AssertionError("no speech started")),
        on_done=lambda: done.append(1),
        on_error=lambda error: errors.append(error),
    )
    assert done == [1]
    assert errors == []


def test_prepared_emoji_only_reply_never_fetches(monkeypatch) -> None:
    # prepare() of the same emoji-only reply is ready at once with no audio and
    # no error, and speak_prepared finishes it without touching an engine.
    service = SpeechService()
    monkeypatch.setattr(speech_mod, "edge_tts", object())
    monkeypatch.setattr(
        service,
        "_fetch_to_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no fetch")),
    )
    item = service.prepare("😊 🎉", 180, "en-US-AriaNeural")
    assert item.ready.wait(1)
    assert item.text == ""
    assert item.error == ""
    assert item.path == ""

    done: list[int] = []
    errors: list[str] = []
    monkeypatch.setattr(speech_mod, "_ui_once", lambda callback, *args: callback(*args))
    service.speak_prepared(item, on_started=lambda: None, on_done=lambda: done.append(1),
                           on_error=lambda error: errors.append(error))

    # the prepared worker runs on its own thread; give it a moment
    import time
    for _ in range(300):
        if done:
            break
        time.sleep(0.01)
    assert done == [1]
    assert errors == []
