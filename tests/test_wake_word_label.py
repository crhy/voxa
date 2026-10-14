from voxa import tips, wakelabel, welcome
from voxa.agent import helptext


def test_hints_follow_the_wake_word():
    wakelabel.set_wake("computer")
    try:
        assert wakelabel.wake() == "Computer"
        ready = tips.tips_for("READY", {})
        assert ready and all("Voxa" not in tip for tip in ready)
        assert any("Computer" in tip for tip in ready)
        assert "Voxa" not in welcome.how_to() and "Computer" in welcome.how_to()
        assert "Computer, pause" in helptext.topic_help("me")
        assert wakelabel.apply("Say “Voxa” to continue") == "Say “Computer” to continue"
    finally:
        wakelabel.set_wake("voxa")
    assert wakelabel.apply("Say “Voxa” to continue") == "Say “Voxa” to continue"
    assert "Voxa" in welcome.how_to()


def test_empty_wake_word_falls_back():
    wakelabel.set_wake("")
    assert wakelabel.wake() == "Voxa"
