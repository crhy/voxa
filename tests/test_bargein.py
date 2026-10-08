from voxa.bargein import BargeInGate, is_own_voice, overlap


def test_overlap_table():
    assert overlap("", "hello world") == 0.0
    assert overlap("a b!", "a b") == 0.0
    assert overlap("Hello, World!", "world") == 0.5
    assert overlap("hello world", "HELLO World") == 1.0
    assert overlap("stop talking now", "talking") == 1 / 3
    assert overlap("tomorrow tomorrow", "tomorrow") == 1.0


def test_is_own_voice_own_sentence():
    sentence = "Please continue with the quarterly report"
    assert is_own_voice(sentence, sentence)
    assert is_own_voice("quarterly report", sentence)


def test_is_own_voice_real_interruption():
    sentence = "Please continue with the quarterly report"
    assert is_own_voice("wait what about tomorrow", sentence) is False
    assert is_own_voice("wait what about tomorrow", "the weather today", sentence) is False


def test_is_own_voice_noise():
    sentence = "Please continue with the quarterly report"
    assert is_own_voice("", sentence)
    assert is_own_voice("stop", sentence) is False  # a bare stop word is a real interruption
    assert is_own_voice("hmm", sentence)


def test_gate_opens_after_four_loud_ticks():
    gate = BargeInGate()
    assert gate.baseline == 0.0
    for _ in range(3):
        assert gate.update(0.5, gate.baseline) is False
    assert gate.update(0.5, gate.baseline) is True


def test_gate_tolerates_dips_between_syllables():
    # "Voxa, stop": loud, dip, loud, loud, dip, loud — four loud ticks among the last eight.
    gate = BargeInGate()
    results = [gate.update(level, gate.baseline) for level in (0.5, 0.01, 0.5, 0.5, 0.01, 0.5)]
    assert results[-1] is True and not any(results[:-1])


def test_gate_closes_again_after_quiet():
    gate = BargeInGate()
    for _ in range(4):
        gate.update(0.5, gate.baseline)
    for _ in range(8):
        last = gate.update(0.01, gate.baseline)
    assert last is False


def test_gate_follows_rising_baseline():
    gate = BargeInGate()
    for _ in range(60):
        gate.note_speaking_level(0.1)
    assert 0.09 < gate.baseline < 0.1
    for _ in range(8):
        assert gate.update(0.15, gate.baseline) is False  # louder than her, but under twice her level
    for _ in range(3):
        assert gate.update(0.5, gate.baseline) is False
    assert gate.update(0.5, gate.baseline) is True


def test_gate_stays_shut_at_baseline_level():
    gate = BargeInGate()
    for _ in range(20):
        gate.note_speaking_level(0.1)
    for _ in range(10):
        assert gate.update(gate.baseline, gate.baseline) is False


def test_gate_reset_clears_streak_only():
    gate = BargeInGate()
    for _ in range(20):
        gate.note_speaking_level(0.1)
    for _ in range(6):
        gate.update(0.5, gate.baseline)
    gate.reset()
    assert gate.update(0.5, gate.baseline) is False
    assert gate.baseline > 0.0


def test_a_single_stop_word_is_a_real_interruption():
    from voxa.bargein import is_own_voice

    speaking = "Larry Silverstein leased the World Trade Center shortly before the attacks."
    assert is_own_voice("stop", speaking) is False
    assert is_own_voice("Voxa", speaking) is False
    assert is_own_voice("uh", speaking) is True
    assert is_own_voice("stop", "Please stop the car now.") is True  # she is saying that word herself


def test_two_stray_words_are_her_own_voice_but_a_stop_word_always_counts():
    from voxa.bargein import is_own_voice

    speaking = "Edgar Allan Poe was a celebrated American author."
    assert is_own_voice("a celibate", speaking) is True
    assert is_own_voice("voxa stop", speaking) is False
    assert is_own_voice("box a stop", speaking) is False
    assert is_own_voice("what about his wife", speaking) is False
    assert is_own_voice("stop", "Say stop dictating when you are done.") is True


def test_wake_word_said_in_one_breath_with_a_soft_first_consonant():
    from voxa.conversation import strip_wake_word

    assert strip_wake_word("Box open Pluma", "voxa") == "open Pluma"
    assert strip_wake_word("Laksa, what's the weather tomorrow?", "voxa") == "what's the weather tomorrow"
    assert strip_wake_word("Locks a open the calculator", "voxa") == "open the calculator"
    assert strip_wake_word("Voxa open Pluma", "voxa") == "open Pluma"
    # Ordinary speech and other names must not wake her.
    for text in ("Lots of people came", "Comment, what time is it?", "Jasper, stop.", "Cosmo, open the calculator.",
                 "Boxing making note", "The box is open", "Open Pluma"):
        assert strip_wake_word(text, "voxa") is None, text


def test_her_own_voice_misheard_into_other_words_is_not_a_person():
    from voxa.bargein import is_own_voice, sounds_like

    # Traced on the real desktop: she said the first, Whisper wrote the second, and she stopped herself.
    assert sounds_like("Bmc, done, themes,", "BLC doesn't seem to be open.") >= 0.5
    assert is_own_voice("Bmc, done, themes,", "BLC doesn't seem to be open.", raw="Bmc, done, themes,") is True
    assert is_own_voice("stop", "Stopping dictation.", raw="Stop indictation.") is True
    # A person still gets through.
    spoken = "Edgar Allan Poe was a celebrated American author."
    assert is_own_voice("voxa stop", spoken, raw="was a celebrated American Voxa stop") is False
    assert is_own_voice("what about his wife", spoken, raw="what about his wife") is False
    assert is_own_voice("stop", spoken, raw="stop") is False
