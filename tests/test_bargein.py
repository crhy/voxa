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


def test_gate_settles_and_recognises_a_towering_voice():
    from voxa.bargein import BargeInGate

    gate = BargeInGate()
    assert not gate.towering(1.0)  # no baseline yet: never certain
    for _ in range(25):
        gate.note_speaking_level(0.03)  # her own voice at the microphone
    assert gate.settled
    assert not gate.towering(0.1)  # louder than her, but not beyond doubt
    assert gate.towering(0.8)  # a person at the desk (measured level)
