from voxa.bargein import SETTLED_SAMPLES, BargeInGate, is_own_voice, overlap


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
    # Task A2 replaced the length rule: a short transcript is UNCERTAINTY, not her voice. "hmm" is
    # no longer deleted as echo (which would also delete "yes"), and it is not heard as a person
    # either - the candidate is held, the audio kept, nothing stripped.
    from voxa.bargein import source_decision
    from voxa.attribution import ECHO, UNCERTAIN, USER

    sentence = "Please continue with the quarterly report"
    assert source_decision("", sentence).source == UNCERTAIN
    assert source_decision("stop", sentence).source == USER  # a bare stop word is a real interruption
    assert source_decision("hmm", sentence).source == UNCERTAIN
    assert is_own_voice("hmm", sentence) is False  # not proven echo
    assert is_own_voice(sentence, sentence) is True


def test_gate_opens_after_four_loud_ticks():
    gate = BargeInGate()
    for _ in range(SETTLED_SAMPLES):  # teach the baseline so the gate trusts it
        gate.note_speaking_level(0.0)
    assert gate.baseline == 0.0
    for _ in range(3):
        assert gate.update(0.5, gate.baseline) is False
    assert gate.update(0.5, gate.baseline) is True


def test_gate_tolerates_dips_between_syllables():
    # "Voxa, stop": loud, dip, loud, loud, dip, loud — four loud ticks among the last eight.
    gate = BargeInGate()
    for _ in range(SETTLED_SAMPLES):  # baseline heard her voice first
        gate.note_speaking_level(0.0)
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
    from voxa.bargein import source_decision
    from voxa.attribution import ECHO, UNCERTAIN, USER

    speaking = "Larry Silverstein leased the World Trade Center shortly before the attacks."
    assert source_decision("stop", speaking, level_ratio=3.0).source == USER
    # Her name is a cue, not evidence: it neither stops her nor proves a second source alone.
    assert source_decision("Voxa", speaking, level_ratio=3.0).source != ECHO
    assert source_decision("uh", speaking, level_ratio=3.0).source == UNCERTAIN
    # She is saying that word herself: the words match her sentence, so the audio and the words
    # agree there is nothing to interrupt with. What must never happen is her stopping herself.
    assert source_decision("stop", "Please stop the car now.", level_ratio=3.0).source != USER


def test_two_stray_words_are_her_own_voice_but_a_stop_word_always_counts():
    from voxa.bargein import source_decision
    from voxa.attribution import ECHO, USER

    speaking = "Edgar Allan Poe was a celebrated American author."
    assert source_decision("a celibate", speaking, level_ratio=3.0).source == ECHO
    for heard in ("voxa stop", "box a stop", "what about his wife"):
        assert source_decision(heard, speaking, level_ratio=3.0).source != ECHO, heard
    # Her own sentence contains the word: never a person, and never a reason to stop her.
    assert source_decision("stop", "Say stop dictating when you are done.",
                           level_ratio=3.0).source != USER


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
    from voxa.bargein import source_decision, sounds_like
    from voxa.attribution import ECHO, USER

    # Traced on the real desktop: she said the first, Whisper wrote the second, and she stopped herself.
    assert sounds_like("Bmc, done, themes,", "BLC doesn't seem to be open.") >= 0.5
    assert source_decision("Bmc, done, themes,", "BLC doesn't seem to be open.",
                           raw="Bmc, done, themes,", level_ratio=3.0).source == ECHO
    # "Stopping dictation" misheard as "Stop indictation": the stop word is hers. The old rule called
    # this her voice outright; the evidence cannot separate the two, so it holds - and holding is what
    # must be enough, because the outcome that matters is that she does not stop herself.
    assert source_decision("stop", "Stopping dictation.", raw="Stop indictation.",
                           level_ratio=3.0).source != USER
    # A person still gets through.
    spoken = "Edgar Allan Poe was a celebrated American author."
    assert source_decision("voxa stop", spoken, raw="was a celebrated American Voxa stop",
                           level_ratio=3.0).source == USER
    assert source_decision("what about his wife", spoken, raw="what about his wife",
                           level_ratio=3.0).source != ECHO
    assert source_decision("stop", spoken, raw="stop", level_ratio=3.0).source == USER


def test_her_own_sentence_with_an_interrupt_word_in_it_is_her():
    # Both traced on 14 October: she stopped herself reading the Ten Commandments. The guarantee the
    # task states is that she never stops herself: the candidate is either her echo or undecided,
    # never a person, and no command runs from an undecided candidate.
    from voxa.bargein import source_decision
    from voxa.attribution import ECHO, USER

    speaking = "**Do not commit adultery.**\n2. **Do not murder.**\n3. **Do not steal.**"
    assert source_decision("3. Do not stop.", speaking, raw="3. Do not stop.",
                           level_ratio=3.0).source != USER
    speaking = "**Do not lie.**\n5. **Do not falsify accounts or cheat.**\n6. **Do not bear false witness.**"
    raw = "Voxa. Commands or cheat. Voxa. Commands or cheat. Voxa. Do not bear false witness."
    assert source_decision("voxa commands or cheat voxa commands or cheat voxa do voxa do", speaking,
                           raw=raw, level_ratio=3.0).source != USER
    # A person interrupting the same sentence still gets through.
    for heard in ("stop", "Voxa, stop.", "Stop, stop.", "Please stop.", "Wait, wait.",
                  "Voxa, what about lying?"):
        assert source_decision(heard, speaking, raw=heard, level_ratio=3.0).source != ECHO, heard