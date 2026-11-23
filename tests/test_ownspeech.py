from __future__ import annotations

from voxa.conversation import ConversationController
from voxa.ownspeech import OwnSpeech, normalise

SENTENCE = "The weather in Portland is sunny with a high of seventy two degrees"


def test_normalise():
    assert normalise("The weather in Portland is sunny with a high of seventy two degrees") == [
        "the", "weather", "in", "portland", "is", "sunny", "with", "a", "high", "of", "seventy", "two", "degrees",
    ]
    assert normalise("Hello, world!") == ["hello", "world"]
    assert normalise("72 degrees") == ["72", "degrees"]
    assert normalise("") == []
    assert normalise("...!?") == []


def test_echo_of_full_sentence():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.active()
    assert os.is_echo("the weather in portland is sunny")


def test_strip_keeps_command_words():
    os = OwnSpeech()
    os.started(SENTENCE)
    remaining, share = os.strip("weather in Portland is sunny Voxa stop", protect=["voxa"])  # noqa: B005
    assert remaining == "voxa stop"
    assert share > 0.0
    assert not os.is_echo("weather in Portland is sunny Voxa stop", protect=["voxa"])


def test_unrelated_untouched():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.strip("what time is it") == ("what time is it", 0.0)  # noqa: B005
    assert not os.is_echo("what time is it")


def test_fuzzy_echo():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.is_echo("the whether in portland is sunni")


def test_chance_words_untouched():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.strip("is it") == ("is it", 0.0)  # noqa: B005
    assert not os.is_echo("is it")


def test_expires_after_keep_seconds():
    fake = [0.0]
    os = OwnSpeech(keep_seconds=6.0, clock=lambda: fake[0])
    os.started(SENTENCE)
    os.finished()
    fake[0] = 7.0
    assert not os.active()
    assert os.strip("the weather in portland is sunny") == ("the weather in portland is sunny", 0.0)  # noqa: B005
    assert not os.is_echo("the weather in portland is sunny")


def test_three_sentences_in_a_row():
    os = OwnSpeech()
    os.started("The weather in Portland is sunny")
    os.started("It is seventy two degrees today")
    os.started("Have a nice afternoon")
    assert os.is_echo("the weather in portland is sunny")
    assert os.is_echo("it is seventy two degrees")
    assert os.is_echo("have a nice afternoon")


def test_empty_and_punctuation():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.strip("") == ("", 0.0)
    assert not os.is_echo("")
    assert os.strip("...!?") == ("...!?", 0.0)  # noqa: B005
    assert not os.is_echo("...!?")


def test_controller_path_drops_echo_and_strips_mixed():
    class _Whisper:
        def transcribe(self, *args, **kwargs):
            return ""

    class _Service:
        def model_name(self):
            return "tiny"

    prompts: list[str] = []
    controller = ConversationController(
        wake_whisper=_Whisper(),
        prompt_whisper=_Whisper(),
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=400,
        max_segment_seconds=8.0,
        on_woken=lambda: None,
        on_prompt=lambda text: prompts.append(text),
        on_status=lambda text: None,
        on_error=lambda text: None,
        on_transcribe_timing=lambda *args: None,
    )
    os = OwnSpeech()
    os.started(SENTENCE)
    controller.own_speech = os

    # A pure echo is dropped: the filter returns None, so no prompt callback runs.
    assert controller._filter_own_speech("the weather in portland is sunny") is None
    assert prompts == []

    # A mixed utterance survives with only her own word runs removed.
    mixed = controller._filter_own_speech("the weather in portland is sunny what time is it")
    assert mixed is not None
    assert "weather" not in mixed
    assert "what time is it" in mixed


def test_a_lone_stop_word_after_her_sentence_is_not_echo():
    from voxa.ownspeech import OwnSpeech

    own = OwnSpeech()
    own.started("and horror novelist. He is best known for his works The Raven and The Tell-Tale Heart.")
    heard = "and horror novelist he is best known stop"
    assert own.is_echo(heard, protect=["voxa"]) is False
    assert own.strip(heard, protect=["voxa"])[0] == "stop"
    assert own.is_echo("and horror novelist he is best known for his works", protect=["voxa"]) is True


# ------------------------------------------------------------------ roadmap task 7: word-timed evidence

def test_word_timeline_matches_only_within_the_window():
    from voxa.ownspeech import OwnSpeech

    own = OwnSpeech()
    own.started("the weather in portland is sunny")
    own.note_played_boundaries(
        [("the", 0.0, 0.3), ("weather", 0.3, 0.4), ("portland", 0.7, 0.4), ("sunny", 1.4, 0.5)],
        generation="reply-1",
    )
    # "weather" heard at 0.35 lands within her audible "weather" onset 0.30 (0.05 s): time-matched.
    # "portland" heard at 2.00 is 1.30 s from her audible "portland" onset 0.70: not matched.
    tight = own.timed_overlap([("weather", 0.35, 0.50), ("portland", 2.0, 2.4)], window_ms=400.0)
    assert tight["matched"] == 1 and tight["words"] == 2 and tight["share"] == 0.5
    # Widening the window to 600 ms reaches the far word too (1.20 - 0.70 = 0.50 s of slack).
    wide = own.timed_overlap([("weather", 0.35, 0.50), ("portland", 1.20, 1.50)], window_ms=600.0)
    assert wide["matched"] == 2 and wide["longest_run"] == 2


def test_word_window_sweep_200_400_600():
    from voxa.ownspeech import OwnSpeech

    own = OwnSpeech()
    own.started("cancel the download")
    own.note_played_boundaries([("cancel", 1.0, 0.4), ("download", 1.5, 0.5)], generation="g")
    heard = [("cancel", 1.15, 1.45), ("download", 1.85, 2.25)]
    # "cancel" onset 1.15 is 0.15 s from hers (every window); "download" onset 1.85 is 0.35 s
    # from hers (1.50), so it reaches at 400/600 but not at 200.
    assert own.timed_overlap(heard, window_ms=200.0)["matched"] == 1
    assert own.timed_overlap(heard, window_ms=400.0)["matched"] == 2
    assert own.timed_overlap(heard, window_ms=600.0)["matched"] == 2


def test_word_timeline_never_deletes_a_confirmed_near_end_utterance():
    from voxa.ownspeech import OwnSpeech

    own = OwnSpeech()
    own.started("the weather in portland is sunny")
    own.note_played_boundaries(
        [("weather", 0.3, 0.4), ("portland", 0.7, 0.4), ("sunny", 1.4, 0.5)], generation="g"
    )
    # A fully time-matched overlap is still only evidence: a CONFIRMED near-end utterance (the A2
    # rule / handoff 5.3) is never rewritten, even when the audio would otherwise say echo.
    heard = "weather portland sunny"
    heard_times = [("weather", 0.30, 0.60), ("portland", 0.70, 1.00), ("sunny", 1.40, 1.70)]
    assert own.timed_overlap(heard_times)["share"] == 1.0
    kept, share = own.strip(heard, evidence="echo", confirmed_user=True)
    assert kept == heard          # nothing removed
    assert share == 0.0           # the bare text overlap is not what decided


def test_item_being_read_resolves_the_anaphor():
    from voxa.ownspeech import OwnSpeech

    own = OwnSpeech()
    own.started("reading the recipe now")
    own.note_played_boundaries(
        [("reading", 0.0, 0.3), ("recipe", 0.4, 0.4), ("now", 0.9, 0.3)], generation="g"
    )
    # "that one" said at 0.5 s resolves to the item on her tongue at that audible moment.
    item = own.item_being_read(0.5)
    assert item is not None and item.word == "recipe"
    assert own.item_being_read(0.35) is None  # between words: nothing was audible then


def test_word_timed_overlap_feeds_attribution_as_one_weighted_feature():
    from voxa.attribution import SourceEvidence, attribute

    def decide(timed):
        ev = SourceEvidence(
            reference_active=True, alignment_healthy=True,
            echo_explained_db=-5.0, residual_over_prediction_db=-2.0, vad_prob=0.2,
            text_overlap=0.7, text_sound=0.7, word_timed_overlap=timed,
        )
        return attribute(ev)

    base = decide(0.0)
    lifted = decide(1.0)
    # Both say echo (the audio already does); the time-matched word only adds confidence, it never
    # flips a user decision on its own.
    assert base.source == lifted.source
    assert lifted.confidence >= base.confidence
