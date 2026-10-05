from __future__ import annotations

from voxa.ui.phonemes import timeline, weights_at, word_to_visemes

EXPECTED = {
    "ship": ["viseme_CH", "viseme_I", "viseme_PP"],  # sh->CH, i->I, p->PP
    "the": ["viseme_TH"],  # th->TH, silent final e
    "movie": ["viseme_PP", "viseme_U", "viseme_FF", "viseme_I"],  # m, o before v->U, v, ie->I
    "hello": ["viseme_E", "viseme_nn", "viseme_O"],  # h silent, e, ll merged, o
    "think": ["viseme_TH", "viseme_I", "viseme_nn", "viseme_kk"],
    "this": ["viseme_TH", "viseme_I", "viseme_SS"],
    "that": ["viseme_TH", "viseme_aa", "viseme_DD"],
    "chat": ["viseme_CH", "viseme_aa", "viseme_DD"],
    "shop": ["viseme_CH", "viseme_O", "viseme_PP"],
    "phone": ["viseme_FF", "viseme_O", "viseme_nn"],  # ph->FF, o, n, silent e
    "photo": ["viseme_FF", "viseme_O", "viseme_DD", "viseme_O"],
    "food": ["viseme_FF", "viseme_U", "viseme_DD"],  # f->FF, oo->U, d->DD
    "book": ["viseme_PP", "viseme_U", "viseme_kk"],
    "out": ["viseme_U", "viseme_DD"],  # ou->U
    "new": ["viseme_nn", "viseme_U"],  # ew->U
    "feet": ["viseme_FF", "viseme_I", "viseme_DD"],  # ee->I
    "read": ["viseme_RR", "viseme_I", "viseme_DD"],  # ea->I
    "rain": ["viseme_RR", "viseme_E", "viseme_nn"],  # ai->E
    "day": ["viseme_DD", "viseme_E"],  # ay->E
    "boat": ["viseme_PP", "viseme_O", "viseme_DD"],  # oa->O
    "cow": ["viseme_kk", "viseme_O"],  # ow->O
    "cat": ["viseme_kk", "viseme_aa", "viseme_DD"],
    "see": ["viseme_SS", "viseme_I"],
    "zoo": ["viseme_SS", "viseme_U"],
    "yes": ["viseme_I", "viseme_E", "viseme_SS"],  # y->I
    "win": ["viseme_U", "viseme_I", "viseme_nn"],  # w->U
    "red": ["viseme_RR", "viseme_E", "viseme_DD"],
    "run": ["viseme_RR", "viseme_U", "viseme_nn"],
    "man": ["viseme_PP", "viseme_aa", "viseme_nn"],
    "sun": ["viseme_SS", "viseme_U", "viseme_nn"],
    "love": ["viseme_nn", "viseme_U", "viseme_FF"],  # o before v->U, silent e
    "hour": ["viseme_U", "viseme_RR"],  # h silent, ou->U, r->RR
    "who": ["viseme_U", "viseme_O"],  # w->U, h silent
    "eye": ["viseme_E", "viseme_I"],  # final e silent
    "box": ["viseme_PP", "viseme_O", "viseme_kk"],  # x->kk
    "quit": ["viseme_kk", "viseme_U", "viseme_I", "viseme_DD"],  # q->kk
}


def test_word_to_visemes_matches_the_rule_table() -> None:
    for word, expected in EXPECTED.items():
        assert word_to_visemes(word) == expected


def test_repeats_merge() -> None:
    assert word_to_visemes("bookkeeper") == [
        "viseme_PP",
        "viseme_U",
        "viseme_kk",
        "viseme_I",
        "viseme_PP",
        "viseme_E",
        "viseme_RR",
    ]  # the two k's merge into one viseme_kk


def test_punctuation_and_case_are_ignored() -> None:
    assert word_to_visemes("Ship!") == word_to_visemes("ship")
    assert word_to_visemes("") == []


def test_timeline_spreads_vowels_wider_than_consonants() -> None:
    segments = timeline([("ba", 0.0, 2.6)])
    assert [shape for _, _, shape in segments] == ["viseme_PP", "viseme_aa"]
    first_end = segments[0][1] - segments[0][0]
    second_end = segments[1][1] - segments[1][0]
    assert abs(first_end - 1.0) < 1e-9  # consonant share
    assert abs(second_end - 1.6) < 1e-9  # vowel share is 1.6x


def test_timeline_covers_each_word_duration() -> None:
    segments = timeline([("hello", 0.5, 1.0)])
    assert segments[0][0] == 0.5
    assert abs(segments[-1][1] - 1.5) < 1e-9


def test_timeline_inserts_silence_only_for_long_gaps() -> None:
    short_gap = timeline([("hi", 0.0, 1.0), ("yo", 1.1, 0.5)])
    assert all(shape != "viseme_sil" for _, _, shape in short_gap)
    long_gap = timeline([("hi", 0.0, 1.0), ("yo", 1.5, 0.5)])
    assert any(shape == "viseme_sil" for _, _, shape in long_gap)


def test_weights_at_anticipates_shapes() -> None:
    segments = [(1.0, 2.0, "viseme_aa")]
    assert weights_at(segments, 0.95) == {}
    assert weights_at(segments, 0.98) == {"viseme_aa": 1.0}


def test_weights_at_crossfades_over_sixty_milliseconds() -> None:
    segments = [(0.0, 1.0, "viseme_PP"), (1.0, 2.0, "viseme_aa")]
    half = weights_at(segments, 0.99)
    assert set(half) == {"viseme_aa", "viseme_PP"}
    assert abs(half["viseme_aa"] - 0.5) < 1e-9
    assert abs(half["viseme_PP"] - 0.5) < 1e-9
    done = weights_at(segments, 1.02)
    assert done == {"viseme_aa": 1.0}


def test_weights_at_returns_at_most_two_shapes() -> None:
    segments = timeline([("hello", 0.0, 1.0), ("world", 1.5, 0.8)])
    for step in range(0, 300):
        weights = weights_at(segments, step / 100.0)
        assert len(weights) <= 2
        for value in weights.values():
            assert 0.0 <= value <= 1.0


def test_weights_at_past_the_end_is_empty() -> None:
    segments = timeline([("hi", 0.0, 0.5)])
    assert weights_at(segments, 5.0) == {}
