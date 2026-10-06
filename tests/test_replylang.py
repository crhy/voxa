from voxa.replylang import LANGUAGES, is_only_a_language_request, requested_language

REQUESTED = [
    ("can I get that in German", "German"),
    ("auf Deutsch", "German"),
    ("en español", "Spanish"),
    ("speak Spanish", "Spanish"),
    ("answer in French", "French"),
    ("can I get that in Korean", "Korean"),
    ("say that in Japanese", "Japanese"),
    ("why are you not speaking Spanish", "Spanish"),
    ("back to English", "English"),
    ("in English please", "English"),
    ("please answer in Russian", "Russian"),
    ("translate to Chinese", "Chinese"),
    ("in Portuguese", "Portuguese"),
    ("speak Hindi", "Hindi"),
    ("in Arabic please", "Arabic"),
    ("say it in Italian", "Italian"),
    ("in Français", "French"),
    ("in italiano", "Italian"),
    ("tell me about German shepherds", None),
    ("the capital of Spain", None),
    ("I like Japanese food", None),
    ("how do you say hello in Korean", "Korean"),
    ("what is the french revolution", None),
    ("speak", None),
]

ONLY = [
    ("can I get that in German", True),
    ("in Spanish please", True),
    ("back to English", True),
    ("answer in French", True),
    ("auf Deutsch", True),
    ("why are you not speaking Spanish", False),
    ("tell me about German shepherds", False),
    ("the capital of Spain", False),
    ("", False),
    ("speak Spanish about the weather", False),
]


def test_requested_language():
    for text, want in REQUESTED:
        assert requested_language(text) == want


def test_only_a_language_request():
    for text, want in ONLY:
        assert is_only_a_language_request(text) is want


def test_unknown_language_is_none():
    assert requested_language("in Klingon") is None
    assert is_only_a_language_request("in Klingon") is False


def test_voice_choice_by_gender():
    for _name, (tag, female, male) in LANGUAGES.items():
        assert female == f"{tag}-" + female.split("-", 2)[2]
        assert female.startswith(tag) and male.startswith(tag)
        assert female != male
    assert LANGUAGES["German"][1] == "de-DE-KatjaNeural"
    assert LANGUAGES["German"][2] == "de-DE-ConradNeural"
    assert LANGUAGES["English"][1] == "en-US-AriaNeural"
    assert LANGUAGES["English"][2] == "en-US-GuyNeural"
