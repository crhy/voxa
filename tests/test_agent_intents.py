from __future__ import annotations

import pytest

from voxa.agent.intents import ROUTED_TOOLS, ToolCall, is_compound, is_start_dictation, route
from voxa.agent.tools import default_registry

CASES: list[tuple[str, ToolCall | None]] = [
    ("open gmail", ToolCall("open_site", {"name": "gmail"})),
    ("Check my inbox", ToolCall("open_site", {"name": "gmail"})),
    ("go to email", ToolCall("open_site", {"name": "gmail"})),
    ("show my mail", ToolCall("open_site", {"name": "gmail"})),
    ("open google mail", ToolCall("open_site", {"name": "gmail"})),
    ("please open gmail!", ToolCall("open_site", {"name": "gmail"})),
    ("can you open my inbox?", ToolCall("open_site", {"name": "gmail"})),
    ("hey, open gmail", ToolCall("open_site", {"name": "gmail"})),
    ("write an email", ToolCall("compose_gmail", {})),
    ("compose a new email in gmail", ToolCall("compose_gmail", {})),
    ("start a message", ToolCall("compose_gmail", {})),
    ("new email", ToolCall("compose_gmail", {})),
    ("write an email to mom about dinner", None),
    ("play Daft Punk on youtube music", ToolCall("play_youtube", {"query": "Daft Punk"})),
    ("play some music", ToolCall("play_youtube", {"query": "music"})),
    ("play music jazz", ToolCall("play_youtube", {"query": "jazz"})),
    ("play the Beatles on youtube", ToolCall("play_youtube", {"query": "the Beatles"})),
    ("play Nirvana", ToolCall("play_youtube", {"query": "Nirvana"})),
    ("put on some jazz", ToolCall("play_youtube", {"query": "some jazz"})),
    ("watch cat videos on youtube", ToolCall("play_youtube", {"query": "cat videos"})),
    ("search youtube for lofi hip hop", ToolCall("search_youtube", {"query": "lofi hip hop"})),
    ("find cat videos on youtube", ToolCall("search_youtube", {"query": "cat videos"})),
    ("search the web for best pizza in Austin", ToolCall("web_search", {"query": "best pizza in Austin"})),
    ("google how old is the moon", ToolCall("web_search", {"query": "how old is the moon"})),
    ("look up the time in Tokyo", ToolCall("web_search", {"query": "the time in Tokyo"})),
    ("open example.com", ToolCall("open_url", {"url": "example.com"})),
    ("go to wikipedia.org", ToolCall("open_url", {"url": "wikipedia.org"})),
    ("open github", ToolCall("open_site", {"name": "github"})),
    ("go to google maps", ToolCall("open_site", {"name": "google maps"})),
    ("open the youtube website", ToolCall("open_site", {"name": "the youtube website"})),
    ("pause", ToolCall("press_key", {"key": "play pause"})),
    ("resume", ToolCall("press_key", {"key": "play pause"})),
    ("play", ToolCall("press_key", {"key": "play pause"})),
    ("pause the music", ToolCall("press_key", {"key": "play pause"})),
    ("next song", ToolCall("press_key", {"key": "next track"})),
    ("skip", ToolCall("press_key", {"key": "next track"})),
    ("previous track", ToolCall("press_key", {"key": "previous track"})),
    ("go back a song", ToolCall("press_key", {"key": "previous track"})),
    ("volume up", ToolCall("press_key", {"key": "volume up"})),
    ("louder", ToolCall("press_key", {"key": "volume up"})),
    ("turn it up", ToolCall("press_key", {"key": "volume up"})),
    ("volume down", ToolCall("press_key", {"key": "volume down"})),
    ("quieter", ToolCall("press_key", {"key": "volume down"})),
    ("turn it down", ToolCall("press_key", {"key": "volume down"})),
    ("mute", ToolCall("press_key", {"key": "mute"})),
    ("unmute", ToolCall("press_key", {"key": "mute"})),
    ("press enter", ToolCall("press_key", {"key": "enter"})),
    ("press escape", ToolCall("press_key", {"key": "escape"})),
    ("select all", ToolCall("press_key", {"key": "select all"})),
    ("copy", ToolCall("press_key", {"key": "copy"})),
    ("paste", ToolCall("press_key", {"key": "paste"})),
    ("undo", ToolCall("press_key", {"key": "undo"})),
    ("save", ToolCall("press_key", {"key": "save"})),
    ("new tab", ToolCall("press_key", {"key": "new tab"})),
    ("close tab", ToolCall("press_key", {"key": "close tab"})),
    ("full screen", ToolCall("press_key", {"key": "full screen"})),
    ("type Hello there", ToolCall("type_text", {"text": "Hello there"})),
    ("write Thanks for the gift", ToolCall("type_text", {"text": "Thanks for the gift"})),
    ("write a report about sales", None),
    ("write a letter to the bank", None),
    ("send it", ToolCall("send_gmail", {})),
    ("send the email", ToolCall("send_gmail", {})),
    ("send this email", ToolCall("send_gmail", {})),
    ("open claude", ToolCall("open_app", {"name": "claude"})),
    ("launch spotify", ToolCall("open_app", {"name": "spotify"})),
    ("start the text editor", ToolCall("open_app", {"name": "the text editor"})),
    ("run Visual Studio Code", ToolCall("open_app", {"name": "Visual Studio Code"})),
    ("close brutal chess", ToolCall("close_app", {"name": "brutal chess"})),
    ("quit Brave", ToolCall("close_app", {"name": "Brave"})),
    ("shut down the text editor", ToolCall("close_app", {"name": "text editor"})),
    ("kill firefox", ToolCall("close_app", {"name": "firefox"})),
    ("close this", ToolCall("close_window", {})),
    ("close it", ToolCall("close_window", {})),
    ("close the window", ToolCall("close_window", {})),
    ("close the app", ToolCall("close_window", {})),
    ("close that window", ToolCall("close_window", {})),
    ("switch to brave", ToolCall("switch_to", {"name": "brave"})),
    ("go to brave", ToolCall("switch_to", {"name": "brave"})),
    ("show me Brutal Chess", ToolCall("switch_to", {"name": "Brutal Chess"})),
    ("go to gmail", ToolCall("open_site", {"name": "gmail"})),
    ("switch to Gmail", ToolCall("open_site", {"name": "gmail"})),
    ("open the file manager", ToolCall("open_app", {"name": "file manager"})),
    ("copy that", ToolCall("press_key", {"key": "copy"})),
    ("play some music please", ToolCall("play_youtube", {"query": "music"})),
    ("check gmail for me", ToolCall("open_site", {"name": "gmail"})),
    ("open my mail please", ToolCall("open_site", {"name": "gmail"})),
    ("skip the track", ToolCall("press_key", {"key": "next track"})),
    ("turn up the volume", ToolCall("press_key", {"key": "volume up"})),
    ("turn down the volume", ToolCall("press_key", {"key": "volume down"})),
    ("type my notes", ToolCall("type_text", {"text": "my notes"})),
    ("open a new tab", ToolCall("press_key", {"key": "new tab"})),
    ("google", ToolCall("web_search", {"query": "google"})),
    ("start dictating", None),
    ("start dictation", None),
    ("begin dictation", None),
    ("how do I close an app", None),
    ("how do I open gmail", None),
    ("what is youtube", None),
    ("why is my screen dark", None),
    ("", None),
    ("blah blah blah", None),
]


@pytest.mark.parametrize(("spoken", "expected"), CASES)
def test_route(spoken: str, expected: ToolCall | None) -> None:
    assert route(spoken) == expected


def test_every_routed_tool_is_registered_and_valid() -> None:
    registry = default_registry()
    names = set(registry.names())
    assert ROUTED_TOOLS <= names
    for _, expected in CASES:
        if expected is None:
            continue
        assert expected.tool in names
        registry.validate(expected.tool, expected.args)


def test_query_case_is_preserved() -> None:
    assert route("Play Radiohead") == ToolCall("play_youtube", {"query": "Radiohead"})
    assert route("Type Hello World") == ToolCall("type_text", {"text": "Hello World"})


def test_is_start_dictation_recognises_the_mode_switch() -> None:
    for phrase in (
        "start dictating",
        "Start Dictation!",
        "begin dictation",
        "take dictation",
        "take a dictation",
        "dictate",
        "dictate this",
        "dictation mode",
        "start typing",
        "type what I say",
        "please start dictating",
        "please dictate.",
    ):
        assert is_start_dictation(phrase), phrase


def test_is_start_dictation_rejects_ordinary_commands() -> None:
    for phrase in (
        "dictate an email to mom",
        "start typing the email",
        "type what I say about the weather",
        "stop dictating",
        "what is dictation",
        "",
        "dictating",
    ):
        assert not is_start_dictation(phrase), phrase


def test_dictation_starts_are_not_routed_to_open_app() -> None:
    for phrase in ("start dictating", "start dictation", "begin dictation", "dictate"):
        assert route(phrase) is None, phrase


COMPOUND: list[str] = [
    "put on some jazz and then turn it down a bit",
    "close brave and open libreoffice writer",
    "open gmail and then compose a message",
    "play some music then turn it up",
    "open brave and also close gmail",
    "copy that, paste it",
    "find cat videos on youtube and then press escape",
    "switch to brave after that close the window",
    "turn up the volume and then mute",
    "open spotify and play some jazz",
    "save the file and then close it",
    "search youtube for cats and then open the first one",
]

NOT_COMPOUND: list[str] = [
    "play simon and garfunkel",
    "play rock and roll",
    "open brave and firefox",
    "write thanks for coming and see you soon",
    "close brave",
    "gmail and then a new message",
    "",
    "the and the",
]


@pytest.mark.parametrize("phrase", COMPOUND)
def test_is_compound_detects_two_commands(phrase: str) -> None:
    assert is_compound(phrase), phrase


@pytest.mark.parametrize("phrase", NOT_COMPOUND)
def test_is_compound_rejects_single_commands(phrase: str) -> None:
    assert not is_compound(phrase), phrase


@pytest.mark.parametrize("phrase", COMPOUND)
def test_compound_commands_are_not_routed_to_one_intent(phrase: str) -> None:
    assert route(phrase) is None, phrase


def test_type_text_survives_the_compound_check() -> None:
    assert route("type hello and then copy it") == ToolCall("type_text", {"text": "hello and then copy it"})
    assert route("write thanks for coming and see you soon") == ToolCall(
        "type_text", {"text": "thanks for coming and see you soon"}
    )
    assert route("write a report about sales and marketing") is None


HEARD_CASES: list[tuple[str, ToolCall | None]] = [
    ("Vox Closed Brutal Chess.", ToolCall("close_app", {"name": "Brutal Chess"})),
    ("Clothes Brutal Chess.", ToolCall("close_app", {"name": "Brutal Chess"})),
    ("those cards with cats.", ToolCall("close_app", {"name": "cards with cats"})),
    ("Lupin Brutal Chess.", ToolCall("open_app", {"name": "Brutal Chess"})),
    ("opened my email", ToolCall("open_site", {"name": "gmail"})),
    ("Open Libra Office, right?", ToolCall("open_app", {"name": "LibreOffice Writer"})),
    ("Open the GAMP.", ToolCall("open_app", {"name": "GIMP"})),
    ("Please some John Barry", ToolCall("play_youtube", {"query": "some John Barry"})),
    ("Vox, save the file.", ToolCall("press_key", {"key": "save"})),
]


@pytest.mark.parametrize(("spoken", "expected"), HEARD_CASES)
def test_route_recovers_misheard_commands(spoken: str, expected: ToolCall) -> None:
    assert route(spoken) == expected, spoken


def test_is_start_dictation_strips_the_wake_word() -> None:
    assert is_start_dictation("Voxet start dictation.")
