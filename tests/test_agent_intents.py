from __future__ import annotations

import pytest

from voxa.agent.intents import (
    ROUTED_TOOLS,
    ToolCall,
    dictation_target,
    is_compound,
    is_start_dictation,
    route,
)
from voxa.agent.tools import default_registry

CASES: list[tuple[str, ToolCall | None]] = [
    ("what is the British pound in USD", ToolCall("exchange_rate", {"base": "GBP", "quote": "USD"})),
    ("what is the price of the brittish pound in us dollars",
     ToolCall("exchange_rate", {"base": "GBP", "quote": "USD"})),
    ("what is Tesla stock price", ToolCall("market", {"what": "Tesla"})),
    ("what's Tesla's stock price", ToolCall("market", {"what": "Tesla's"})),
    ("stock price of Tesla", ToolCall("market", {"what": "Tesla"})),
    ("how much is Tesla stock", ToolCall("market", {"what": "Tesla"})),
    ("what is the current price of one share of Tesla",
     ToolCall("market", {"what": "one share of Tesla"})),
    ("what is the current price of one share of Mercedes",
     ToolCall("market", {"what": "one share of Mercedes"})),
    ("how is Nvidia stock doing", ToolCall("market", {"what": "Nvidia"})),
    ("what's Ford at", ToolCall("market", {"what": "Ford"})),
    ("what is the stock price for Ford Motor Company", ToolCall("market", {"what": "Ford Motor Company"})),
    ("what is the share price right now", ToolCall("market", {"what": "share price"})),
    ("what is the price of one Toyota share", ToolCall("market", {"what": "one Toyota share"})),
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
    ("play some music", ToolCall("play_music", {"query": "music"})),
    ("what song is playing", ToolCall("now_playing", {})),
    ("play music jazz", ToolCall("play_music", {"query": "jazz"})),
    ("play the Beatles on youtube", ToolCall("play_youtube", {"query": "the Beatles"})),
    ("play Nirvana", ToolCall("play_music", {"query": "Nirvana"})),
    ("put on some jazz", ToolCall("play_music", {"query": "jazz music"})),
    ("watch cat videos on youtube", ToolCall("play_youtube", {"query": "cat videos"})),
    ("search youtube for lofi hip hop", ToolCall("search_youtube", {"query": "lofi hip hop"})),
    ("find cat videos on youtube", ToolCall("search_youtube", {"query": "cat videos"})),
    ("search the web for best pizza in Austin", ToolCall("web_search", {"query": "best pizza in Austin"})),
    ("google how old is the moon", ToolCall("web_search", {"query": "how old is the moon"})),
    ("look up the time in Tokyo", ToolCall("web_search", {"query": "the time in Tokyo"})),
    ("Lookup which is better the iPhone or the Android",
     ToolCall("web_search", {"query": "which is better the iPhone or the Android"})),
    ("lookup how to fix a null pointer exception",
     ToolCall("web_search", {"query": "how to fix a null pointer exception"})),
    ("lookup the company Nvidia", ToolCall("look_up", {"name": "Nvidia", "kind": "company"})),
    ("open example.com", ToolCall("open_url", {"url": "example.com"})),
    ("go to wikipedia.org", ToolCall("open_url", {"url": "wikipedia.org"})),
    ("open github", ToolCall("open_site", {"name": "github"})),
    ("go to google maps", ToolCall("open_site", {"name": "google maps"})),
    ("open the youtube website", ToolCall("open_site", {"name": "the youtube website"})),
    ("pause the music", ToolCall("media_control", {"action": "pause"})),
    ("pause music", ToolCall("media_control", {"action": "pause"})),
    ("pause radio", ToolCall("media_control", {"action": "pause"})),
    ("pause TV", ToolCall("media_control", {"action": "pause"})),
    ("pause the television", ToolCall("media_control", {"action": "pause"})),
    ("pause VLC", ToolCall("media_control", {"action": "pause"})),
    ("pause", None),  # a bare "pause" is for Voxa herself, handled by the window
    ("resume", ToolCall("media_control", {"action": "resume"})),
    ("resume the TV", ToolCall("media_control", {"action": "resume"})),
    ("play", ToolCall("press_key", {"key": "play pause"})),
    ("pause the music", ToolCall("media_control", {"action": "pause"})),
    ("next song", ToolCall("media_control", {"action": "next"})),
    ("skip", ToolCall("media_control", {"action": "next"})),
    ("previous track", ToolCall("media_control", {"action": "previous"})),
    ("go back a song", ToolCall("media_control", {"action": "previous"})),
    ("volume up", ToolCall("system_volume", {"action": "louder"})),
    ("louder", ToolCall("system_volume", {"action": "louder"})),
    ("turn it up", ToolCall("system_volume", {"action": "louder"})),
    ("volume down", ToolCall("system_volume", {"action": "quieter"})),
    ("quieter", ToolCall("system_volume", {"action": "quieter"})),
    ("turn it down", ToolCall("system_volume", {"action": "quieter"})),
    ("mute", ToolCall("system_volume", {"action": "mute"})),
    ("unmute", ToolCall("system_volume", {"action": "unmute"})),
    ("press enter", ToolCall("press_key", {"key": "enter"})),
    ("press escape", ToolCall("press_key", {"key": "escape"})),
    ("select all", ToolCall("press_key", {"key": "select all"})),
    ("copy", ToolCall("press_key", {"key": "copy"})),
    ("paste", ToolCall("press_key", {"key": "paste"})),
    ("undo", ToolCall("press_key", {"key": "undo"})),
    ("save", ToolCall("press_key", {"key": "save"})),
    ("new tab", ToolCall("press_key", {"key": "new tab"})),
    ("close tab", ToolCall("press_key", {"key": "close tab"})),
    ("full screen", ToolCall("maximize_app", {"name": ""})),
    ("type Hello there", ToolCall("type_text", {"text": "Hello there"})),
    ("write Thanks for the gift", ToolCall("type_text", {"text": "Thanks for the gift"})),
    ("write a report about sales", None),
    ("write a letter to the bank", None),
    ("send it", ToolCall("send_gmail", {})),
    ("send the email", ToolCall("send_gmail", {})),
    ("send this email", ToolCall("send_gmail", {})),
    ("open claude", ToolCall("open_app", {"name": "claude"})),
    ("launch spotify", ToolCall("open_app", {"name": "spotify"})),
    ("start the text editor", ToolCall("open_app", {"name": "text editor"})),
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
    ("open the file manager", ToolCall("open_app", {"name": "file browser"})),
    ("What is today\u2019s date", ToolCall("tell_date", {})),
    ("what is todays date", ToolCall("tell_date", {})),
    ("playscar radio", ToolCall("play_radio", {"name": "ska"})),
    ("PlaySka Radio", ToolCall("play_radio", {"name": "ska"})),
    ("take a note, call the vet", ToolCall("take_note", {"text": "call the vet"})),
    ("what is 15% of 80", ToolCall("calculate", {"expression": "what is 15% of 80"})),
    ("What is it my documents folder", ToolCall("list_folder", {"folder": "documents"})),
    ("Watch Nassatini", ToolCall("play_tv", {"name": "NASA TV"})),
    ("Open my documents older", ToolCall("open_folder", {"folder": "documents"})),
    ("Composer, new email", ToolCall("compose_gmail", {})),
    ("Health with Music", ToolCall("voxa_help", {"topic": "Music"})),
    ("what are my reminders", ToolCall("list_reminders", {})),
    ("what is the square root of 81", ToolCall("calculate", {"expression": "what is the square root of 81"})),
    ("PlayScore Radio", ToolCall("play_radio", {"name": "ska"})),
    ("open, gitHub", ToolCall("open_site", {"name": "gitHub"})),
    ("What is 7x8", ToolCall("calculate", {"expression": "What is 7x8"})),
    ("stop the radio", ToolCall("media_control", {"action": "stop"})),
    ("Set the volume of 40%", ToolCall("system_volume", {"action": "set", "percent": "40"})),
    ("Please open the terminal window.", ToolCall("open_app", {"name": "terminal"})),
    ("Open my email client quickly.", ToolCall("open_app", {"name": "email client"})),
    ("Start a new document editor instance.", ToolCall("open_app", {"name": "LibreOffice Writer"})),
    ("I need to open Photoshop right now.", ToolCall("open_app", {"name": "Photoshop"})),
    ("Open the Notepad app.", ToolCall("open_app", {"name": "text editor"})),
    ("Try to open my nonexistent app called FooBar.", ToolCall("open_app", {"name": "FooBar"})),
    ("Open the media player for playback.", ToolCall("open_app", {"name": "media player"})),
    ("copy that", ToolCall("press_key", {"key": "copy"})),
    ("play some music please", ToolCall("play_music", {"query": "music"})),
    ("check gmail for me", ToolCall("open_site", {"name": "gmail"})),
    ("open my mail please", ToolCall("open_site", {"name": "gmail"})),
    ("skip the track", ToolCall("media_control", {"action": "next"})),
    ("turn up the volume", ToolCall("system_volume", {"action": "louder"})),
    ("turn down the volume", ToolCall("system_volume", {"action": "quieter"})),
    ("type my notes", ToolCall("type_text", {"text": "my notes"})),
    ("open a new tab", ToolCall("press_key", {"key": "new tab"})),
    ("google", ToolCall("web_search", {"query": "google"})),
    ("play the latest video from TechLinked", ToolCall("play_latest", {"channel": "TechLinked"})),
    ("play latest from TechLinked", ToolCall("play_latest", {"channel": "TechLinked"})),
    ("listen to jazz", ToolCall("play_music", {"query": "jazz"})),
    ("stop the music", ToolCall("media_control", {"action": "stop"})),
    ("stop the video", ToolCall("media_control", {"action": "stop"})),
    ("skip ahead", ToolCall("media_control", {"action": "forward"})),
    ("go back 30 seconds", ToolCall("media_control", {"action": "back"})),
    ("forward 30 seconds", ToolCall("media_control", {"action": "forward"})),
    ("start dictating", None),
    ("start dictation", None),
    ("begin dictation", None),
    ("how do I close an app", None),
    ("how do I open gmail", None),
    ("what is youtube", None),
    ("why is my screen dark", None),
    ("browse to en.wikipedia.org", ToolCall("browse", {"url": "en.wikipedia.org"})),
    ("go to the website example.com", ToolCall("browse", {"url": "example.com"})),
    ("open the page devuan.org", ToolCall("browse", {"url": "devuan.org"})),
    ("browse to https://example.com", ToolCall("browse", {"url": "https://example.com"})),
    ("read this text file", ToolCall("read_text_file", {})),
    ("recite a poem by edgar allen poe", ToolCall("recite_poem", {"author": "edgar allen poe"})),
    ("read me a poem by emily dickinson", ToolCall("recite_poem", {"author": "emily dickinson"})),
    ("recite the raven by edgar allan poe", ToolCall("recite_poem", {"title": "the raven", "author": "edgar allan poe"})),
    ("recite annabel lee", ToolCall("recite_poem", {"title": "annabel lee"})),
    ("recite a poem", ToolCall("recite_poem", {})),
    ("read the file called meeting notes", ToolCall("read_text_file", {"name": "meeting notes"})),
    ("read me the text file named intro out loud", ToolCall("read_text_file", {"name": "intro"})),
    ("read this page", ToolCall("read_page", {})),
    ("read the page", ToolCall("read_page", {})),
    ("read this to me", ToolCall("read_page", {})),
    ("what does this page say", ToolCall("read_page", {})),
    ("click History", ToolCall("click_on", {"text": "History"})),
    ("click on Search", ToolCall("click_on", {"text": "Search"})),
    ("press the Save button", ToolCall("click_on", {"text": "Save"})),
    ("follow the Next link", ToolCall("click_on", {"text": "Next"})),
    ("search this site for Devuan", ToolCall("search_site", {"query": "Devuan"})),
    ("search here for systemd", ToolCall("search_site", {"query": "systemd"})),
    ("search this page for Devuan", ToolCall("search_site", {"query": "Devuan"})),
    ("scroll down", ToolCall("scroll", {"direction": "down", "amount": "page"})),
    ("scroll up", ToolCall("scroll", {"direction": "up", "amount": "page"})),
    ("go to the top", ToolCall("scroll", {"direction": "up", "amount": "top"})),
    ("go to the bottom of the page", ToolCall("scroll", {"direction": "down", "amount": "bottom"})),
    ("go back", ToolCall("go_back", {})),
    ("previous page", ToolCall("go_back", {})),
    ("please go back", ToolCall("go_back", {})),
    ("", None),
    ("blah blah blah", None),
    ("play some jazz music", ToolCall("play_music", {"query": "jazz music"})),
    ("play jazz music", ToolCall("play_music", {"query": "jazz music"})),
    ("play a little rock music", ToolCall("play_music", {"query": "rock music"})),
    ("play a bit of classical music", ToolCall("play_music", {"query": "classical music"})),
    ("play music by jazz", ToolCall("play_music", {"query": "jazz music"})),
    ("play some blues", ToolCall("play_music", {"query": "blues music"})),
    ("play some lofi", ToolCall("play_music", {"query": "lofi music"})),
    ("play some lo-fi", ToolCall("play_music", {"query": "lo-fi music"})),
    ("play some hip hop", ToolCall("play_music", {"query": "hip hop music"})),
    ("play some country", ToolCall("play_music", {"query": "country music"})),
    ("play some metal", ToolCall("play_music", {"query": "metal music"})),
    ("play some ambient", ToolCall("play_music", {"query": "ambient music"})),
    ("play some reggae", ToolCall("play_music", {"query": "reggae music"})),
    ("play some soul", ToolCall("play_music", {"query": "soul music"})),
    ("play some funk", ToolCall("play_music", {"query": "funk music"})),
    ("play some pop", ToolCall("play_music", {"query": "pop music"})),
    ("play some techno", ToolCall("play_music", {"query": "techno music"})),
    ("play some house", ToolCall("play_music", {"query": "house music"})),
    ("play some folk", ToolCall("play_music", {"query": "folk music"})),
    ("play some piano", ToolCall("play_music", {"query": "piano music"})),
    ("play some guitar", ToolCall("play_music", {"query": "guitar music"})),
    ("put on some jazz", ToolCall("play_music", {"query": "jazz music"})),
    ("play something relaxing", ToolCall("play_music", {"query": "relaxing music"})),
    ("play something upbeat", ToolCall("play_music", {"query": "upbeat music"})),
    ("play something chill", ToolCall("play_music", {"query": "chill music"})),
    ("play something calm", ToolCall("play_music", {"query": "calm music"})),
    ("play the song bohemian rhapsody", ToolCall("play_music", {"query": "bohemian rhapsody"})),
    ("play bohemian rhapsody by queen", ToolCall("play_music", {"query": "bohemian rhapsody queen"})),
    ("play the video cats", ToolCall("play_video", {"query": "cats"})),
    ("watch cat videos", ToolCall("play_video", {"query": "cat videos"})),
    ("play cat video", ToolCall("play_video", {"query": "cat"})),
    ("play the movie inception", ToolCall("play_video", {"query": "inception"})),
    ("play the trailer inception", ToolCall("play_video", {"query": "inception"})),
    ("play the episode one", ToolCall("play_video", {"query": "one"})),
    ("play rock and roll", ToolCall("play_music", {"query": "rock and roll"})),
    ("play simon and garfunkel", ToolCall("play_music", {"query": "simon and garfunkel"})),
    ("show me images of a pony", ToolCall("image_search", {"query": "a pony"})),
    ("show me pictures of cats", ToolCall("image_search", {"query": "cats"})),
    ("find images of the moon", ToolCall("image_search", {"query": "the moon"})),
    ("look for photos of sunsets", ToolCall("image_search", {"query": "sunsets"})),
    ("search for pictures of mountains", ToolCall("image_search", {"query": "mountains"})),
    ("get images of a red fox", ToolCall("image_search", {"query": "a red fox"})),
    ("image search for golden retriever", ToolCall("image_search", {"query": "golden retriever"})),
    ("picture search for the Eiffel Tower", ToolCall("image_search", {"query": "the Eiffel Tower"})),
    ("photo search Niagara Falls", ToolCall("image_search", {"query": "Niagara Falls"})),
    ("do an image search for a unicorn", ToolCall("image_search", {"query": "a unicorn"})),
    ("do a brave image search of the great pyramids", ToolCall("image_search", {"query": "the great pyramids"})),
    ("what does a pony look like", ToolCall("image_search", {"query": "a pony"})),
    ("what does the Eiffel Tower look like", ToolCall("image_search", {"query": "the Eiffel Tower"})),
    ("show me cute images of puppies", ToolCall("image_search", {"query": "puppies"})),
    ("find funny pictures of llamas", ToolCall("image_search", {"query": "llamas"})),
    ("look for vintage photos of cars", ToolCall("image_search", {"query": "cars"})),
    ("show me some pictures of the ocean", ToolCall("image_search", {"query": "the ocean"})),
    ("search for images of a lighthouse", ToolCall("image_search", {"query": "a lighthouse"})),
    ("image search for lofi album art", ToolCall("image_search", {"query": "lofi album art"})),
    ("what does a capybara look like", ToolCall("image_search", {"query": "a capybara"})),
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
    assert route("Play Radiohead") == ToolCall("play_music", {"query": "Radiohead"})
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
        "dictation",
        "start typing",
        "type what I say",
        "please start dictating",
        "please dictate.",
        "start dictation in Pluma",
        "start dictating into the Codex window",
        "please start dictation in Pluma",
        "start dictation in Write",
    ):
        assert is_start_dictation(phrase), phrase


def test_dictation_target_preserves_the_destination_name() -> None:
    assert dictation_target("start dictation in Pluma") == "Pluma"
    assert dictation_target("start dictating into the Codex window") == "Codex window"
    assert dictation_target("start dictation in Write") == "LibreOffice Writer"
    assert dictation_target("start dictation") is None


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
    ("Please some John Barry", ToolCall("play_music", {"query": "some John Barry"})),
    ("Vox, save the file.", ToolCall("file_dialog", {"action": "save"})),
]


@pytest.mark.parametrize(("spoken", "expected"), HEARD_CASES)
def test_route_recovers_misheard_commands(spoken: str, expected: ToolCall) -> None:
    assert route(spoken) == expected, spoken


REMINDER_CASES: list[tuple[str, ToolCall | None]] = [
    ("set a timer for 10 minutes", ToolCall("set_timer", {"duration": "10 minutes"})),
    ("timer for 2 hours", ToolCall("set_timer", {"duration": "2 hours"})),
    ("remind me to call the dentist at 3pm", ToolCall("set_reminder", {"when": "3pm", "text": "call the dentist"})),
    ("remind me in 10 minutes to stretch", ToolCall("set_reminder", {"when": "in 10 minutes", "text": "stretch"})),
    ("what reminders do I have", ToolCall("list_reminders", {})),
    ("list my reminders", ToolCall("list_reminders", {})),
    ("cancel my reminders", ToolCall("cancel_reminders", {})),
    ("cancel all timers", ToolCall("cancel_reminders", {})),
]


@pytest.mark.parametrize(("spoken", "expected"), REMINDER_CASES)
def test_route_recognises_reminder_commands(spoken: str, expected: ToolCall) -> None:
    assert route(spoken) == expected, spoken


def test_is_start_dictation_strips_the_wake_word() -> None:
    assert is_start_dictation("Voxet start dictation.")


def test_a_request_is_found_among_other_peoples_words():
    from voxa.agent.intents import route_spoken

    call, said = route_spoken("Add pay the rent to my to-do list. Dude, it's bothering you, like")
    assert call == ToolCall("list_add", {"item": "pay the rent", "list": "to-do"}) and said == "Add pay the rent to my to-do list."
    call, _ = route_spoken("You can go ahead and watch it if you want. , what's on my to-do list")
    assert call == ToolCall("list_read", {"list": "to-do"})
    assert route_spoken("What time is it")[0] == ToolCall("tell_time", {})
    # two requests among the chatter are not guessed at, and plain chatter is still nothing
    assert route_spoken("So anyway what time is it. And then what day is it. He said no.")[0] is None
    assert route_spoken("I went to the shop. It was closed.")[0] is None
