from __future__ import annotations

from voxa.agent.helptext import ALIASES, TOPICS, overview, topic_for, topic_help
from voxa.agent.intents import route
from voxa.agent.tools.help import help_tools, voxa_help


def test_topic_for_keys():
    for key in TOPICS:
        assert topic_for(key) == key


def test_topic_for_aliases():
    for alias, key in ALIASES.items():
        assert topic_for(alias) == key


def test_topic_for_plurals():
    assert topic_for("app") == "apps"
    assert topic_for("folder") == "files"
    assert topic_for("program") == "apps"
    assert topic_for("video") == "music"
    assert topic_for("timer") == "time"
    assert topic_for("reminder") == "time"


def test_topic_for_leading_words():
    assert topic_for("the files") == "files"
    assert topic_for("my files") == "files"
    assert topic_for("your files") == "files"
    assert topic_for("The Apps") == "apps"


def test_topic_for_unknown():
    assert topic_for("taxes") is None
    assert topic_for("weather") is None
    assert topic_for("") is None


def test_texts_short_and_ended():
    texts = [overview()] + [topic_help(key) for key in TOPICS]
    for text in texts:
        assert len(text) < 330
        assert text.endswith((".", "?"))


def test_topic_help_lists_examples():
    for key in TOPICS:
        summary = TOPICS[key][0]
        assert topic_help(key).startswith(summary)
        for example in TOPICS[key][1][:3]:
            assert example in topic_help(key)


def test_tool_overview_for_blank_and_unknown():
    assert voxa_help({}).speech == overview()
    assert voxa_help({"topic": ""}).speech == overview()
    assert voxa_help({"topic": "taxes"}).speech == overview()


def test_tool_topic_text():
    assert voxa_help({"topic": "folders"}).speech == topic_help("files")
    assert voxa_help({"topic": "music"}).speech == topic_help("music")


def test_tool_registration():
    tools = help_tools()
    assert len(tools) == 1
    assert tools[0].name == "voxa_help"
    assert tools[0].risk == 0


def test_router_plain_help():
    for text in ("what can you do", "help", "what can I say"):
        call = route(text)
        assert call is not None
        assert call.tool == "voxa_help"
        assert "topic" not in call.args


def test_router_topic_help():
    call = route("what can you do with files")
    assert call is not None and call.tool == "voxa_help" and call.args == {"topic": "files"}
    call = route("help with music")
    assert call is not None and call.tool == "voxa_help" and call.args == {"topic": "music"}
    call = route("files help")
    assert call is not None and call.tool == "voxa_help" and call.args == {"topic": "files"}


def test_router_other_sentences_keep_their_own_path():
    for text in ("help me with my taxes", "what can you do about the weather"):
        call = route(text)
        assert call is None or "topic" not in call.args
