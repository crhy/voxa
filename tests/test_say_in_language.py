from voxa.agent import intents
from voxa.agent.tools import translate


def test_language_and_tidy():
    assert translate.language_name("Japanese.") == "Japanese"
    assert translate.language_name("spanish") == "Spanish"
    assert translate.language_name("klingon") is None
    assert translate.tidy('"¿Dónde está la estación?"\nExplanation: ...') == "¿Dónde está la estación?"
    assert translate.tidy("Translation: Guten Morgen") == "Guten Morgen"
    assert translate.tidy("") == ""


def test_tool(monkeypatch):
    asked, spoken = [], []
    monkeypatch.setattr(translate, "ask_model", lambda messages: asked.append(messages) or '"Guten Morgen"')
    monkeypatch.setattr(translate, "speak_in", spoken.append)
    result = translate.say_in_language({"phrase": "good morning", "language": "german"})
    assert result.ok and result.speech == "Guten Morgen" and spoken == ["German"]
    assert "German" in asked[0][0]["content"] and asked[0][1]["content"] == "good morning"
    assert not translate.say_in_language({"phrase": "hi", "language": "klingon"}).ok and spoken == ["German"]
    monkeypatch.setattr(translate, "ask_model", lambda messages: "")
    assert not translate.say_in_language({"phrase": "hi", "language": "french"}).ok
    monkeypatch.setattr(translate, "ask_model", None)
    assert not translate.say_in_language({"phrase": "hi", "language": "french"}).ok


def test_router():
    call = intents.route("how do I say where is the station in Japanese")
    assert call.tool == "say_in_language" and call.args == {"phrase": "where is the station", "language": "Japanese"}
    assert intents.route("say good morning in spanish").tool == "say_in_language"
    assert intents.route("translate thank you into French").args["phrase"] == "thank you"
    other = intents.route("what is the weather in Portland")
    assert other is None or other.tool != "say_in_language"
    typed = intents.route("say hello to Bob")
    assert typed is None or typed.tool != "say_in_language"
