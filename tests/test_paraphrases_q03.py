from voxa.agent.hearing import normalize
from voxa.agent.intents import route


def _tool(prompt):
    call = route(normalize(prompt))
    return call.tool if call else None


PARAPHRASES = {
    "15 percent of 80": "calculate",
    "what is 15 percent of 80 please": "calculate",
    "144 divided by 12 please": "calculate",
    "144 divided by 12": "calculate",
    "convert 100 kilometers into miles": "calculate",
    "how many miles is 100 kilometers": "calculate",
    "square root of 81 please": "calculate",
    "what is the square root of 81 please": "calculate",
    "250 plus 175": "calculate",
    "what is 250 plus 175 please": "calculate",
    "20 percent of 150": "calculate",
    "what is 20 percent of 150 please": "calculate",
    "convert 5 miles into kilometers": "calculate",
    "how many kilometers is 5 miles": "calculate",
    "12 squared": "calculate",
    "what is 12 squared please": "calculate",
    "seven times eight": "calculate",
    "what is seven times eight please": "calculate",
    "look up recipes for banana bread": "web_search",
    "search the web for banana bread recipes": "web_search",
    "what is the biggest planet": None,
    "which planet is the largest": None,
    "whats the capital of France": None,
    "name the capital of France": None,
    "what is the tallest mountain": None,
    "which mountain is the highest": None,
    "play the local news please": "local_news",
    "can you play the local news": "local_news",
    "cheapest flight to Hawaii": "find_deal",
    "whats the cheapest ticket to Hawaii": "find_deal",
    "cheapest 4070 Ti Super": "find_deal",
    "best price on the 4070 Ti Super": "find_deal",
    "look up how to fix a null pointer exception please": "web_search",
    "search the web for how to fix a null pointer exception": "web_search",
    "where is the nearest coffee shop": "find_nearby",
    "find the closest coffee shop": "find_nearby",
    "where is the nearest pharmacy": "find_nearby",
    "find the closest pharmacy": "find_nearby",
    "walking directions to the park": "directions",
    "how do I get to the park": "directions",
    "live music tonight near me": "local_events",
    "any live music tonight in my area": "local_events",
}


def test_every_paraphrase_routes():
    for prompt, expected in PARAPHRASES.items():
        assert _tool(prompt) == expected, prompt


def test_guard_phrases_still_route():
    assert _tool("what's on my to-do list") == "list_read"
    assert _tool("what time is it") == "tell_time"
    assert _tool("what is the capital of France") is None
    assert _tool("open the calculator") == "open_app"
