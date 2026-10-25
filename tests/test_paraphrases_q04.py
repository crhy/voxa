from voxa.agent.hearing import normalize
from voxa.agent.intents import route


def _tool(prompt):
    call = route(normalize(prompt))
    return call.tool if call else None


PARAPHRASES = {
    "how do I get to the public library": "directions",
    "please give me directions to the library": "directions",
    "can you show me the way to the park": "directions",
    "hey can you give me walking directions to the park": "directions",
    "hey can you navigate me from Portland to Seattle": "directions",
    "please show me how to get from Portland to Seattle": "directions",
    "hey can you give me directions to the airport": "directions",
    "hey can you take me to the airport": "directions",
    "hey can you give me directions to Denver": "directions",
    "please show me the way to Denver": "directions",
    "hey can you put milk on my shopping list": "list_add",
    "please add milk to the shopping list": "list_add",
    "hey can you add eggs to my shopping list": "list_add",
    "please put eggs on the shopping list": "list_add",
    "hey can you read my shopping list": "list_read",
    "please show me my shopping list": "list_read",
    "my electricity bill is due on the 15th of every month": "date_remember",
    "the electric bill is due on the 15th of every month": "date_remember",
    "when is my electricity bill due": "date_when",
    "when is the electric bill due": "date_when",
    "when is the water bill due": "date_when",
    "what bills are coming up": "date_upcoming",
    "what bills do I have due this month": "date_upcoming",
    "what bills are due this month": "date_upcoming",
    "I paid twelve dollars for lunch": "expense_add",
    "add an expense of twelve dollars for lunch": "expense_add",
    "how much have I spent on food this month": "expense_total",
    "what did I spend on food this month": "expense_total",
    "show my expenses this month": "expense_breakdown",
    "what's my spending breakdown this month": "expense_breakdown",
    "my car service is due on march 3rd": "date_remember",
    "the oil change is due on march 3rd": "date_remember",
}


def test_every_paraphrase_routes():
    for prompt, expected in PARAPHRASES.items():
        assert _tool(prompt) == expected, prompt


def test_guard_phrases_still_route():
    assert _tool("what's on my to-do list") == "list_read"
    assert _tool("what time is it") == "tell_time"
    assert _tool("what is the capital of France") is None
    assert _tool("open the calculator") == "open_app"
