from voxa.agent.intents import route


def test_cap41_trip_plan_paraphrases():
    r = route("can you plan a three day vacation to Rome")
    assert r and r.tool == "trip_plan" and r.args["place"] == "Rome"
    r = route("hey please make a three day trip to Rome")
    assert r and r.tool == "trip_plan" and r.args["place"] == "Rome"


def test_cap41_trip_read_paraphrases():
    r = route("show me my trip to Rome")
    assert r and r.tool == "trip_read" and r.args["place"] == "Rome"
    r = route("can you read my vacation to Rome")
    assert r and r.tool == "trip_read" and r.args["place"] == "Rome"


def test_cap09_knowledge_paraphrases():
    assert route("what's the capital of Italy") is None
    assert route("can you tell me the capital of Italy") is None


def test_cap09_conversation_summary_paraphrases():
    r = route("can you summarize our conversation")
    assert r and r.tool == "conversation_summary"
    r = route("hey please recap our conversation")
    assert r and r.tool == "conversation_summary"


def test_cap70_event_plan_paraphrases():
    r = route("can you plan a wedding for Saturday")
    assert r and r.tool == "event_plan"
    r = route("hey please organize a wedding for Saturday")
    assert r and r.tool == "event_plan"
    r = route("can you plan a barbecue")
    assert r and r.tool == "event_plan"
    r = route("hey please organize a bbq")
    assert r and r.tool == "event_plan"


def test_cap34_look_up_paraphrases():
    r = route("can you research the company Nvidia")
    assert r and r.tool == "look_up"
    r = route("hey please look up the company Nvidia")
    assert r and r.tool == "look_up"
    r = route("can you research Ada Lovelace on wikipedia")
    assert r and r.tool == "look_up"
    r = route("hey please look up Ada Lovelace on wikipedia")
    assert r and r.tool == "look_up"


def test_cap35_compare_prices_paraphrases():
    r = route("can you compare prices for a cordless drill")
    assert r and r.tool == "compare_prices"
    r = route("hey please compare prices on a cordless drill")
    assert r and r.tool == "compare_prices"


def test_cap35_find_deal_paraphrases():
    r = route("can you find me a deal on headphones")
    assert r and r.tool == "find_deal"
    r = route("hey please find a deal on headphones")
    assert r and r.tool == "find_deal"


def test_guard_phrases_still_route():
    r = route("what's on my to-do list")
    assert r and r.tool == "list_read"
    r = route("what time is it")
    assert r and r.tool == "tell_time"
    assert route("what is the capital of France") is None
    r = route("open the calculator")
    assert r and r.tool == "open_app"
    r = route("look up quantum computing")
    assert r and r.tool == "web_search"
