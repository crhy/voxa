from voxa.agent.intents import route


def test_cap39_market_paraphrases():
    r = route("hey how is the market doing")
    assert r and r.tool == "market"
    r = route("can you check the stock market please")
    assert r and r.tool == "market"


def test_cap39_bitcoin_paraphrases():
    r = route("hey how much is bitcoin")
    assert r and r.tool == "market" and r.args["what"] == "bitcoin"
    r = route("hey what's bitcoin at")
    assert r and r.tool == "market" and r.args["what"] == "bitcoin"


def test_cap40_fact_check_paraphrases():
    r = route("can you fact check the goldfish memory thing please")
    assert r and r.tool == "fact_check"
    r = route("hey fact check that goldfish have a three second memory")
    assert r and r.tool == "fact_check"


def test_cap45_flight_status_paraphrases():
    r = route("hey check flight BA 117 please")
    assert r and r.tool == "flight_status" and r.args["flight"] == "BA 117"
    r = route("hey track flight BA 117")
    assert r and r.tool == "flight_status" and r.args["flight"] == "BA 117"
    r = route("hey is flight BA 117 delayed")
    assert r and r.tool == "flight_status" and r.args["flight"] == "BA 117"
    r = route("can you check if flight BA 117 is delayed please")
    assert r and r.tool == "flight_status" and r.args["flight"] == "BA 117"


def test_cap49_trip_cost_paraphrases():
    r = route("hey how much did the Paris trip cost")
    assert r and r.tool == "trip_cost" and r.args["name"] == "Paris"
    r = route("hey what did i spend on the Paris trip")
    assert r and r.tool == "trip_cost" and r.args["name"] == "Paris"


def test_cap53_compare_prices_paraphrases():
    r = route("hey compare prices for a PS5 please")
    assert r and r.tool == "compare_prices"
    r = route("hey compare prices on a PS5")
    assert r and r.tool == "compare_prices"


def test_guards_still_route():
    r = route("what's on my to-do list")
    assert r and r.tool == "list_read"
    r = route("what time is it")
    assert r and r.tool == "tell_time"
    assert route("what is the capital of France") is None
    r = route("open the calculator")
    assert r and r.tool == "open_app"
