from voxa.agent.intents import ToolCall, route


def test_cap92_p1():
    r = route("Anything I should know")
    assert r == ToolCall(tool="heads_up", args={})


def test_cap92_p2():
    r = route("What is going on today")
    assert r == ToolCall(tool="heads_up", args={})


def test_cap94_p1():
    r = route("Did I forget anything")
    assert r == ToolCall(tool="what_did_i_forget", args={})


def test_cap94_p2():
    r = route("What have I missed")
    assert r == ToolCall(tool="what_did_i_forget", args={})


def test_cap95_p1():
    r = route("What would you do next")
    assert r == ToolCall(tool="suggest_next_action", args={})


def test_cap95_p2():
    r = route("What do you suggest I do next")
    assert r == ToolCall(tool="suggest_next_action", args={})


def test_cap89_p1():
    r = route("What is the status of the Voxa repo")
    assert r == ToolCall(tool="project_status", args={"name": "Voxa"})


def test_cap89_p2():
    r = route("How is the Voxa project doing")
    assert r == ToolCall(tool="project_status", args={"name": "Voxa"})


def test_cap90_p1():
    r = route("Is Strata running")
    assert r == ToolCall(tool="model_status", args={})


def test_cap90_p2():
    r = route("Is the coding model up")
    assert r == ToolCall(tool="model_status", args={})
