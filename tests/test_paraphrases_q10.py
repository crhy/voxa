from voxa.agent.intents import ToolCall, route


def test_cap98_01_p1():
    r = route("is there anything wrong with my computer")
    assert r == ToolCall(tool="system_problems", args={})


def test_cap98_01_p2():
    r = route("check for problems with my computer")
    assert r == ToolCall(tool="system_problems", args={})


def test_cap98_02_p1():
    r = route("do you see anything unusual going on")
    assert r == ToolCall(tool="system_problems", args={})


def test_cap98_02_p2():
    r = route("anything weird going on with my system")
    assert r == ToolCall(tool="system_problems", args={})


def test_cap100_01_p1():
    r = route("hey what should i know")
    assert r == ToolCall(tool="heads_up", args={})


def test_cap100_01_p2():
    r = route("can you give me a heads up please")
    assert r == ToolCall(tool="heads_up", args={})


def test_cap96_02_p1():
    r = route("hey tell me about my dentist appointment")
    assert r == ToolCall(tool="meeting_brief", args={"title": "dentist"})


def test_cap96_02_p2():
    r = route("can you brief me on my dentist appointment please")
    assert r == ToolCall(tool="meeting_brief", args={"title": "dentist"})


def test_cap97_x2_p1():
    r = route("hey what did i tell you about window seats")
    assert r == ToolCall(tool="recall_fact", args={"topic": "window seats"})


def test_cap97_x2_p2():
    r = route("do you know anything about window seats")
    assert r == ToolCall(tool="recall_fact", args={"topic": "window seats"})


def test_guard_todo_list():
    r = route("what's on my to-do list")
    assert r == ToolCall(tool="list_read", args={"list": "to-do"})


def test_guard_time():
    r = route("what time is it")
    assert r == ToolCall(tool="tell_time", args={})


def test_guard_capital():
    assert route("what is the capital of France") is None


def test_guard_open_app():
    r = route("open the calculator")
    assert r == ToolCall(tool="open_app", args={"name": "calculator"})
