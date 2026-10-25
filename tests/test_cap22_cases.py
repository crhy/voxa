from voxa.agent.hearing import normalize
from voxa.agent.intents import route


def test_cap22_kitchen_list_cases_route() -> None:
    cases = [
        ("Add coffee to my kitchen list", "list_add"),
        ("can you add coffee to my kitchen list please", "list_add"),
        ("hey can you add coffee to my kitchen list", "list_add"),
        ("What's on my kitchen list", "list_read"),
        ("hey what is on my kitchen list", "list_read"),
        ("can you read my kitchen list", "list_read"),
        ("Clear my kitchen list", "list_clear"),
        ("please clear my kitchen list", "list_clear"),
        ("hey can you clear my kitchen list", "list_clear"),
    ]
    for phrase, expected in cases:
        call = route(normalize(phrase))
        assert call is not None and call.tool == expected, f"{phrase!r} -> {call}"


def test_cap22_clear_is_idempotent() -> None:
    assert route("clear my kitchen list") == route("please clear my kitchen list")


def test_cap22_does_not_swallow_ordinary_questions() -> None:
    assert route("what's on my to-do list").tool == "list_read"
    assert route("what time is it").tool == "tell_time"
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"
