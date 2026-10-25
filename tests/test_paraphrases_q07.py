from voxa.agent.intents import route


def test_cap14_conflicts_p1():
    assert route("are there any conflicts this week").tool == "calendar_conflicts"
    assert route("do i have any clashes this week").tool == "calendar_conflicts"


def test_cap14_conflicts_p2():
    assert route("hey can you check my calendar for clashes").tool == "calendar_conflicts"
    assert route("check my calendar for conflicts").tool == "calendar_conflicts"


def test_cap22_list_add_p1():
    assert route("can you add buy paint to my kitchen list please").tool == "list_add"
    assert route("hey can you add buy paint to my kitchen list").tool == "list_add"


def test_cap22_list_read_p1():
    assert route("hey what is on my kitchen list").tool == "list_read"
    assert route("can you read my kitchen list").tool == "list_read"


def test_cap22_list_clear_p1():
    assert route("please clear my kitchen list").tool == "list_clear"
    assert route("hey can you clear my kitchen list").tool == "list_clear"


def test_cap25_pending_p1():
    assert route("hey what is still pending").tool == "list_pending"
    assert route("what do i still have to do").tool == "list_pending"


def test_cap27_next_p1():
    assert route("hey what is my top priority").tool == "list_next"
    assert route("what should i work on next").tool == "list_next"


def test_guards_still_route():
    assert route("what's on my to-do list").tool == "list_read"
    assert route("what time is it").tool == "tell_time"
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"
