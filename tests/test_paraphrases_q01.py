from voxa.agent.intents import route


def test_cap2_paraphrases():
    assert route("write a new email").tool == "compose_gmail"
    assert route("hey can you write a new email please").tool == "compose_gmail"


def test_cap15_paraphrases():
    assert route("what do i have tomorrow").tool == "calendar_day"
    assert route("hey what's on my calendar tomorrow please").tool == "calendar_day"
    assert route("what is my next appointment").tool == "calendar_next"
    assert route("hey what's next on my calendar please").tool == "calendar_next"


def test_cap17_paraphrases():
    assert route("when is my mom's birthday").tool == "date_when"
    assert route("hey when is mom's birthday please").tool == "date_when"
    assert route("any birthdays this week").tool == "date_upcoming"
    assert route("what birthdays are coming up").tool == "date_upcoming"


def test_cap18_paraphrases():
    assert route("what's the time in Tokyo").tool == "world_time"
    assert route("hey can you tell me the time in Tokyo please").tool == "world_time"
    assert route("what time is it in London").tool == "world_time"
    assert route("hey can you tell me the time in London please").tool == "world_time"
    assert route("what is the time difference between here and London").tool == "time_difference"
    assert route("hey what's the time difference with London please").tool == "time_difference"
    assert route("how far ahead is Tokyo").tool == "time_difference"
    assert route("hey how many hours ahead is Tokyo please").tool == "time_difference"
