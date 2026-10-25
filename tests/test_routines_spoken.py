from voxa.agent.routines import create_reply, delete_reply, list_reply


def test_create_reply_names_the_trigger_and_steps():
    reply = create_reply("good morning", ["play the news", "make coffee"])
    assert reply == "Okay. When you say \u201cgood morning\u201c I'll play the news, then make coffee."


def test_create_reply_single_step():
    reply = create_reply("focus", ["start the timer"])
    assert reply == "Okay. When you say \u201cfocus\u201c I'll start the timer."


def test_list_reply_empty():
    assert list_reply([]) == "No routines yet."


def test_list_reply_names_each_routine():
    assert list_reply(["good morning", "wind down"]) == "Routines: good morning, wind down"


def test_delete_reply_success():
    assert delete_reply("good morning", True) == "Deleted the good morning routine."


def test_delete_reply_missing():
    assert delete_reply("good morning", False) == "There is no good morning routine."
