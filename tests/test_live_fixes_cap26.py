from voxa.agent import intents


def test_glued_greeting_before_mark_done():
    # "Heymark finish the report is done" is "hey mark finish the report as done" heard as one word.
    call = intents.route("Heymark finish the report is done")
    assert call is not None and call.tool == "list_done" and call.args["item"] == "finish the report"


def test_list_add_recovers_dropped_letter_in_add():
    call = intents.route("can you at finish the report to my to-do list")
    assert call is not None and call.tool == "list_add"
    assert call.args["item"] == "finish the report" and call.args["list"] == "to-do"


def test_list_add_recovers_mangled_finished():
    call = intents.route("hey I've finished the report to my to-do list")
    assert call is not None and call.tool == "list_add"
    assert call.args["item"] == "the report" and call.args["list"] == "to-do"
    call = intents.route("I've just watered the plants to my to-do list")
    assert call is None or call.tool != "list_add"


def test_plain_i_finished_still_marks_done():
    call = intents.route("I finished call the vet")
    assert call is not None and call.tool == "list_done" and call.args["item"] == "call the vet"
