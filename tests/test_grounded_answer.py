from voxa import websearch

RESULTS = [
    websearch.SearchResult(
        "Padres beat Giants 7-3 today",
        "https://www.espn.com/mlb",
        "The Padres beat the Giants 7 to 3 on Tuesday. Final score 7-3.",
    )
]


def test_invented_score_is_replaced_by_the_snippet() -> None:
    reply = "The Dodgers won 5 to 1 against the Mets yesterday."
    answer = websearch.grounded_answer(reply, RESULTS, "who won the padres game today")
    assert "Dodgers" not in answer
    assert "Padres" in answer
    assert "espn.com" in answer


def test_faithful_reply_is_kept_and_shortened() -> None:
    reply = "The Padres beat the Giants 7 to 3 on Tuesday. It was a close game. The final score was 7 to 3."
    answer = websearch.grounded_answer(reply, RESULTS, "who won the padres game today")
    assert answer == "The Padres beat the Giants 7 to 3 on Tuesday. It was a close game."


def test_empty_results_leave_the_reply_unchanged() -> None:
    assert websearch.grounded_answer("just a model answer", [], "who won") == "just a model answer"


def test_a_number_not_in_the_results_triggers_replacement() -> None:
    reply = "The Padres beat the Giants 99 to 3 on Tuesday."
    answer = websearch.grounded_answer(reply, RESULTS, "who won the padres game today")
    assert "99" not in answer
    assert "espn.com" in answer
