"""Meal ideas, weekly plans and shopping-list handoff route and work end to end."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.meals import RECIPES, find, plan, spoken_plan, spoken_suggestion, suggest, tag_for
from voxa.agent.tools.meals import (
    _meal_idea_handler,
    _meal_shop_handler,
)


def test_suggest_is_deterministic() -> None:
    assert suggest(0) == suggest(0)
    assert suggest(7, "quick") == suggest(7, "quick")
    assert suggest(3, "salad") is not None
    assert suggest(0, "nope") is None


def test_plan_has_no_repeats() -> None:
    week = plan(100, 7)
    names = [recipe[0] for recipe in week]
    assert len(names) == len(set(names))


def test_tag_for() -> None:
    assert tag_for("something quick") == "quick"
    assert tag_for("with chicken") == "chicken"
    assert tag_for("a soup") == "soup"
    assert tag_for("dinner") is None


def test_find() -> None:
    assert find("chicken stir fry") == RECIPES[0]
    assert find("stir fry") == RECIPES[0]
    assert find("chicken stir fri") == RECIPES[0]
    assert find("zzzz") is None


def test_spoken_strings() -> None:
    recipe = RECIPES[0]
    assert spoken_suggestion(recipe) == (
        "How about chicken stir fry? You need chicken, peppers, soy sauce, rice and broccoli."
    )
    week = plan(0, 3)
    assert spoken_plan(week, "Monday") == "Monday: chicken stir fry. Tuesday: vegetable soup. Wednesday: grilled salmon."


def test_meal_phrases_route() -> None:
    assert route("what should i cook tonight") == ToolCall("meal_idea", {"tag": ""})
    assert route("what's for dinner") == ToolCall("meal_idea", {"tag": ""})
    assert route("suggest a quick dinner idea") == ToolCall("meal_idea", {"tag": "quick"})
    assert route("plan my meals for the week") == ToolCall("meal_plan", {})
    assert route("make a meal plan") == ToolCall("meal_plan", {})
    assert route("add the ingredients to my shopping list") == ToolCall("meal_shop", {"name": ""})
    assert route("i'll make that") == ToolCall("meal_shop", {"name": "that"})


def test_meal_rules_do_not_steal_other_phrases() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_meal_idea_remembers_last(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.meals.LAST_MEAL", None)
    result = _meal_idea_handler({"tag": ""})
    assert result.ok
    assert result.speech == spoken_suggestion(suggest(__import__("datetime").date.today().toordinal()))


def test_meal_shop_uses_last_suggestion(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: store["content"])
    monkeypatch.setattr("voxa.agent.tools.lists._write", lambda content: store.__setitem__("content", content))
    monkeypatch.setattr("voxa.agent.tools.meals.LAST_MEAL", RECIPES[0])
    result = _meal_shop_handler({"name": ""})
    assert result.ok
    assert result.speech == f"I added {len(RECIPES[0][2])} things for chicken stir fry to your shopping list."
    assert "Chicken" in store["content"]


def test_meal_shop_names_a_recipe(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: store["content"])
    monkeypatch.setattr("voxa.agent.tools.lists._write", lambda content: store.__setitem__("content", content))
    monkeypatch.setattr("voxa.agent.tools.meals.LAST_MEAL", None)
    result = _meal_shop_handler({"name": "garden salad"})
    assert result.ok
    assert result.speech == "I added 5 things for garden salad to your shopping list."


def test_meal_shop_without_a_meal_fails(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.meals.LAST_MEAL", None)
    result = _meal_shop_handler({"name": ""})
    assert not result.ok
    assert result.speech == "Which meal?"
