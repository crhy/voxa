from __future__ import annotations

import logging
from datetime import date

from voxa.agent.meals import find, plan, spoken_plan, spoken_suggestion, suggest, tag_for
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.lists import _list_add_handler

log = logging.getLogger(__name__)

LAST_MEAL: tuple | None = None


def _meal_idea_handler(args: dict[str, str]) -> ToolResult:
    tag = tag_for(args.get("tag", ""))
    recipe = suggest(date.today().toordinal(), tag)
    if recipe is None:
        return ToolResult.failure("I have no idea for that.")
    global LAST_MEAL
    LAST_MEAL = recipe
    return ToolResult.success(spoken_suggestion(recipe))


def _meal_plan_handler(args: dict[str, str]) -> ToolResult:
    tag = tag_for(args.get("tag", ""))
    recipes = plan(date.today().toordinal(), 7, tag)
    if not recipes:
        return ToolResult.failure("I have no ideas for that.")
    return ToolResult.success(spoken_plan(recipes, date.today().strftime("%A")))


def _meal_shop_handler(args: dict[str, str]) -> ToolResult:
    global LAST_MEAL
    name = args.get("name", "").strip()
    if not name or name.lower() == "that":
        recipe = LAST_MEAL
    else:
        recipe = find(name)
    if recipe is None:
        return ToolResult.failure("Which meal?")
    count = 0
    for ingredient in recipe[2]:
        result = _list_add_handler({"item": ingredient, "list": "shopping"})
        if result.ok:
            count += 1
    if count == 0:
        return ToolResult.failure("Which meal?")
    return ToolResult.success(f"I added {count} things for {recipe[0]} to your shopping list.")


def meals_tools() -> list[Tool]:
    return [
        Tool(
            name="meal_idea",
            description="Suggest one dinner for tonight, optionally filtered by a tag.",
            parameters={"tag": "a preference, e.g. quick or vegetarian"},
            risk=RiskLevel.READ_ONLY,
            handler=_meal_idea_handler,
        ),
        Tool(
            name="meal_plan",
            description="Plan the dinners for the week, optionally filtered by a tag.",
            parameters={"tag": "a preference, e.g. quick or vegetarian"},
            risk=RiskLevel.READ_ONLY,
            handler=_meal_plan_handler,
        ),
        Tool(
            name="meal_shop",
            description="Add a recipe's ingredients to the shopping list.",
            parameters={"name": "the recipe name, or empty for the last suggestion"},
            risk=RiskLevel.REVERSIBLE,
            handler=_meal_shop_handler,
        ),
    ]
