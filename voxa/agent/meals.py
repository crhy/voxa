from __future__ import annotations

import difflib
import re

RECIPES: list[tuple[str, tuple[str, ...], list[str]]] = [
    ("chicken stir fry", ("quick", "chicken"), ["chicken", "peppers", "soy sauce", "rice", "broccoli"]),
    ("spaghetti bolognese", ("pasta", "beef"), ["pasta", "beef", "tomato sauce", "onion", "garlic"]),
    ("vegetable soup", ("vegetarian", "soup"), ["carrot", "potato", "onion", "celery", "stock"]),
    ("garden salad", ("vegetarian", "salad"), ["lettuce", "tomato", "cucumber", "carrot", "dressing"]),
    ("grilled salmon", ("fish", "quick"), ["salmon", "lemon", "butter", "asparagus"]),
    ("beef tacos", ("beef", "quick"), ["beef", "tortilla", "cheese", "lettuce", "salsa"]),
    ("chicken curry", ("chicken", "quick"), ["chicken", "curry paste", "coconut milk", "rice", "onion"]),
    ("mac and cheese", ("pasta", "vegetarian"), ["pasta", "cheese", "butter", "milk"]),
    ("tomato soup", ("vegetarian", "soup"), ["tomato", "onion", "garlic", "stock", "cream"]),
    ("fish and chips", ("fish",), ["fish", "potato", "flour", "oil", "lemon"]),
    ("caesar salad", ("salad",), ["lettuce", "parmesan", "croutons", "dressing"]),
    ("chicken pasta bake", ("chicken", "pasta"), ["chicken", "pasta", "tomato sauce", "cheese", "onion"]),
    ("beef stew", ("beef", "soup"), ["beef", "potato", "carrot", "onion", "stock"]),
    ("mushroom risotto", ("vegetarian", "pasta"), ["rice", "mushroom", "onion", "stock", "cheese"]),
    ("tuna salad", ("fish", "salad"), ["tuna", "lettuce", "tomato", "egg", "mayo"]),
    ("chicken nuggets", ("chicken", "quick"), ["chicken", "flour", "egg", "oil"]),
    ("lentil soup", ("vegetarian", "soup"), ["lentil", "carrot", "onion", "celery", "stock"]),
    ("beef stir fry", ("beef", "quick"), ["beef", "peppers", "soy sauce", "rice", "onion"]),
    ("pasta salad", ("pasta", "salad", "vegetarian"), ["pasta", "tomato", "cucumber", "onion", "dressing"]),
    ("fish pie", ("fish",), ["fish", "potato", "butter", "flour", "milk"]),
    ("chicken soup", ("chicken", "soup"), ["chicken", "carrot", "celery", "onion", "noodles"]),
    ("bean chili", ("vegetarian", "quick"), ["bean", "tomato", "onion", "peppers", "spices"]),
]

_TAGS = ("quick", "vegetarian", "chicken", "beef", "fish", "pasta", "soup", "salad")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def suggest(day_number: int, tag: str | None = None, avoid: set[str] = frozenset()) -> tuple | None:
    matching = [recipe for recipe in RECIPES if (tag is None or tag in recipe[1]) and recipe[0] not in avoid]
    if not matching:
        return None
    return matching[day_number % len(matching)]


def plan(start_day_number: int, days: int = 7, tag: str | None = None) -> list:
    out: list = []
    used: set[str] = set()
    for i in range(days):
        recipe = suggest(start_day_number + i, tag, used)
        if recipe is None:
            break
        out.append(recipe)
        used.add(recipe[0])
    return out


def tag_for(spoken: str) -> str | None:
    low = spoken.lower()
    for tag in _TAGS:
        if re.search(rf"\b{tag}\b", low):
            return tag
    return None


def spoken_suggestion(recipe: tuple) -> str:
    name, _tags, ingredients = recipe
    if len(ingredients) == 1:
        joined = ingredients[0]
    else:
        joined = ", ".join(ingredients[:-1]) + " and " + ingredients[-1]
    return f"How about {name}? You need {joined}."


def spoken_plan(recipes: list, first_weekday_name: str) -> str:
    try:
        start = next(i for i, day in enumerate(_WEEKDAYS) if day.lower() == first_weekday_name.lower())
    except StopIteration:
        start = 0
    parts = [f"{_WEEKDAYS[(start + i) % 7]}: {recipe[0]}." for i, recipe in enumerate(recipes)]
    return " ".join(parts)


def find(name: str) -> tuple | None:
    low = name.lower().strip()
    for recipe in RECIPES:
        if recipe[0].lower() == low:
            return recipe
    for recipe in RECIPES:
        if low in recipe[0].lower() or recipe[0].lower() in low:
            return recipe
    close = difflib.get_close_matches(low, [recipe[0].lower() for recipe in RECIPES], n=1, cutoff=0.7)
    if close:
        for recipe in RECIPES:
            if recipe[0].lower() == close[0]:
                return recipe
    return None
