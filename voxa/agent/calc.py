"""Arithmetic and unit conversions answered exactly, without the AI model."""

from __future__ import annotations

import re

_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "a": 1,
    "an": 1,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}

_LEAD_INS = (
    "what does",
    "how much is",
    "what is",
    "what's",
    "what's",
    "calculate",
    "work out",
)
_TRAILINGS = ("equals", "equal", "make")

_OP_PHRASES: list[tuple[tuple[str, ...], str]] = [
    (("to", "the", "power", "of"), "^"),
    (("square", "root", "of"), "sqrt"),
    (("multiplied", "by"), "*"),
    (("divided", "by"), "/"),
    (("take", "away"), "-"),
    (("percent", "of"), "%of"),
    (("percent", "off"), "%off"),
    (("a", "quarter", "of"), "qtr"),
    (("a", "third", "of"), "third"),
    (("half", "of"), "half"),
    (("plus",), "+"),
    (("and",), "+"),
    (("minus",), "-"),
    (("times",), "*"),
    (("over",), "/"),
    (("squared",), "sqr"),
    (("cubed",), "cub"),
    (("double",), "dbl"),
    (("triple",), "tri"),
    (("+",), "+"),
    (("-",), "-"),
    (("*",), "*"),
    (("/",), "/"),
    (("^",), "^"),
]

_NUMBER_WORDS = frozenset(_ONES) | frozenset(_TENS) | frozenset(_SCALES) | {"point"}


def _words_to_value(tokens: list[str]) -> float | None:
    """Value of number words like "three hundred and five", or None."""
    if not tokens:
        return None
    if "point" in tokens:
        split = tokens.index("point")
        head = _words_value(tokens[:split]) if tokens[:split] else None
        tail = tokens[split + 1 :]
        if head is None or not tail or head != int(head) or abs(head) >= 1e15:
            return None
        digits = 0
        fraction = 0.0
        for tok in tail:
            if tok in _ONES and _ONES[tok] < 10:
                fraction = fraction * 10 + _ONES[tok]
                digits += 1
            elif re.fullmatch(r"[0-9]", tok):
                fraction = fraction * 10 + int(tok)
                digits += 1
            else:
                return None
        return head + fraction / 10**digits
    return _words_value(tokens)


def _words_value(tokens: list[str]) -> float | None:
    """Whole-number value of number words, or None when not all are number words."""
    if not tokens:
        return None
    total = 0
    current = 0
    seen_hundred = False
    for tok in tokens:
        if tok in _ONES:
            current += _ONES[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            current = max(current, 1) * 100
            seen_hundred = True
        elif tok in _SCALES:
            current = max(current, 1) * _SCALES[tok]
            total += current
            current = 0
            seen_hundred = False
        elif tok == "and":
            if not seen_hundred:
                return None
        else:
            return None
    return float(total + current)


def parse_number(text: str) -> float | None:
    """The number a phrase says ("532", "1,200", "minus 4", "twenty one"), or None."""
    s = text.strip().lower().replace(",", "")
    if not s:
        return None
    if s in ("a half", "half"):
        return 0.5
    if s in ("a quarter", "quarter"):
        return 0.25
    sign = 1.0
    if s.startswith("minus "):
        sign, s = -1.0, s[len("minus ") :]
    elif s.startswith("-"):
        sign, s = -1.0, s[1 :]
    if not s:
        return None
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?|\.[0-9]+", s):
        return sign * float(s)
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?|\.[0-9]+", s.replace(" point ", ".")) and " point " not in s:
        return None
    if " point " in s:
        head, _, tail = s.partition(" point ")
        head_value = parse_number(head) if head else None
        if head_value is None or head_value != int(head_value):
            return None
        digits = 0
        fraction = 0.0
        for tok in tail.split():
            value = _ONES.get(tok)
            if value is None and re.fullmatch(r"[0-9]", tok):
                value = int(tok)
            if value is None or value >= 10:
                return None
            fraction = fraction * 10 + value
            digits += 1
        if digits == 0:
            return None
        return sign * (head_value + fraction / 10**digits)
    value = _words_value(s.split())
    return sign * value if value is not None else None


def _strip_lead_in(s: str) -> str:
    """Remove a spoken lead-in ("what is", "calculate") and trailing "equals"/"?"."""
    s = s.strip().lower()
    s = s.rstrip("?").strip()
    for lead in _LEAD_INS:
        if s == lead or s.startswith(lead + " "):
            s = s[len(lead) :].strip()
            break
    s = s.rstrip("?").strip()
    for trail in _TRAILINGS:
        if s.endswith(" " + trail):
            s = s[: -len(trail)].strip()
            break
    return s.strip(".!?,;").strip()


def _tokenise(s: str) -> list[tuple[str, float | None]] | None:
    """Tokens (kind, value) with kind "num"/"op"; None when a word is not a number or operator."""
    words = re.findall(r"[a-z0-9.,%]+|[+\-*/^]", s)
    if "".join(words) != s.replace(" ", ""):
        return None
    tokens: list[tuple[str, float | None]] = []
    i = 0
    while i < len(words):
        matched = False
        for phrase, op in _OP_PHRASES:
            if tuple(words[i : i + len(phrase)]) == phrase:
                tokens.append((op, None))
                i += len(phrase)
                matched = True
                break
        if matched:
            continue
        run = [words[i]]
        j = i + 1
        while j < len(words):
            tok = words[j]
            if tok in _NUMBER_WORDS or re.fullmatch(r"[0-9][0-9.]*", tok):
                run.append(tok)
                j += 1
            elif tok == "and" and "hundred" in run and j + 1 < len(words) and (
                words[j + 1] in _NUMBER_WORDS or re.fullmatch(r"[0-9][0-9.]*", words[j + 1])
            ):
                run.append(tok)
                j += 1
            else:
                break
        value = parse_number(" ".join(run))
        if value is None:
            return None
        tokens.append(("num", value))
        i = j
    return tokens


class _Parse:
    def __init__(self, tokens: list[tuple[str, float | None]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        kind, _ = self.tokens[self.pos]
        return kind if kind != "num" else "num"

    def value(self) -> float:
        value = self.tokens[self.pos][1]
        self.pos += 1
        return float(value or 0.0)

    def expr(self) -> float:
        total = self.term()
        while self.peek() in ("+", "-"):
            op = self.tokens[self.pos][0]
            self.pos += 1
            other = self.term()
            total = total + other if op == "+" else total - other
        return total

    def term(self) -> float:
        total = self.power()
        while self.peek() in ("*", "/", "%of", "%off"):
            op = self.tokens[self.pos][0]
            self.pos += 1
            other = self.power()
            if op == "*":
                total *= other
            elif op == "/":
                if other == 0:
                    raise ZeroDivisionError
                total /= other
            elif op == "%of":
                total = total * other / 100.0
            else:
                total = other - other * total / 100.0
        return total

    def power(self) -> float:
        base = self.unary()
        while self.peek() in ("sqr", "cub"):
            op = self.tokens[self.pos][0]
            self.pos += 1
            base = base**2 if op == "sqr" else base**3
        if self.peek() == "^":
            self.pos += 1
            base = base**self.power()
        return base

    def unary(self) -> float:
        kind = self.peek()
        if kind == "-":
            self.pos += 1
            return -self.unary()
        if kind == "sqrt":
            self.pos += 1
            value = self.unary()
            if value < 0:
                raise ValueError
            return value**0.5
        if kind == "half":
            self.pos += 1
            return self.unary() / 2.0
        if kind == "qtr":
            self.pos += 1
            return self.unary() / 4.0
        if kind == "third":
            self.pos += 1
            return self.unary() / 3.0
        if kind == "dbl":
            self.pos += 1
            return self.unary() * 2.0
        if kind == "tri":
            self.pos += 1
            return self.unary() * 3.0
        if kind == "num":
            return self.value()
        raise ValueError


def evaluate(text: str) -> float | None:
    """The value of a spoken sum, or None when the text is not purely arithmetic."""
    s = _strip_lead_in(text)
    if not s:
        return None
    tokens = _tokenise(s)
    if tokens is None:
        return None
    if not any(kind == "num" for kind, _ in tokens):
        return None
    ops = {"+", "-", "*", "/", "^", "%of", "%off", "sqr", "cub", "sqrt", "half", "qtr", "third", "dbl", "tri"}
    if not any(kind in ops for kind, _ in tokens):
        return None
    try:
        parser = _Parse(tokens)
        value = parser.expr()
        if parser.pos != len(tokens):
            return None
        if not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
            return None
        return float(value)
    except (ZeroDivisionError, ValueError, IndexError, RecursionError):
        return None


def _unit_kind(unit: str) -> str | None:
    kinds = {
        "length": {"km": "kilometers", "kms": "kilometers", "kilometer": "kilometers", "kilometers": "kilometers", "mi": "miles", "miles": "miles", "mile": "miles", "m": "meters", "meters": "meters", "meter": "meters", "ft": "feet", "foot": "feet", "feet": "feet", "in": "inches", "inch": "inches", "inches": "inches", "cm": "centimeters", "centimeter": "centimeters", "centimeters": "centimeters", "yd": "yards", "yard": "yards", "yards": "yards"},
        "mass": {"kg": "kilograms", "kgs": "kilograms", "kilo": "kilograms", "kilos": "kilograms", "kilogram": "kilograms", "kilograms": "kilograms", "lb": "pounds", "lbs": "pounds", "pound": "pounds", "pounds": "pounds", "g": "grams", "gram": "grams", "grams": "grams", "oz": "ounces", "ounce": "ounces", "ounces": "ounces"},
        "volume": {"l": "liters", "liter": "liters", "liters": "liters", "litre": "liters", "litres": "liters", "gal": "gallons", "gallon": "gallons", "gallons": "gallons", "ml": "milliliters", "milliliter": "milliliters", "milliliters": "milliliters", "cup": "cups", "cups": "cups"},
        "speed": {"mph": "mph", "km/h": "kilometers per hour", "kph": "kilometers per hour"},
        "temp": {"c": "degrees Celsius", "celsius": "degrees Celsius", "f": "degrees Fahrenheit", "fahrenheit": "degrees Fahrenheit"},
    }
    for kind, table in kinds.items():
        if unit in table:
            return kind
    return None


def _unit_name(unit: str) -> str:
    for table in (
        {"km": "kilometers", "kms": "kilometers", "kilometer": "kilometers", "kilometers": "kilometers", "mi": "miles", "mile": "miles", "miles": "miles", "m": "meters", "meter": "meters", "meters": "meters", "ft": "feet", "foot": "feet", "feet": "feet", "in": "inches", "inch": "inches", "inches": "inches", "cm": "centimeters", "centimeter": "centimeters", "centimeters": "centimeters", "yd": "yards", "yard": "yards", "yards": "yards"},
        {"kg": "kilograms", "kgs": "kilograms", "kilo": "kilograms", "kilos": "kilograms", "kilogram": "kilograms", "kilograms": "kilograms", "lb": "pounds", "lbs": "pounds", "pound": "pounds", "pounds": "pounds", "g": "grams", "gram": "grams", "grams": "grams", "oz": "ounces", "ounce": "ounces", "ounces": "ounces"},
        {"l": "liters", "liter": "liters", "liters": "liters", "litre": "liters", "litres": "liters", "gal": "gallons", "gallon": "gallons", "gallons": "gallons", "ml": "milliliters", "milliliter": "milliliters", "milliliters": "milliliters", "cup": "cups", "cups": "cups"},
        {"mph": "mph", "km/h": "kilometers per hour", "kph": "kilometers per hour"},
        {"c": "degrees Celsius", "celsius": "degrees Celsius", "f": "degrees Fahrenheit", "fahrenheit": "degrees Fahrenheit"},
    ):
        if unit in table:
            return table[unit]
    return unit


_KM_PER_MILE = 1.609344
_M_PER_FT = 0.3048
_M_PER_IN = 0.0254
_CM_PER_IN = 2.54
_YD_PER_M = 1.093613333
_KG_PER_LB = 0.45359237
_G_PER_OZ = 28.349312375
_L_PER_GAL = 3.78541178
_ML_PER_CUP = 236.588236588
_KMPH_PER_MPH = 1.609344


def _unit_key(unit: str) -> str | None:
    """Canonical abbreviation for a unit word, or None when it is not a unit."""
    table = {
        "km": "km", "kms": "km", "kilometer": "km", "kilometers": "km",
        "mi": "mi", "mile": "mi", "miles": "mi",
        "m": "m", "meter": "m", "meters": "m",
        "ft": "ft", "foot": "ft", "feet": "ft",
        "in": "in", "inch": "in", "inches": "in",
        "cm": "cm", "centimeter": "cm", "centimeters": "cm",
        "yd": "yd", "yard": "yd", "yards": "yd",
        "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "kilograms": "kg",
        "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
        "g": "g", "gram": "g", "grams": "g",
        "oz": "oz", "ounce": "oz", "ounces": "oz",
        "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
        "gal": "gal", "gallon": "gal", "gallons": "gal",
        "ml": "ml", "milliliter": "ml", "milliliters": "ml",
        "cup": "cup", "cups": "cup",
        "mph": "mph", "kph": "kph",
        "c": "c", "celsius": "c",
        "f": "f", "fahrenheit": "f",
    }
    return table.get(unit)


def _convert_value(value: float, src: str, dst: str) -> float | None:
    src, dst = _unit_key(src), _unit_key(dst)
    if src is None or dst is None or _unit_kind(src) != _unit_kind(dst):
        return None
    if src == dst:
        return value
    if _unit_kind(src) == "temp":
        if src == "c":
            return value * 1.8 + 32.0
        return (value - 32.0) / 1.8
    pairs = {
        ("km", "mi"): lambda v: v / _KM_PER_MILE,
        ("mi", "km"): lambda v: v * _KM_PER_MILE,
        ("m", "ft"): lambda v: v / _M_PER_FT,
        ("ft", "m"): lambda v: v * _M_PER_FT,
        ("in", "m"): lambda v: v * _M_PER_IN,
        ("m", "in"): lambda v: v / _M_PER_IN,
        ("cm", "in"): lambda v: v / _CM_PER_IN,
        ("in", "cm"): lambda v: v * _CM_PER_IN,
        ("yd", "m"): lambda v: v / _YD_PER_M,
        ("m", "yd"): lambda v: v * _YD_PER_M,
        ("kg", "lb"): lambda v: v / _KG_PER_LB,
        ("lb", "kg"): lambda v: v * _KG_PER_LB,
        ("g", "oz"): lambda v: v / _G_PER_OZ,
        ("oz", "g"): lambda v: v * _G_PER_OZ,
        ("l", "gal"): lambda v: v / _L_PER_GAL,
        ("gal", "l"): lambda v: v * _L_PER_GAL,
        ("ml", "cup"): lambda v: v / _ML_PER_CUP,
        ("cup", "ml"): lambda v: v * _ML_PER_CUP,
        ("mph", "kph"): lambda v: v * _KMPH_PER_MPH,
        ("kph", "mph"): lambda v: v / _KMPH_PER_MPH,
    }
    return pairs.get((src, dst), lambda v: None)(value)


_CONNECTORS = frozenset({"in", "to", "is", "into", "as", "equal", "equals", "be", "'s"})


def _strip_words(s: str, drop: tuple[str, ...]) -> str:
    for word in drop:
        s = re.sub(rf"\b{re.escape(word)}\b", " ", s)
    return " ".join(s.split())


def convert(text: str) -> tuple[float, str] | None:
    """(value, spoken target unit) for a spoken conversion, or None."""
    s = _strip_lead_in(text).replace("km/h", "kph").replace("°", "")
    s = re.sub(r"^(how many|how much)\s+", "", s)
    s = re.sub(r"^(convert|change|turn)\s+", "", s)
    words = s.split()
    units = [i for i, tok in enumerate(words) if _unit_kind(tok) is not None]
    if len(units) != 2:
        return None
    i, j = units
    first, second = words[i], words[j]
    before = " ".join(words[:i])
    between = _strip_words(" ".join(words[i + 1 : j]), ("is", "in", "to", "be", "'s"))
    after = " ".join(words[j + 1 :])
    value = src = dst = None
    if parse_number(before) is not None and not after:
        value, src, dst = parse_number(before), first, second
    elif parse_number(after) is not None and not before:
        value, src, dst = parse_number(after), second, first
    elif parse_number(between) is not None:
        value, src, dst = parse_number(between), second, first
    if value is None or src is None or dst is None:
        return None
    if _unit_kind(src) != _unit_kind(dst):
        return None
    converted = _convert_value(value, src, dst)
    if converted is None:
        return None
    return (converted, _unit_name(dst))


def spoken_number(value: float) -> str:
    """A number as it should be spoken: "1,321", "2.5", "1.23e+16"."""
    if value != value or value in (float("inf"), float("-inf")):
        return str(value)
    if abs(value) >= 1e15 or (0 < abs(value) < 0.01):
        return f"{value:.3g}"
    if value == int(value):
        return f"{int(value):,d}"
    return f"{round(value, 2):.2f}".rstrip("0").rstrip(".")


def answer(text: str) -> str | None:
    """A one-line spoken answer to a sum or conversion, or None when neither applies."""
    converted = convert(text)
    if converted is not None:
        value, unit = converted
        return f"{spoken_number(value)} {unit}."
    value = evaluate(text)
    if value is None:
        return None
    s = _strip_lead_in(text)
    return f"{s} is {spoken_number(value)}."