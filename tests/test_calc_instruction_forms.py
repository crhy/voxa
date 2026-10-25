"""Tests for instruction-form arithmetic routed to the calculator (s1)."""

from __future__ import annotations

from voxa.agent import calc, intents


def test_instruction_forms():
    """The five failing phrases now return correct results."""
    cases = {
        "Divide 100 by 4": 25.0,
        "Add 15 and 20": 35.0,
        "Subtract 10 from 50": 40.0,
        "Multiply 7 by 8": 56.0,
        "What is the sum of five and seven": 12.0,
    }
    for text, expected in cases.items():
        assert calc.evaluate(text) == expected, f"evaluate({text!r})"


def test_answer_contains_result():
    """answer() returns a reply containing the numeric result."""
    assert "25" in (calc.answer("Divide 100 by 4") or "")
    assert "35" in (calc.answer("Add 15 and 20") or "")
    assert "40" in (calc.answer("Subtract 10 from 50") or "")
    assert "56" in (calc.answer("Multiply 7 by 8") or "")
    assert "12" in (calc.answer("What is the sum of five and seven") or "")


_number_words = {
    "Add five and seven": 12.0,
    "Multiply three by four": 12.0,
    "Divide twenty by five": 4.0,
    "Subtract three from twenty one": 18.0,
    "the product of six and seven": 42.0,
    "the difference between fifty and thirty": 20.0,
}


def test_number_words():
    for text, expected in _number_words.items():
        assert calc.evaluate(text) == expected, f"evaluate({text!r})"


def test_polite_wrappers():
    """Leading 'please'/'can you'/'could you' and trailing 'please'/'for me'."""
    assert calc.evaluate("please divide 100 by 4") == 25.0
    assert calc.evaluate("can you add 15 and 20") == 35.0
    assert calc.evaluate("could you multiply 7 by 8") == 56.0
    assert calc.evaluate("divide 100 by 4 please") == 25.0
    assert calc.evaluate("add 15 and 20 for me") == 35.0


def test_non_arithmetic_guards():
    """Non-arithmetic sentences must NOT route to calculate."""
    guards = [
        "add milk to my shopping list",
        "add a reminder",
        "divide the work between us",
        "subtract that from my budget",
    ]
    for text in guards:
        assert calc.answer(text) is None, f"answer({text!r}) should be None"
        result = intents.route(text)
        # Must not be a calculate call
        if result is not None:
            assert result.tool != "calculate", f"route({text!r}) should not be calculate"
