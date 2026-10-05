"""claims_action flags replies that pretend an action was done."""

from __future__ import annotations

import pytest

from voxa.agent.claims import claims_action

CLAIMS = [
    "Done. Brutal Chess is closed.",
    "Done, the window is closed.",
    "I've closed Brutal Chess.",
    "I have sent the email.",
    "I opened the file manager.",
    "I started the timer.",
    "I saved the document.",
    "I deleted the folder.",
    "I booked a table for two.",
    "I turned on the light.",
    "I set an alarm for seven.",
    "I created a new note.",
    "I launched the app.",
    "I played the video.",
    "Opening it now.",
    "Sending it now.",
    "Playing that song.",
    "Brutal Chess is now closed.",
    "The email is sent.",
    "The song is playing.",
]

NOT_CLAIMS = [
    "The capital of France is Paris.",
    "You can open it by clicking the icon.",
    "Sure! I can help with that.",
    "I don't know what you mean.",
    "Hello there.",
    "What would you like to do?",
    "It is three o'clock.",
    "I will open it when you say so.",
    "Try saying close Brutal Chess.",
    "I could not do that.",
    "Paris",
    "An open question is fine.",
    "Open the door now.",
    "Sounds good to me.",
    "I am ready.",
]


@pytest.mark.parametrize("text", CLAIMS)
def test_claims_detected(text: str) -> None:
    assert claims_action(text) is True


@pytest.mark.parametrize("text", NOT_CLAIMS)
def test_claims_not_detected(text: str) -> None:
    assert claims_action(text) is False


def test_empty_is_not_a_claim() -> None:
    assert claims_action("") is False
    assert claims_action("   ") is False
