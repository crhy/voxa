from __future__ import annotations

import re
from collections.abc import Callable

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.replylang import ALIASES, LANGUAGES

ask_model: Callable | None = None                 # set by the window: messages -> text
speak_in: Callable[[str], None] | None = None     # set by the window: speak the NEXT reply with this language's voice

PROMPT = (
    "Translate the user's text into {language}. Reply with ONLY the translation in {language}: "
    "no quotes, no explanation, no transliteration, no extra words."
)


def language_name(spoken: str) -> str | None:
    """The language Voxa has a voice for, from how it was said ("japanese", "Spanish."); None when she has none."""
    wanted = re.sub(r"[^a-z ]", "", (spoken or "").lower()).strip()
    for name, aliases in ALIASES.items():
        if wanted == name.lower() or wanted in {a.lower() for a in aliases}:
            return name
    return None


def tidy(answer: str) -> str:
    """The model's reply with wrapping quotes and a leading "Translation:" removed; first line only."""
    text = (answer or "").strip().splitlines()[0].strip() if (answer or "").strip() else ""
    text = re.sub(r"^(?:translation|here(?:'s| is)[^:]*)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip("\"'“”«»").strip()


def say_in_language(args: dict[str, str]) -> ToolResult:
    phrase = (args.get("phrase") or "").strip().strip("\"'“”")
    language = language_name(args.get("language", ""))
    if language is None:
        return ToolResult.failure("I can speak " + ", ".join(sorted(LANGUAGES)) + ". Which one would you like?")
    if not phrase:
        return ToolResult.failure("What should I say?")
    if ask_model is None:
        return ToolResult.failure("I cannot translate without an AI model.")
    try:
        answer = ask_model([
            {"role": "system", "content": PROMPT.format(language=language)},
            {"role": "user", "content": phrase},
        ])
    except Exception as exc:  # noqa: BLE001 - the model server is outside our control
        return ToolResult.failure("I could not reach the AI model to translate that.", detail=str(exc))
    translation = tidy(answer)
    if not translation:
        return ToolResult.failure("I could not translate that.")
    if speak_in is not None:
        speak_in(language)
    return ToolResult.success(translation, detail=f"{phrase} → {language}")


def translate_tools() -> list[Tool]:
    return [
        Tool(name="say_in_language", description="Translate a phrase and say it in another language.",
             parameters={"phrase": "what to say", "language": "the target language"},
             risk=RiskLevel.READ_ONLY, handler=say_in_language, required=("phrase", "language")),
    ]
