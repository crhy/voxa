"""Interpreter mode: one session that alternates two languages (issue #66).

Two people take turns at the microphone. Each utterance is transcribed in the
language of whoever spoke, translated by the AI model, and answered in the other
language's voice, so the two voices alternate the way one reply language already
alternates for "how do I say X in Y".

Pure module: no GTK, no audio, no I/O. A session holds only which two languages
are in play and which side spoke last.
"""

from __future__ import annotations

import re

from voxa.replylang import LANGUAGES, language_name

# Whisper's own codes for the languages Voxa has voices for ("Spanish" -> "es").
WHISPER_CODE: dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Arabic": "ar",
    "Hindi": "hi",
    "Russian": "ru",
}


def pair(spoken_a: str, spoken_b: str) -> tuple[str, str] | None:
    """The two languages of a session, from how they were said ("english", "español").

    None when either name is unknown to Voxa, or both name the same language.
    """
    first = language_name(spoken_a)
    second = language_name(spoken_b)
    if first is None or second is None or first == second:
        return None
    return (first, second)


def pair_in(spoken: str) -> tuple[str, str] | None:
    """The first two languages named anywhere inside a tail such as 'between french and japanese'."""
    found: list[str] = []
    for word in re.findall(r"[^\W\d_]+", spoken or "", re.UNICODE):
        name = language_name(word)
        if name is not None:
            found.append(name)
    return (found[0], found[1]) if len(found) >= 2 else None


def other(pair: tuple[str, str] | None, language: str | None) -> str | None:
    """The far side of the pair from the language given; None when it is not one of them."""
    if pair is None or language not in pair:
        return None
    return pair[1] if language == pair[0] else pair[0]


def target_language(pair: tuple[str, str] | None, transcript_language: str | None) -> str | None:
    """The language to answer in, given the language the utterance was transcribed in."""
    return other(pair, transcript_language)


def prompt_for(language: str) -> str:
    """The instruction that keeps the model translating and nothing else.

    The strict translation-only wording of the "say X in Y" tool, so an interpreter
    turn answers with the translation alone; imported here rather than at the top
    because the tool layer is what pulls this module in.
    """
    from voxa.agent.tools.translate import PROMPT

    return PROMPT.format(language=language)


def transcription_code(language: str | None) -> str | None:
    """Whisper's code for a language name (Japanese -> ja); None when Voxa has none."""
    return WHISPER_CODE.get(language or "")


def spoken_languages() -> str:
    """Every language Voxa can interpret between, as one list she can say back."""
    return ", ".join(sorted(LANGUAGES))


class Session:
    """Which two languages are being interpreted, and which side spoke last.

    ``last_side`` is the language the microphone should be transcribing in, so
    the window can hand it straight to Whisper; ``switch`` flips it, and every
    turn flips it. ``stop`` clears everything.
    """

    def __init__(self) -> None:
        self.active = False
        self.a: str | None = None
        self.b: str | None = None
        self.last_side: str | None = None

    def start(self, language_a: str, language_b: str) -> bool:
        """Open a session; False, with nothing changed, when the pair is unusable."""
        resolved = pair(language_a, language_b)
        if resolved is None:
            return False
        self.a, self.b = resolved
        self.last_side = self.a
        self.active = True
        return True

    def sides(self) -> tuple[str, str] | None:
        """The pair as (first, second), or None when no session is open."""
        return (self.a, self.b) if self.active else None

    def switch(self) -> None:
        """Flip which side speaks next."""
        if not self.active:
            return
        self.last_side = self.b if self.last_side == self.a else self.a

    def stop(self) -> None:
        """Close the session: nothing remembered, the microphone goes back to normal."""
        self.active = False
        self.a = self.b = self.last_side = None
