"""Language requests: "can I get that in German" -> German, with a voice that speaks it.

Pure module: no GTK, no state. It answers two questions about a sentence —
which language (if any) the sentence asks the answer to be in, and whether
nothing but that request remains.
"""

from __future__ import annotations

import re

LANGUAGES: dict[str, tuple[str, str, str]] = {
    "English": ("en-US", "en-US-AriaNeural", "en-US-GuyNeural"),
    "Spanish": ("es-MX", "es-MX-DaliaNeural", "es-MX-JorgeNeural"),
    "French": ("fr-FR", "fr-FR-DeniseNeural", "fr-FR-HenriNeural"),
    "German": ("de-DE", "de-DE-KatjaNeural", "de-DE-ConradNeural"),
    "Italian": ("it-IT", "it-IT-ElsaNeural", "it-IT-DiegoNeural"),
    "Portuguese": ("pt-BR", "pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"),
    "Japanese": ("ja-JP", "ja-JP-NanamiNeural", "ja-JP-KeitaNeural"),
    "Korean": ("ko-KR", "ko-KR-SunHiNeural", "ko-KR-InJoonNeural"),
    "Chinese": ("zh-CN", "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"),
    "Arabic": ("ar-SA", "ar-SA-ZariyahNeural", "ar-SA-HamedNeural"),
    "Hindi": ("hi-IN", "hi-IN-SwaraNeural", "hi-IN-MadhurNeural"),
    "Russian": ("ru-RU", "ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"),
}

# Words a sentence may use to name the language, lowercase, accents included.
ALIASES: dict[str, tuple[str, ...]] = {
    "English": ("english",),
    "Spanish": ("spanish", "español"),
    "French": ("french", "français"),
    "German": ("german", "deutsch"),
    "Italian": ("italian", "italiano"),
    "Portuguese": ("portuguese", "português"),
    "Japanese": ("japanese",),
    "Korean": ("korean",),
    "Chinese": ("chinese",),
    "Arabic": ("arabic",),
    "Hindi": ("hindi",),
    "Russian": ("russian",),
}

# Prepositions and verbs that turn a bare language name into a request for it.
# A bare name alone ("tell me about German shepherds") is never a request.
_TRIGGERS = (
    r"in\s+{alias}",
    r"into\s+{alias}",
    r"(?:speak|speaks|speaking)\s+{alias}",
    r"back\s+to\s+{alias}",
    r"auf\s+{alias}",
    r"en\s+{alias}",
    r"translate\s+to\s+{alias}",
)

# Words that carry no content: a sentence made only of these plus the language
# name is nothing but a language request ("can I get that in German").
_FILLERS = frozenset(
    """
    can i get that it this please the a answer say says speak speaking in into
    back to auf en could you would will for me now just only translate respond
    reply switch change turn mode language talk use keep stay always again do
    """.split()
)


def requested_language(text: str) -> str | None:
    """The language NAME the sentence asks for, or None."""
    lowered = text.lower()
    for name, aliases in ALIASES.items():
        for alias in aliases:
            for pattern in _TRIGGERS:
                if re.search(pattern.format(alias=alias), lowered):
                    return name
    return None


def is_only_a_language_request(text: str) -> bool:
    """True when nothing but the language request remains in the sentence."""
    name = requested_language(text)
    if name is None:
        return False
    aliases = set(ALIASES[name])
    for token in re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE):
        if token not in aliases and token not in _FILLERS:
            return False
    return True
