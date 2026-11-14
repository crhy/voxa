"""The personal profile: the plain facts Voxa keeps about you.

What to call you, where you live, how to ring you. These are saved in the
settings file with the other preferences, so they are readable by anyone who
opens it; anything secret belongs in the keyring instead (see
``voxa/agent/secrets.py``).

Pure Python: no GTK, no host access, plain strings in and out.
"""

from __future__ import annotations

__all__ = [
    "NAME_FIELD",
    "ADDRESS_FIELD",
    "PHONE_FIELD",
    "PROFILE_FIELDS",
    "clean_value",
    "field_label",
    "profile_forget",
    "profile_get",
    "profile_set",
    "spoken_profile",
]

NAME_FIELD = "name"
ADDRESS_FIELD = "home_address"
PHONE_FIELD = "phone_number"
PROFILE_FIELDS = (NAME_FIELD, ADDRESS_FIELD, PHONE_FIELD)

_LABELS = {NAME_FIELD: "name", ADDRESS_FIELD: "home address", PHONE_FIELD: "phone number"}
# A profile line is a short fact, not a document; anything longer was caught by
# mistake (dictation left running, a whole page read out).
_MAX_VALUE = 200


def field_label(field: str) -> str:
    """How the field is said out loud, or "" when Voxa has no such field."""
    return _LABELS.get(field, "")


def clean_value(value: str) -> str:
    """A profile value as plain text: trimmed, single spaces, capped."""
    return " ".join((value or "").split())[:_MAX_VALUE]


def profile_get(settings: object, field: str) -> str:
    """The saved value of one profile field, or ""."""
    if field not in _LABELS:
        return ""
    return clean_value(str(getattr(settings, field, "") or ""))


def profile_set(settings: object, field: str, value: str) -> bool:
    """Save one profile field; False when the field or the value is unusable."""
    if field not in _LABELS:
        return False
    cleaned = clean_value(value)
    if not cleaned:
        return False
    setattr(settings, field, cleaned)
    return True


def profile_forget(settings: object, field: str) -> bool:
    """Clear one profile field; False when there was nothing to clear."""
    if field not in _LABELS:
        return False
    if not profile_get(settings, field):
        return False
    setattr(settings, field, "")
    return True


def spoken_profile(field: str, value: str) -> str:
    """The read-back wording: "Your home address is 12 oak street."."""
    label = field_label(field)
    return f"Your {label} is {value}." if label else ""
