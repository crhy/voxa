from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date

# The spoken word that authorises a spend, and how long after the read-back it
# may arrive (issue #43: "require the spoken word 'confirm' within 20 seconds").
CONFIRM_WORD = "confirm"
CONFIRM_WINDOW_SECONDS = 20.0

# Defaults for the Preferences caps when the user has not set them, used by the
# tool layer as a safety net. A cap of 0 or less means no cap.
DEFAULT_PER_ORDER_CAP = 50.0
DEFAULT_PER_DAY_CAP = 200.0


class RiskLevel(enum.IntEnum):
    READ_ONLY = 0
    REVERSIBLE = 1
    EXTERNAL_WRITE = 2
    SENSITIVE = 3


def needs_confirmation(
    level: RiskLevel,
    granted: frozenset[str] = frozenset(),
    tool_name: str = "",
) -> bool:
    """Level 0-1: never. Level 2: unless tool_name is in `granted`. Level 3: always."""
    if level >= RiskLevel.SENSITIVE:
        return True
    if level == RiskLevel.EXTERNAL_WRITE:
        return tool_name not in granted
    return False


@dataclass(frozen=True, slots=True)
class IntendedSpend:
    """An action about to spend money: what is bought, for how much, from
    where, and to whom it is paid or sent."""

    what: str
    price: float
    source: str = ""
    payee: str = ""
    address: str = ""
    reference: str = ""

    def readback(self) -> str:
        """Read the order back: what, price, from where, to which payee/address."""
        line = f"{(self.what or 'this').strip()} costs {self.price:.2f}"
        if self.source:
            line += f", from {self.source.strip()}"
        destination = self.payee or self.address
        if destination:
            line += f", to {destination.strip()}"
        if self.reference:
            line += f" (ref {self.reference.strip()})"
        return line


@dataclass(frozen=True, slots=True)
class SpendDecision:
    allowed: bool
    reason: str = ""
    readback: str = ""


def spend_guard(
    spend: IntendedSpend,
    *,
    confirm_word: str = "",
    confirm_seconds: float | None = None,
    per_order_cap: float = DEFAULT_PER_ORDER_CAP,
    per_day_cap: float = DEFAULT_PER_DAY_CAP,
    spent_today: float = 0.0,
) -> SpendDecision:
    """The one reusable guard for any action that spends money (issue #43).

    It (a) renders a read-back of what/price/source/payee-or-address, (b)
    enforces a per-order cap and a rolling per-day total cap, and (c) refuses
    unless the spoken word ``confirm`` arrived within 20 seconds of the
    read-back. The caps are read from Preferences by the caller; 0 or less is
    treated as no cap.
    """
    readback = spend.readback()

    # Hard limits first: never even prompt to confirm something we would refuse.
    if per_order_cap > 0 and spend.price > per_order_cap:
        return SpendDecision(False, f"That is over the limit for one order, {per_order_cap:.2f}.", readback)
    if per_day_cap > 0 and spent_today + spend.price > per_day_cap:
        return SpendDecision(False, f"That is over today's spending limit, {per_day_cap:.2f}.", readback)

    confirmed = (
        confirm_word.strip().casefold() == CONFIRM_WORD
        and confirm_seconds is not None
        and 0.0 <= confirm_seconds <= CONFIRM_WINDOW_SECONDS
    )
    if not confirmed:
        if confirm_seconds is not None and confirm_seconds > CONFIRM_WINDOW_SECONDS:
            reason = "That confirmation came too late; say the word again."
        else:
            reason = f'Say "{CONFIRM_WORD}" within {int(CONFIRM_WINDOW_SECONDS)} seconds: {readback}.'
        return SpendDecision(False, reason, readback)

    return SpendDecision(True, "", readback)


def day_spend_total(action_log, day: date | None = None) -> float:
    """Total already spent on ``day`` (today by default), summed from the action
    log's ``spend`` records, for the rolling per-day cap."""
    target = (day or date.today()).isoformat()
    total = 0.0
    for record in action_log.read():
        if record.get("route") != "spend" or not record.get("ok"):
            continue
        if str(record.get("time", ""))[:10] != target:
            continue
        try:
            total += float(record.get("args", {}).get("amount", 0) or 0)
        except (TypeError, ValueError):
            continue
    return round(total, 2)

