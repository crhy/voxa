"""The tokens that make stale worker completions no-op (issue #138, part 2).

``MainWindow`` starts many background ``threading.Thread`` workers. Every
completion that touches the UI, speech or settings must first ask whether it is
still current: a worker started before OFFLINE (or before a newer request of the
same kind) must do nothing at all when it finally reports back. This is the
mechanical version of the rule the assistant controller already applies to
microphone/model/speech callbacks (see ``voxa/controller.py``).

A worker captures the ``value`` when it starts; its idle() completion calls
:meth:`is_current` before changing anything. Starting the work again, or
pressing OFFLINE, calls :meth:`bump`.

This module is GTK-free (see tests/test_imports.py): the guard itself can be
tested and reused without a display.
"""

from __future__ import annotations


class Generation:
    """A monotonically increasing integer token.

    ``None`` is the "always current" placeholder for completions invoked
    directly on the UI thread, where nothing can have overtaken them.
    """

    __slots__ = ("value",)

    def __init__(self, start: int = 0) -> None:
        self.value = start

    def bump(self) -> int:
        """Retire everything in flight and return the new current token."""
        self.value += 1
        return self.value

    def is_current(self, token: int | None) -> bool:
        """True while ``token`` (captured at worker start) is still current."""
        return token is None or token == self.value
