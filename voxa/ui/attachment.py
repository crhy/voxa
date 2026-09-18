"""Attachment escape hatch (issue #6 target tree).

A small file picker that files the pick as a task. Not the primary
workflow: Voxa should usually locate files itself through tools.
"""

from __future__ import annotations

from collections.abc import Callable

from gi.repository import Gtk

from voxa.tasks import TaskStore


def attach_files(
    parent: Gtk.Window,
    store: TaskStore,
    notify: Callable[[str], None],
) -> None:
    """Show an Open dialog; each accepted file becomes a task."""
    dialog = Gtk.FileChooserNative(
        title="Attach a file",
        transient_for=parent,
        action=Gtk.FileChooserAction.OPEN,
    )

    def on_response(native: Gtk.FileChooserNative, response: int) -> None:
        if response == Gtk.ResponseType.ACCEPT:
            file = native.get_file()
            if file is not None:
                path = file.get_path() or file.get_uri()
                name = file.get_basename() or path
                store.add_task(f"Attached {name}", detail=path)
                notify(f"Attached {name}.")
        native.destroy()

    dialog.connect("response", on_response)
    dialog.show()
