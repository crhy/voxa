"""The first-boot welcome dialog: greet once, offer the one missing piece."""
from __future__ import annotations

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk  # noqa: E402

from .. import welcome  # noqa: E402


class WelcomeDialog(Adw.Dialog):
    def __init__(self, name: str, step: str, on_primary, on_tutorial=None):
        super().__init__(title="Welcome to Voxa", content_width=460)
        self.on_primary = on_primary
        self.on_tutorial = on_tutorial

        headline, explanation, primary_label = welcome.step_text(step)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        for margin in (box.set_margin_top, box.set_margin_bottom, box.set_margin_start, box.set_margin_end):
            margin(24)
        self.set_child(box)

        greet_label = Gtk.Label(label=welcome.greeting(name), wrap=True, xalign=0)
        greet_label.add_css_class("title-3")
        box.append(greet_label)

        self.headline = Gtk.Label(label=headline)
        self.headline.add_css_class("heading")
        box.append(self.headline)

        self.explanation = Gtk.Label(label=explanation, wrap=True, xalign=0)
        box.append(self.explanation)

        tips = Gtk.Label(label=welcome.HOW_TO, wrap=True, xalign=0)
        tips.add_css_class("dim-label")
        if step == "ready":
            tips.set_visible(False)
        box.append(tips)

        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, halign=Gtk.Align.END)
        box.append(buttons)

        self.tutorial_button = None
        if welcome.TUTORIAL_URL and on_tutorial is not None:
            self.tutorial_button = Gtk.Button(label="Watch the tutorial")
            self.tutorial_button.connect("clicked", lambda *_: on_tutorial())
            buttons.append(self.tutorial_button)

        self.later_button = Gtk.Button(label="Later")
        self.later_button.connect("clicked", lambda *_: self.close())
        buttons.append(self.later_button)

        self.primary_button = Gtk.Button(label=primary_label)
        self.primary_button.add_css_class("suggested-action")
        self.primary_button.connect(
            "clicked",
            lambda *_: (self.close(), on_primary(step)),
        )
        buttons.append(self.primary_button)
