"""Tests for the Spaced desktop-palette loader (task th05-voxa-palette).

Run with: python3 -m unittest discover -s tests -p "test_spaced_palette*.py"
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gio", "2.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():  # pragma: no cover - headless guard
    raise unittest.SkipTest("no GTK available")
Adw.init()

from voxa.ui import spaced_palette  # noqa: E402
from voxa.ui.spaced_palette import SpacedPalette, read_dark, read_css, spaced_config_dir  # noqa: E402


def _make_manager() -> Adw.StyleManager:
    # A standalone manager keeps the test from mutating the process-wide default.
    return Adw.StyleManager()


class HelperTests(unittest.TestCase):
    def test_config_dir_honours_xdg(self) -> None:
        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = "/custom/config"
        try:
            self.assertEqual(spaced_config_dir(), Path("/custom/config/spaced"))
        finally:
            if old is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = old

    def test_read_dark_variants(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "theme.json"
            p.write_text(json.dumps({"dark": True}), encoding="utf-8")
            self.assertIs(read_dark(p), True)
            p.write_text(json.dumps({"dark": False}), encoding="utf-8")
            self.assertIs(read_dark(p), False)
            p.write_text(json.dumps({"dark": "true"}), encoding="utf-8")
            self.assertIs(read_dark(p), True)
            p.write_text("{not json", encoding="utf-8")
            self.assertIsNone(read_dark(p))
            self.assertIsNone(read_dark(Path(d) / "missing.json"))

    def test_read_css_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(read_css(Path(d) / "libadwaita.css"))
            p = Path(d) / "libadwaita.css"
            p.write_text("@define-color window_bg_color #101820;", encoding="utf-8")
            self.assertIn("@define-color", read_css(p))


class LoadTests(unittest.TestCase):
    def _spaced_dir(self, d: str, css: str | None, dark: bool | None) -> Path:
        base = Path(d) / "spaced"
        base.mkdir()
        if css is not None:
            (base / "libadwaita.css").write_text(css, encoding="utf-8")
        if dark is not None:
            (base / "theme.json").write_text(json.dumps({"dark": dark}), encoding="utf-8")
        return base

    def test_missing_files_leave_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "spaced"
            base.mkdir()  # dir exists but no files
            sm = _make_manager()
            before = sm.get_color_scheme()
            pal = SpacedPalette(config_dir=base, style_manager=sm)
            self.assertFalse(pal.install())
            self.assertIsNone(pal.provider)
            self.assertIsNone(pal.theme_dark)
            self.assertEqual(sm.get_color_scheme(), before)

    def test_load_sets_provider_and_dark_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = self._spaced_dir(
                d, "@define-color window_bg_color #101820;", dark=True
            )
            sm = _make_manager()
            pal = SpacedPalette(config_dir=base, style_manager=sm)
            self.assertTrue(pal.install())
            self.assertIsInstance(pal.provider, Gtk.CssProvider)
            self.assertIs(pal.theme_dark, True)
            self.assertEqual(sm.get_color_scheme(), Adw.ColorScheme.FORCE_DARK)
            self.assertIsNotNone(pal.monitor)

    def test_load_sets_light_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = self._spaced_dir(d, "@define-color window_bg_color #f3f3f3;", dark=False)
            sm = _make_manager()
            pal = SpacedPalette(config_dir=base, style_manager=sm)
            self.assertTrue(pal.install())
            self.assertIs(pal.theme_dark, False)
            self.assertEqual(sm.get_color_scheme(), Adw.ColorScheme.FORCE_LIGHT)

    def test_css_only_keeps_default_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = self._spaced_dir(d, "@define-color window_bg_color #101820;", dark=None)
            sm = _make_manager()
            before = sm.get_color_scheme()
            pal = SpacedPalette(config_dir=base, style_manager=sm)
            self.assertTrue(pal.install())
            self.assertIsInstance(pal.provider, Gtk.CssProvider)
            self.assertEqual(sm.get_color_scheme(), before)


class LiveReloadTests(unittest.TestCase):
    def test_file_change_triggers_one_reload(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "spaced"
            base.mkdir()
            (base / "libadwaita.css").write_text("@define-color a #000;", encoding="utf-8")
            (base / "theme.json").write_text(json.dumps({"dark": True}), encoding="utf-8")
            sm = _make_manager()
            pal = SpacedPalette(config_dir=base, style_manager=sm)
            self.assertTrue(pal.install())
            self.assertEqual(pal.reloads, 0)

            # A burst of change notifications (css + json rewritten in
            # succession) must collapse into a single debounced reload.
            pal._on_change()
            pal._on_change()
            pal._on_change()
            pal._flush()
            self.assertEqual(pal.reloads, 1)

    def test_reload_picks_up_new_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "spaced"
            base.mkdir()
            (base / "libadwaita.css").write_text("@define-color a #000;", encoding="utf-8")
            (base / "theme.json").write_text(json.dumps({"dark": True}), encoding="utf-8")
            sm = _make_manager()
            pal = SpacedPalette(config_dir=base, style_manager=sm)
            pal.install()
            self.assertEqual(sm.get_color_scheme(), Adw.ColorScheme.FORCE_DARK)
            (base / "theme.json").write_text(json.dumps({"dark": False}), encoding="utf-8")
            pal.reload()
            self.assertIs(pal.theme_dark, False)
            self.assertEqual(sm.get_color_scheme(), Adw.ColorScheme.FORCE_LIGHT)


if __name__ == "__main__":
    unittest.main()
