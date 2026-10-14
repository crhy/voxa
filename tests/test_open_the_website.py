from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

Adw.init()

from voxa.window import WEBSITE_MEMORY_SECONDS, site_of_url, website_fresh, website_request  # noqa: E402

WEBSITE_PHRASES = [
    "open the website",
    "open that website",
    "open the site",
    "open their website",
    "go to the website",
    "show me the website",
]

PAGE_PHRASES = ["open that page"]

NOT_WEBSITE = [
    "open the door",
    "what is the website",
    "open spaced linux website",
    "open the website of the website",
    "the website",
    "open the website for",
]


def test_website_phrases_match_as_site():
    for phrase in WEBSITE_PHRASES:
        assert website_request(phrase) == ("site", None)
        assert website_request(phrase.capitalize()) == ("site", None)


def test_page_phrases_match_as_page():
    for phrase in PAGE_PHRASES:
        assert website_request(phrase) == ("page", None)


def test_for_words_search_terms():
    assert website_request("open the website for spaced linux") == ("site", "spaced linux")
    assert website_request("open that page for spaced linux news") == ("page", "spaced linux news")
    assert website_request("show me the site for  spaced linux ") == ("site", "spaced linux")


def test_not_website():
    for phrase in NOT_WEBSITE:
        assert website_request(phrase) is None


def test_site_of_url():
    assert site_of_url("https://spacedlinux.com/2026/01/news") == "https://spacedlinux.com"
    assert site_of_url("http://example.com") == "http://example.com"
    assert site_of_url("https://a.b.c/x?y=1") == "https://a.b.c"
    assert site_of_url("spacedlinux.com") == "spacedlinux.com"


def test_expiry():
    now = 1000.0
    assert website_fresh(now - 1.0, now)
    assert website_fresh(now - WEBSITE_MEMORY_SECONDS, now)
    assert not website_fresh(now - WEBSITE_MEMORY_SECONDS - 1.0, now)
