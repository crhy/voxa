from __future__ import annotations

from voxa.agent.progress import working_caption
from voxa.agent.spoken_text import site_label


def test_browse_caption_says_website():
    assert working_caption("browse", {"url": "https://example.com/a/b"}) == "Opening website…"


def test_site_label_address_with_path():
    assert site_label("https://example.com/a/b") == "website"


def test_site_label_bare_domain():
    assert site_label("example.com") == "website"


def test_site_label_friendly_name():
    assert site_label("YouTube") == "YouTube"
