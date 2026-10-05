"""Debugging aid: open a URL in the headless test browser and print page facts.

Usage: python tools/browser_probe.py <url> [js-expression]
Prints the title, the viewport, the first 40 outline entries, and — when a
JavaScript expression is given — session.evaluate(expression) as JSON.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxa.agent.browser_session import BrowserSession
from voxa.agent.cdp import CdpError

PROFILE = os.path.expanduser("~/.var/app/com.brave.Browser/config/voxa-test-profile")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: browser_probe.py <url> [js-expression]")
        return 1
    url = sys.argv[1]
    expression = sys.argv[2] if len(sys.argv) > 2 else None

    session = BrowserSession(port=9333, headless=True, profile=PROFILE)
    try:
        session.ensure()
        session.goto(url)
        print(f"title: {session.title()}")
        viewport = session.evaluate("(() => ({w: innerWidth, h: innerHeight}))()")
        print(f"viewport: {viewport}")
        for item in session.outline(limit=40):
            print(f"outline: {json.dumps(item, separators=(',', ':'))}")
        if expression:
            value = session.evaluate(expression)
            print(f"evaluate: {json.dumps(value)}")
        return 0
    except CdpError as exc:
        print(f"PROBE FAIL: {exc}")
        return 1
    finally:
        session.close_browser()


if __name__ == "__main__":
    raise SystemExit(main())
