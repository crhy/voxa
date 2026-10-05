"""Real end-to-end browser check against a live site (Wikipedia)."""

from __future__ import annotations

import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxa.agent.browser_session import BrowserSession
from voxa.agent.cdp import CdpError
from voxa.agent.tools.web import browse, click_on, go_back, search_site, set_session, wait_for

PROFILE = os.path.expanduser("~/.var/app/com.brave.Browser/config/voxa-test-profile")


def network_ok() -> bool:
    # A plain TCP connect: Wikipedia answers urllib's default User-Agent with 403, which would be
    # mistaken for "no network" and silently skip the whole check.
    try:
        with socket.create_connection(("en.wikipedia.org", 443), timeout=10):
            return True
    except OSError:
        return False


def main() -> int:
    if not network_ok():
        print("REAL SMOKE SKIPPED (no network)")
        return 0

    session = BrowserSession(port=9333, headless=True, profile=PROFILE)
    set_session(session)

    def timed(label: str, fn) -> object:
        started = time.perf_counter()
        result = fn()
        ms = int((time.perf_counter() - started) * 1000)
        print(f"{label}: {ms} ms")
        return result

    try:
        session.ensure()

        result = timed("browse", lambda: browse({"url": "en.wikipedia.org"}))
        if not result.ok:
            print("FAIL at browse")
            return 1

        result = timed("search_site", lambda: search_site({"query": "Devuan"}))
        if not result.ok:
            print("FAIL at search_site")
            return 1

        result = timed("wait_for", lambda: wait_for({"text": "systemd"}))
        if not result.ok:
            print("FAIL at wait_for")
            return 1

        if "Devuan" not in session.title():
            print("FAIL at title (Devuan not in title)")
            return 1

        result = timed("click_on History", lambda: click_on({"text": "History"}))
        if not result.ok:
            result = timed("click_on View history", lambda: click_on({"text": "View history"}))
            if not result.ok:
                print("FAIL at click_on")
                return 1

        if "action=history" not in session.url():
            print("FAIL at url (action=history not in url)")
            return 1

        result = timed("go_back", lambda: go_back({}))
        if not result.ok:
            print("FAIL at go_back")
            return 1

        if "action=history" in session.url():
            print("FAIL at url after back (action=history still in url)")
            return 1

        session.close_browser()
        print("REAL SMOKE OK")
        return 0
    except CdpError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        session.close_browser()


if __name__ == "__main__":
    raise SystemExit(main())
