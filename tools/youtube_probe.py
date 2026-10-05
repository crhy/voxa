"""Run youtube.play against a real headless test browser and report every step."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxa.agent.browser_session import BrowserSession
from voxa.agent.cdp import CdpError
from voxa.agent.youtube import play


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('usage: youtube_probe.py "<query or url>"')
        return 2
    query = argv[1]
    profile = os.path.expanduser("~/.var/app/com.brave.Browser/config/voxa-test-profile")
    extra = ("--autoplay-policy=no-user-gesture-required", "--mute-audio")
    session = BrowserSession(port=9333, headless=True, profile=profile, extra_args=extra)

    def log(step: str, state: dict) -> None:
        print(f"STEP {step}: {state}")

    try:
        session.ensure()
        ok, how, url = play(query, session, log=log)
        print(f"RESULT: ok={ok} how={how} url={url}")
        session.close_browser()
        return 0 if ok else 1
    except CdpError as exc:
        print(f"FAIL: {exc} (YouTube may be unreachable from here)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
