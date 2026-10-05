"""Real end-to-end browser check: serve a page, drive it over CDP, verify each step."""

from __future__ import annotations

import http.server
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxa.agent.browser_session import BrowserSession
from voxa.agent.cdp import CdpError

PAGE1 = """<!doctype html>
<html>
<head><title>Voxa test page</title></head>
<body>
<h1>Voxa test page</h1>
<input id="name" type="text" placeholder="Your name" />
<button id="greet" onclick="document.getElementById('out').textContent = 'Hello, ' + document.getElementById('name').value + '!'">Greet</button>
<div id="out"></div>
<a href="/next">Next page</a>
</body>
</html>
"""

PAGE2 = """<!doctype html>
<html>
<head><title>Next page</title></head>
<body><h1>Second page</h1></body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        body = PAGE2 if self.path == "/next" else PAGE1
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())


def main() -> int:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    import threading

    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{port}/"

    profile = os.path.expanduser("~/.var/app/com.brave.Browser/config/voxa-test-profile")
    session = BrowserSession(port=9333, headless=True, profile=profile)

    def step(label, ok):
        if not ok:
            print(f"FAIL at {label}")
            return False
        return True

    try:
        session.ensure()
        session.goto(url)

        title = session.title()
        if not step("title", title == "Voxa test page"):
            return 1
        if not step("text", "Voxa test page" in session.text()):
            return 1

        print("OUTLINE:", session.outline())

        if not step("type_into", session.type_into("Your name", "rhY")):
            return 1
        if not step("click Greet", session.click("Greet")):
            return 1
        if not step("wait Hello", session.wait_for_text("Hello, rhY!")):
            return 1

        before = session.url()
        if not step("click Next page", session.click("Next page")):
            return 1
        after = session.url()
        if not step("url changed", after != before):
            return 1

        session.evaluate("history.back()")
        session.close_browser()
        print("SMOKE OK")
        return 0
    except CdpError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        server.shutdown()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
