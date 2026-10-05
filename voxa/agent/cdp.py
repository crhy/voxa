"""A small synchronous Chrome DevTools Protocol client.

One background thread runs an asyncio loop that owns the WebSocket;
``call`` is thread-safe and may be used from worker threads, never on the
GTK thread.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.parse
import urllib.request
from collections import deque

import aiohttp

EVENT_QUEUE_LIMIT = 200


class CdpError(Exception):
    """Raised when a CDP command fails, times out, or the socket dies."""


class CdpConnection:
    def __init__(self, websocket_url: str, timeout: float = 15.0) -> None:
        self._websocket_url = websocket_url
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()
        self._boxes: dict[int, _ReplyBox] = {}
        self._events: dict[str, deque[dict]] = {}
        self._next_id = 1
        self._closed = False
        self._ws = None
        self._ready = threading.Event()
        self._connect_error: str | None = None
        self._thread = threading.Thread(target=self._pump, name="cdp-pump", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise CdpError(f"could not connect to {websocket_url} in time")
        if self._connect_error is not None:
            raise CdpError(self._connect_error)

    def call(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        timeout = self._timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        with self._lock:
            if self._closed:
                raise CdpError("the browser socket is closed")
            msg_id = self._next_id
            self._next_id += 1
            box = _ReplyBox()
            self._boxes[msg_id] = box
        payload = {"id": msg_id, "method": method}
        if params is not None:
            payload["params"] = params
        try:
            asyncio.run_coroutine_threadsafe(
                self._send(json.dumps(payload)),
                self._loop,
            ).result(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            with self._lock:
                self._boxes.pop(msg_id, None)
            raise CdpError("the browser did not accept the command") from None
        if not box.event.wait(max(0.0, deadline - time.monotonic())):
            with self._lock:
                self._boxes.pop(msg_id, None)
            raise CdpError(f"no reply to {method} within {timeout} seconds")
        with self._lock:
            self._boxes.pop(msg_id, None)
        if box.error is not None:
            raise CdpError(box.error)
        return box.result

    def wait_event(self, method: str, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                queue = self._events.get(method)
                if queue:
                    return queue.popleft()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for box in self._boxes.values():
                box.fail("the browser socket was closed")
            self._boxes.clear()
        try:
            asyncio.run_coroutine_threadsafe(self._close_ws(), self._loop).result(timeout=5)
        except Exception:
            pass

    async def _send(self, payload: str) -> None:
        await self._ws.send_str(payload)

    async def _close_ws(self) -> None:
        if self._ws is not None:
            await self._ws.close()

    def _pump(self) -> None:
        try:
            self._loop.run_until_complete(self._pump_async())
        except Exception as exc:
            if not self._ready.is_set():
                self._connect_error = str(exc) or exc.__class__.__name__
                self._ready.set()
        finally:
            with self._lock:
                for box in self._boxes.values():
                    box.fail("the browser socket was closed")
                self._boxes.clear()

    async def _pump_async(self) -> None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self._websocket_url) as ws:
                    self._ws = ws
                    self._ready.set()
                    async for msg in ws:
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            break
                        self._handle(msg.data)
        except Exception as exc:
            if not self._ready.is_set():
                self._connect_error = str(exc) or exc.__class__.__name__
                self._ready.set()
                raise
        finally:
            self._ws = None

    def _handle(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not isinstance(msg, dict):
            return
        msg_id = msg.get("id")
        if isinstance(msg_id, int):
            with self._lock:
                box = self._boxes.get(msg_id)
            if box is not None:
                error = msg.get("error")
                if error is not None:
                    box.fail(str(error.get("message", error)) if isinstance(error, dict) else str(error))
                else:
                    box.done(msg.get("result", {}))
            return
        method = msg.get("method")
        if isinstance(method, str):
            with self._lock:
                self._events.setdefault(method, deque(maxlen=EVENT_QUEUE_LIMIT)).append(msg)


class _ReplyBox:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: dict = {}
        self.error: str | None = None

    def done(self, result: dict) -> None:
        self.result = result
        self.event.set()

    def fail(self, error: str) -> None:
        self.error = error
        self.event.set()


def _http_json(method: str, path: str, port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> dict:
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local CDP endpoint
        return json.loads(response.read().decode("utf-8", errors="replace"))


def list_pages(port: int, host: str = "127.0.0.1") -> list[dict]:
    """GET /json/list: the pages the browser exposes for debugging."""
    return _http_json("GET", "/json/list", host=host, port=port)


def new_page(port: int, url: str = "about:blank") -> dict:
    """PUT /json/new?<url>: open a new debuggable page, return its info."""
    return _http_json("PUT", f"/json/new?{urllib.parse.quote(url, safe='')}", port=port)


def browser_alive(port: int) -> bool:
    """True when something answers the CDP /json/version probe."""
    try:
        _http_json("GET", "/json/version", port=port, timeout=1.0)
    except (urllib.error.HTTPError, OSError, ValueError, TimeoutError):
        return False
    return True


def page_socket(port: int, host: str = "127.0.0.1") -> str | None:
    """The webSocketDebuggerUrl of the first page, or None."""
    try:
        pages = list_pages(port, host)
    except (urllib.error.HTTPError, OSError, ValueError, TimeoutError):
        return None
    for page in pages:
        if page.get("type") == "page":
            return page.get("webSocketDebuggerUrl")
    return None


def browser_socket(port: int, host: str = "127.0.0.1") -> str | None:
    """The browser-level webSocketDebuggerUrl, or None."""
    try:
        version = _http_json("GET", "/json/version", port=port)
    except (urllib.error.HTTPError, OSError, ValueError, TimeoutError):
        return None
    return version.get("webSocketDebuggerUrl")
