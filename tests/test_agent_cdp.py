"""Tests for the CDP client: fake WebSocket server (subprocess), fake HTTP endpoints."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from voxa.agent.cdp import CdpConnection, CdpError, browser_alive, browser_socket, list_pages, new_page, page_socket

# The fake server runs in a SUBPROCESS so its asyncio loop and the CdpConnection's
# pump loop do not share a socket fd (two sock_recv on one fd is undefined).
FAKE_SERVER_SRC = """
import asyncio, json, sys
from aiohttp import web

port = int(sys.argv[1])
marker = sys.argv[2]

async def cdp_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for raw in ws:
        try:
            msg = json.loads(raw.data)
        except (ValueError, TypeError):
            continue
        msg_id = msg.get("id")
        method = msg.get("method", "")
        if msg_id is None:
            continue
        if method == "Page.navigate":
            await ws.send_str(json.dumps({"method": "Page.loadEventFired", "params": {}}))
        if method == "Runtime.evaluate":
            await ws.send_str(json.dumps({"id": msg_id, "error": {"message": "bad expression"}}))
        elif method == "Page.noReply":
            pass
        else:
            await ws.send_str(json.dumps({"id": msg_id, "result": {"echo": method}}))
    return ws

async def json_list(request):
    return web.json_response([
        {"type": "page", "webSocketDebuggerUrl": f"ws://127.0.0.1:{port}/ws"}])

async def json_new(request):
    return web.json_response({"type": "page", "webSocketDebuggerUrl": f"ws://127.0.0.1:{port}/ws"})

async def json_version(request):
    return web.json_response({"webSocketDebuggerUrl": f"ws://127.0.0.1:{port}/browser"})

async def main():
    app = web.Application()
    app.router.add_get("/ws", cdp_ws)
    app.router.add_get("/json/list", json_list)
    app.router.add_put("/json/new", json_new)
    app.router.add_get("/json/version", json_version)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    with open(marker, "w") as fh:
        fh.write("ready")
    while True:
        await asyncio.sleep(0.1)

asyncio.run(main())
"""


def _with_fake_server(port: int, fn):
    """Start the fake CDP server in a subprocess, run fn(port), then stop it."""
    marker = os.path.join(os.path.dirname(__file__), f".fake_server_{port}.marker")
    if os.path.exists(marker):
        os.remove(marker)
    proc = subprocess.Popen(
        [sys.executable, "-c", FAKE_SERVER_SRC, str(port), marker],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not os.path.exists(marker):
            if time.monotonic() >= deadline:
                raise AssertionError("fake server did not start")
            if proc.poll() is not None:
                raise AssertionError("fake server exited before ready")
            time.sleep(0.05)
        return fn(port)
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        if os.path.exists(marker):
            os.remove(marker)


def test_cdp_connection_call():
    port = 19001

    def run(p):
        conn = CdpConnection(f"ws://127.0.0.1:{p}/ws")
        try:
            result = conn.call("Page.enable")
            assert result == {"echo": "Page.enable"}
        finally:
            conn.close()

    _with_fake_server(port, run)


def test_cdp_connection_wait_event():
    port = 19002

    def run(p):
        conn = CdpConnection(f"ws://127.0.0.1:{p}/ws")
        try:
            conn.call("Page.navigate", {"url": "http://example.com"})
            event = conn.wait_event("Page.loadEventFired", timeout=2.0)
            assert event is not None
            assert event["method"] == "Page.loadEventFired"
        finally:
            conn.close()

    _with_fake_server(port, run)


def test_cdp_connection_error():
    port = 19003

    def run(p):
        conn = CdpConnection(f"ws://127.0.0.1:{p}/ws")
        try:
            with pytest.raises(CdpError, match="bad expression"):
                conn.call("Runtime.evaluate", {"expression": "1+1"})
        finally:
            conn.close()

    _with_fake_server(port, run)


def test_cdp_connection_timeout():
    port = 19004

    def run(p):
        conn = CdpConnection(f"ws://127.0.0.1:{p}/ws", timeout=0.5)
        try:
            with pytest.raises(CdpError, match="no reply"):
                conn.call("Page.noReply", timeout=0.1)
        finally:
            conn.close()

    _with_fake_server(port, run)


def test_cdp_connection_close():
    port = 19005

    def run(p):
        conn = CdpConnection(f"ws://127.0.0.1:{p}/ws")
        conn.close()
        assert conn.is_closed()
        with pytest.raises(CdpError, match="closed"):
            conn.call("Page.enable")

    _with_fake_server(port, run)


def test_cdp_connection_two_threads():
    port = 19006

    def run(p):
        conn = CdpConnection(f"ws://127.0.0.1:{p}/ws")
        results = {}
        errors = {}

        def worker(name):
            try:
                results[name] = conn.call("Page.enable")
            except CdpError as exc:
                errors[name] = str(exc)

        try:
            t1 = threading.Thread(target=worker, args=("a",), daemon=True)
            t2 = threading.Thread(target=worker, args=("b",), daemon=True)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)
            assert results.get("a") == {"echo": "Page.enable"}
            assert results.get("b") == {"echo": "Page.enable"}
            assert not errors
        finally:
            conn.close()

    _with_fake_server(port, run)


def test_list_pages():
    port = 19007

    def run(p):
        pages = list_pages(p)
        assert len(pages) == 1
        assert pages[0]["type"] == "page"
        assert pages[0]["webSocketDebuggerUrl"] == f"ws://127.0.0.1:{p}/ws"

    _with_fake_server(port, run)


def test_new_page():
    port = 19008

    def run(p):
        page = new_page(p, url="http://example.com")
        assert page["type"] == "page"
        assert page["webSocketDebuggerUrl"] == f"ws://127.0.0.1:{p}/ws"

    _with_fake_server(port, run)


def test_browser_alive():
    port = 19009

    def run(p):
        assert browser_alive(p)

    _with_fake_server(port, run)


def test_page_socket():
    port = 19010

    def run(p):
        url = page_socket(p)
        assert url == f"ws://127.0.0.1:{p}/ws"

    _with_fake_server(port, run)


def test_browser_socket():
    port = 19011

    def run(p):
        url = browser_socket(p)
        assert url == f"ws://127.0.0.1:{p}/browser"

    _with_fake_server(port, run)
