"""Tests for voxa/ui/live_face.py: encode, MessageReader, FrameBuffer, LiveFaceClient."""

from __future__ import annotations

import json
import socket
import threading
import time

from tools.fake_face_server import serve
from voxa.ui.live_face import FrameBuffer, LiveFaceClient, MessageReader, encode


def test_encode_no_payload():
    raw = encode({"op": "hello", "version": 1})
    assert raw.endswith(b"\n")
    assert b"bytes" not in raw


def test_encode_with_payload():
    payload = b"\x00\x01\x02"
    raw = encode({"op": "audio", "id": 7}, payload)
    line, sep, rest = raw.partition(b"\n")
    msg = json.loads(line)
    assert msg["bytes"] == len(payload)
    assert rest == payload


def test_message_reader_splits():
    reader = MessageReader()
    data = encode({"op": "hello", "version": 1}) + encode({"op": "frame", "id": 1, "index": 0}, b"\xff\xd8")
    pairs = reader.feed(data)
    assert len(pairs) == 2
    assert pairs[0][0]["op"] == "hello"
    assert pairs[0][1] == b""
    assert pairs[1][0]["op"] == "frame"
    assert pairs[1][1] == b"\xff\xd8"


def test_message_reader_partial():
    reader = MessageReader()
    full = encode({"op": "frame", "id": 1, "index": 0}, b"\xff\xd8\xff")
    reader.feed(full[:10])
    assert reader.feed(full[10:]) == [({"op": "frame", "id": 1, "index": 0, "bytes": 3}, b"\xff\xd8\xff")]


def test_frame_buffer_put_get():
    buf = FrameBuffer(42, fps=25)
    buf.put(0, b"a")
    buf.put(1, b"b")
    assert buf.get(0) == b"a"
    assert buf.get(1) == b"b"
    assert buf.get(2) is None


def test_frame_buffer_index_at():
    buf = FrameBuffer(1, fps=25)
    buf.put(0, b"x")
    buf.put(5, b"y")
    assert buf.index_at(0.0) == 0
    assert buf.index_at(0.1) == 0
    assert buf.index_at(0.2) == 5
    assert buf.index_at(0.04) == 0
    assert buf.index_at(0.19) == 0
    assert buf.index_at(0.24) == 5


def test_frame_buffer_buffered_seconds():
    buf = FrameBuffer(1, fps=25)
    for i in range(10):
        buf.put(i, b"\x00")
    assert buf.buffered_seconds() == 10 / 25.0


def test_frame_buffer_done_fail():
    buf = FrameBuffer(1)
    buf.finish(394)
    assert buf.done()
    assert not buf.failed()
    buf2 = FrameBuffer(2)
    buf2.fail("bad")
    assert buf2.failed()
    assert buf2.error() == "bad"


def test_live_client_available():
    client = LiveFaceClient(host="127.0.0.1", port=19999)
    assert not client.available()


def _free_port():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()
    return port


def test_live_client_with_fake_server():
    stop = threading.Event()
    port = _free_port()
    t = threading.Thread(
        target=lambda: serve(port=port, delay_ms=0, stop_event=stop),
        daemon=True,
    )
    t.start()
    client = LiveFaceClient(host="127.0.0.1", port=port)
    try:
        for _ in range(100):  # the server thread may not be listening yet
            if client.available():
                break
            time.sleep(0.02)
        assert client.available()
        assert "aoife" in client.characters()
    finally:
        client.close()
        stop.set()
        t.join(timeout=2)


def test_live_client_start_utterance():
    stop = threading.Event()
    port = _free_port()
    t = threading.Thread(
        target=lambda: serve(port=port, delay_ms=0, stop_event=stop),
        daemon=True,
    )
    t.start()
    client = LiveFaceClient(host="127.0.0.1", port=port)
    try:
        for _ in range(100):  # the server thread may not be listening yet
            if client.available():
                break
            time.sleep(0.02)
        assert client.available()
        silence = b"\x00" * (2 * 16000 * 2)
        buf = client.start_utterance("aoife", silence, size=512)
        assert buf is not None
        deadline = time.monotonic() + 5.0
        while not buf.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert buf.done()
        assert buf.buffered_seconds() == 2.0
        frame = buf.frame_at(0.5)
        assert frame is not None
        assert frame[:2] == b"\xff\xd8"
    finally:
        client.close()
        stop.set()
        t.join(timeout=2)


def test_live_client_cancel():
    client = LiveFaceClient(host="127.0.0.1", port=19998)
    assert not client.available()
    client.cancel()


def test_live_client_close():
    client = LiveFaceClient(host="127.0.0.1", port=19997)
    client.close()
