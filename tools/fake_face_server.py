"""Fake neural face server for tests and demos.

Serves the live-face protocol over TCP. Reads real face-pack JPEGs when
available, otherwise generates synthetic JPEGs with PIL. Testable without
a GPU.

Usage: python3 tools/fake_face_server.py [--port 8765] [--delay-ms 15]
"""

import io
import json
import math
import socket
import time
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None

PACK_ROOT = Path.home() / ".local" / "share" / "voxa" / "faces"


def encode(message, payload=b""):
    line = dict(message)
    if payload:
        line["bytes"] = len(payload)
    raw = json.dumps(line, separators=(",", ":")) + "\n"
    return raw.encode() + payload


class MessageReader:
    def __init__(self):
        self._buf = b""

    def feed(self, data):
        self._buf += data
        out = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            try:
                msg = json.loads(self._buf[:nl].decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._buf = self._buf[nl + 1:]
                continue
            if not isinstance(msg, dict):
                self._buf = self._buf[nl + 1:]
                continue
            nbytes = msg.get("bytes", 0)
            if not isinstance(nbytes, int) or nbytes < 0:
                self._buf = self._buf[nl + 1:]
                continue
            if len(self._buf) < nl + 1 + nbytes:
                break
            payload = self._buf[nl + 1:nl + 1 + nbytes]
            self._buf = self._buf[nl + 1 + nbytes:]
            out.append((msg, payload))
        return out


def _pack_frames(character):
    directory = PACK_ROOT / character
    if not directory.is_dir():
        return []
    return sorted(
        (p.as_posix(), p.stat().st_size)
        for p in directory.glob("m*.jpg")
    )


def _synthetic_jpeg(index, size):
    if Image is None:
        return b""
    img = Image.new("RGB", (size, size), (38, 42, 52))
    draw = ImageDraw.Draw(img)
    openness = 10 + (index % 40) * 4
    cx, cy = size // 2, int(size * 0.62)
    draw.ellipse(
        (cx - size // 8, cy - openness // 2, cx + size // 8, cy + openness // 2),
        fill=(140, 70, 70),
    )
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _characters():
    if not PACK_ROOT.is_dir():
        return []
    return sorted(d.name for d in PACK_ROOT.iterdir() if d.is_dir())


def _handle_connection(conn, delay_ms):
    reader = MessageReader()
    utterances = {}
    try:
        while True:
            data = conn.recv(65536)
            if not data:
                break
            for msg, payload in reader.feed(data):
                op = msg.get("op")
                if op == "hello":
                    conn.sendall(encode({
                        "op": "hello",
                        "version": 1,
                        "ready": True,
                        "characters": _characters(),
                    }))
                elif op == "utterance":
                    utterances[msg.get("id")] = {
                        "character": msg.get("character", ""),
                        "size": int(msg.get("size", 512)),
                        "fps": int(msg.get("fps", 25)),
                        "sample_rate": int(msg.get("sample_rate", 16000)),
                        "audio": b"",
                        "cancelled": False,
                    }
                elif op == "audio":
                    entry = utterances.get(msg.get("id"))
                    if entry is not None:
                        entry["audio"] += payload
                elif op == "cancel":
                    entry = utterances.get(msg.get("id"))
                    if entry is not None:
                        entry["cancelled"] = True
                elif op == "end":
                    _send_frames(conn, msg.get("id"), utterances, delay_ms)
    finally:
        conn.close()


def _send_frames(conn, uid, utterances, delay_ms):
    entry = utterances.pop(uid, None)
    if entry is None or entry["cancelled"]:
        return
    fps = entry["fps"]
    sample_rate = entry["sample_rate"]
    audio_seconds = len(entry["audio"]) / float(sample_rate * 2)
    frame_count = max(1, math.ceil(audio_seconds * fps))
    pack = _pack_frames(entry["character"])
    size = entry["size"]
    for index in range(frame_count):
        if entry["cancelled"]:
            return
        if pack:
            path, _ = pack[index % len(pack)]
            jpeg = Path(path).read_bytes()
        else:
            jpeg = _synthetic_jpeg(index, size)
        conn.sendall(encode(
            {"op": "frame", "id": uid, "index": index},
            jpeg,
        ))
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
    conn.sendall(encode({"op": "done", "id": uid, "frames": frame_count}))


def serve(port=8765, delay_ms=15, stop_event=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(8)
    try:
        while stop_event is None or not stop_event.is_set():
            conn, _ = sock.accept()
            _handle_connection(conn, delay_ms)
    finally:
        sock.close()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--delay-ms", type=int, default=15)
    args = parser.parse_args(argv)
    print(f"fake face server listening on 127.0.0.1:{args.port}", flush=True)
    serve(port=args.port, delay_ms=args.delay_ms)


if __name__ == "__main__":
    main()
