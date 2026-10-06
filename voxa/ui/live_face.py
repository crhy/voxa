"""Live neural face client: talks to a face server over the wire protocol.

Every message is one JSON line ending in ``\\n``; when the line carries a
``bytes`` field, exactly that many raw bytes follow it. See
:mod:`tools.fake_face_server` for the matching server.
"""

from __future__ import annotations

import json
import socket
import threading

CHUNK = 65536


def encode(message, payload=b""):
    line = dict(message)
    if payload:
        line["bytes"] = len(payload)
    raw = json.dumps(line, separators=(",", ":")) + "\n"
    return raw.encode() + payload


class MessageReader:
    """Splits a byte stream into (message, payload) pairs; keeps the remainder."""

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
            if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
                self._buf = self._buf[nl + 1:]
                continue
            if len(self._buf) < nl + 1 + nbytes:
                break
            payload = self._buf[nl + 1:nl + 1 + nbytes]
            self._buf = self._buf[nl + 1 + nbytes:]
            out.append((msg, payload))
        return out


class FrameBuffer:
    """Frames arriving for one utterance, addressable by playback seconds."""

    def __init__(self, utterance_id, fps=25):
        self.utterance_id = utterance_id
        self.fps = fps
        self._frames: dict[int, bytes] = {}
        self._lock = threading.Lock()
        self._done = False
        self._failed = False
        self._error = ""

    def put(self, index, jpeg) -> None:
        with self._lock:
            self._frames[int(index)] = jpeg

    def get(self, index):
        with self._lock:
            return self._frames.get(int(index))

    def finish(self, total=0) -> None:
        with self._lock:
            self._done = True

    def fail(self, message) -> None:
        with self._lock:
            self._failed = True
            self._error = str(message)

    def index_at(self, seconds):
        want = int(float(seconds) * self.fps)
        with self._lock:
            if not self._frames:
                return None
            if want in self._frames:
                return want
            earlier = [index for index in self._frames if index <= want]
            if not earlier:
                return None
            return max(earlier)

    def frame_at(self, seconds):
        index = self.index_at(seconds)
        if index is None:
            return None
        return self._frames.get(index)

    def buffered_seconds(self) -> float:
        with self._lock:
            count = 0
            while count in self._frames:
                count += 1
            return count / float(self.fps)

    def done(self) -> bool:
        with self._lock:
            return self._done

    def failed(self) -> bool:
        with self._lock:
            return self._failed

    def error(self) -> str:
        with self._lock:
            return self._error


def decode_audio_to_pcm16k(path) -> bytes:
    """Decode a media file to mono 16 kHz s16le PCM bytes using GStreamer."""

    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    # parse_launch links decodebin's pads when they appear; linking it by hand fails (its pads are dynamic).
    try:
        pipeline = Gst.parse_launch(
            "filesrc name=src ! decodebin ! audioconvert ! audioresample ! "
            "audio/x-raw,format=S16LE,channels=1,rate=16000 ! appsink name=sink sync=false"
        )
    except Exception as exc:  # noqa: BLE001 - missing plugins and the like
        raise RuntimeError(f"Could not build the audio decode pipeline: {exc}") from exc
    pipeline.get_by_name("src").set_property("location", str(path))
    sink = pipeline.get_by_name("sink")
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("The audio decode pipeline could not start.")

    chunks = []
    try:
        while True:
            sample = sink.emit("try-pull-sample", 5 * Gst.SECOND)
            if sample is None:  # end of stream (or nothing for 5 s: a broken file)
                break
            buffer = sample.get_buffer()
            success, mapped = buffer.map(Gst.MapFlags.READ)
            if not success:
                break
            try:
                chunks.append(bytes(mapped.data))
            finally:
                buffer.unmap(mapped)
    finally:
        pipeline.set_state(Gst.State.NULL)
    if not chunks:
        raise RuntimeError(f"No audio could be decoded from {path}")
    return b"".join(chunks)


class LiveFaceClient:
    """A face server connection that never raises: unreachable means unavailable."""

    def __init__(self, host="127.0.0.1", port=8765, connect_timeout=0.4):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self._sock = None
        self._reader = MessageReader()
        self._ready = False
        self._characters: list[str] = []
        self._buffer = None
        self._buffers: dict[int, FrameBuffer] = {}
        self._next_id = 0

    def available(self) -> bool:
        if self._ready and self._sock is not None:
            return True
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            sock.connect((self.host, self.port))
            # A fresh parser for a fresh connection: bytes left over from an attempt that failed half-way
            # would otherwise make every later reply unreadable.
            self._reader = MessageReader()
            sock.sendall(encode({"op": "hello", "version": 1}))
            messages = []
            while not messages:
                data = sock.recv(65536)
                if not data:
                    break
                messages = self._reader.feed(data)
            for msg, _payload in messages:
                if msg.get("op") == "hello" and msg.get("ready"):
                    self._characters = list(msg.get("characters", []))
                    self._ready = True
                    self._sock = sock
                    # The connect timeout must not apply to frames: the first one can take a second.
                    sock.settimeout(None)
                    threading.Thread(target=self._pump, args=(sock,), name="voxa-live-face", daemon=True).start()
                    sock = None
                    return True
        except OSError:
            pass
        if sock is not None:
            sock.close()
        self._ready = False
        return False

    def characters(self) -> list[str]:
        return list(self._characters)

    def start_utterance(self, character, pcm16k, size=512):
        if self._sock is None:
            return None
        uid = self._next_id
        self._next_id += 1
        buffer = FrameBuffer(uid)
        self._buffer = buffer
        self._buffers[uid] = buffer
        for old in [key for key in self._buffers if key < uid - 4]:
            del self._buffers[old]  # keep only the last few: the current one and those being prepared
        try:
            self._sock.sendall(
                encode(
                    {
                        "op": "utterance",
                        "id": uid,
                        "character": character,
                        "size": size,
                        "fps": 25,
                        "sample_rate": 16000,
                    }
                )
            )
            for offset in range(0, max(1, len(pcm16k)), CHUNK):
                self._sock.sendall(encode({"op": "audio", "id": uid}, pcm16k[offset:offset + CHUNK]))
            self._sock.sendall(encode({"op": "end", "id": uid}))
        except OSError:
            self._sock = None
            self._ready = False
            return None
        return buffer

    def _pump(self, sock) -> None:
        """One reader per connection: hands each message to the utterance it belongs to."""
        try:
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                for msg, payload in self._reader.feed(data):
                    buffer = self._buffers.get(msg.get("id"))
                    if buffer is None:
                        continue  # a frame of an utterance that was cancelled or is long gone
                    op = msg.get("op")
                    if op == "frame":
                        buffer.put(msg.get("index", 0), payload)
                    elif op == "done":
                        buffer.finish(msg.get("frames", 0))
                    elif op == "error":
                        buffer.fail(msg.get("message", "face server error"))
        except OSError:
            pass
        # The connection is gone: say so, so the next check reconnects and the mouth falls back at once.
        if self._sock is sock:
            self._sock = None
            self._ready = False
            self._reader = MessageReader()
        for buffer in list(self._buffers.values()):
            if not (buffer.done() if callable(buffer.done) else buffer.done):
                buffer.fail("face server disconnected")

    def cancel(self) -> None:
        """Stop every utterance that is playing or being prepared."""
        buffers, self._buffers = self._buffers, {}
        if self._sock is None:
            return
        try:
            for uid in buffers:
                self._sock.sendall(encode({"op": "cancel", "id": uid}))
        except OSError:
            self._sock = None
            self._ready = False

    def shutdown_server(self) -> None:
        """Ask the face server to exit (used only for a server Voxa started itself)."""
        if self._sock is not None:
            try:
                self._sock.sendall(encode({"op": "shutdown"}))
            except OSError:
                pass
        self.close()

    def close(self) -> None:
        sock, self._sock = self._sock, None
        self._ready = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
