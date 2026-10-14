"""Live neural face client: talks to a face server over the wire protocol.

Every message is one JSON line ending in ``\\n``; when the line carries a
``bytes`` field, exactly that many raw bytes follow it. See
:mod:`tools.fake_face_server` for the matching server.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from bisect import bisect_left, bisect_right, insort

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

    def __init__(self, utterance_id, fps=25, duration_seconds=0.0):
        self.utterance_id = utterance_id
        self.fps = fps
        self.duration_seconds = max(0.0, float(duration_seconds))
        self._frames: dict[int, bytes] = {}
        self._indices: list[int] = []
        self._contiguous = 0
        self._lock = threading.Lock()
        self._done = False
        self._failed = False
        self._error = ""
        self._started_at: float | None = None

    def mark_started(self) -> None:
        with self._lock:
            self._started_at = time.monotonic()

    def started_at(self) -> float | None:
        with self._lock:
            return self._started_at

    def put(self, index, jpeg) -> None:
        index = int(index)
        with self._lock:
            if index not in self._frames:
                insort(self._indices, index)
            self._frames[index] = jpeg
            if index == self._contiguous:
                while self._contiguous in self._frames:
                    self._contiguous += 1

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
            position = bisect_right(self._indices, want)
            if position == 0:
                return None
            return self._indices[position - 1]

    def frame_at(self, seconds):
        _index, frame = self.frame_with_index_at(seconds, retain_behind=None)
        return frame

    def frame_with_index_at(self, seconds, retain_behind=2):
        """Return the closest available frame and optionally discard played frames.

        Live frames arrive in order, but lookup still needs to tolerate a missing
        frame.  Keeping a sorted index makes this logarithmic instead of scanning
        every frame on every GTK tick.  Once playback advances, only a tiny
        rewind window is retained; future frames remain intact.
        """
        want = int(float(seconds) * self.fps)
        with self._lock:
            position = bisect_right(self._indices, want)
            if position == 0:
                return None, None
            index = self._indices[position - 1]
            frame = self._frames.get(index)
            if retain_behind is not None:
                cutoff = index - max(0, int(retain_behind))
                discard = bisect_left(self._indices, cutoff)
                for old in self._indices[:discard]:
                    self._frames.pop(old, None)
                if discard:
                    del self._indices[:discard]
            return index, frame

    def buffered_seconds(self) -> float:
        with self._lock:
            return self._contiguous / float(self.fps)

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
        self._protocol_version = 0
        self._characters: list[str] = []
        self._buffer = None
        self._buffers: dict[int, FrameBuffer] = {}
        self._buffers_lock = threading.Lock()
        self._send_lock = threading.Lock()
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
                    self._protocol_version = int(msg.get("version", 0) or 0)
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
        self._protocol_version = 0
        return False

    def protocol_version(self) -> int:
        return self._protocol_version

    def characters(self) -> list[str]:
        return list(self._characters)

    def start_utterance(self, character, pcm16k, size=512, fps=25):
        if self._sock is None:
            return None
        with self._buffers_lock:
            uid = self._next_id
            self._next_id += 1
            buffer = FrameBuffer(uid, fps=fps, duration_seconds=len(pcm16k) / (16000.0 * 2.0))
            self._buffer = buffer
            self._buffers[uid] = buffer
        try:
            # Prepared speech can finish on more than one worker. Keep each
            # utterance's header/audio/end sequence together on the wire.
            with self._send_lock:
                self._sock.sendall(
                    encode(
                        {
                            "op": "utterance",
                            "id": uid,
                            "character": character,
                            "size": size,
                            "fps": fps,
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
            with self._buffers_lock:
                self._buffers.pop(uid, None)
            buffer.fail("face server disconnected")
            return None
        return buffer

    def report_playback(self, buffer: FrameBuffer, seconds: float) -> None:
        """Tell the renderer how far this utterance's audio has actually played.

        The server uses this clock to stop neural work when it gets more than a
        small prebuffer ahead.  Progress is best-effort: losing a report must
        never interrupt speech playback.
        """
        sock = self._sock
        if sock is None:
            return
        try:
            with self._send_lock:
                sock.sendall(
                    encode(
                        {
                            "op": "progress",
                            "id": buffer.utterance_id,
                            "seconds": max(0.0, min(float(seconds), buffer.duration_seconds)),
                        }
                    )
                )
        except (OSError, TypeError, ValueError):
            self._sock = None
            self._ready = False

    def _pump(self, sock) -> None:
        """One reader per connection: hands each message to the utterance it belongs to."""
        try:
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                for msg, payload in self._reader.feed(data):
                    uid = msg.get("id")
                    with self._buffers_lock:
                        buffer = self._buffers.get(uid)
                    if buffer is None:
                        continue  # a frame of an utterance that was cancelled or is long gone
                    op = msg.get("op")
                    if op == "started":
                        buffer.mark_started()
                    elif op == "frame":
                        buffer.put(msg.get("index", 0), payload)
                    elif op == "done":
                        buffer.finish(msg.get("frames", 0))
                        # The caller/renderer owns the completed buffer. The
                        # client only needs in-flight buffers for dispatch.
                        with self._buffers_lock:
                            self._buffers.pop(uid, None)
                    elif op == "error":
                        buffer.fail(msg.get("message", "face server error"))
                        with self._buffers_lock:
                            self._buffers.pop(uid, None)
        except OSError:
            pass
        # The connection is gone: say so, so the next check reconnects and the mouth falls back at once.
        buffers = []
        if self._sock is sock:
            # Drain this connection's buffers before advertising that a new
            # connection may be opened, or the old pump could clear new work.
            with self._buffers_lock:
                buffers, self._buffers = list(self._buffers.values()), {}
            self._sock = None
            self._ready = False
            self._protocol_version = 0
            self._reader = MessageReader()
        for buffer in buffers:
            if not (buffer.done() if callable(buffer.done) else buffer.done):
                buffer.fail("face server disconnected")

    def cancel(self) -> None:
        """Stop every utterance that is playing or being prepared."""
        with self._buffers_lock:
            buffers, self._buffers = self._buffers, {}
        if self._sock is None:
            return
        try:
            with self._send_lock:
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
        self._protocol_version = 0
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
