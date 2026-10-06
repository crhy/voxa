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
    source = Gst.ElementFactory.make("filesrc")
    decode = Gst.ElementFactory.make("decodebin")
    convert = Gst.ElementFactory.make("audioconvert")
    resample = Gst.ElementFactory.make("audioresample")
    capsfilter = Gst.ElementFactory.make("capsfilter")
    sink = Gst.ElementFactory.make("appsink")
    if not all((source, decode, convert, resample, capsfilter, sink)):
        raise RuntimeError("Required GStreamer audio elements are unavailable.")
    source.set_property("location", str(path))
    capsfilter.set_property("caps", Gst.Caps.from_string("audio/x-raw,format=S16LE,channels=1,rate=16000"))
    sink.set_property("emit-signals", True)

    pipeline = Gst.Pipeline.new("voxa-live-decode")
    for element in (source, decode, convert, resample, capsfilter, sink):
        pipeline.add(element)
    links = (
        source.link(decode),
        decode.link(convert),
        convert.link(resample),
        resample.link(capsfilter),
        capsfilter.link(sink),
    )
    if not all(links):
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("Could not connect the GStreamer decode pipeline.")
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("The decode pipeline could not start.")

    chunks = []
    for _ in range(100000):
        sample = sink.emit("pull-sample")
        if sample is None:
            break
        buffer = sample.get_buffer()
        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            break
        try:
            chunks.append(bytes(mapped.data))
        finally:
            buffer.unmap(mapped)
    pipeline.set_state(Gst.State.NULL)
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
        self._next_id = 0

    def available(self) -> bool:
        if self._ready and self._sock is not None:
            return True
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            sock.connect((self.host, self.port))
            sock.sendall(encode({"op": "hello", "version": 1}))
            data = sock.recv(65536)
            for msg, _payload in self._reader.feed(data):
                if msg.get("op") == "hello" and msg.get("ready"):
                    self._characters = list(msg.get("characters", []))
                    self._ready = True
                    self._sock = sock
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
        threading.Thread(
            target=self._pump,
            args=(self._sock, buffer),
            name="voxa-live-face",
            daemon=True,
        ).start()
        return buffer

    def _pump(self, sock, buffer) -> None:
        try:
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                for msg, payload in self._reader.feed(data):
                    if msg.get("id") != buffer.utterance_id:
                        continue
                    op = msg.get("op")
                    if op == "frame":
                        buffer.put(msg.get("index", 0), payload)
                    elif op == "done":
                        buffer.finish(msg.get("frames", 0))
                    elif op == "error":
                        buffer.fail(msg.get("message", "face server error"))
        except OSError:
            pass

    def cancel(self) -> None:
        if self._sock is None or self._buffer is None:
            return
        try:
            self._sock.sendall(encode({"op": "cancel", "id": self._buffer.utterance_id}))
        except OSError:
            self._sock = None
            self._ready = False

    def close(self) -> None:
        sock, self._sock = self._sock, None
        self._ready = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
