"""Voxa face server: animates a portrait from speech audio with Ditto and streams JPEG frames.

Runs INSIDE the no-network sandbox (see run-face-server.sh) and listens on a Unix socket; face_relay.py, outside
the sandbox, forwards 127.0.0.1:8765 to that socket. Protocol (one JSON line per message, optionally followed by
`bytes` raw bytes) is described in /Voxa/.voxa-spec/tasks/agent/l02-live-face-client.md.

Usage (cwd = ditto-talkinghead): python face_server.py <unix-socket-path> [--fake] [--steps N] [--max-fps N]
  --fake    no neural model: returns the source portrait for every frame (for wiring tests without a GPU)
  --steps   diffusion sampling steps (Ditto's default is 50; fewer is faster and a little less refined)
  --max-fps leave a small gap between generated frames so CUDA does not starve the desktop compositor
  --no-compile  skip torch.compile of the render networks (fp16 only; slower per frame, faster first start)
"""
import collections
import itertools
import json
import math
import os
import socket
import sys
import threading
import time
import zlib

import cv2
import numpy as np

N = os.environ.get("VOXA_FACE_HOME") or os.path.dirname(os.path.abspath(__file__))
PORTRAITS = f"{N}/portraits"          # 768x768 JPEGs, the same crop as Voxa's still portraits and face packs
FPS = 25
CHUNK = (3, 5, 2)                      # Ditto's streaming window: history, new, look-ahead (in 40 ms frames)
# Keep only a short cushion of generated video ahead of real audio playback.
# This gate is applied before the costly warp/decode stages as well as at the
# sender, so a prepared sentence cannot monopolise the GPU in a large burst.
MAX_RENDER_AHEAD_SECONDS = 0.75
# A frame is skipped when its moment will have been played before it can be on
# screen: it is admitted roughly this long before the client draws it (the GPU
# stages, JPEG encoding and delivery).  Skipping is what lets a face that fell
# behind -- a slow start on a long reply, or the GPU shared with the language
# model -- catch up with the voice instead of staying late for the whole reply.
CATCH_UP_SECONDS = 0.15
# Compiled render kernels are cached on disk.  The sandbox's HOME and /tmp are
# thrown away on exit, so without this every start would recompile (about 15 s)
# instead of loading the cache (about 3 s).  Set before torch is imported.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{N}/out/torch-cache")
os.environ.setdefault("TRITON_CACHE_DIR", f"{N}/out/torch-cache/triton")


def natural_head_motion(frames: int, pcm: bytes) -> dict[int, dict[str, float]]:
    """Return smooth, visible pose controls for a spoken utterance.

    Ditto's expression model is lively but this checkpoint produces very little
    head pose on its own. A pair of slow, incommensurate curves avoids the
    mechanical fixed-head result without turning speech into a repeating bob.
    The audio-derived phase varies the gesture between utterances, and the
    envelope makes both ends line up with Voxa's resting portrait.
    """
    if frames <= 0:
        return {}
    phase = zlib.crc32(pcm) / 0xFFFFFFFF * math.tau
    controls = {}
    for index in range(frames):
        seconds = index / FPS
        # A long edge ramp made short answers spend their entire duration near
        # zero pose. Six frames is still a quick transition at 25 fps, while
        # leaving enough of even a one-second answer for the gesture to read.
        edge = min(1.0, index / 6.0, max(0, frames - 1 - index) / 6.0)
        envelope = edge * edge * (3.0 - 2.0 * edge)  # smoothstep
        # Ditto's pose deltas are degrees. Keep the gesture visible at portrait
        # size, but restrained enough that High feels alive rather than restless.
        # Motion frequency is unchanged; only its range is reduced.
        yaw = 3.8 * math.sin(math.tau * seconds / 3.2 + phase)
        yaw += 0.7 * math.sin(math.tau * seconds / 1.55 + phase * 0.37)
        pitch = 1.5 * math.sin(math.tau * seconds / 2.7 + phase * 0.61)
        pitch += 0.4 * math.sin(math.tau * seconds / 1.35 + phase * 1.13)
        roll = 0.75 * math.sin(math.tau * seconds / 4.1 - phase * 0.43)
        controls[index] = {
            "delta_yaw": envelope * yaw,
            "delta_pitch": envelope * pitch,
            "delta_roll": envelope * roll,
        }
    return controls


def encode(message: dict, payload: bytes = b"") -> bytes:
    if payload:
        message = dict(message, bytes=len(payload))
    return json.dumps(message, separators=(",", ":")).encode() + b"\n" + payload


class Reader:
    def __init__(self, conn):
        self.conn, self.buf = conn, b""

    def _fill(self) -> bool:
        data = self.conn.recv(1 << 16)
        if not data:
            return False
        self.buf += data
        return True

    def next(self):
        while b"\n" not in self.buf:
            if not self._fill():
                return None, b""
        line, self.buf = self.buf.split(b"\n", 1)
        message = json.loads(line)
        need = int(message.get("bytes", 0))
        while len(self.buf) < need:
            if not self._fill():
                return None, b""
        payload, self.buf = self.buf[:need], self.buf[need:]
        return message, payload


class PlaybackGate:
    """Backpressure rendering when it gets too far ahead of played audio."""

    def __init__(self):
        self.seconds = 0.0
        self.reported_at = None
        self.condition = threading.Condition()

    def update(self, seconds: float) -> None:
        with self.condition:
            self.seconds = max(self.seconds, max(0.0, float(seconds)))
            if self.seconds > 0.0:
                self.reported_at = time.monotonic()
            self.condition.notify_all()

    def played(self) -> float | None:
        """Estimated audio position now (reports arrive every ~0.15 s); None before playback starts."""
        with self.condition:
            if self.reported_at is None:
                return None
            return self.seconds + (time.monotonic() - self.reported_at)

    def wait_for(self, frame_seconds: float, cancelled: threading.Event) -> bool:
        with self.condition:
            while frame_seconds > self.seconds + MAX_RENDER_AHEAD_SECONDS:
                if cancelled.is_set():
                    return False
                self.condition.wait(0.1)
        return not cancelled.is_set()

    def wake(self) -> None:
        with self.condition:
            self.condition.notify_all()


class Turns:
    """Lets numbered jobs through one at a time, strictly in number order."""

    def __init__(self):
        self.serving = 0
        self.condition = threading.Condition()

    def wait(self, ticket: int) -> None:
        with self.condition:
            while self.serving != ticket:
                self.condition.wait()

    def advance(self) -> None:
        with self.condition:
            self.serving += 1
            self.condition.notify_all()


class FrameSender:
    """Stands in for Ditto's video writer: each finished frame goes straight to the client as a JPEG."""

    def __init__(self, conn, lock, utterance_id, size, cancelled, gate, fps=FPS, max_fps=0, on_first_frame=None):
        self.conn, self.lock, self.id, self.size, self.cancelled = conn, lock, utterance_id, size, cancelled
        self.on_first_frame = on_first_frame
        self.gate = gate
        self.index = 0  # frames sent
        self.skipped = 0
        self.first_at = None
        self.fps = min(float(fps), max_fps) if max_fps > 0 else float(fps)
        self.fps = max(1.0, self.fps)
        self.min_interval = 1.0 / self.fps
        self.last_render_release_at = None
        # Timeline positions of admitted frames, in order: the pipeline keeps
        # order, so each finished frame takes the next one.  Skipped frames
        # leave gaps, which the client fills by holding the previous frame.
        self.admitted = collections.deque()

    def wait_to_render(self, index: int):
        """Admit, skip or stop one frame before its costly GPU work.

        Returns True to render, "skip" for a frame whose moment has already
        been played, False to stop.  Admission is spaced by one frame interval
        so a newly released job cannot fill every CUDA stage in one burst that
        starves Compiz and GTK.  That spacing alone never lets a late face
        catch up, so frames that are already late are skipped instead.  Frame
        0 is always rendered: Ditto takes its motion reference from the first
        frame it stitches.
        """
        seconds = float(index) / self.fps
        if not self.gate.wait_for(seconds, self.cancelled):
            return False
        played = self.gate.played()
        if index > 0 and played is not None and seconds < played + CATCH_UP_SECONDS:
            self.skipped += 1
            return "skip"
        now = time.monotonic()
        if self.last_render_release_at is not None:
            delay = self.last_render_release_at + self.min_interval - now
            if delay > 0 and self.cancelled.wait(delay):
                return False
        self.last_render_release_at = time.monotonic()
        if self.cancelled.is_set():
            return False
        self.admitted.append(index)
        return True

    def __call__(self, frame_rgb, fmt="rgb"):
        if self.cancelled.is_set():
            return
        # Without a render gate (warm-up, older pipelines) frames are consecutive.
        index = self.admitted.popleft() if self.admitted else self.index
        # Frames are timed by the client against the audio it is playing, so
        # the only pacing needed here is the cushion limit; a timeline anchored
        # to the first frame kept a late start late for the whole reply.
        if not self.gate.wait_for(index / self.fps, self.cancelled):
            return
        if self.first_at is None:
            self.first_at = time.time()
        img = frame_rgb if fmt == "bgr" else cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self.lock:
                self.conn.sendall(encode({"op": "frame", "id": self.id, "index": index}, jpeg.tobytes()))
            self.index += 1
            if self.index == 1 and self.on_first_frame is not None:
                self.on_first_frame()

    def close(self):
        pass


def accelerate(sdk, compile_models: bool = True) -> str:
    """Make the per-frame warp and decode passes cheaper on the GPU; returns the mode in use.

    These two networks are nearly all of High's GPU time (about 32 ms a frame
    as Ditto ships them).  Ditto runs them under autocast, which re-casts every
    fp32 weight on every call; it also uploads the portrait's 3D features again
    for every frame and copies the warped features to the CPU and back between
    the two stages.  Here the weights become fp16 once, the features stay on
    the GPU, the warped features go straight to the decoder, and both networks
    are compiled: about 22 ms a frame for the same picture.  Each step falls
    back to the next simpler one, finally to Ditto's own path.
    """
    import torch

    original = sdk.warp_f3d, sdk.decode_f3d
    warp_model = sdk.warp_f3d.warp_net.model.eval()
    decode_model = sdk.decode_f3d.decoder.model.eval()
    features = {}  # id(f_s array) -> (array, GPU tensor); holding the array keeps its id from being reused

    def gpu(array):
        return torch.from_numpy(array).to("cuda").half()

    def make(warp, decode):
        def warp_f3d(f_s, x_s, x_d):
            hit = features.get(id(f_s))
            if hit is None or hit[0] is not f_s:
                if len(features) >= 8:
                    features.pop(next(iter(features)))
                hit = features[id(f_s)] = (f_s, gpu(f_s))
            with torch.no_grad():
                return warp(hit[1], gpu(x_s), gpu(x_d))

        def decode_f3d(f_3d):
            # Same result as Ditto's Decoder: an HxWx3 float32 image in 0..255.
            with torch.no_grad():
                image = decode(f_3d)[0].float().clamp_(0, 1).mul_(255).permute(1, 2, 0).contiguous()
            return image.cpu().numpy()

        return warp_f3d, decode_f3d

    def ready(stages) -> bool:
        warp_f3d, decode_f3d = stages
        pose = np.zeros((1, 21, 3), dtype=np.float32)
        decode_f3d(warp_f3d(np.zeros((1, 32, 16, 64, 64), dtype=np.float32), pose, pose))
        features.clear()
        return True

    warp_model.half(), decode_model.half()
    attempts = [("fp16", (warp_model, decode_model))]
    if compile_models:
        # Compiled on this thread now, not lazily inside the first utterance.
        attempts.insert(0, ("fp16+compiled", (torch.compile(warp_model), torch.compile(decode_model))))
    for mode, (warp, decode) in attempts:
        stages = make(warp, decode)
        try:
            if ready(stages):
                sdk.warp_f3d, sdk.decode_f3d = stages
                return mode
        except Exception as exc:  # noqa: BLE001 - each step is only an optimisation
            print(f"{mode} rendering unavailable:", repr(exc)[:300], flush=True)
    warp_model.float(), decode_model.float()
    sdk.warp_f3d, sdk.decode_f3d = original
    return "ditto"


def characters() -> list[str]:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PORTRAITS) if f.endswith(".jpg"))


class Engine:
    def __init__(self, fake: bool, steps: int, max_fps: float = 0, compile_models: bool = True):
        self.fake, self.steps, self.max_fps, self.sdk = fake, steps, max_fps, None
        self.render_mode = "fake" if fake else "ditto"
        self.features_lock = threading.Lock()
        if not fake:
            sys.path.insert(0, ".")
            # The offline pipeline: Voxa always has the whole sentence before it plays, and this one returns
            # exactly one frame per 40 ms of audio (the streaming one swallows its first window).
            from stream_pipeline_offline import StreamSDK

            self.sdk = StreamSDK("./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", "./checkpoints/ditto_pytorch")
            # Registering a portrait (face detection, appearance features) costs seconds: do it once each.
            register, cache = self.sdk.avatar_registrar, {}

            def cached(path, **kwargs):
                key = (path, tuple(sorted((k, str(v)) for k, v in kwargs.items())))
                if key not in cache:
                    cache[key] = register(path, **kwargs)
                return cache[key]

            self.sdk.avatar_registrar = cached
            started = time.time()
            self.render_mode = accelerate(self.sdk, compile_models)
            print(f"rendering: {self.render_mode} (ready in {time.time() - started:.1f}s)", flush=True)

    def features(self, pcm: bytes):
        """Speech features for one utterance, or None in --fake mode.

        Computed before the utterance waits for the GPU renderer, so a reply's
        next sentence is prepared while the current one is still being drawn
        instead of adding this time to the start of every sentence.
        """
        if self.fake:
            return None
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        with self.features_lock:
            return self.sdk.wav2feat.wav2feat(audio)

    def run(self, character: str, pcm: bytes, sender: FrameSender, feat=None) -> int:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        frames = math.ceil(len(audio) / 16000 * FPS)
        source = f"{PORTRAITS}/{character}.jpg"
        if self.fake:
            frames = math.ceil(len(audio) / 16000 * sender.fps)
            still = cv2.imread(source)
            for _ in range(frames):
                sender(still, fmt="bgr")
                time.sleep(0.01)
            return frames
        sdk = self.sdk
        t0 = time.time()
        sdk.setup(
            source,
            f"{N}/out/live-{os.getpid()}.mp4",
            sampling_timesteps=self.steps,
            # One slot per stage carries the output clock's backpressure all
            # the way to CUDA. _put() uses timed writes, so shutdown markers
            # can no longer deadlock a full queue.
            pipeline_queue_size=1,
            render_fps=sender.fps,
            # Backpressure before motion stitching, warping and decoding.  An
            # output-only gate still allowed every CUDA stage to start another
            # frame before the full pipeline finally filled up.
            render_gate=getattr(sender, "wait_to_render", None),
        )
        try:
            sdk.writer.close()
        except Exception:
            pass
        sdk.writer = sender
        # Ease out of and back into the portrait's own pose so Voxa can cross-fade to its resting frame.
        # This checkpoint otherwise changes the mouth while leaving the head almost perfectly fixed.
        sdk.setup_Nd(N_d=frames, fade_in=6, fade_out=10, ctrl_info=natural_head_motion(frames, pcm))
        t1 = time.time()
        if feat is None:
            with self.features_lock:
                feat = sdk.wav2feat.wav2feat(audio)
        t2 = time.time()
        sdk.audio2motion_queue.put(feat)
        print(f"  setup {t1 - t0:.2f}s, audio features {t2 - t1:.2f}s", flush=True)
        finished = threading.Event()

        def stop_cancelled_render():
            while not finished.is_set():
                if sender.cancelled.wait(0.05):
                    sdk.stop_event.set()
                    return

        threading.Thread(target=stop_cancelled_render, daemon=True).start()
        try:
            sdk.close()
        finally:
            finished.set()
        return sender.index


def serve(conn, engine: Engine, busy: threading.Lock):
    reader, lock = Reader(conn), threading.Lock()
    current = {"id": None, "pcm": bytearray(), "character": "", "size": 512,
               "cancelled": threading.Event(), "gate": PlaybackGate()}
    jobs_lock = threading.Lock()
    jobs = {}
    # A reply's sentences arrive in order and must be prepared and drawn in
    # that order; plain locks let whichever thread wins go first.
    tickets = itertools.count()
    feature_turns, render_turns = Turns(), Turns()
    # Set once the previous sentence shows its first frame (or ends): the next
    # sentence is prepared after that, never competing with that first frame.
    previous_ready = None
    try:
        while True:
            message, payload = reader.next()
            if message is None:
                return
            op = message.get("op")
            if op == "shutdown":
                print("shutdown requested by the client", flush=True)
                os._exit(0)
            if op == "hello":
                conn.sendall(encode({"op": "hello", "version": 5, "ready": True, "characters": characters(),
                                     "fake": engine.fake}))
            elif op == "utterance":
                current = {"id": message["id"], "pcm": bytearray(), "character": message.get("character", ""),
                           "size": int(message.get("size", 512)), "fps": float(message.get("fps", FPS)),
                           "cancelled": threading.Event(), "gate": PlaybackGate()}
            elif op == "audio" and message.get("id") == current["id"]:
                current["pcm"] += payload
            elif op == "cancel":
                uid = message.get("id")
                if uid == current["id"]:
                    current["cancelled"].set()
                    current["gate"].wake()
                with jobs_lock:
                    job = jobs.get(uid)
                if job is not None:
                    job["cancelled"].set()
                    job["gate"].wake()
            elif op == "progress":
                uid = message.get("id")
                with jobs_lock:
                    job = jobs.get(uid)
                if job is None and uid == current["id"]:
                    job = current
                if job is not None:
                    job["gate"].update(message.get("seconds", 0.0))
            elif op == "end" and message.get("id") == current["id"]:
                job = current
                job["ticket"] = next(tickets)
                job["after"], job["ready"] = previous_ready, threading.Event()
                previous_ready = job["ready"]
                with jobs_lock:
                    jobs[job["id"]] = job

                def work(job=job):
                    feat = None
                    if job["after"] is not None:
                        job["after"].wait()
                    feature_turns.wait(job["ticket"])
                    try:
                        if not job["cancelled"].is_set():
                            feat = engine.features(bytes(job["pcm"]))
                    except Exception as exc:  # noqa: BLE001 - run() computes them again and reports
                        print("speech features failed:", repr(exc), flush=True)
                    finally:
                        feature_turns.advance()
                    render_turns.wait(job["ticket"])
                    try:
                        with busy:  # one utterance at a time on the GPU
                            if job["cancelled"].is_set():
                                return
                            started = time.time()
                            sender = FrameSender(
                                conn, lock, job["id"], job["size"], job["cancelled"], job["gate"],
                                job["fps"], engine.max_fps, on_first_frame=job["ready"].set
                            )
                            try:
                                with lock:
                                    conn.sendall(encode({"op": "started", "id": job["id"]}))
                                if job["character"] not in characters():
                                    raise ValueError(f"unknown character {job['character']!r}")
                                count = engine.run(job["character"], bytes(job["pcm"]), sender, feat)
                                took = time.time() - started
                                first = (sender.first_at - started) if sender.first_at else -1
                                line = (f"utterance {job['id']} {job['character']}: {count} frames in {took:.2f}s "
                                        f"({count / max(took, 1e-6):.1f} fps), first frame after {first:.2f}s, "
                                        f"{sender.skipped} skipped to keep up")
                                print(line, flush=True)
                                try:  # kept on disk too: when Voxa starts the server its output goes nowhere
                                    with open(f"{N}/out/face-usage.log", "a") as log:
                                        log.write(time.strftime("%H:%M:%S ") + line + "\n")
                                except OSError:
                                    pass
                                with lock:
                                    conn.sendall(encode({"op": "done", "id": job["id"], "frames": count,
                                                         "seconds": round(took, 3), "first_frame": round(first, 3)}))
                            except Exception as exc:  # noqa: BLE001 - report to the client, keep serving
                                print("utterance failed:", repr(exc), flush=True)
                                try:
                                    with lock:
                                        conn.sendall(encode({"op": "error", "id": job["id"], "message": str(exc)[:300]}))
                                except OSError:
                                    pass
                    finally:
                        job["ready"].set()
                        render_turns.advance()
                        with jobs_lock:
                            jobs.pop(job["id"], None)

                threading.Thread(target=work, daemon=True).start()
    except (OSError, ValueError) as exc:
        print("connection closed:", repr(exc), flush=True)
    finally:
        current["cancelled"].set()
        with jobs_lock:
            active_jobs = list(jobs.values())
        for job in active_jobs:
            job["cancelled"].set()
            job["gate"].wake()
        try:
            conn.close()
        except OSError:
            pass


def main():
    args = sys.argv[1:]
    path = args[0]
    fake = "--fake" in args
    steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 50
    max_fps = float(args[args.index("--max-fps") + 1]) if "--max-fps" in args else 0.0
    engine = Engine(fake, steps, max_fps, compile_models="--no-compile" not in args)
    if not fake:
        # The first run pays for CUDA start-up (about 1.5 s). Pay it now, not on the user's first sentence.
        class Discard:
            index, first_at, cancelled = 0, None, threading.Event()
            fps = engine.max_fps if engine.max_fps > 0 else FPS

            def __call__(self, frame, fmt="rgb"):
                self.index += 1

            def close(self):
                pass

        try:
            warm = time.time()
            engine.run(characters()[0], b"\x00\x00" * 16000, Discard())
            print(f"warm-up done in {time.time() - warm:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failed warm-up only costs speed later
            print("warm-up failed:", repr(exc), flush=True)
    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o600)
    server.listen(4)
    busy = threading.Lock()
    print(f"face server ready on {path} (fake={fake}, steps={steps}, rendering={engine.render_mode}, "
          f"characters={len(characters())})", flush=True)
    while True:
        conn, _ = server.accept()
        threading.Thread(target=serve, args=(conn, engine, busy), daemon=True).start()


if __name__ == "__main__":
    main()
