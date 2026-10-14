"""Send real speech to the face server and report speed and delay. Usage: live_probe.py <character> [seconds]"""
import json, socket, sys, time, wave, subprocess
import numpy as np
char = sys.argv[1]; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
import librosa
audio, _ = librosa.load("/home/rhy/voxa-neural/ditto-talkinghead/example/audio.wav", sr=16000)
pcm = (np.clip(audio[: int(secs * 16000)], -1, 1) * 32767).astype("<i2").tobytes()
s = socket.create_connection(("127.0.0.1", 8765), timeout=5); s.settimeout(120)
def send(m, p=b""):
    if p: m = dict(m, bytes=len(p))
    s.sendall(json.dumps(m).encode() + b"\n" + p)
buf = b""
def recv():
    global buf
    while b"\n" not in buf:
        d = s.recv(1 << 16)
        if not d: raise SystemExit("the face server closed the connection")
        buf += d
    line, buf = buf.split(b"\n", 1); m = json.loads(line); n = m.get("bytes", 0)
    while len(buf) < n:
        d = s.recv(1 << 16)
        if not d: raise SystemExit("the face server closed the connection")
        buf += d
    p, buf = buf[:n], buf[n:]; return m, p
send({"op": "hello", "version": 1}); recv()
t0 = time.time()
send({"op": "utterance", "id": 1, "character": char, "size": 512, "fps": 25, "sample_rate": 16000})
send({"op": "audio", "id": 1}, pcm); send({"op": "end", "id": 1})
times = []; keep = {}
while True:
    m, p = recv()
    if m["op"] == "frame":
        times.append(time.time() - t0)
        if m["index"] in (0, 20, 40, 60, 80, 100): keep[m["index"]] = p
    else:
        print(m); break
n = len(times)
late = max((times[i] - i / 25) for i in range(n))  # how long the voice must be held so no frame arrives late
print(f"{char}: {n} frames for {secs:.1f}s audio | first frame {times[0]:.2f}s | last {times[-1]:.2f}s | "
      f"{n / times[-1]:.1f} fps overall | hold voice {late:.2f}s for smooth playback")
for i, p in keep.items(): open(f"/home/rhy/voxa-neural/out/live-{char}-{i:03d}.jpg", "wb").write(p)
