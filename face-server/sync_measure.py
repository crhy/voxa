"""Estimate how far the generated mouth lags the voice: correlate mouth opening with loudness per frame."""
import glob, sys, numpy as np, cv2, mediapipe as mp, librosa
secs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path="/home/rhy/voxa-characters/analysis/models/face_landmarker.task"), num_faces=1)
lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
opens = []
for f in sorted(glob.glob("/home/rhy/voxa-neural/out/sync/*.jpg")):
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if r.face_landmarks:
        p = r.face_landmarks[0]; e = abs(p[33].x - p[263].x)
        opens.append(abs(p[13].y - p[14].y) / e)
    else:
        opens.append(0.0)
opens = np.array(opens)
audio, _ = librosa.load("/home/rhy/voxa-neural/ditto-talkinghead/example/audio.wav", sr=16000)
audio = audio[: secs * 16000]
rms = np.array([np.sqrt(np.mean(audio[i * 640:(i + 1) * 640] ** 2) + 1e-12) for i in range(len(opens))])
a = (opens - opens.mean()) / opens.std(); b = (rms - rms.mean()) / rms.std()
def corr(k):
    return float(np.mean(a[max(0, k):len(a) + min(0, k)] * b[max(0, -k):len(b) + min(0, -k)]))
for k in (-6, -4, -2, 0, 2, 4, 6, 8, 10, 12):
    print(f"mouth {k * 40:+5d} ms after voice: correlation {corr(k):.3f}")
best = max(range(-15, 16), key=corr)
print("best fit: mouth is", best * 40, "ms after the voice")
