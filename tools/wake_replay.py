"""Replay the owner's read-aloud rounds through the app's own wake-word path.

For every round x microphone x {raw, webrtc}, each line window is cut exactly
like score.py does (xc alignment) and run through the app's own functions
(WhisperService.transcribe, strip_wake_word) plus four variants:

  a  app as is (tiny wake model, hint "Voxa.", strip_wake_word)
  b  app + faster-whisper hotwords="Voxa" on top of the app's hint
  c  app + near-miss wake words accepted ONLY as the first word(s)
  d  b + c together

Also counts false triggers on stretches with no owner speech (first 2.5 s of
every round, and the echo-only clips cut into 4 s windows). Writes
.voxa-spec/out/anclab/WAKE.md.
"""

import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Voxa")
sys.path.insert(0, "/Voxa/tools")

from anc_lab import SR, read_wav  # noqa: E402

from voxa.conversation import strip_wake_word  # noqa: E402  (wake word default: voxa/config.py:53)
from voxa.transcription import WhisperService  # noqa: E402

OUT = Path("/Voxa/.voxa-spec/out/anclab/readscript")
REC = Path("/Voxa/.voxa-spec/out/anclab/recordings")
WAKE_MD = Path("/Voxa/.voxa-spec/out/anclab/WAKE.md")

NEAR_MISSES = (
    "boxer",
    "boxa",
    "box a",
    "box to",
    "box of",
    "loxa",
    "luxa",
    "locks a",
    "vox",
    "vox a",
    "voxer",
    "bucks a",
)


def near_miss_wake(text: str) -> str | None:
    """Command text after a listed near-miss wake word, or None.

    Only the very start of the utterance counts: a near-miss word in the
    middle of a sentence never wakes the app. The phrase must be followed by
    a word boundary, so "vox" does not match inside "voxel".
    """
    t = re.sub(r"^[^a-z0-9]+", "", text, flags=re.IGNORECASE)
    low = t.lower()
    for phrase in sorted(NEAR_MISSES, key=len, reverse=True):
        if re.match(re.escape(phrase) + r"(?![\w'])", low):
            return t[len(phrase) :].strip(" ,.!?—-\t\n")
    return None


def _words(text: str) -> list[str]:
    """Words of a transcript, same normalisation as score.py."""
    return re.findall(
        r"[a-z']+",
        text.lower().replace("voxer", "voxa").replace("vox a", "voxa"),
    )


def acc(text: str, line: str) -> float:
    """Fraction of the line's command words present, same scoring as score.py."""
    got = _words(text)
    want = _words(line)[1:]
    pool = list(got)
    hit = 0
    for w in want:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit / len(want)


def xc(a: np.ndarray, b: np.ndarray) -> int:
    """Alignment offset in samples, exactly as score.py computes it."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    k = int(np.argmax(np.abs(c)))
    return k if k < n // 2 else k - n


def wake(word: str, text: str) -> tuple[bool, str]:
    """Variant a/b: the app's own decision — strip_wake_word gates the prompt."""
    rem = strip_wake_word(text, word)
    if rem is None:
        return False, text
    return True, rem


def wake_nm(word: str, text: str) -> tuple[bool, str]:
    """Variants c/d: the app's matcher OR the near-miss first-word matcher."""
    rem = strip_wake_word(text, word)
    if rem is not None:
        return True, rem
    nm = near_miss_wake(text)
    if nm is None:
        return False, text
    return True, nm


def main() -> None:
    wake_word = "voxa"
    hint = f"{wake_word.strip().capitalize()}."

    man = json.load(open(OUT / "manifest.json"))
    lines = man["lines"]

    svc = WhisperService()
    done = threading.Event()
    svc.load_async("tiny", lambda _n, _b: done.set(), lambda _e: done.set())
    for _ in range(3000):
        if done.is_set():
            break
        time.sleep(0.1)
    if not svc.ready:
        raise SystemExit("tiny model did not load")

    def app_transcribe(seg: np.ndarray) -> str:
        pcm = (np.clip(np.asarray(seg, np.float32), -1.0, 1.0) * 32768).astype("<i2")
        return svc.transcribe(pcm.tobytes(), "en", hint)

    def hot_transcribe(seg: np.ndarray) -> str:
        segments, _info = svc._model.transcribe(
            np.asarray(seg, np.float32),
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
            condition_on_previous_text=False,
            word_timestamps=False,
            initial_prompt=hint,
            hotwords="Voxa",
        )
        return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()

    rows = []
    mishear: dict[str, dict[str, int]] = {r: {} for r in man["rounds"]}
    for rnd, info in man["rounds"].items():
        ref = np.load(OUT / f"{rnd}_ref.npy").astype(np.float32)
        played = np.asarray(read_wav(OUT / f"{rnd}_played.wav"), np.float32)
        off = xc(ref, played)
        for m in ("blue", "camera"):
            mic = np.load(OUT / f"{rnd}_{m}.npy").astype(np.float32)
            n = min(len(mic), len(ref))
            audios = {"raw": mic[:n]}
            p = OUT / f"{rnd}_{m}.webrtc.npy"
            if p.exists():
                audios["webrtc"] = np.load(p).astype(np.float32)[:n]
            for aname, x in audios.items():
                hits = {k: 0 for k in "abcd"}
                scores = {k: [] for k in "abcd"}
                for line, (a, b) in zip(lines, info["line_windows_s"], strict=True):
                    lo = max(0, off + int(a * SR))
                    hi = off + int(b * SR)
                    seg = x[lo:hi] if hi > lo else x[:0]
                    t_plain = app_transcribe(seg)
                    t_hot = hot_transcribe(seg)
                    wa, act_a = wake(wake_word, t_plain)
                    wb, act_b = wake(wake_word, t_hot)
                    wc, act_c = wake_nm(wake_word, t_plain)
                    wd, act_d = wake_nm(wake_word, t_hot)
                    for k, (w, act) in {"a": (wa, act_a), "b": (wb, act_b),
                                        "c": (wc, act_c), "d": (wd, act_d)}.items():
                        hits[k] += int(w)
                        scores[k].append(acc(act, line))
                    rows.append((rnd, m, aname, hits, scores))
                    if aname == "raw":
                        heard = " ".join(_words(t_plain)[:2])
                        if heard and "voxa" not in heard.split():
                            mishear[rnd][heard] = mishear[rnd].get(heard, 0) + 1
                print(f"{rnd:8s} {m:6s} {aname:6s} "
                      + " ".join(f"{k}: {hits[k]}/10 {np.mean(scores[k])*100:5.1f}%" for k in "abcd"),
                      flush=True)

    # False triggers: first 2.5 s of every round (no owner speech yet).
    ft = {k: {"lead": [0, 0], "echo": [0, 0]} for k in "abcd"}
    for rnd, _info in man["rounds"].items():
        ref = np.load(OUT / f"{rnd}_ref.npy").astype(np.float32)
        played = np.asarray(read_wav(OUT / f"{rnd}_played.wav"), np.float32)
        off = xc(ref, played)
        for m in ("blue", "camera"):
            mic = np.load(OUT / f"{rnd}_{m}.npy").astype(np.float32)
            n = min(len(mic), len(ref))
            audios = {"raw": mic[:n]}
            p = OUT / f"{rnd}_{m}.webrtc.npy"
            if p.exists():
                audios["webrtc"] = np.load(p).astype(np.float32)[:n]
            for x in audios.values():
                lo = max(0, off)
                hi = min(off + int(2.5 * SR), len(x))
                if hi <= lo:
                    continue
                seg = x[lo:hi]
                t_plain = app_transcribe(seg)
                t_hot = hot_transcribe(seg)
                for k, (w, _act) in {
                    "a": wake(wake_word, t_plain), "b": wake(wake_word, t_hot),
                    "c": wake_nm(wake_word, t_plain), "d": wake_nm(wake_word, t_hot),
                }.items():
                    ft[k]["lead"][0] += int(w)
                    ft[k]["lead"][1] += 1
                print(f"lead {rnd} {m} done", flush=True)

    for f in sorted(REC.glob("speech_0*_v0.7.mic.npy")) + sorted(REC.glob("speech_0*_v1.0.mic.npy")):
        clip = np.load(f).astype(np.float32)
        for i in range(0, len(clip), 4 * SR):
            seg = clip[i:i + 4 * SR]
            if len(seg) < SR:
                continue
            t_plain = app_transcribe(seg)
            t_hot = hot_transcribe(seg)
            for k, (w, _act) in {
                "a": wake(wake_word, t_plain), "b": wake(wake_word, t_hot),
                "c": wake_nm(wake_word, t_plain), "d": wake_nm(wake_word, t_hot),
            }.items():
                ft[k]["echo"][0] += int(w)
                ft[k]["echo"][1] += 1
        print(f"echo {f.name} done", flush=True)

    lines_out = []
    lines_out.append("# Q21: wake-word replay of the owner's read-aloud rounds\n")
    lines_out.append("""## The app's wake chain (file:line)

- `voxa/window.py:94` `WAKE_WHISPER_MODEL = "tiny"`; `voxa/window.py:398`
  `_load_wake_whisper` loads it into the wake `WhisperService`.
- `voxa/transcription.py:55` `WhisperService`; `transcribe()` at
  `voxa/transcription.py:164` runs faster-whisper with `beam_size=1`,
  `vad_filter=True` (`min_silence_duration_ms=350`),
  `condition_on_previous_text=False`, `initial_prompt=hint or None`.
- Wake hint for the wake phase: `voxa/conversation.py:368` and
  `voxa/conversation.py:449` build `f"{wake_word.strip().capitalize()}."`
  (= `"Voxa."` for the default wake word, `voxa/config.py:53`).
  The prompt-phase hint is `voxa/conversation.py:242` `_build_prompt_hint`
  -> `voxa/vocabulary.py:29` `build_hint` (wake word + command verbs + app
  names); it is NOT used while waiting for the wake word.
- Early gate: `voxa/conversation.py:360` `_early_check` transcribes the
  segment so far with the tiny model + wake hint, then
  `voxa/endpoint.py:82` `is_complete(text, wake_word)` decides the utterance
  is finished (routes the command via `route()`).
- Final decision: `voxa/conversation.py:479` `strip_wake_word(text, wake_word)`
  (`voxa/conversation.py:18`, exact case-insensitive substring, else
  `_fuzzy_strip` at `voxa/conversation.py:37`: first one/two words must have
  `SequenceMatcher` ratio >= 0.65 vs the wake word). Not-None = wake; the
  remainder is the command the app acts on.
- Agent-side ASR fixes (not the wake gate): `voxa/agent/hearing.py:41`
  `WAKE_VARIANTS` / `_WAKE_ALT` used by `strip_wake`/`normalize`, and
  `voxa/agent/hearing.py:151` `direct_command` with `WAKE_SOUNDALIKES`.\n""")
    lines_out.append("## Wake hits and command words per round/microphone/audio (a=app, b=hotwords, c=near-miss first word, d=b+c)\n")
    lines_out.append("| round | mic | audio | a wake/10 | a cmd% | b wake/10 | b cmd% | c wake/10 | c cmd% | d wake/10 | d cmd% |")
    lines_out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rnd, m, aname, hits, scores in rows:
        lines_out.append(f"| {rnd} | {m} | {aname} | {hits['a']}/10 | {np.mean(scores['a'])*100:5.1f}% "
                         f"| {hits['b']}/10 | {np.mean(scores['b'])*100:5.1f}% "
                         f"| {hits['c']}/10 | {np.mean(scores['c'])*100:5.1f}% "
                         f"| {hits['d']}/10 | {np.mean(scores['d'])*100:5.1f}% |")
    lines_out.append("\n## False triggers (wake detections on stretches with no owner speech)\n")
    lines_out.append("| variant | first 2.5 s of each round | echo-only clips in 4 s windows |")
    lines_out.append("|---|---|---|")
    for k in "abcd":
        lead_h, lead_n = ft[k]["lead"]
        echo_h, echo_n = ft[k]["echo"]
        lines_out.append(f"| {k} | {lead_h}/{lead_n} | {echo_h}/{echo_n} |")
    lines_out.append("\n## Most common wake mishearings per round (raw mic, app hint, first two words)\n")
    for rnd in man["rounds"]:
        top = sorted(mishear[rnd].items(), key=lambda kv: -kv[1])[:3]
        lines_out.append(f"- {rnd}: " + (", ".join(f'"{w}" x{c}' for w, c in top) or "none"))
    lines_out.append("\nNotes: command words are scored on the text the app would act on "
                     "(the stripped remainder when the wake matched, otherwise the full "
                     "transcription, since without a wake the app acts on nothing). "
                     "Scoring is score.py's (wake word excluded from the wanted words).")
    WAKE_MD.write_text("\n".join(lines_out) + "\n")
    print("WAKE.md written", flush=True)


if __name__ == "__main__":
    main()
