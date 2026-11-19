"""B2's own measurement: the runner's transcripts against a direct WhisperModel.transcribe.

``python -m voxa.bench agree --suite smoke`` takes the NST clips (clean near-end speech, no echo
in the room), reads the request text the runner produced for each of them from the
``events.jsonl`` the last ``run`` wrote, and transcribes the same ``mic.wav`` directly with the
same pinned faster-whisper settings (one beam, ``temperature=0``, no conditioning on previous
text, no VAD filter, Voxa's default model).

Roadmap B2's gate: the two must agree on at least 98 % of the clean NST clips, so this exits 1
below that. The gated comparison uses the runner's ``clip-transcript`` event: the pinned ASR over
the whole stream its pipeline delivered for the clip. That is what proves the instrument - 20 ms
chunking, int16 handling, the pinned parameters, the echo stages - because the only thing left that
can differ is the audio the pipeline itself changed.

The per-utterance comparison (the ``transcript`` events, one per turn the speech-pause rules
produced) is printed as well and NOT gated: a difference there is a finding about the product's
turn-taking (a sentence cut at a pause, a pre-roll trimmed), which B2 measures and later tasks fix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import wave
from pathlib import Path

from .render import DEFAULT_DATA_ROOT
from .runner import PinnedASR

WORD = re.compile(r"[a-z0-9']+")


def normalize_words(text: str) -> list[str]:
    """Lower-case words of a transcript, the way the bench compares two of them."""
    return WORD.findall(text.casefold())


def direct_transcribe(asr: PinnedASR, wav_path: Path) -> dict:
    """``WhisperModel.transcribe`` of the whole mic WAV, pinned the same way as the runner.

    The frames are taken straight out of the WAV: the exact bytes the runner's pipeline was
    handed, so a difference in the transcript can only come from the pipeline, not from a
    re-encoding in the comparison itself.
    """
    with wave.open(str(wav_path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("only 16-bit mono WAVs are supported")
        frames = handle.readframes(handle.getnframes())
    return asr.transcribe_words(frames)


def agreement(runner_text: str, direct_text: str) -> tuple[bool, float]:
    """Exact match after normalisation, plus the word-level overlap of the two transcripts."""
    want = normalize_words(direct_text)
    got = normalize_words(runner_text)
    if not want:
        return (not got, 1.0 if not got else 0.0)
    pool = list(want)
    hits = 0
    for word in got:
        if word in pool:
            pool.remove(word)
            hits += 1
    return (got == want, hits / len(want))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voxa.bench agree",
        description="runner transcripts vs direct WhisperModel.transcribe of the mic WAV")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--scenario", default="nst", help="which clips to compare (default: the clean NST ones)")
    parser.add_argument("--min", type=float, default=0.98, help="agreement required to pass (roadmap: 0.98)")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("VOXA_BENCH_DATA", str(DEFAULT_DATA_ROOT))))
    parser.add_argument("--events", type=Path, default=Path(".voxa-bench"),
                        help="where the last run wrote <suite>/<clip>/events.jsonl")
    parser.add_argument("--model", default="", help="faster-whisper model (default: Voxa's, config.py:22)")
    args = parser.parse_args(argv)

    suite_dir = args.data_root / args.suite
    if not suite_dir.is_dir():
        print(f"{suite_dir} does not exist: render the suite first", file=sys.stderr)
        return 2
    asr = PinnedASR(model_name=args.model)
    clip_dirs = sorted(path for path in suite_dir.iterdir()
                       if (path / "truth.json").exists()
                       and json.loads((path / "truth.json").read_text()).get("scenario") == args.scenario)
    if not clip_dirs:
        print(f"no {args.scenario} clips in {suite_dir}", file=sys.stderr)
        return 2

    exact = 0
    word_total = 0.0
    turns_exact = 0
    turn_words_total = 0.0
    rows: list[str] = []
    started = time.perf_counter()
    for clip_dir in clip_dirs:
        events_path = args.events / args.suite / clip_dir.name / "events.jsonl"
        if not events_path.exists():
            print(f"{events_path} missing: run python -m voxa.bench run --suite {args.suite} first",
                  file=sys.stderr)
            return 2
        heard: list[str] = []
        whole: str | None = None
        for line in events_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "clip-transcript":
                whole = str(event.get("text", ""))
            elif event.get("event") == "transcript":
                heard.append(str(event.get("text", "")))
        direct = direct_transcribe(asr, clip_dir / "mic.wav")
        clip_match, clip_words = agreement(whole if whole is not None else "", direct["text"])
        exact += int(clip_match)
        word_total += clip_words
        turn_match, turn_words = agreement(" ".join(heard), direct["text"])
        turns_exact += int(turn_match)
        turn_words_total += turn_words
        rows.append(f"| {clip_dir.name} | clip {'exact' if clip_match else 'differs'} {clip_words:.3f} "
                    f"| turns {'exact' if turn_match else 'differs'} {turn_words:.3f} "
                    f"| {direct['text'][:44]!r} |")
    rate = exact / len(clip_dirs)
    print(f"agree {args.suite} {args.scenario}: clip {exact}/{len(clip_dirs)} exact = {rate:.3f} "
          f"(word overlap {word_total / len(clip_dirs):.3f}); "
          f"per-turn {turns_exact}/{len(clip_dirs)} exact = {turns_exact / len(clip_dirs):.3f} "
          f"(word overlap {turn_words_total / len(clip_dirs):.3f}) in {time.perf_counter() - started:.1f} s")
    for line in rows:
        print(line)
    if rate < args.min:
        print(f"agree FAILED: {rate:.3f} < {args.min}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
