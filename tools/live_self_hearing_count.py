#!/usr/bin/env python3
"""Count Voxa's self-hearing events in one live run (issue #72, task v09).

Reads Voxa's barge.log for the run (lines from --from-line on) plus the VoxaTest
JSON reports that say what the test user asked. Per `words heard=` line written
while `speaking=` is set it classifies, exactly as the task defines:

  echo  - the heard words mostly match what she said in that turn
          (the `speaking=` text, the turn's speak-start texts, or a turn from
          the last 10 s: the piece-gap echo v08 fixed);
  user  - the heard words match one of VoxaTest's prompts (with and without the
          wake word, and the `say <reply>` lead prompts of hearing cases).

and counts, for the run:
  A  echo lines the app decided `user`
  B  INTERRUPT events caused by an echo line
  C  replies she gave to her own words (a `request` whose heard text matches a
     turn of hers, not a VoxaTest prompt, that she then answered)
  D  real barge-ins she obeyed, of the yield-during-speech cases VoxaTest made
     (the live hearing block says she yielded; the words reached her)
  E  `reference silent` decisions taken while she was speaking

self-hearing events = A + B + C. Prints one JSON summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

WORD_RE = re.compile(r"[a-z0-9']+")
LINE_RE = re.compile(r"^(\d\d:\d\d:\d\d) (\S+)(?: (.*))?$")
KEY_RE = re.compile(r"\b(\w+)=(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|(\S+))")


def norm(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def overlap(heard: list[str], said: list[str]) -> float:
    """Fraction of heard words that sit inside what she said (mostly-match rule)."""
    if not heard or not said:
        return 0.0
    said_set = set(said)
    return sum(1 for word in heard if word in said_set) / len(heard)


def match_score(heard: list[str], source: list[str]) -> float:
    """How much of `source` the heard line carries (user match, tolerant of mishears)."""
    if not heard or not source:
        return 0.0
    heard_set = set(heard)
    hits = sum(1 for word in source if word in heard_set)
    return hits / len(source)


def parse_barge(path: Path, from_line: int) -> list[dict]:
    events = []
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if index <= from_line:
            continue
        mo = LINE_RE.match(line)
        if not mo:
            continue
        time_s, event, rest = mo.group(1), mo.group(2), mo.group(3) or ""
        fields = {}
        for key_re in KEY_RE.finditer(rest):
            value = key_re.group(2)
            if value is None:
                value = key_re.group(3) if key_re.group(3) is not None else key_re.group(4)
            fields[key_re.group(1)] = value
        events.append({"line": index, "time": time_s, "event": event, "fields": fields, "raw": rest})
    return events


def load_reports(patterns: list[str]) -> list[dict]:
    cases = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            results = data.get("results", data) if isinstance(data, dict) else data
            if isinstance(results, list):
                for record in results:
                    record["_report"] = path
                    cases.append(record)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="barge.log (or assembled run log) to read")
    ap.add_argument("--from-line", type=int, default=0,
                    help="only lines after this one belong to the run")
    ap.add_argument("--reports", nargs="+", required=True,
                    help="VoxaTest JSON report files (glob ok)")
    args = ap.parse_args()

    events = parse_barge(Path(args.log), args.from_line)
    reports = load_reports(args.reports)

    # What the test user said (user evidence), and what she said (echo evidence).
    wake = ("voxa", "bazaar", "boxa", "boxer", "dirty", "azar", "bazaar,")
    user_phrases: list[list[str]] = []
    for record in reports:
        for text in (record.get("prompt", ""),):
            words = [w for w in norm(text) if w not in wake]
            if words:
                user_phrases.append(words)
        hearing = record.get("hearing") or {}
        expected = norm(str(hearing.get("expected_request", "")))
        expected = [w for w in expected if w not in ("voxa", "voxas")]
        if expected:
            user_phrases.append(expected)

    turns: list[dict] = []          # {"texts": [...], "start_line": n, "end_line": n}
    current: dict | None = None
    a = b = c = e = 0
    other_words = 0
    classified: list[dict] = []

    def turn_words(line: int) -> list[str]:
        words: list[str] = []
        for turn in turns[-3:]:
            if turn["start_line"] <= line or abs(turn["end_line"] - line) <= 400:
                words += turn["texts"]
        if current is not None:
            words += current["texts"]
        return words

    for event in events:
        name, fields = event["event"], event["fields"]
        if name == "speak-start":
            text = norm(fields.get("text", ""))
            if current is None:
                current = {"texts": list(text), "start_line": event["line"], "end_line": 10 ** 9}
                turns.append(current)
            else:
                current["texts"] += text
        elif name == "reply-end":
            if current is not None:
                current["end_line"] = event["line"]
                current = None
        elif name == "INTERRUPT":
            if current is not None:
                current["end_line"] = event["line"]
                current = None
        elif name == "words":
            heard = norm(fields.get("heard", ""))
            speaking = norm(fields.get("speaking", ""))
            if not speaking:
                continue
            said = turn_words(event["line"]) + speaking
            echo = overlap(heard, said)
            user = max([match_score(heard, phrase) for phrase in user_phrases] or [0.0])
            decided_user = fields.get("source", "") == "user"
            kind = "other"
            if echo >= 0.5 and echo > user:
                kind = "echo"
            elif user >= 0.5:
                kind = "user"
            if "reference silent" in fields.get("reasons", ""):
                e += 1
            if kind == "other":
                other_words += 1
            elif kind == "echo" and decided_user:
                a += 1
            classified.append({"line": event["line"], "time": event["time"], "kind": kind,
                               "source": fields.get("source", ""), "heard": fields.get("heard", "")[:80]})
            event["kind"] = kind
        elif name == "request":
            heard = norm(fields.get("heard", ""))
            if not heard:
                continue
            said = turn_words(event["line"])
            echo = overlap(heard, said)
            user = max([match_score(heard, phrase) for phrase in user_phrases] or [0.0])
            if echo >= 0.5 and echo > user:
                # she took her own voice as input; did she then answer it?
                for later in events:
                    if later["line"] <= event["line"]:
                        continue
                    if later["event"] == "speak-start":
                        c += 1
                        classified.append({"line": event["line"], "time": event["time"],
                                           "kind": "self-request-answered",
                                           "heard": fields.get("heard", "")[:80]})
                        break
                    if later["event"] in ("request",):
                        break

    # B: INTERRUPTs caused by an echo line.
    by_line = {event["line"]: event for event in events}
    for index, event in enumerate(events):
        if event["event"] != "INTERRUPT":
            continue
        for back in events[max(0, index - 4):index][::-1]:
            if back["event"] == "words":
                if back.get("kind") == "echo":
                    b += 1
                break

    # D: real barge-ins she obeyed, of those VoxaTest made.
    d_total = d_obeyed = 0
    for record in reports:
        hearing = record.get("hearing") or {}
        if hearing.get("expect") != "yield" or hearing.get("user_onset_s") is None:
            continue
        d_total += 1
        words = [w for w in norm(record.get("prompt", "")) if w not in wake]
        reached = any(any(word in norm(record_heard) for word in words)
                      for record_heard in record.get("heard_records", []) if words)
        if hearing.get("yielded") and reached:
            d_obeyed += 1

    out = {
        "words_lines_while_speaking": sum(1 for x in classified if x.get("kind") in ("echo", "user", "other")),
        "echo_lines_decided_user_A": a,
        "echo_interrupts_B": b,
        "self_replies_C": c,
        "self_hearing_events": a + b + c,
        "real_barge_ins_obeyed_D": d_obeyed,
        "real_barge_ins_total": d_total,
        "reference_silent_while_speaking_E": e,
        "unmatched_words_lines": other_words,
        "detail": classified,
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
