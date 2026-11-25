"""Scratch analysis: how do the turn-wide text metrics look on the rc4 live lines?"""
import json
import re
import sys

sys.path.insert(0, ".")
from voxa.bargein import overlap, sounds_like, _words  # noqa: E402

FIELD = re.compile(r"(?P<key>heard|raw|source|confidence|reasons|seconds|speaking)=")


def parse(line):
    head, rest = line.split(" ", 1)
    hh, mm, ss = head.split(":")
    when = int(hh) * 3600 + int(mm) * 60 + int(ss)
    kind = rest.split(" ", 1)[0].rstrip(",")
    fields = {}
    for m in re.finditer(r"(\w+)=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)", rest):
        key, value = m.group(1), m.group(2)
        if len(value) > 1 and value[0] in "'\"" and value[-1] == value[0]:
            value = value[1:-1]
        fields.setdefault(key, value)
    return when, kind, fields


def text_of(fields):
    for key in ("text", "speaking"):
        if key in fields:
            return fields[key]
    return ""


def main():
    lines = open(".voxa-bench/live-rc4-0-8b/barge.log").read().splitlines()
    sec = [l for l in lines if l[:5] >= "18:20"]
    events = []
    for line in sec:
        when, kind, fields = parse(line)
        if kind in ("speak-start", "words", "INTERRUPT"):
            events.append((when, kind, fields))

    turns = []
    current = None
    for when, kind, fields in events:
        if kind != "speak-start":
            continue
        if current is None or when - current["start"] > 25:
            current = {"start": when, "pieces": []}
            turns.append(current)
        current["pieces"].append((when, text_of(fields)))

    rows = []
    for when, kind, fields in events:
        if kind != "words":
            continue
        idx = [i for i, tr in enumerate(turns) if tr["start"] <= when]
        cur = turns[idx[-1]] if idx else None
        prev = turns[idx[-2]] if len(idx) > 1 else None
        cur_text = " ".join(t for _, t in cur["pieces"]) if cur else ""
        prev_text = ""
        if prev and when - prev["start"] <= 10:
            prev_text = " ".join(t for _, t in prev["pieces"])
        turn_text = (cur_text + " " + prev_text).strip()
        heard = fields.get("heard", "")
        rows.append({
            "t": f"{when // 3600:02d}:{(when % 3600) // 60:02d}:{when % 60:02d}",
            "heard": heard,
            "src": fields.get("source", ""),
            "reasons": fields.get("reasons", "")[:40],
            "turn": turn_text,
            "overlap": round(overlap(heard, turn_text), 3),
            "sound": round(sounds_like(heard, turn_text), 3),
        })
    json.dump(rows, open(".voxa-bench/veto-analysis.json", "w"), indent=1)
    for r in rows:
        print(f"{r['t']} {r['src']:<9} ov={r['overlap']:.2f} sl={r['sound']:.2f} | {r['heard'][:70]}")
    print(len(rows))


main()
