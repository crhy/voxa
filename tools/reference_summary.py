"""Aggregate per-photo measurements into per-group reference faces.

Per person: median across their photos. Per group: mean and standard
deviation across people (so one person with many photos does not dominate).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REF_DIR = Path("/Voxa/.voxa-spec/out/reference")
DOCS_DIR = Path("/Voxa/docs")


def _person_key(path):
    """photos/<group>/<name>/N.jpg -> (group, name)."""
    parts = Path(path).parts
    i = parts.index("photos")
    return parts[i + 1], parts[i + 2]


def median_per_person(photo_measurements):
    """Median of each numeric field across one person's photos.

    photo_measurements: {path: {"proportions": {...}, "colors": {...}}}
    Returns {(group, name): {"proportions": {...}, "colors": {...}}}.
    """
    per_person = {}
    for path, m in photo_measurements.items():
        per_person.setdefault(_person_key(path), []).append(m)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    out = {}
    for key, ms in per_person.items():
        props = {}
        names = set()
        for m in ms:
            names.update(m["proportions"])
        for n in sorted(names):
            props[n] = med([m["proportions"].get(n) for m in ms])
        colors = {}
        cnames = set()
        for m in ms:
            cnames.update(k for k, v in m["colors"].items() if v is not None)
        for c in sorted(cnames):
            chans = [[m["colors"][c][i] for m in ms if m["colors"].get(c) is not None]
                      for i in range(3)]
            colors[c] = [round(statistics.median(ch), 1) for ch in chans if ch]
        out[key] = {"proportions": props, "colors": colors}
    return out


def group_stats(person_medians):
    """Mean and sd across people for each proportion and colour channel."""
    groups = {}
    for (group, _name), m in person_medians.items():
        groups.setdefault(group, []).append(m)

    out = {}
    for group, ms in groups.items():
        props = {}
        names = set()
        for m in ms:
            names.update(k for k, v in m["proportions"].items() if v is not None)
        for n in sorted(names):
            vals = [m["proportions"][n] for m in ms if m["proportions"].get(n) is not None]
            if vals:
                sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                props[n] = {"mean": round(statistics.mean(vals), 4),
                            "sd": round(sd, 4)}
        colors = {}
        for c in ("skin", "lips", "iris", "brow", "hair"):
            chans = [[m["colors"][c][i] for m in ms
                      if m["colors"].get(c) is not None and len(m["colors"][c]) > i]
                     for i in range(3)]
            if all(chans):
                colors[c] = [round(statistics.mean(ch), 1) for ch in chans]
        out[group] = {
            "n_people": len(ms),
            "proportions": props,
            "colors": colors,
            "people_used": sorted(name for (g, name) in person_medians if g == group),
        }
    return out


def _write_markdown(summary, path):
    lines = ["# Reference face amalgams\n"]
    for group in sorted(summary):
        s = summary[group]
        lines.append(f"\n## {group} ({s['n_people']} people)\n")
        lines.append("| proportion | mean | sd |")
        lines.append("|---|---|---|")
        for n, v in s["proportions"].items():
            lines.append(f"| {n} | {v['mean']} | {v['sd']} |")
        for c, rgb in s["colors"].items():
            lines.append(f"\n**{c}**: rgb({rgb[0]}, {rgb[1]}, {rgb[2]})")
        lines.append(f"\npeople: {', '.join(s['people_used'])}")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    measurements = json.loads((REF_DIR / "measurements.json").read_text())
    persons = median_per_person(measurements)
    groups = group_stats(persons)
    n_photos = {}
    for path in measurements:
        g, _ = _person_key(path)
        n_photos[g] = n_photos.get(g, 0) + 1
    for g in groups:
        groups[g]["n_photos"] = n_photos.get(g, 0)

    (REF_DIR / "reference_faces.json").write_text(json.dumps(groups, indent=1))
    _write_markdown(groups, REF_DIR / "reference_faces.md")
    (DOCS_DIR / "reference_faces.json").write_text((REF_DIR / "reference_faces.json").read_text())
    (DOCS_DIR / "reference_faces.md").write_text((REF_DIR / "reference_faces.md").read_text())
    for g in sorted(groups):
        s = groups[g]
        print(f"{g}: {s['n_people']} people, {s['n_photos']} photos")


if __name__ == "__main__":
    main()
