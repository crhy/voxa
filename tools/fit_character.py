"""Fit one Voxa character's MakeHuman face targets to its country's measured
reference (an amalgam, not a likeness of any single person).

Run with the analysis venv's Python (it has mediapipe, opencv, numpy):

    /home/rhy/voxa-characters/analysis-venv/bin/python fit_character.py <id> \
        [--passes N] [--step S] [--max-builds N]

Pipeline:
  1. Build the character (Blender/MPFB) and measure <id>-neutral.png with the
     SAME proportion function as P07 (tools/measure_faces.proportions).
  2. Error = sum over proportions of ((character - group mean) / max(group sd, 0.03))^2.
  3. Coordinate descent over the MakeHuman face-target axes: for each axis try
     current value +-step (clamped 0..1 per side), keep the best; default schedule
     is two passes at 0.2 then one pass at 0.1. Rebuild+remeasure per trial
     (~10 s each); measurements are cached keyed by the targets dict. Stop early
     when a whole pass improves the error by less than 2%.
  4. Colours: hair/brow from the group's measured colour, eye_color nearest of the
     available names by measured iris colour, skin_gain so measured skin moves
     70% of the way to the group's. Freckles/beauty-marks/acne settings are kept.
  5. Write the tuned spec back (original kept as <id>.before.json), rebuild, save
     fit/<id>-before.png and <id>-after.png, and append a line to fit/report.md.

If the face tracker finds no face in a render the character is skipped (reported),
never guessed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

# Reuse the exact P07 proportion + colour-sampling code (pure, no mediapipe).
sys.path.insert(0, "/Voxa/tools")
from measure_faces import proportions, sample_colors

REF_DIR = Path("/Voxa/.voxa-spec/out/reference")
SPECS_DIR = Path("/home/rhy/voxa-characters/specs")
BUILD_SCRIPT = "/home/rhy/voxa-characters/scripts/build_character.py"
MODEL = "/home/rhy/voxa-characters/analysis/models/face_landmarker.task"
AVATAR_DIR = Path("/home/rhy/.local/share/voxa/avatars")

GROUP_MAP = {
    "grace": "us_f", "jack": "us_m", "aoife": "ie_f", "seamus": "ie_m",
    "charlotte": "uk_f", "oliver": "uk_m", "valentina": "mx_f", "juan": "mx_m",
    "camille": "fr_f", "etienne": "fr_m", "greta": "de_f", "klaus": "de_m",
    "sakura": "jp_f", "hiroshi": "jp_m",
}

# (stem, positive-suffix, negative-suffix). suffix None => single-sided axis.
AXES = [
    ("eye-scale", "incr", None),
    ("eye-height2", "incr", None),
    ("mouth-lowerlip-volume", "incr", None),
    ("mouth-upperlip-volume", "incr", None),
    ("mouth-scale-horiz", "incr", "decr"),
    ("nose-scale-horiz", "incr", "decr"),
    ("nose-scale-vert", "incr", "decr"),
    ("nose-width2", "incr", "decr"),
    ("chin-width", "incr", "decr"),
    ("chin-height", "incr", "decr"),
    ("chin-bones", "incr", "decr"),
    ("chin-prominent", "incr", "decr"),
    ("cheek-bones", "incr", "decr"),
    ("cheek-volume", "incr", "decr"),
    ("head-oval", None, None),
    ("head-square", None, None),
    ("head-round", None, None),
    ("head-scale-horiz", "incr", "decr"),
    ("head-scale-vert", "incr", "decr"),
    ("eyebrows-trans", "up", "down"),
    ("eyebrows-angle", "up", "down"),
    ("eye-trans", "in", "out"),
]
AXIS_BY_STEM = {stem: (stem, pos, neg) for (stem, pos, neg) in AXES}

# Approximate iris colours (0..255) for the named eye colours the spec allows.
EYE_COLORS = {
    "blue": (60, 95, 150),
    "bluegreen": (55, 120, 120),
    "brown": (70, 45, 30),
    "brownlight": (125, 90, 55),
    "deepblue": (30, 45, 95),
    "green": (50, 95, 55),
    "grey": (110, 110, 110),
}


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def axis_bounds(stem):
    """(lo, hi) for an axis value: single-sided axes are 0..1, paired are -1..1."""
    _, pos, neg = AXIS_BY_STEM[stem]
    return (0.0, 1.0) if neg is None else (-1.0, 1.0)


def parse_targets(spec_targets):
    """Split a spec's targets dict into tunable axis values + fixed leftovers.

    Paired axes are stored as a signed value (positive => the +suffix side,
    negative => the -suffix side); single-sided axes store a 0..1 value.
    """
    axes = {stem: 0.0 for (stem, _, _) in AXES}
    fixed = {}
    for key, val in spec_targets.items():
        matched = False
        for (stem, pos, neg) in AXES:
            if pos is None and key == stem:
                axes[stem] = float(val)
                matched = True
                break
            if pos and key == f"{stem}-{pos}":
                axes[stem] = float(val)
                matched = True
                break
            if neg and key == f"{stem}-{neg}":
                axes[stem] = -float(val)
                matched = True
                break
        if not matched:
            fixed[key] = float(val)
    return axes, fixed


def targets_from_axes(axes, fixed):
    """Rebuild a spec targets dict from axis values + fixed leftovers."""
    out = dict(fixed)
    for (stem, pos, neg) in AXES:
        v = axes.get(stem, 0.0)
        if abs(v) < 1e-9:
            continue
        if pos is None:
            out[stem] = round(v, 4)
        elif v > 0:
            out[f"{stem}-{pos}"] = round(v, 4)
        else:
            out[f"{stem}-{neg}"] = round(-v, 4)
    return out


def error(props, group):
    """Sum of squared z-scores of the character's proportions vs the group."""
    total = 0.0
    for name, stats in group["proportions"].items():
        if name not in props:
            continue
        sd = max(stats["sd"], 0.03)
        z = (props[name] - stats["mean"]) / sd
        total += z * z
    return total


def build(spec, targets, out_dir):
    """Run Blender/MPFB for one spec+targets; return the neutral PNG path."""
    spec = dict(spec)
    spec["targets"] = targets
    tmp = Path(out_dir) / "_spec.json"
    tmp.write_text(json.dumps(spec, indent=1))
    cmd = [
        "nice", "-n", "12", "flatpak", "run", "--command=blender",
        "org.blender.Blender", "-b", "--python", BUILD_SCRIPT, "--",
        str(tmp), str(out_dir),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return Path(out_dir) / f"{spec['id']}-neutral.png"


_MARKER = None


def _get_marker():
    global _MARKER
    if _MARKER is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
        _MARKER = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            num_faces=1,
        ))
    return _MARKER


def measure(png):
    """Landmark-detect a render and return (proportions, colors) or None."""
    import cv2
    import mediapipe as mp
    marker = _get_marker()
    bgr = cv2.imread(str(png))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = marker.detect(img)
    if len(res.face_landmarks) != 1:
        return None
    lm = [(p.x, p.y) for p in res.face_landmarks[0]]
    return proportions(lm, rgb.shape[1], rgb.shape[0]), sample_colors(rgb, lm)


def nearest_eye(iris):
    if not iris:
        return "brown"
    best, best_d = "brown", 1e18
    for name, c in EYE_COLORS.items():
        d = sum((a - b) ** 2 for a, b in zip(iris, c, strict=True))
        if d < best_d:
            best, best_d = name, d
    return best


def apply_colors(spec, cols, group):
    """Return the spec unchanged: the colours sampled from the reference photos
    are not trustworthy (the "hair" patch lands on forehead skin), so only the
    face proportions are fitted and each character keeps its own colouring."""
    return dict(spec)


def descend(axes, fixed, group, measure_fn, schedule, max_builds, cache=None):
    """Pure coordinate descent over the face-target axes.

    measure_fn(targets_dict) -> proportions dict (or None if unmeasurable).
    Returns (axes, error, builds_used). Paired -incr/-decr axes are a single
    signed value, so they can never both be set at once. A caller may pass a
    pre-seeded cache keyed the same way to avoid re-measuring the start state.
    """
    cache = cache if cache is not None else {}

    def props_for(axes_state):
        targets = targets_from_axes(axes_state, fixed)
        key = tuple(sorted(targets.items()))
        if key in cache:
            return cache[key], 0
        p = measure_fn(targets)
        cache[key] = p
        return p, 1

    base, builds = props_for(axes)
    if base is None:
        return dict(axes), float("inf"), builds
    cur_err = error(base, group)
    cur_axes = dict(axes)
    for step in schedule:
        pass_start = cur_err
        for (stem, _, _) in AXES:
            lo, hi = axis_bounds(stem)
            cur = cur_axes.get(stem, 0.0)
            best_v, best_e = cur, cur_err
            for delta in (step, -step):
                trial = _clamp(cur + delta, lo, hi)
                if abs(trial - cur) < 1e-9:
                    continue
                trial_axes = dict(cur_axes)
                trial_axes[stem] = trial
                p, used = props_for(trial_axes)
                builds += used
                if p is None:
                    continue
                e = error(p, group)
                if e < best_e:
                    best_e, best_v = e, trial
                if builds >= max_builds:
                    break
            cur_axes[stem] = best_v
            cur_err = best_e
            if builds >= max_builds:
                break
        if pass_start <= 1e-9:
            break
        if (pass_start - cur_err) / pass_start < 0.02:
            break
        if builds >= max_builds:
            break
    return cur_axes, cur_err, builds


def fit(cid, schedule, max_builds):
    group_id = GROUP_MAP[cid]
    ref = json.loads((REF_DIR / "reference_faces.json").read_text())
    group = ref[group_id]
    spec = json.loads((SPECS_DIR / f"{cid}.json").read_text())

    axes, fixed = parse_targets(spec.get("targets", {}))
    build_dir = REF_DIR / "fit" / "builds" / cid
    build_dir.mkdir(parents=True, exist_ok=True)
    orig_targets = targets_from_axes(axes, fixed)

    # Build the original spec once to sample its colours.
    build(spec, orig_targets, build_dir)
    m0 = measure(build_dir / f"{cid}-neutral.png")
    if m0 is None:
        print(f"SKIP {cid}: no face found in render")
        return False
    cols0 = m0[1]

    # Colour-tune the spec once; descent optimises this colour-tuned render,
    # so the before/after errors are measured on the same render and are
    # monotonic by construction (descent only accepts improvements).
    spec_col = apply_colors(spec, cols0, group)

    def measure_fn(targets):
        png = build(spec_col, targets, build_dir)
        m = measure(png)
        return None if m is None else m[0]

    cache = {}
    base = measure_fn(orig_targets)
    if base is None:
        print(f"SKIP {cid}: no face found in colour-tuned render")
        return False
    cache[tuple(sorted(orig_targets.items()))] = base
    err0 = error(base, group)
    shutil.copyfile(str(build_dir / f"{cid}-neutral.png"),
                    str(REF_DIR / "fit" / f"{cid}-before.png"))

    cur_axes, cur_err, builds = descend(axes, fixed, group, measure_fn,
                                        schedule, max_builds - 2, cache)

    final_targets = targets_from_axes(cur_axes, fixed)
    final_spec = dict(spec_col)
    final_spec["targets"] = final_targets
    build(final_spec, final_targets, build_dir)
    m = measure(build_dir / f"{cid}-neutral.png")
    err_after = error(m[0], group) if m else float("inf")
    after_png = REF_DIR / "fit" / f"{cid}-after.png"
    if m is None or err_after > err0:
        # The final rebuild measured no better (or no face at all): keep the
        # ORIGINAL targets; the before render already shows that state.
        final_targets = orig_targets
        final_spec["targets"] = final_targets
        err_after = err0
        shutil.copyfile(str(REF_DIR / "fit" / f"{cid}-before.png"),
                        str(after_png))
    else:
        shutil.copyfile(str(build_dir / f"{cid}-neutral.png"), str(after_png))

    before_props = base
    after_props = m[0] if m and err_after < err0 else before_props
    deltas = []
    for name in group["proportions"]:
        if name in before_props and name in after_props:
            sd = max(group["proportions"][name]["sd"], 0.03)
            deltas.append((abs(after_props[name] - before_props[name]) / sd, name))
    deltas.sort(reverse=True)
    changed = [f"{n2}:{before_props[n2]:.3f}->{after_props[n2]:.3f}"
               for _, n2 in deltas[:5]]

    (SPECS_DIR / f"{cid}.before.json").write_text(
        (SPECS_DIR / f"{cid}.json").read_text())
    (SPECS_DIR / f"{cid}.json").write_text(json.dumps(final_spec, indent=1))

    line = (f"| {cid} | {group_id} | {err0:.2f} -> {err_after:.2f} | "
            f"{', '.join(changed)} |\n")
    report = REF_DIR / "fit" / "report.md"
    if not report.exists():
        report.write_text("# Fit report\n\n| id | group | error before -> after | top-5 changed proportions |\n")
    report.open("a").write(line)
    print(f"FIT {cid}: err {err0:.2f} -> {err_after:.2f}, builds={builds + 2}")
    return True


def main():
    args = sys.argv[1:]
    cid = args[0]
    passes, step = None, None
    max_builds = 10 ** 9
    i = 1
    while i < len(args):
        if args[i] == "--passes":
            passes = int(args[i + 1])
            i += 2
        elif args[i] == "--step":
            step = float(args[i + 1])
            i += 2
        elif args[i] == "--max-builds":
            max_builds = int(args[i + 1])
            i += 2
        else:
            i += 1
    if passes is not None and step is not None:
        schedule = [step] * passes
    else:
        schedule = [0.2, 0.2, 0.1]
    ok = fit(cid, schedule, max_builds)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
