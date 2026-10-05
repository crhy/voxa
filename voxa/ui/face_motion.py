"""Deterministic idle + speaking motion model for Voxa's talking face.

Pure Python, no per-frame state: :meth:`FaceMotion.pose` computes a full
:class:`FacePose` from ``(seed, t, flags)`` alone, so it can be sampled at any
time in any order.  The model layers the natural behaviours described in
``docs/FACE_NATURALNESS.md`` (blinks, gaze fixations/saccades, head drift and
nods, brow emphasis, resting smile, breathing) on top of the ARKit-style blend
shapes the renderer already drives.

Measured values from ``voxa/ui/assets/face_stats.json`` (when present) override
the rule-of-thumb defaults; without them the bracketed defaults in the task
spec are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .speech_motion import speech_level


def _flatten_stats(stats: dict | None) -> dict:
    """Map the nested layout from ``face_stats.py`` onto this model's flat keys.

    If the dict already uses flat keys (no ``blink``/``head`` sections) it is
    returned unchanged.
    """
    if not stats:
        return {}
    if not any(k in stats for k in ("blink", "head", "gaze", "mouth", "correlations")):
        return stats

    def _section(key: str) -> dict:
        sec = stats.get(key, {})
        if sec.get("plausible") is False:
            return {}
        return sec

    blink = _section("blink")
    gaze = _section("gaze")
    head = _section("head")
    mouth = _section("mouth")
    corr = _section("correlations")
    out: dict = {}
    if blink:
        med = blink.get("interval_median_s")
        if med is not None:
            out["blink_mean_speak"] = med
            out["blink_mean_rest"] = med * 1.5
        if blink.get("interval_p10_s") is not None:
            out["blink_min"] = blink["interval_p10_s"]
        if blink.get("duration_p10_ms") is not None:
            out["blink_dur_lo"] = blink["duration_p10_ms"] / 1000.0
        if blink.get("duration_p90_ms") is not None:
            out["blink_dur_hi"] = blink["duration_p90_ms"] / 1000.0
        if blink.get("double_fraction") is not None:
            out["double_blink_p"] = blink["double_fraction"]
    if gaze:
        if gaze.get("fixation_p10_s") is not None:
            out["fix_lo"] = gaze["fixation_p10_s"]
        if gaze.get("fixation_p90_s") is not None:
            out["fix_hi"] = gaze["fixation_p90_s"]
    if corr.get("blink_on_gaze_shift_fraction") is not None:
        out["gaze_shift_blink_p"] = corr["blink_on_gaze_shift_fraction"]
    for src, dst in (("yaw_std", "head_std_yaw"), ("pitch_std", "head_std_pitch"), ("roll_std", "head_std_roll")):
        if head.get(src) is not None:
            out[dst] = head[src]
    if head.get("pitch_period_s") is not None:
        out["breath_period"] = head["pitch_period_s"]
    if mouth.get("mouthSmile_mean") is not None:
        out["smile_lo"] = mouth["mouthSmile_mean"]
        out["smile_hi"] = mouth["mouthSmile_mean"]
    return out


@dataclass(slots=True)
class FacePose:
    """A single face pose: blend-shape morphs plus head and gaze rotation."""

    morphs: dict[str, float]
    head: tuple[float, float, float]
    gaze: tuple[float, float]


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth (cosine-like) ease between ``edge0`` and ``edge1``."""
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hash(index: int, salt: int, seed: int) -> int:
    """Stable 32-bit-ish hash of ``(index, salt, seed)``."""
    h = (index * 2654435761 + salt * 40503 + seed * 2246822519 + 0x9E3779B9)
    h &= 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= h >> 16
    return h & 0xFFFFFFFF


def _unit(index: int, salt: int, seed: int) -> float:
    """Deterministic float in [0, 1) from a hash."""
    return (_hash(index, salt, seed) % 1_000_000) / 1_000_000.0


def _range(index: int, salt: int, seed: int, lo: float, hi: float) -> float:
    return lo + (hi - lo) * _unit(index, salt, seed)


def _signed_noise(t: float, salt: int, seed: int, cycles: float) -> float:
    """Smooth summed-noise in [-1, 1] with ``cycles`` full cycles per second."""
    if cycles <= 0.0:
        return 0.0
    phase = t * cycles * math.tau
    base = math.sin(phase)
    slow = math.sin(phase * 0.5 + 0.7)
    fast = math.sin(phase * 2.0 + 1.3)
    return _clamp((0.6 * base + 0.3 * slow + 0.1 * fast) / 0.48, -2.0, 2.0)


class FaceMotion:
    """Natural face motion: blinks, gaze, head, brows, smile and breathing."""

    def __init__(self, seed: int = 0, stats: dict | None = None) -> None:
        self.seed = int(seed)
        s = _flatten_stats(stats)

        self.blink_mean_rest = _clamp(float(s.get("blink_mean_rest", 4.0)), 1.5, 10.0)
        self.blink_mean_speak = _clamp(float(s.get("blink_mean_speak", 2.5)), 1.5, 10.0)
        self.blink_min = _clamp(float(s.get("blink_min", 1.0)), 0.5, 10.0)
        self.blink_dur_lo = _clamp(float(s.get("blink_dur_lo", 0.14)), 0.08, 0.45)
        self.blink_dur_hi = _clamp(float(s.get("blink_dur_hi", 0.22)), 0.08, 0.45)
        self.double_blink_p = _clamp(float(s.get("double_blink_p", 0.10)), 0.0, 1.0)
        if self.blink_dur_lo > self.blink_dur_hi:
            self.blink_dur_lo, self.blink_dur_hi = self.blink_dur_hi, self.blink_dur_lo

        self.fix_lo = float(s.get("fix_lo", 1.0))
        self.fix_hi = float(s.get("fix_hi", 3.5))
        self.glance_lo = float(s.get("glance_lo", 8.0))
        self.glance_hi = float(s.get("glance_hi", 15.0))
        self.micro_lo = float(s.get("micro_lo", 0.2))
        self.micro_hi = float(s.get("micro_hi", 0.5))
        self.gaze_shift_blink_p = float(s.get("gaze_shift_blink_p", 0.6))

        self.head_std_yaw = float(s.get("head_std_yaw", 1.5))
        self.head_std_pitch = float(s.get("head_std_pitch", 1.0))
        self.head_std_roll = float(s.get("head_std_roll", 0.7))
        self.head_speak_mult = float(s.get("head_speak_mult", 2.0))
        self.nod_lo = float(s.get("nod_lo", 2.0))
        self.nod_hi = float(s.get("nod_hi", 4.0))
        self.head_follow_p = float(s.get("head_follow_p", 0.3))
        self.head_lag = float(s.get("head_lag", 0.12))

        self.brow_lo = float(s.get("brow_lo", 0.2))
        self.brow_hi = float(s.get("brow_hi", 0.5))

        self.smile_lo = float(s.get("smile_lo", 0.05))
        self.smile_hi = float(s.get("smile_hi", 0.12))
        self.smile_listen = float(s.get("smile_listen", 0.25))
        self.cheek_ratio = float(s.get("cheek_ratio", 0.4))

        self.breath_period = float(s.get("breath_period", 4.0))

    # ---- blink schedule -------------------------------------------------
    def _blink_closures(self, t: float, speaking: bool) -> list[tuple[float, float]]:
        """Return ``(start, duration)`` for every blink closure up to ``t``."""
        mean = self.blink_mean_speak if speaking else self.blink_mean_rest
        closures: list[tuple[float, float]] = []
        i = 0
        cur = _range(i, 11, self.seed, 0.8, mean * 0.9)
        while cur <= t + 0.5:
            dur = _range(i, 12, self.seed, self.blink_dur_lo, self.blink_dur_hi)
            closures.append((cur, dur))
            if _unit(i, 13, self.seed) < self.double_blink_p:
                gap = _range(i, 14, self.seed, 0.15, 0.25)
                dur2 = _range(i, 15, self.seed, self.blink_dur_lo, self.blink_dur_hi)
                closures.append((cur + dur + gap, dur2))
            interval = max(self.blink_min, mean * _range(i, 16, self.seed, 0.5, 1.5))
            cur += interval
            i += 1
        return closures

    def _blink_amount(self, t: float, start: float, dur: float) -> float:
        """Fast close (40%) then slower open (60%), eased."""
        if t < start or t >= start + dur:
            return 0.0
        frac = (t - start) / dur
        if frac < 0.4:
            return _smoothstep(0.0, 0.4, frac)
        return 1.0 - _smoothstep(0.4, 1.0, frac)

    # ---- gaze schedule --------------------------------------------------
    def _gaze_at(self, t: float, thinking: bool) -> tuple[float, float, float, float]:
        """Return (gx, gy, shift_deg, fixation_start) for the fixation at ``t``."""
        lo, hi = self.fix_lo, self.fix_hi
        i = 0
        cur = 0.0
        while True:
            span = _range(i, 21, self.seed, lo, hi)
            if cur + span > t:
                break
            cur += span
            i += 1
        fix_start = cur
        away = _unit(i, 22, self.seed)
        think_boost = 0.35 if thinking else 0.0
        is_glance = away < (0.18 + think_boost)
        if is_glance:
            mag = _range(i, 23, self.seed, self.glance_lo, self.glance_hi)
            ang = _range(i, 24, self.seed, 0.0, math.tau)
            gx = mag * math.cos(ang)
            gy = mag * math.sin(ang) * 0.6
        else:
            gx = _range(i, 25, self.seed, -2.0, 2.0)
            gy = _range(i, 26, self.seed, -2.0, 2.0)
        drift = _range(i, 27, self.seed, self.micro_lo, self.micro_hi)
        gx += _signed_noise(t, 31, self.seed, 0.5) * drift
        gy += _signed_noise(t, 32, self.seed, 0.5) * drift
        prev = self._gaze_center(i - 1) if i > 0 else (0.0, 0.0)
        shift = math.hypot(gx - prev[0], gy - prev[1])
        return gx, gy, shift, fix_start

    def _gaze_center(self, i: int) -> tuple[float, float]:
        if i < 0:
            return (0.0, 0.0)
        away = _unit(i, 22, self.seed)
        if away < 0.18:
            mag = _range(i, 23, self.seed, self.glance_lo, self.glance_hi)
            ang = _range(i, 24, self.seed, 0.0, math.tau)
            return (mag * math.cos(ang), mag * math.sin(ang) * 0.6)
        return (_range(i, 25, self.seed, -2.0, 2.0), _range(i, 26, self.seed, -2.0, 2.0))

    # ---- public API -----------------------------------------------------
    def pose(self, t: float, speaking: bool = False, listening: bool = False, energy: float = 0.0) -> FacePose:
        if t < 0.0:
            t = 0.0
        energy = _clamp(energy, 0.0, 1.0)
        if speaking and energy <= 0.0:
            energy = speech_level(t, self.seed)
        thinking = speaking and energy < 0.35

        morphs: dict[str, float] = {}

        # Breathing: slow cycle, exposed as _breath in 0..1.
        breath = 0.5 + 0.5 * math.sin(t / self.breath_period * math.tau - math.tau * 0.25)
        breath = _clamp(breath, 0.0, 1.0)
        morphs["_breath"] = breath

        # Gaze.
        gx, gy, shift, fix_start = self._gaze_at(t, thinking)
        gaze = (gx, gy)

        # Blink closures (gaze-shift-triggered blinks fold into the schedule).
        closures = self._blink_closures(t, speaking)
        blink_l = 0.0
        blink_r = 0.0
        squint_after = 0.0
        for start, dur in closures:
            amt = self._blink_amount(t, start, dur)
            if amt > blink_l:
                blink_l = amt
            asym = _range(int(start * 1000), 41, self.seed, 0.96, 1.0)
            if amt * asym > blink_r:
                blink_r = amt * asym
            end = start + dur
            if end <= t < end + 0.3:
                glow = 0.08 * (1.0 - (t - end) / 0.3)
                if glow > squint_after:
                    squint_after = glow
        # Gaze-shift-triggered blink (60% of large shifts, just after the saccade).
        if shift > 6.0 and _unit(int(fix_start * 1000), 42, self.seed) < self.gaze_shift_blink_p:
            since = t - fix_start
            if 0.0 <= since < 0.16:
                shift_blink = self._blink_amount(t, fix_start, 0.16)
                if shift_blink > blink_l:
                    blink_l = shift_blink
                if shift_blink > blink_r:
                    blink_r = shift_blink
        morphs["eyeBlinkLeft"] = _clamp(blink_l, 0.0, 1.0)
        morphs["eyeBlinkRight"] = _clamp(blink_r, 0.0, 1.0)

        # eyeLook* morphs follow gaze so lids track the eyes.
        look = _clamp(abs(gx) / 15.0, 0.0, 1.0)
        look_up = _clamp(gy / 15.0, -1.0, 1.0)
        morphs["eyeLookOutLeft"] = _clamp(look if gx < 0 else 0.0, 0.0, 1.0)
        morphs["eyeLookInLeft"] = _clamp(look if gx > 0 else 0.0, 0.0, 1.0)
        morphs["eyeLookOutRight"] = _clamp(look if gx > 0 else 0.0, 0.0, 1.0)
        morphs["eyeLookInRight"] = _clamp(look if gx < 0 else 0.0, 0.0, 1.0)
        morphs["eyeLookUpLeft"] = _clamp(look_up if look_up > 0 else 0.0, 0.0, 1.0)
        morphs["eyeLookUpRight"] = _clamp(look_up if look_up > 0 else 0.0, 0.0, 1.0)
        morphs["eyeLookDownLeft"] = _clamp(-look_up if look_up < 0 else 0.0, 0.0, 1.0)
        morphs["eyeLookDownRight"] = _clamp(-look_up if look_up < 0 else 0.0, 0.0, 1.0)

        # Squint afterglow + smile coupling added below.
        base_squint = squint_after

        # Head: summed smooth noise, ~2x while speaking.
        mult = self.head_speak_mult if speaking else 1.0
        yaw = _signed_noise(t, 51, self.seed, 0.15) * self.head_std_yaw * mult
        pitch = _signed_noise(t, 52, self.seed, 0.12) * self.head_std_pitch * mult
        roll = _signed_noise(t, 53, self.seed, 0.10) * self.head_std_roll * mult
        # Breathing adds a tiny pitch cue.
        pitch += (breath - 0.5) * 0.6
        # Nods on energy peaks while speaking (continuous bump per peak).
        if speaking and energy > 0.5:
            peak = int(t * 2.0)
            nod = _range(peak, 61, self.seed, self.nod_lo, self.nod_hi)
            env = max(0.0, math.sin((t * 2.0 - peak) * math.pi))
            pitch -= nod * _smoothstep(0.5, 1.0, energy) * env
        # Head follows 30% of large gaze shifts, trailing the eyes by ~120 ms.
        if shift > 6.0 and _unit(int(fix_start * 1000), 43, self.seed) < self.head_follow_p:
            follow = _smoothstep(fix_start + self.head_lag, fix_start + self.head_lag + 0.3, t)
            yaw += gx * 0.15 * follow
            pitch += gy * 0.10 * follow
        head = (yaw, pitch, roll)

        # Brows rise on speech energy peaks (emphasis).
        brow = 0.0
        if speaking and energy > 0.4:
            brow = _range(int(t * 5), 71, self.seed, self.brow_lo, self.brow_hi)
            brow *= _smoothstep(0.4, 1.0, energy)
        asym_brow = _range(int(t * 1000), 72, self.seed, 1.10, 1.20)
        morphs["browInnerUp"] = _clamp(brow, 0.0, 1.0)
        morphs["browOuterUpLeft"] = _clamp(brow * 0.6, 0.0, 1.0)
        morphs["browOuterUpRight"] = _clamp(brow * 0.6 * asym_brow, 0.0, 1.0)

        # Expression: faint resting smile that breathes; warmer while listening.
        smile_base = _range(0, 81, self.seed, self.smile_lo, self.smile_hi)
        if listening:
            smile_base = max(smile_base, self.smile_listen * (0.6 + 0.4 * breath))
        smile = smile_base * (0.85 + 0.15 * breath)
        smile = _clamp(smile, 0.0, 1.0)
        morphs["mouthSmileLeft"] = smile
        morphs["mouthSmileRight"] = _clamp(smile * _range(0, 82, self.seed, 0.95, 1.05), 0.0, 1.0)
        cheek = smile * self.cheek_ratio
        morphs["cheekSquintLeft"] = _clamp(cheek, 0.0, 1.0)
        morphs["cheekSquintRight"] = _clamp(cheek * _range(0, 83, self.seed, 0.95, 1.05), 0.0, 1.0)

        # Squint afterglow coupled onto the eye squint units.
        morphs["eyeSquintLeft"] = _clamp(base_squint, 0.0, 1.0)
        morphs["eyeSquintRight"] = _clamp(base_squint, 0.0, 1.0)

        return FacePose(morphs=morphs, head=head, gaze=gaze)
