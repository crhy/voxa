from __future__ import annotations

from voxa.ui.face_motion import FaceMotion

SMOOTH = (
    "_breath",
    "mouthSmileLeft",
    "mouthSmileRight",
    "cheekSquintLeft",
    "cheekSquintRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
)


def _blink_onsets(m: FaceMotion, span: float, speaking: bool) -> list[float]:
    onsets: list[float] = []
    prev = 0.0
    i = 0
    while i * 0.01 <= span:
        t = i * 0.01
        v = m.pose(t, speaking).morphs["eyeBlinkLeft"]
        if v > 0.5 and prev <= 0.5:
            onsets.append(t)
        prev = v
        i += 1
    return onsets


def test_pose_is_deterministic() -> None:
    a = FaceMotion(7).pose(3.11, True, False, 0.6)
    b = FaceMotion(7).pose(3.11, True, False, 0.6)
    assert a.morphs == b.morphs
    assert a.head == b.head
    assert a.gaze == b.gaze
    assert FaceMotion(0).pose(0.0) == FaceMotion(0).pose(0.0)


def test_all_values_in_bounds() -> None:
    for i in range(4000):
        t = i * 0.03
        pose = FaceMotion(5).pose(t, i % 2 == 0, i % 3 == 0, (i % 10) / 10)
        for value in pose.morphs.values():
            assert 0.0 <= value <= 1.0
        for axis in pose.head:
            assert -30.0 <= axis <= 30.0
        for axis in pose.gaze:
            assert -20.0 <= axis <= 20.0


def test_blink_count_at_rest_in_band() -> None:
    onsets = _blink_onsets(FaceMotion(2), 120.0, False)
    assert 20 <= len(onsets) <= 45


def test_inter_blink_intervals_unique() -> None:
    onsets = _blink_onsets(FaceMotion(2), 120.0, False)
    gaps = [round(b - a, 4) for a, b in zip(onsets, onsets[1:], strict=False)]
    assert len(set(gaps)) == len(gaps)


def test_double_blinks_occur() -> None:
    onsets = _blink_onsets(FaceMotion(2), 120.0, False)
    doubles = sum(1 for a, b in zip(onsets, onsets[1:], strict=False) if b - a < 0.4)
    assert doubles >= 1


def test_smooth_channels_are_continuous() -> None:
    for speaking, energy in ((False, 0.0), (True, 0.8)):
        m = FaceMotion(4)
        prev = m.pose(0.0, speaking, False, energy)
        for i in range(1, 1900):
            t = i * 0.016
            cur = m.pose(t, speaking, False, energy)
            for key in SMOOTH:
                delta = abs(cur.morphs[key] - prev.morphs[key])
                assert delta <= 0.35
            prev = cur


def test_head_std_near_target() -> None:
    m = FaceMotion(3)
    ys = [m.pose(i * 0.05).head[0] for i in range(2000)]
    ps = [m.pose(i * 0.05).head[1] for i in range(2000)]
    rs = [m.pose(i * 0.05).head[2] for i in range(2000)]
    assert abs(_pstdev(ys) - m.head_std_yaw) <= 0.6
    assert abs(_pstdev(ps) - m.head_std_pitch) <= 0.6
    assert abs(_pstdev(rs) - m.head_std_roll) <= 0.6


def _pstdev(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def test_speaking_increases_motion() -> None:
    m = FaceMotion(2)
    rest = len(_blink_onsets(m, 120.0, False))
    speak = len(_blink_onsets(m, 120.0, True))
    assert speak > rest
    rest_yaw = [m.pose(i * 0.05).head[0] for i in range(1200)]
    speak_yaw = [m.pose(i * 0.05, True).head[0] for i in range(1200)]
    assert _pstdev(speak_yaw) > _pstdev(rest_yaw)


def test_gaze_mostly_near_centre() -> None:
    m = FaceMotion(3)
    near = sum(
        1 for i in range(2000)
        if abs(m.pose(i * 0.05).gaze[0]) < 3.0 and abs(m.pose(i * 0.05).gaze[1]) < 3.0
    )
    assert near / 2000 > 0.6


def test_smile_cheek_coupling() -> None:
    m = FaceMotion(3)
    for i in range(500):
        pose = m.pose(i * 0.1)
        smile = pose.morphs["mouthSmileLeft"]
        cheek = pose.morphs["cheekSquintLeft"]
        assert abs(cheek - m.cheek_ratio * smile) <= 1e-6


def test_nested_stats_are_consumed() -> None:
    stats = {
        "blink": {
            "interval_median_s": 2.0,
            "interval_p10_s": 0.5,
            "duration_p10_ms": 100.0,
            "duration_p90_ms": 200.0,
            "double_fraction": 0.2,
        },
        "gaze": {"fixation_p10_s": 1.0, "fixation_p90_s": 4.0},
        "head": {"yaw_std": 5.0, "pitch_std": 3.0, "roll_std": 2.0, "pitch_period_s": 6.0},
        "mouth": {"mouthSmile_mean": 0.3},
        "correlations": {"blink_on_gaze_shift_fraction": 0.7},
    }
    m = FaceMotion(1, stats=stats)
    assert m.blink_mean_speak == 2.0
    assert m.blink_mean_rest == 3.0
    assert m.blink_min == 0.5
    assert m.blink_dur_lo == 0.1
    assert m.blink_dur_hi == 0.2
    assert m.double_blink_p == 0.2
    assert m.fix_lo == 1.0
    assert m.fix_hi == 4.0
    assert m.head_std_yaw == 5.0
    assert m.head_std_pitch == 3.0
    assert m.head_std_roll == 2.0
    assert m.breath_period == 6.0
    assert m.gaze_shift_blink_p == 0.7
    assert m.smile_lo == 0.3
    assert m.smile_hi == 0.3


def test_implausible_section_is_ignored() -> None:
    stats = {
        "blink": {
            "plausible": False,
            "interval_median_s": 0.05,
            "interval_p10_s": 0.01,
            "duration_p10_ms": 700.0,
            "duration_p90_ms": 800.0,
            "double_fraction": 0.9,
        },
        "gaze": {"plausible": False, "fixation_p10_s": 0.0, "fixation_p90_s": 0.0},
        "head": {"plausible": False, "yaw_std": 999.0},
        "mouth": {"plausible": False, "mouthSmile_mean": 0.9},
        "correlations": {"plausible": False, "blink_on_gaze_shift_fraction": 0.0},
    }
    m = FaceMotion(1, stats=stats)
    assert m.blink_mean_speak == 2.5
    assert m.blink_mean_rest == 4.0
    assert m.blink_min == 1.0
    assert m.blink_dur_lo == 0.14
    assert m.blink_dur_hi == 0.22
    assert m.double_blink_p == 0.10
    assert m.fix_lo == 1.0
    assert m.fix_hi == 3.5
    assert m.head_std_yaw == 1.5
    assert m.smile_lo == 0.05
    assert m.smile_hi == 0.12
    assert m.gaze_shift_blink_p == 0.6


def test_implausible_blink_values_are_clamped() -> None:
    stats = {
        "blink": {
            "plausible": True,
            "interval_median_s": 0.05,
            "interval_p10_s": 0.01,
            "duration_p10_ms": 700.0,
            "duration_p90_ms": 800.0,
            "double_fraction": 1.5,
        }
    }
    m = FaceMotion(1, stats=stats)
    assert m.blink_mean_speak == 1.5
    assert m.blink_mean_rest == 1.5
    assert m.blink_min == 0.5
    assert m.blink_dur_lo == 0.45
    assert m.blink_dur_hi == 0.45
    assert m.double_blink_p == 1.0
