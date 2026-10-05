from __future__ import annotations

from tools import measure_faces, reference_summary


def test_proportions_synthetic():
    lm = [(0.0, 0.0)] * 478
    lm[468] = (0.0, 0.0)
    lm[473] = (100.0, 0.0)
    lm[133] = (20.0, 0.0)
    lm[362] = (80.0, 0.0)
    lm[33] = (-20.0, 0.0)
    lm[263] = (120.0, 0.0)
    lm[145] = (0.0, -10.0)
    lm[159] = (0.0, 10.0)
    lm[374] = (100.0, -10.0)
    lm[386] = (100.0, 10.0)
    lm[105] = (0.0, -30.0)
    lm[107] = (20.0, -25.0)
    lm[70] = (-20.0, -25.0)
    lm[334] = (100.0, -30.0)
    lm[336] = (80.0, -25.0)
    lm[300] = (120.0, -25.0)
    lm[168] = (50.0, 0.0)
    lm[2] = (50.0, 40.0)
    lm[129] = (40.0, 40.0)
    lm[358] = (60.0, 40.0)
    lm[0] = (50.0, 55.0)
    lm[13] = (50.0, 62.0)
    lm[14] = (50.0, 62.0)
    lm[17] = (50.0, 70.0)
    lm[152] = (50.0, 120.0)
    lm[61] = (20.0, 55.0)
    lm[291] = (80.0, 55.0)
    lm[9] = (50.0, -30.0)
    lm[234] = (-30.0, 30.0)
    lm[454] = (130.0, 30.0)
    lm[172] = (0.0, 55.0)
    lm[397] = (100.0, 55.0)
    p = measure_faces.proportions(lm, 1.0, 1.0)
    expected = {
        "face_height": 1.5, "face_width": 1.6, "jaw_width": 1.0,
        "chin_height": 0.5, "nose_length": 0.4, "nose_width": 0.2,
        "mouth_width": 0.6, "upper_lip_height": 0.07,
        "lower_lip_height": 0.08, "eye_width": 0.4, "eye_height": 0.2,
        "eye_spacing": 0.6, "brow_height": 0.4, "brow_arch": 0.05,
        "philtrum_length": 0.15, "cheekbone_to_jaw_taper": 1.6,
        "face_height_over_width": 0.9375,
    }
    for k, v in expected.items():
        assert abs(p[k] - v) < 1e-6


def test_sanity_violations():
    good = {k: (lo + hi) / 2 for k, (lo, hi) in measure_faces.EXPECTED_RANGES.items()}
    assert measure_faces.sanity_violations(good) == []
    bad = dict(good)
    bad["nose_length"] = 0.0
    bad["brow_height"] = 0.0
    bad["face_height"] = 0.0
    assert len(measure_faces.sanity_violations(bad)) == 3


def test_median_per_person():
    photos = {
        "/Voxa/.voxa-spec/out/reference/photos/us_m/Alice/1.jpg":
            {"proportions": {"x": 1.0}, "colors": {"skin": [10.0, 10.0, 10.0]}},
        "/Voxa/.voxa-spec/out/reference/photos/us_m/Alice/2.jpg":
            {"proportions": {"x": 2.0}, "colors": {"skin": [20.0, 20.0, 20.0]}},
        "/Voxa/.voxa-spec/out/reference/photos/us_m/Alice/3.jpg":
            {"proportions": {"x": 10.0}, "colors": {"skin": [30.0, 30.0, 30.0]}},
        "/Voxa/.voxa-spec/out/reference/photos/us_m/Bob/1.jpg":
            {"proportions": {"x": 4.0}, "colors": {"skin": [40.0, 40.0, 40.0]}},
    }
    med = reference_summary.median_per_person(photos)
    assert med[("us_m", "Alice")]["proportions"]["x"] == 2.0
    assert med[("us_m", "Alice")]["colors"]["skin"] == [20.0, 20.0, 20.0]
    assert med[("us_m", "Bob")]["proportions"]["x"] == 4.0


def test_group_stats():
    persons = {
        ("us_m", "Alice"): {"proportions": {"x": 2.0}, "colors": {"skin": [10.0, 10.0, 10.0]}},
        ("us_m", "Bob"): {"proportions": {"x": 4.0}, "colors": {"skin": [30.0, 30.0, 30.0]}},
    }
    g = reference_summary.group_stats(persons)
    assert g["us_m"]["n_people"] == 2
    assert g["us_m"]["proportions"]["x"]["mean"] == 3.0
    assert g["us_m"]["proportions"]["x"]["sd"] == 1.0
    assert g["us_m"]["colors"]["skin"] == [20.0, 20.0, 20.0]
    assert g["us_m"]["people_used"] == ["Alice", "Bob"]
