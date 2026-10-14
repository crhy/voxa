"""Tests for the WebRTC lab: the echo-removed metric and the resumable batch runner.

The batch runner only needs GStreamer for jobs whose output is missing, so the skip path
is exercised without webrtcdsp; the run path is skipped when webrtcdsp is unavailable.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from tools import webrtc_lab


def test_erle_tail_db_identical_is_zero():
    mic = np.full(1000, 0.25, dtype=np.float32)
    assert webrtc_lab.erle_tail_db(mic, mic.copy()) == 0.0


def test_erle_tail_db_halved_energy_is_six_db():
    mic = np.full(1000, 0.4, dtype=np.float32)
    cleaned = mic * 0.5
    assert abs(webrtc_lab.erle_tail_db(mic, cleaned) - 10.0 * np.log10(4.0)) < 1e-6


def test_erle_tail_db_ignores_the_head():
    # head is loud noise, tail is a clean echo; only the tail share counts
    mic = np.full(1000, 0.4, dtype=np.float32)
    mic[:400] = 10.0
    cleaned = mic.copy()
    cleaned[400:] = mic[400:] * 0.5
    assert abs(webrtc_lab.erle_tail_db(mic, cleaned) - 10.0 * np.log10(4.0)) < 1e-6


def test_batch_skips_existing_outputs(tmp_path):
    out_a = tmp_path / "a.out.npy"
    out_b = tmp_path / "b.out.npy"
    np.save(out_a, np.zeros(8, dtype=np.float32))
    np.save(out_b, np.ones(8, dtype=np.float32))
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([
        {"mic": str(tmp_path / "missing.mic.npy"), "ref": str(tmp_path / "missing.ref.npy"),
         "out": str(out_a), "props": {}},
        {"mic": str(tmp_path / "missing2.mic.npy"), "ref": str(tmp_path / "missing2.ref.npy"),
         "out": str(out_b), "props": {}},
    ]))
    assert webrtc_lab._batch(str(jobs)) == 0
    # skipped jobs are left untouched and the missing inputs were never loaded
    assert np.array_equal(np.load(out_a), np.zeros(8, dtype=np.float32))
    assert np.array_equal(np.load(out_b), np.ones(8, dtype=np.float32))
    assert not (tmp_path / "missing.mic.npy").is_file()


def test_batch_processes_when_available(tmp_path):
    if not webrtc_lab.available():
        return
    rng = np.random.default_rng(1)
    ref = (0.1 * rng.standard_normal(2 * webrtc_lab.SR)).astype(np.float32)
    mic = ref * 0.5
    micp, refp, outp = tmp_path / "m.npy", tmp_path / "r.npy", tmp_path / "o.npy"
    np.save(micp, mic)
    np.save(refp, ref)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"mic": str(micp), "ref": str(refp), "out": str(outp), "props": {}}]))
    assert webrtc_lab._batch(str(jobs)) == 0
    assert outp.is_file()
