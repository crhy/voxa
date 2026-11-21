---
name: voxa-hearing-profile
description: Voxa's echo canceller starts from a hearing profile saved per output/microphone pair, keeps adapting during use, and drops the profile when the pair or volume changes (issue #72)
metadata:
  type: project
---

Issue crhy/voxa#72 asks Voxa to hear only the user. Her canceller (`voxa/anc.py`) adapts while she listens; rhY's self-calibration comment (2026-10-06) requires "**Save the result per output/microphone pair**, so each speaker and microphone combination starts from a good filter instead of adapting from scratch; recalibrate automatically when the device or volume range changes" and "Keep adapting quietly during normal use, using the saved profile as the starting point".

**How it works now:** `voxa/anccal.py` keys a profile by sink and microphone (`profile_path`, names made filesystem-safe by `_safe`) under `$XDG_DATA_HOME/voxa/hearing/<sink>__<source>.json`. The payload is `{"delay", "state", "sink", "source", "level", "saved_at"}`; `level` is the loudest microphone level the filter was learned at. `voxa/audio.py:AudioCapture.apply_hearing_profile(sink, source, level)` loads it, `profile_is_current` rejects a profile from another pair or from a volume more than about two stops away, and `stop()` writes back what the running canceller learned (`save_live_state`) when it actually removed at least 3 dB. `_maybe_estimate_delay` carries the old weights over the rebuilt delay instead of starting blank. When the profile is stale, `window.py:_prepare_microphone` sets `audio.hearing_needs_calibration` and shows a status line; the calibration routine itself is not started by this task.

**How to apply:** keep the pair identity (sink name from `voxa/echo.py:default_monitor_source`, microphone from `settings.microphone_id`/`microphone_name`) as the profile key, and never let a saved filter make the residual worse — the regression test in `tests/test_hearing_profile.py` compares the saved-start and blank-start residual with `anc.erle_db` on the same synthetic pair. Still open in #72: the by-ear proof and published decibel figure, the full calibrate-until-it-stops-improving loop, and first-boot microphone and gain set-up (#76). Related: [[voxa-tell-before-live-audio-tests]], [[voxatest-never-lower-volume]], [[voxa-0-1-6-handoff]].
