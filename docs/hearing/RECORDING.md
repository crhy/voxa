# Recording a real Voxa hearing session (task r01, crhy/voxa#72)

One guided ~10-minute terminal session replaces the synthetic clip set the hearing gates measure
(pair_accuracy is over 3 pairs today - one clip swings a score). rhY records ONCE; the tool makes
that one session count.

## One command to start

```
cd /Voxa && .venv/bin/python3 tools/record_session.py --prepare   # once, online: fetches the CC0
                                                                # music into /home/rhy/voxa-bench-data/music/
                                                                # (URL + SHA-256 in music.json) and
                                                                # renders Voxa's lines with her own TTS
.venv/bin/python3 tools/record_session.py                        # the session itself (~10 minutes)
```

The tool is GTK-free and terminal-only. It tells rhY before anything opens the microphone: run it
only when the room can be quiet for 10 minutes.

## What rhY will be asked to do (17 takes, printed one line at a time)

- 2 takes reading a line alone in the quiet room (the reference "user" clips);
- 2 takes listening to Voxa speak alone (her real TTS voice - the echo the bench must cancel);
- 2 takes with only the CC0 music playing (Aysu - Sinusitis, Cold Unity CUM 131, CC0);
- 3 barge-in takes: Voxa reads a long line, rhY interrupts ~4 s in with "stop", "wait", or a real
  request ("Voxa, what time is it?");
- 2 takes talking OVER the music;
- 2 back-channel takes: quietly "mm-hm" / "okay" ONCE while Voxa talks;
- 2 takes reading a line two or three meters from the mic;
- 2 noise takes: one cough; one knock/door/keyboard.

Each take: the line is printed, rhY presses ENTER, the tool counts down 3-2-1, then records
~5-25 s. Nothing to press during a take.

## What is recorded

- 48 kHz WAV per take: the default microphone AND the exact playback reference, sample-aligned -
  the sink's monitor source (`parec --device <sink>.monitor`, like tools/anc_lab.py) or, if the
  sound server has none, the samples the tool played itself (`--ref synthetic`);
- one JSON per take: what played (file + SHA-256 + exact start sample), rhY's prompted text, the
  expected label (user/echo/noise/bargein/backchannel), devices, sample rates, the measured
  playback->mic delay by cross-correlation, and the system volume AS READ - the tool never lowers,
  sets or mutes the volume (rhY's rule);
- everything it opened (recorders, players) is closed when each take and the session ends.

Output lands in `/home/rhy/voxa-recordings/<date>/<take-id>/`.

## Then make it a bench suite

```
.venv/bin/python3 tools/record_session.py --to-bench /home/rhy/voxa-recordings/<date>
```

writes `/home/rhy/voxa-bench-data/real/<clip>/` - mic.wav + ref.wav at 16 kHz int16 mono plus
truth.json (id, scenario, expect mapped from the label, plus the recording metadata) - the exact
format voxa/bench's runner.load_clip reads, so `python -m voxa.bench run --suite real` can measure
the real room once the bench lands on main.
