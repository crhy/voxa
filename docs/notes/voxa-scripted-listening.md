---
name: voxa-scripted-listening
description: Live tests switch Voxa's Active on themselves - the start-listening/stop-listening app actions, the listening.json state file, and the VoxaTest preflight that never plays into a deaf Voxa (issue #130)
---

Voxa never starts listening on launch (privacy) — that is exactly why v09's live run started while she sat at "Press Active to start listening" and every case came back `not_heard`. Since task v10 (2026-11-25) live tests press Active themselves:

**The app actions** (voxa/application.py, voxa/window.py, voxa/listening.py)

```sh
# while Voxa is running (flatpak shares the session bus, so the host command reaches her):
gapplication action io.github.crhy.voxa start-listening
gapplication action io.github.crhy.voxa stop-listening
```

Verified on rhY's machine 2026-11-25 against a real running Voxa: `start-listening` performs the whole Active-button path
(checks included - if the brain or mic is not ready it says why instead of faking it), `stop-listening` is the same kill
switch as turning listening off, and `quit` still quits. With Voxa NOT running both plain `gapplication` and
`flatpak run --command=gapplication io.github.crhy.voxa action ...` fail with `ServiceUnknown` (there is no .service file to
activate her), so a live run must launch Voxa first (`flatpak run io.github.crhy.voxa`) - the preflight names that reason
and refuses before playing anything. Never fake-listen; never play into a deaf Voxa.

**The state file** - written from the same code path the button uses, on every change:

```
$XDG_STATE_HOME/voxa/listening.json   {"active": bool, "since": epoch, "reason": str}
```

Host Voxa: `~/.local/state/voxa/listening.json`. Flatpak Voxa: `~/.var/app/io.github.crhy.voxa/.local/state/voxa/listening.json`
(that is what `XDG_STATE_HOME` is inside the sandbox - verified with `flatpak run --command=printenv`). Quitting while
listening rewrites it with `reason: "Voxa closed"` so it can never claim she is active afterwards.

**VoxaTest live runs** (voxatest/listening.py; wired in voxatest/__main__.py `run`, the GUI preflight and `voxatest doctor`):
before the first case VoxaTest sends `start-listening` (plain `gapplication` while she owns her D-Bus name, otherwise the
flatpak form; `VOXATEST_VOXA_ACTION_CMD` overrides), then polls listening.json up to 60 s for `active: true`. Not active:
refuse, naming the reason, exit non-zero, nothing played (verified live against the 0.1.9 flatpak, which lacks the action:
`Unknown action 'start-listening'` -> refused, exit 2, no report, no sound). It rechecks before every case; if she went
inactive mid-run it re-sends `start-listening` once and stops the run with that reason if she stays deaf. At the end it
sends `stop-listening` only if VoxaTest was the one that started her. `voxatest doctor` gained the `scripted listening`
check (PASS/WARN/FAIL; SKIP in `--dry-run`). No v07-style live task notes template exists in the fleet tasks; this note is
where the live-run commands live.

Related: [[voxa-tell-before-live-audio-tests]], [[voxatest-never-lower-volume]], [[voxatest-clean-up-after-each-run]].
