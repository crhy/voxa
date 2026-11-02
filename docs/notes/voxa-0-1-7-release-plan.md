---
name: voxa-0-1-7-release-plan
description: "Voxa 0.1.7's main goal (end users can get Ollama models and the High face), the GitHub-hosting decision, and the later torrent update backend"
metadata:
  node_type: memory
  type: project
  originSessionId: 4c9169c2-e4e4-57e7-8197-531ede3203ea
  modified: 2026-10-24T00:18:24.003Z
---

On 2026-10-23 rhY set the main goal for Voxa 0.1.7: **end users must be able to download and install Ollama models and the High neural face from inside Voxa** (#112/#71 and #73/#110). Fixes ship only through the Spaced Bazaar Flatpak (remote `spaced-github`); test with that build and a clean profile, never a `--user` dev build. The differences found between rhY's machine and the Bazaar build, and the plan, are in `/Voxa/.voxa-spec/local-vs-bazaar-2026-10-23.md`. rhY's local Voxa data is in `~/voxa-local-backup-2026-10-23/`, and the dev face server was renamed to `~/voxa-neural.hidden-for-bazaar-test`.

**Decision (rhY, 2026-10-23):** host the High-face server as a prebuilt bundle in GitHub release downloads (split under GitHub's 2 GB per-file limit), not downloaded piecemeal from PyPI and Hugging Face. **Later:** a torrent file-sharing backend for Voxa and Spaced Linux that carries all their updates, Voxa included.

**Why:** today the face server only runs on rhY's machine (hard-coded `/home/rhy/voxa-neural`, a venv from another project), and the bundled Medium face packs (`faces-1`) are rejected for lacking `portrait_sha256`, so every Bazaar user sees a still face.

**How to apply:** do not run VMs while Strata is loaded; that combination froze the machine on 2026-10-23. Fix branch: `voxa-0.1.7-fixes` in `/Voxa`. Related: [[voxa-close-issues-with-summary]], [[voxa-spec-lives-in-github-issues]], [[delegate-coding-to-local-qwen]].

**Lock-ups on 2026-10-23 (twice; rhY had to hard-reset):** both were memory exhaustion with Strata loaded (it locks ~18 GB of RAM that cannot be swapped). Morning: Strata plus Spaced review VMs. 22:59: a gate test run; under xvfb GTK's GPU renderer took several GB outside the pytest process and kept growing. Fixed: `tests/conftest.py` sets `GSK_RENDERER=cairo` (suite ~1 GB, 75 s), and every gate runs through `/Voxa/.voxa-spec/tasks/agent/guarded-pytest.sh`, which kills the tests below 1.5 GB of free memory. Never run a VM or flatpak-builder while Strata is loaded (`~/voxa-vmtest/vmtest.sh` refuses).

**Working state:** `/Voxa/.voxa-spec/STATE-0.1.7.md` — read it first after a reset (queue fqueue2, VM test loop, A/B, what waits on rhY).
