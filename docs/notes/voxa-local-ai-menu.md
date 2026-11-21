---
name: voxa-local-ai-menu
description: Voxa lists installed Ollama models by reading Ollama's own model folder, so the picker is never empty just because the server is stopped (crhy/voxa#27).
metadata:
  type: project
---

Installed Ollama models are listed from disk, not from the server. `voxa/ollama.py: ollama_models_dir()` follows Ollama's own rule (`$OLLAMA_MODELS`, else `~/.ollama/models`) and `disk_models()` walks that folder's `manifests/` tree — one JSON manifest per tag, the content-addressed `blobs/` beside it carries no names. It never touches the network and never raises: a missing, empty or unreadable store is simply `[]`.

**The manifest path is `<registry>/<library>/<name>/<tag>`, not `<family>/<name>/<tag>`.** Ollama 0.32 on rhY's machine stores `manifests/registry.ollama.ai/library/qwen2.5/0.5b` and `manifests/hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/IQ3_S`; `ollama list` shows `qwen2.5:0.5b` and `hf.co/ISTA-DASLab/…:IQ3_S`, so the default registry and the default `library` are dropped and everything else stays. That rule lives once, in `voxa/ollama.py: ollama_display_name()`, and `voxa/backup.py: _ollama_display_name()` (the issue #26 backup inventory) now calls it — a second parser would drift and the picker would offer names `ollama pull` cannot use.

`voxa/window.py: _refresh_ollama_models` seeds the picker with that disk list immediately for the Ollama backend, then runs the existing `wait_for_models` HTTP path; `voxa/modelwait.py: models_to_show(disk, live)` decides the outcome — a live list always wins, a server that never answered keeps the disk list, and "No models available" is only reached when the disk holds nothing either. The status line "The AI server (Ollama) is offline. Dictation is still available." stays, and `_apply_ollama_models` never re-points `settings.ollama_model` away from a model that is still installed.

Because a non-empty list no longer proves the engine is up, `voxa/ui/model_selector.py: ai_status_for_backend` reads the server's health for **both** `llamacpp` and `ollama` (`SERVER_READY` → online, `SERVER_STARTING` → starting, anything else → offline). Only a backend Voxa does not manage (Strata) keeps the old "model names have no spaces" rule.

In Preferences the Local AI page order is now: Model server → **Ollama** (install, pull or remove models) → Answers → Voxa brains, so the Ollama choices are at the top of the menu as the issue asks; `update_backend_rows` keeps the same per-backend visibility rules.

**Why:** rhY saw "no models" with Ollama stopped, then the same models appear the moment the server starts — the list was only ever read over HTTP. Installed models are a fact about the disk, and the dot must not lie once that fact is on screen.

**How to apply:** `tests/test_ollama_disk_models.py` pins the layout and the names, including the `hf.co` case `tests/test_backup.py` already asserts, so a future Ollama that changes the tree (a non-empty `manifests-v2`, for instance) is caught there and fixed in `ollama_display_name`/`disk_models`, never in the HTTP path. Inside the Flatpak the store is read through the host's `HOME`; a custom `OLLAMA_MODELS` set only on the host is not in the sandbox environment, so Voxa would fall back to `~/.ollama/models` — pass it through the manifest if users hit that. Still open on #27: always offering `qwen2.5:0.5b` as a pull option, and a clickable green button that takes Ollama offline. Related: [[voxa-spec-lives-in-github-issues]], [[voxa-0-1-6-handoff]].
