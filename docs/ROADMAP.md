# Voxa Roadmap Status (tracks issue #6)

Baseline: test suite green throughout (111 → 143 tests, ruff clean).
Verified from source under Xvfb: startup, shell layout, state routing,
task flows, avatar spike (WebKitGTK + VRM + visemes, see docs/AVATAR.md).
Verified in the installed Flatpak under Xvfb: launch, Whisper load,
GPU readout, ACTIVE/OFFLINE styling.

## Phase 0 — Protect the baseline: DONE

No regressions: Whisper/Ollama/audio/TTS/GPU/conversation/config/
Flatpak backends untouched by the shell rebuild; old transcript panes
became headless buffers consumed via the same methods.

## Phase 1 — Assistant state model: DONE

`voxa/ui/state.py`: OFFLINE/READY/LISTENING/THINKING/SPEAKING/WORKING/
WAITING/ERROR + captions. `MainWindow.set_assistant_state()` is the
single entry point; conversation events map per the spec (wake-word
wait → READY, woken → LISTENING, inference → THINKING, TTS → SPEAKING,
cancel → READY, stopped → READY/OFFLINE).

## Phase 2 — Task model: DONE (in-memory)

`voxa/tasks.py`: id/title/detail/state/progress/created_at/updated_at/
requires_user_input/choices/result/error; QUEUED/RUNNING/WAITING/DONE/
FAILED/CANCELLED; add/start/update/request_choice/resolve_choice/
complete/fail/cancel; thread-safe with post-lock subscribe notifications.

## Phases 3–8 — Shell rebuild: DONE (milestone 1)

Centered `AssistantView`, overlay task panel (250–290px), bottom bar
(attachment / Local AI selector / ACTIVE / OFFLINE), header menu
preserving Ask AI / Speak reply / Manage models / Shortcuts /
Preferences. Old panes removed from the main view only.

## Phase 9 — Conversation mode: DONE (mapping; behavior preserved)

Wake word, tiny wake model, bounded history, streaming, TTS, barge-in,
spoken cancel/goodbye all preserved and mapped to AssistantState.

## Phase 10+ — Agent controller/tools, 3D avatar swap-in: OPEN

`TaskStore.subscribe` and `set_audio_level()` are the integration
seams. Next: `voxa/agent/` (controller, permissions, tools), then the
vendored Three.js renderer behind the `AssistantView` hooks.
