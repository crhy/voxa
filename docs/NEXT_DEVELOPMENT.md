# Voxa: Next Development Series

Updated 2026-09-22 after the character-picker, local-server manager, and
0.1.3 release work. This is the short execution queue for the next
development cycle; the broader product direction remains in GitHub issues
[#7](https://github.com/crhy/voxa/issues/7),
[#8](https://github.com/crhy/voxa/issues/8),
[#9](https://github.com/crhy/voxa/issues/9), and
[#10](https://github.com/crhy/voxa/issues/10).

## Working rules

- Keep GTK work on the GTK thread and blocking work in workers.
- Treat OFFLINE as a hard privacy and cancellation boundary.
- Keep each milestone reviewable, tested, and usable before starting the next.
- Preserve Ollama and llama.cpp support behind the same UI and controller
  boundaries.

## 1. Finish local AI server lifecycle and health

`voxa/server.py` provides an ownership-aware `AiServerManager`, but the running
application does not use it yet.

- Wire one manager into `MainWindow` startup, backend changes, and shutdown.
- Add a backend health state: starting, ready, unavailable, and failed.
- Surface concise recovery actions without blocking the UI.
- Preserve server logs in a bounded diagnostics file instead of discarding all
  output.
- Add lifecycle, backend-switch, startup-failure, and stale-callback tests.

Acceptance: a fresh launch can bring up the configured local backend, OFFLINE
does not accidentally kill a shared server.

## 2. Add explicit accelerator/GPU selection

Implement GitHub issue #9 through backend-neutral settings.

- Discover usable CPU/GPU devices and stable identifiers.
- Offer Auto, CPU, and individual GPUs in Preferences.
- Map the choice to supported Ollama and llama.cpp controls without embedding
  backend flags in widgets.
- Validate unavailable saved devices and fall back to Auto with a visible
  explanation.
- Document multi-GPU behavior instead of promising unsupported combinations.

Acceptance: the chosen device survives restart, produces the expected backend
launch configuration, and safely falls back when hardware changes.

## 3. Deliver the first complete safe voice workflow

Use the Gmail scenario in issue #10 as the vertical slice, building on the
existing mail drafting support and capability policy.

- Resolve a recipient from an explicit local contact source.
- Collect and visibly preview subject and body.
- Require an unambiguous confirmation immediately before sending.
- Support cancel, correction, timeout, and OFFLINE at every stage.
- Keep browser automation/provider details behind an adapter.
- Record user-visible task events, not hidden reasoning or credentials.

Acceptance: a user can say “send an email to Mom,” dictate it, review it, and
confirm or cancel it; tests prove that no send occurs without final consent.

## 4. Complete character and avatar delivery

The picker and 14-character registry now exist, but production model assets
need a managed lifecycle.

- Define a versioned asset manifest with URLs, licenses, sizes, and SHA-256
  checksums.
- Download on demand with progress, cancellation, atomic install, and cleanup.
- Show thumbnails and distinguish installed from downloadable characters.
- Keep the procedural face as the reliable fallback.
- Drive mouth movement from TTS audio/viseme timing rather than assistant state
  alone.

Acceptance: selecting an uninstalled character offers a verified download and
the application remains functional after a failed or interrupted download.

## 5. Add attachments as a safe, useful capability

- Introduce a GTK-independent attachment model with MIME type, size, source,
  and trust metadata.
- Start with user-selected local text, PDF, and image files.
- Enforce size/context budgets and make truncation visible.
- Treat document contents as untrusted data, never as tool instructions.
- Connect the existing disabled paperclip only after cancellation and error
  behavior are tested.

Acceptance: users can attach a supported file and ask a local model about it
without freezing the UI or silently exceeding context limits.

## 6. Diagnostics, accessibility, and repeatable releases

- Add structured, privacy-aware logs and an opt-in diagnostics bundle.
- Cover keyboard navigation, screen-reader labels, reduced motion, and contrast.
- Run lint, unit tests, release metadata checks, and Flatpak build/install smoke
  tests in CI.
- Keep generated build trees and bundles out of Git; publish versioned bundles
  only as release artifacts.

Acceptance: one command validates a release candidate and failures provide
enough redacted evidence to diagnose backend, audio, avatar, and packaging
problems.
