# Voxa Architecture

This document describes Voxa's current architecture and the direction new code should follow.

Voxa began as a Linux dictation/local-AI application and is becoming a local-first voice assistant capable of carrying out any digital task. As its capabilities grow, the separation between **model reasoning**, **application state**, **permission**, and **execution** becomes a core product requirement.

> **The model may propose an action. Deterministic Voxa code decides whether that action is valid, permitted, executed, cancelled, or presented to the user for confirmation.**

## Goals

Voxa should support voice-first interaction, local-first speech recognition and inference, interchangeable AI backends, responsive GTK 4/libadwaita UI, observable assistant/task state, cancellation, desktop capabilities with explicit permissions, graceful fallback, testable components, and Flatpak distribution without making core logic Flatpak-specific.

## High-level flow

```text
Microphone
    |
    v
AudioCapture -> Faster Whisper
                    |
                    v
          Request / Conversation
              Orchestration
               /        \
              v          v
       AI Backend     safe/direct
   llama.cpp/Ollama   capability match
              \          /
               v        v
          Structured request
                 |
                 v
         Capability registry
                 |
                 v
          Schema validation
                 |
                 v
          Permission policy
          /       |       \
       allow    confirm    deny
          \       |       /
                 v
            Voxa task
                 |
                 v
              Executor
                 |
                 v
        Skill / capability
                 |
                 v
        Structured result
            /         \
           v           v
   AssistantModel   SpeechService
           |           |
           v           v
          UI         Speaker
```

Important parts already exist. The normalized capability registry, permission layer and executor are the next major boundary before Voxa gains broad write/control privileges.

## Current components

### Application and integration

- `voxa/application.py` — GTK/libadwaita lifecycle.
- `voxa/window.py` — current integration point for configuration, audio, transcription, speech, conversation, inference backends, hardware/model discovery, assistant UI, legacy UI and desktop capabilities.

`MainWindow` currently owns too many responsibilities. New work should extract orchestration behind controllers/services instead of making `window.py` larger.

### State and controller

- `voxa/controller.py` — ACTIVE/OFFLINE and assistant lifecycle through explicit ports.
- `voxa/ui/state.py` — assistant/task state consumed by the UI.

Preferred dependency:

```text
service event -> controller/orchestrator -> AssistantModel -> UI
```

Avoid services locating and mutating GTK widgets directly.

OFFLINE is a hard behavioral boundary, not merely a red visual state. It should stop listening/speech, cancel inference and applicable work, invalidate stale callbacks, and prevent cancelled work from resurrecting assistant state, and turn off the AI server

### Production UI

Important modules under `voxa/ui/` include:

- `shell.py` — primary assistant shell
- `assistant_view.py` — assistant presence/renderer abstraction
- `model_selector.py` — backend/model selection
- `task_panel.py` — task presentation
- `status_controls.py` — ACTIVE/OFFLINE controls
- `exchange_panel.py` — interaction presentation
- `choice_overlay.py` — choices/confirmations
- `notice.py` — notices/status
- `legacy_view.py` — original transcript/dictation interface

New product features should target shared state and the assistant shell, not expand the legacy UI.

### Avatar/rendering

- `assistant_view.py` owns the renderer-neutral interface and static fallback.
- `avatar_3d.py` provides experimental `Gtk.GLArea` rendering.
- `gltf.py` handles GLTF/GLB support.
- `avatars.py` contains avatar support.

Application code should communicate in renderer-neutral concepts: state, listening, thinking, speaking, audio level, emotion, viseme, gaze and activity intensity.

A GL failure or bad avatar asset must never make Voxa unusable.

### Audio, transcription and conversation

- `audio.py` — capture/devices
- `transcription.py` — Faster Whisper
- `dictation.py` — dictation orchestration
- `conversation.py` — conversation control and bounded history
- `speech.py` — spoken output

Long-running audio/transcription/inference work must not block GTK. Conversation callbacks must be session/generation aware so old results cannot overwrite newer state.

## AI backend abstraction

Current backends include `voxa/ollama.py` and `voxa/llamacpp.py`.

Backend-specific branching should shrink. Before adding more providers, introduce a normalized contract conceptually like:

```python
class AIBackend(Protocol):
    def health(self) -> BackendHealth: ...
    def list_models(self) -> list[ModelInfo]: ...
    def chat(self, request: ChatRequest, *, cancel) -> ChatResult: ...
    def stream_chat(self, request, *, cancel, on_chunk) -> ChatResult: ...
    def capabilities(self) -> BackendCapabilities: ...
```

Conversation, skills and UI should not need to understand Ollama versus llama.cpp details.

Structured health/error states should distinguish backend unavailable, model unavailable/loading, context-limit errors, resource exhaustion/OOM, cancellation, malformed responses and generic failures.

## Desktop capabilities

Existing desktop-oriented modules include `apps.py`, `documents.py`, `mail.py` and `websearch.py`.

They should converge on one shared capability/permission architecture rather than becoming independent special dispatch paths inside `MainWindow`.

## Dependency direction

```text
UI
 |
 v
controllers / orchestrators
 |
 +-------------------+
 |                   |
 v                   v
services          agent/capabilities
 |                   |
 v                   v
audio/AI/TTS      policy + executor
 |                   |
 +---------+---------+
           |
           v
       OS / network
```

Dependencies point downward. Lower layers should not import the main window merely to update presentation.

## Hard rules

1. Widgets do not own business state.
2. Services do not manipulate GTK widgets.
3. Model output is untrusted data.
4. The LLM does not receive unrestricted execution authority.
5. Every side-effecting capability has explicit policy.
6. Cancellation is part of an operation's contract.
7. Long work never blocks GTK's main loop.
8. Optional components fail safely and degrade gracefully.
9. Backend-specific details remain behind backend boundaries.
10. Stale asynchronous callbacks cannot mutate newer sessions.
11. OFFLINE is a real kill boundary.
12. Important logic should be testable without GTK where practical.

## Agent and capability foundation

A possible future structure:

```text
voxa/
  agent/
    request.py
    result.py
    registry.py
    policy.py
    planner.py
    executor.py

  skills/
    apps.py
    clipboard.py
    notifications.py
    system_info.py
    documents.py
    mail.py
    web.py
```

This is direction, not a requirement to move every existing module immediately.

### Capability definition

Each executable capability should have a stable name, description, validated schema, risk classification, confirmation policy, cancellation behavior, deterministic handler and structured result.

Conceptually:

```python
Capability(
    name="open_application",
    description="Open an installed desktop application",
    input_schema={...},
    risk_level=RiskLevel.LOW,
    requires_confirmation=False,
    handler=open_application,
)
```

### Structured requests

Prefer:

```json
{
  "capability": "open_application",
  "arguments": {"application": "Firefox"}
}
```

over model prose such as “run whatever command is necessary.” Structured requests can be validated, authorized, logged and tested.

## Permission policy

Risk is determined by Voxa code, not by the model.

| Level | Examples | Default |
| --- | --- | --- |
| READ | system information, inspect state | Allow |
| LOW | open app, clipboard copy, notification | Allow |
| WRITE | create/modify user data | Confirm/policy |
| COMMUNICATION | send email/message | Confirm |
| INSTALL | install/remove software | Confirm |
| DESTRUCTIVE | delete/overwrite important data | Explicit confirmation |
| PRIVILEGED | elevated system changes | Confirmation + OS authorization |

Exact enum names can change; explicit deterministic policy cannot.

## Confirmation

Confirmation should expose the actual side effect. Sending mail should show recipient, subject, content/preview, attachments and relevant account—not merely “Voxa wants to perform an action.”

The confirmation UI renders structured policy/execution data rather than trusting an LLM-generated description.

## Task lifecycle

Actionable work should receive a stable task ID.

```text
QUEUED
  -> PLANNING
  -> WAITING_FOR_CONFIRMATION
  -> RUNNING
  -> COMPLETED

or -> CANCELLED
or -> FAILED
```

Assistant presentation state and task lifecycle are related but separate concepts.

## Executor

Only the executor invokes capability handlers. It should locate the capability, validate arguments, evaluate policy, request confirmation when needed, honor cancellation, execute the deterministic handler, capture structured results/errors, update task state and return a result suitable for narration.

Safe first capabilities include opening an application/URL, copying to clipboard, showing a notification and retrieving system information.

## Security boundary

Local inference does not make model output inherently safe.

Treat LLM output, web pages, email bodies, documents, retrieved text and model-generated arguments as untrusted data. Content being summarized never gains authority merely because the model read it.

Do not make arbitrary model-generated shell commands the normal execution mechanism. Prefer narrow structured tools. Any future shell capability should be high-risk and separately restricted.

Sending communications is consequential. Drafting and sending should be separate capabilities; sending should normally require confirmation.

Skills receive only the minimum credentials they require.

## Threading and cancellation

GTK objects belong to the GTK main thread. Inference waits, network requests, model loading and transcription run away from it, with results marshalled back through the main loop.

Asynchronous operations should carry enough identity—request generation, session token, task ID and/or cancellation event—to reject stale completion.

OFFLINE must invalidate work that should no longer affect the assistant.

## Error handling

Prefer structured internal errors plus concise user-facing rendering. Expected failures—missing AI server/model/microphone, network TTS failure, 3D failure, denied capability, cancellation or unavailable host permissions—should not crash Voxa.

## Testing

Use pure/unit tests for state transitions, schemas, permissions, registry behavior, backend normalization, task lifecycle, cancellation and stale-result rejection.

Use GTK tests for shell rendering, state-to-widget behavior, ACTIVE/OFFLINE controls, confirmation UI, model selector, task panel and renderer fallback.

Use fake backends/skills for integration tests so architectural correctness does not require a live Ollama or llama.cpp server.

Continue packaging tests for AppStream, desktop metadata, Flatpak manifest, bundle import and OSTree checkout.

## Spaced Linux integration

Voxa can be a major Spaced Linux feature without making the core application Spaced-specific.

Prefer:

```text
generic Voxa capability
       |
       +--> generic Linux implementation
       |
       +--> optional Spaced Linux enhancement
```

Distribution-specific behavior should live behind adapters/providers where practical.

## Observability

Long local-model work should expose reliable progress when available: backend health, model loading, prompt processing, generation, elapsed time, token count/speed, task phase and cancellation state.

Do not invent progress percentages when the backend cannot provide them.

## Migration plan

### Stage 1 — stabilize
- keep `AssistantModel` authoritative;
- preserve deterministic OFFLINE behavior;
- reduce `window.py` responsibilities;
- normalize errors;
- preserve CI/Flatpak reliability.

### Stage 2 — backend abstraction
- common backend protocol;
- normalized health/model discovery;
- normalized streaming/cancellation;
- backend manager.

### Stage 3 — agent foundation
- capability registry;
- schemas;
- permission policy;
- confirmation requests;
- executor;
- task lifecycle;
- safe initial capabilities.

### Stage 4 — capability expansion
- documents;
- richer app control;
- web workflows;
- email draft/send;
- filesystem operations;
- optional Spaced Linux integrations.

### Stage 5 — richer presence
- Preferences-driven avatar selection;
- audio envelope;
- speaking animation;
- visemes;
- emotion/gaze/activity;
- avatar packages.

## Review checklist

Before merging a capability/subsystem, ask whether it adds inappropriate logic to `MainWindow`, directly manipulates GTK from lower layers, lets model output bypass validation, has explicit side-effect policy, can be cancelled, behaves safely when Voxa goes OFFLINE, rejects stale callbacks, avoids unnecessary backend coupling, degrades gracefully and has meaningful tests.

## Product principle

As Voxa becomes more powerful, the interface should become **simpler**, not more complicated.

Complexity belongs behind clear state, predictable controls, visible tasks and explicit permission boundaries.

> Speak naturally. See what Voxa is doing. Approve consequential actions when necessary. Interrupt or take it OFFLINE at any time.
