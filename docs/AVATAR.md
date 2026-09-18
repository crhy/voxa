# Voxa 3D Avatar Plan (voxa#2, voxa#4)

The avatar is the main focus of the new interface (voxa#3): it listens,
thinks (Ollama), and visibly talks. This doc records what we stole from
whom, the architecture, and the fallback chain. Smallest path that still
looks like talking.

## What we reviewed

Issue #4 listed seven projects. Verdict after review:

| Project | License | Verdict |
|---|---|---|
| SethRobinson/GPTAvatar (Unity, SALSA, cloud APIs) | BSD (code) | Reject: Windows-only, proprietary lip-sync, cloud cost |
| Vongoori/OpenAI_3DAvatarChatBot_Unity | GPL-3.0 | Reject: Unity + GPL conflict + Oculus proprietary |
| Daniel-Kennison/avatar-chatbot (Python+Blender) | GPL-3.0 | Reject: needs Blender at runtime, GPL conflict |
| myned-ai/avatar-chat-widget (Gaussian Splatting) | MIT | Partial: `mountAvatar` renderer-only pattern; full widget needs a server |
| ruslanmv/3D-Avatar-Chatbot (Three.js, Ollama) | Apache-2.0 | Pattern only: GLB/VRM morph visemes + Ollama; app itself is a kitchen sink |
| danieloquelis/chat-avatar-ai (wav2vec2 phonemes) | MIT | Pattern only: viseme morph targets; 330 MB model is anti-lean |
| kenken64/remotion-3d-AI-avatar | MIT/ISC | Reject: offline video render, not a live UI |

Better sources found outside the issue list (all MIT, all offline):

- **DanielSWolf/rhubarb-lip-sync** (`v1.9.1`): wav + dialog text
  → timestamped Preston-Blair mouth shapes (A–F + optional G/H/X).
  CLI, ~MBs, PocketSphinx bundled. This is the timing engine.
- **pixiv/three-vrm + vlapky/three-vrm-lip-sync**: VRM loading and a
  drop-in lip-sync integration pattern for Three.js. Renderer side.
- **tenjerma/tts-avatar**: converged on the same architecture
  (TTS → phonemes → RMS-anchored visemes, ~60 ms mouth lookahead,
  live-level modulation). Validates the timing constants below.
- **buildfastwithai/talking-avatar**: validates the 2D fallback
  (static portrait + 3 mouth sprites, ~96 ms pose stepping).

Rejected additionally: `Scthe/ai-iris-avatar` (GPL + Unity),
`majidmanzarpour/threejs-talking-avatar` (needs WebGPU + GBs of
models), neural talking-heads (SadTalker/Wav2Lip class: GPU-heavy).

## Architecture

```
Ollama reply text ─┬─► Edge/espeak TTS ──► wav ──► GStreamer playback
                   └─► rhubarb -d dialog.txt voice.wav ──► JSON cues ──► scheduler
                                                                        (GStreamer clock)
                     WebKitGTK WebView ◄── setViseme() over JS bridge ──┘
                     (vendored three.js + three-vrm + one VRM model, no CDN)
```

- Timeline is **baked before playback** (we hold both inputs already),
  scheduled against the audio clock — deterministic and unit-testable.
- Mouth shapes lead audio by ~60 ms (mouths move before sound).
- Live RMS level modulates openness each frame (quiet moments close
  the mouth even mid-cue).
- States: `idle` / `listening` / `speaking`, plus blink/idle motion.

## Fallback chain (never a frozen face)

1. Rhubarb timeline (English; `phonetic` recognizer for other languages
   when configured).
2. RMS jaw flap (existing level meter, no new deps).
3. Static portrait.

## Flatpak notes

- `rhubarb-lip-sync` module ships the upstream 1.9.1 release binary
  (pinned URL + sha256) plus its `res/` acoustic models next to the
  binary, where its executable-relative lookup finds them. Building
  from source was tried and abandoned: the SDK ships no Boost, its
  iconv check fails, and its CMake targets pre-3.5 policies. The
  binary was verified to run and analyze real TTS audio inside the
  GNOME 50 runtime.
- `voxa/lipsync.py`: `analyze()` returns `None` when the binary is
  absent (dev machines); raises `LipSyncError` on failure. Shapes are
  validated against `ABCDEFGHX`. `ensure_valid_wav()` repairs the
  placeholder RIFF data length `espeak-ng --stdout` writes (Rhubarb
  rejects it; GStreamer tolerates it).
- Renderer page + model are vendored (offline rule); WebKitGTK comes
  from the GNOME runtime at no extra bundle cost.

## Models

VRM from VRoid Hub (check per-model terms) or the MIT Nyx avatar from
`avatar-chat-widget`. Procedural fallback while no model is bundled.

## Spike verdict (proven, 2026-09-18)

A VRM avatar renders in WebKitGTK 6.0 under the GNOME 50 SDK with
JS-driven visemes verified via snapshots (rest vs `aa`/`oh`/`ee` —
mouth visibly moves). Production notes:

- Vendor ~1 MB JS: `three.module.min.js` + `three.core.min.js` (r180
  splits the build) + `GLTFLoader.js` + `utils/BufferGeometryUtils.js`
  (its relative import) + `three-vrm.module.min.js`. No CDN (offline).
- Serve the page over loopback HTTP from the app: `file://` blocks ES
  modules via CORS. Loopback needs no extra Flatpak permission.
- WebKit 6.0 API deltas vs 4.1: `evaluate_javascript()` (not
  `run_javascript`), `SnapshotOptions` (not `SnapshotFormat`),
  `get_snapshot_finish()` returns `Gdk.MemoryTexture` (use
  `save_to_png()`).

## Issue #5 contract (implemented foundation)

- `voxa/ui/state.py`: `AssistantState` (OFFLINE/READY/LISTENING/
  THINKING/SPEAKING/WORKING/ERROR) + caption text. UI renders from
  state; no scattered label poking.
- `voxa/ui/assistant_view.py`: `AssistantView` center-stage widget
  (placeholder avatar + caption + listening level) with
  `set_listening/set_thinking/set_speaking/set_audio_level()` hooks.
  The 3D page later replaces the placeholder behind the same hooks;
  `set_audio_level()` + Rhubarb cues drive the lips.
- `voxa/tasks.py`: `TaskStore` (`add/update/complete/fail/cancel`,
  `request_choice/resolve_choice`) backing the compact task panel.
- `voxa/lipsync.py` stays backend-side: `analyze()` bakes timelines
  ahead of playback; the view only consumes cues + levels.
