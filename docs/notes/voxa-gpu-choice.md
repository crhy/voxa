---
name: voxa-gpu-choice
description: How Voxa's "Graphics card" choice (crhy/voxa#9) is stored and handed to the server it starts, and what stays open (Whisper's device, combining cards, AMD hand-off).
metadata:
  type: project
---

crhy/voxa#9 asks for a "Graphics card" row in Preferences → Local AI that lists the
detected cards plus "Automatic" and "Processor only", passes the choice to the server
Voxa starts, and shows which device is in use. The owner's plan comment (2026-10-07) is
the contract; its first step is what shipped.

**The stored value** is `Settings.gpu_choice`, one of three forms: `"auto"` (the backend
decides, exactly as before), `"cpu"` ("Processor only"), `"gpu:<index>"` (the card whose
index `hardware.detect_gpus()` listed). `config.normalize_gpu_choice()` repairs anything
else to `"auto"`; a card index that is not currently present is kept, so a card added
later still gets its choice, and the Preferences row says "the saved card is not in
this machine".

**Detection is split, not refactored.** `detect_gpu_vram_gb()` collapses the machine
into one number and its probes are pinned by byte-compatible tests, so the per-card
listing is a separate `detect_gpus()` with its own multi-column nvidia-smi query
(`--query-gpu=index,name,memory.total`) through `_nvidia_smi_command()`, which is what
keeps the bundled `/app/lib/nvml/nvidia-smi` copy working inside the Flatpak. AMD cards
come from `rocm-smi --showmeminfo vram --json`, else `/sys/class/drm/card*/device/mem_info_vram_total`
named by `detect_graphics_adapter`. Nothing found returns `()`, never an exception.

**How each backend is steered** (`server.py:_command`, `engine.server_command`):

- Ollama: `CUDA_VISIBLE_DEVICES` in the environment — `<index>` for a chosen card,
  empty for "Processor only", absent for "Automatic". Inside the Flatpak the same
  variable is also named in the `flatpak-spawn --host … env NAME=value` argument list,
  because `flatpak-spawn` starts the host process with the host's own environment.
- llama.cpp: `-ngl 0` instead of `-ngl 99` for "Processor only", and
  `--device cuda:<index>` for a chosen card. The env variable is deliberately **not**
  sent together with `--device`: `CUDA_VISIBLE_DEVICES` renumbers the cards a process
  can see, so `cuda:<index>` would then name the wrong card or none at all.

**Show which device is in use:** `ui/model_selector.py:gpu_status_detail()` turns the
choice into "card 0", "processor", "card 3 (not in this machine)" or "" for "Automatic",
and `ModelSelector.set_status(state, detail)` appends it to the dot: "● Online · card 0".
Strata's server is not Voxa's, so the row hides itself and the status says nothing there.

**Still open, deliberately:**

- Whisper's device (`transcription.py` tries its own `(device, compute_type)` attempts)
  is a separate decision and untouched by this slice.
- Combining cards is left to the backends; the owner doubts Nvidia/AMD mixing, so no
  cross-vendor combination is invented here.
- The AMD hand-off: an AMD card can be listed, saved and shown, but `CUDA_VISIBLE_DEVICES`
  does not steer it, and llama.cpp's device name for a ROCm build may not be `cuda:N`.
  Needs a real AMD card to settle.
- A card picked in Preferences restarts Voxa's server so the choice takes effect;
  whether that restart should be deferred until the next question is rhY's call.

**Why:** rhY wants to offload to the built-in GPU, assign a specific card, or keep the
AI on the processor, and to see what is actually running.

**How to apply:** keep `detect_gpu_vram_gb` and its probes byte-compatible; put new
per-card work in `detect_gpus()`. Never send both `CUDA_VISIBLE_DEVICES` and
`--device` for the same server. There is no GPU on the development machine, so tests
prove parsing, the stored choice and the exact command line only — a real card must be
checked by rhY. Related: [[voxa-speed-is-top-priority]], [[voxa-flatpak-host-access-approved]],
[[voxa-separate-from-voice2text]].
