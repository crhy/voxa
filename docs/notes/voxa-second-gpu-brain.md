---
name: voxa-second-gpu-brain
description: How Voxa enumerates the machine's GPUs and puts her own brain on a card the other workload is not using (crhy/voxa#113).
metadata:
    type: project
---

Voxa's device layer (crhy/voxa#113, task f063) keeps the machine's graphics cards apart instead
of collapsing them into one number, so her small model can sit on a card while rhY's big coding
model works on another.

**How it works**

- `hardware.GpuDevice` is one card: `index`, `vendor`, `name`, `total_vram_gb`, `used_vram_gb`,
  `utilization_percent`, and a `device_id` such as `nvidia:1` or `amd:0`. The vendor is part of
  the id because GPU 0 on the Nvidia side and GPU 0 on the AMD side are different cards, and a
  saved choice has to survive a reboot into the other tool's listing.
- `hardware.detect_gpu_devices()` reads `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu`,
  then `rocm-smi --showmeminfo vram --json`, then the sysfs `card*/device/{vendor,name,mem_info_vram_total,mem_info_vram_used}`
  tree. The old single-value probes (`detect_gpu_vram_gb`, `sample_gpu_usage`, `_sysfs_amdgpu_vram_gb`)
  now read the same listing, so the model suggestions and the window's gauge keep behaving as before.
  `detect_graphics_adapter` deliberately stays sysfs-only: it names the machine's adapter and must not
  depend on a GPU tool being installed.
- `pick_brain_device(devices, reserved_gb=..., busy=...)` returns the card plus a plain-language reason,
  and `resolve_brain_device(saved, devices, ...)` is the single place that says what Voxa will really do,
  so the launch and the spoken status cannot disagree.
- `Settings.gpu_device` is `"auto"`, `"cpu"` or a `device_id`. A malformed value becomes `"auto"` in
  `normalized()`; a card that is simply not in this machine today is left saved and falls back to
  automatic at launch, because hardware changes between launches.
- The launch mapping lives in `server.py:_command`, not in widgets: Ollama gets
  `CUDA_VISIBLE_DEVICES=<index>` (or `OLLAMA_NUM_GPU=0` for the processor), llama.cpp gets
  `--device cuda --main-gpu <index>` (`hip` for AMD, `vulkan` for Intel, only `--main-gpu` when the
  vendor is unknown). llama.cpp never gets `CUDA_VISIBLE_DEVICES` as well: that variable renumbers
  the cards a process can see, so the index would name the wrong card or none.
- `"cpu"` adds no device-pinning flags anywhere; in Voxa's own engine it is `-ngl 0`.
- `gpu_status` says which card Voxa is on, its vendor and VRAM, and the reason. `gpu_choose`
  (risk REVERSIBLE) saves a new choice for "put voxa on the other gpu" / "keep voxa on the cpu";
  the reply says the card applies when her AI server starts next, because a server that is already
  running keeps its card.

**Why:** rhY wants the large local model to keep coding on the main GPU while Voxa runs
continuous background tasks; a single "the GPU" number cannot express that.

**How to apply / what stays open**

- Nothing here has been measured on two real cards: the live split, VRAM pressure, and whether
  Ollama and llama.cpp honour these flags as assumed are rhY's to verify.
- `"auto"` sends no card at all, so the backend keeps its own pick; the split only happens once
  a card is chosen. `pick_brain_device` still reports the card Voxa would take, and says so.
- `busy_devices()` infers another workload from `used_vram_gb >= BUSY_VRAM_GB`; that reading also
  covers Voxa's own model once it is running, which is why moving her off a card treats her
  current card as free.
- Whisper's own device choice (`transcription.py`) and the window's gauge still use the single
  busiest-card sample; they are not steered by this setting yet.
- There is no Preferences row for the choice yet: it is set by voice or by the config file.
  The sibling branch `strata/f024-voxa-gpu-choice-plumbing` (crhy/voxa#9) adds the UI and a
  `gpu:<index>` choice; the two overlap in `hardware.py`, `config.py`, `engine.py` and `server.py`
  and must be reconciled when both merge. Related: [[voxa-two-brains-plan]], [[voxa-gpu-choice]].
