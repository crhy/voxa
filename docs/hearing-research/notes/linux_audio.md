# Linux audio plumbing for echo cancellation in Voxa (PipeWire 1.x / PulseAudio)

Research notes, 2026-11-17. Note on access: gitlab.freedesktop.org and wiki.archlinux.org now sit behind Anubis bot protection and refused automated fetches. Source code was therefore read from the GitHub mirrors (github.com/PipeWire/pipewire, github.com/GStreamer/gstreamer, "master"/"main" branch, current as of fetch) and Arch wiki text from a 2026-04 Kiwix mirror. Line-level claims about code describe those mirror snapshots.

## 1. PipeWire libpipewire-module-echo-cancel (webrtc backend): config, alignment, latency, pros/cons, private node

### Takeaway
The PipeWire module creates four streams (capture, source, sink, playback) and runs WebRTC AEC on 10 ms blocks. It aligns reference and mic only by graph cycle (it processes when the capture and sink cycles match), plus an optional fixed `buffer.play_delay`. It does not measure device latency or correct drift, and leaves the rest to WebRTC's internal delay estimator. It works well when mic and speaker share one clock (one device, one driver), and is known to work poorly across devices. Loaded as configured in the docs, it adds globally visible nodes, which is the routing-hijack problem Voxa wants to avoid.

### Cited Findings
- Module options: `capture.props`, `source.props`, `sink.props`, `playback.props`, `library.name` (only `aec/libspa-aec-webrtc` supported, also the default), `aec.args`, and `monitor.mode`. `monitor.mode` means: "Instead of making a sink, make a stream that captures from the monitor ports of the default sink." General keys honoured: `node.latency`, `node.group`, `node.link-group`, `node.virtual`, `audio.rate`, `audio.channels`, `audio.position`, `remote.name`. — [PipeWire docs: echo-cancel](https://docs.pipewire.org/page_module_echo_cancel.html)
- Doc example (drop-in `~/.config/pipewire/pipewire.conf.d/my-echo-cancel.conf`):
  ```
  context.modules = [
  { name = libpipewire-module-echo-cancel
    args = {
      # library.name = aec/libspa-aec-webrtc
      # node.latency = 1024/48000
      # monitor.mode = false
      capture.props  = { node.name = "Echo Cancellation Capture" }
      source.props   = { node.name = "Echo Cancellation Source" }
      sink.props     = { node.name = "Echo Cancellation Sink" }
      playback.props = { node.name = "Echo Cancellation Playback" }
    }
  }
  ]
  ```
  The four streams are Capture (mic in), Sink (the app plays the far-end/reference here), Playback (forwards the Sink to the speaker), and Source (echo-cancelled mic out). — [PipeWire docs](https://docs.pipewire.org/page_module_echo_cancel.html)
- Undocumented options found in the source: `buffer.max_size` (default `MAX_BUFSIZE_MS` = 100 ms) and `buffer.play_delay` (a fraction, e.g. `"480/48000"`, converted to `buffer_delay` samples, default 0). `aec.method` has been removed; the code logs "aec.method is not supported anymore use library.name". — [module-echo-cancel.c (GitHub mirror)](https://raw.githubusercontent.com/PipeWire/pipewire/master/src/modules/module-echo-cancel.c)
- How it aligns: `capture_process()` and `sink_process()` each record `clock.cycle` from `SPA_IO_Position`, and `process()` runs only when `capture_cycle == sink_cycle`. Sink audio goes into a `play_ring` and a `play_delayed_ring`. The play ring is pre-advanced by `buffer_delay` so the reference lags the capture by that fixed amount. During warm-up it outputs silence. The module does not call `pw_stream_get_time` or any `pw_time` delay query, so there is **no measurement of hardware latency**. In `monitor.mode` the sink becomes `Stream/Input/Audio`, a stream on the default sink's monitor. — [module-echo-cancel.c](https://raw.githubusercontent.com/PipeWire/pipewire/master/src/modules/module-echo-cancel.c)
- Clock and drift: default rate 48000, format F32P, position `[ FL FR ]`. Node latency is snapped to a multiple of the AEC block. Capture data beyond two blocks is dropped. "the source contains no explicit clock drift compensation". `resample.prefill` defaults to true. — [module-echo-cancel.c](https://raw.githubusercontent.com/PipeWire/pipewire/master/src/modules/module-echo-cancel.c)
- The WebRTC backend reports `latency = "480/48000"` (10 ms), and "Buffers must be multiples of 10ms in length". It calls `apm->set_stream_delay_ms((num_blocks - 1) * 10)`, a fixed value from the block count, not a measured device delay. The legacy-API build turns on `DelayAgnostic` by default and turns off WebRTC's drift compensation, with a comment that PipeWire handles drift itself. — [aec-webrtc.cpp (GitHub mirror)](https://raw.githubusercontent.com/PipeWire/pipewire/master/spa/plugins/aec/aec-webrtc.cpp)
- `aec.args` keys and defaults:
  - All builds: `webrtc.high_pass_filter=true`, `webrtc.noise_suppression=true`, `webrtc.gain_control=false`.
  - webrtc-audio-processing 1.x API: adds `webrtc.voice_detection=true`, `webrtc.transient_suppression=true`, `webrtc.mobile_mode=false`.
  - 2.x API: adds `webrtc.mobile_mode`, the `webrtc.gain_control1.*` and `webrtc.gain_control2.adaptive_digital.*` families, and `webrtc.capture_level_adjust.*`.
  - Legacy (0.x) API: adds `webrtc.extended_filter=true`, `webrtc.delay_agnostic=true`, `webrtc.experimental_agc`, `webrtc.experimental_ns`, `webrtc.beamforming`.
  - Echo cancellation is always on (`config.echo_canceller.enabled = true`).

  — [aec-webrtc.cpp](https://raw.githubusercontent.com/PipeWire/pipewire/master/spa/plugins/aec/aec-webrtc.cpp)
- Arch wiki guidance:
  - Put the config in `/etc/pipewire/pipewire.conf.d/60-echo-cancel.conf`.
  - `monitor.mode` reads the audio to cancel straight from the fallback output, so apps don't have to play into a virtual sink.
  - Pin devices with a target property, and set `node.passive` on capture so the mic isn't kept open while idle.
  - Set `node.force-quantum` if the voice crackles.
  - The resulting source is named `source_ec` ("Echo-cancelled source").
  - Leave `webrtc.gain_control` off: the wiki says the PipeWire developers advise against it because it interferes with delay detection.

  — [Arch wiki PipeWire/Examples (2026-04 Kiwix mirror)](https://wikipedia.jakami.de/content/archlinux_en_all_maxi_2026-04/PipeWire/Examples)
- Known problems (from issue titles and snippets, issues not readable because of Anubis): "WebRTC Echo Cancellation only works if Microphone and Speakers are on the same device" — [pipewire#2165](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/2165). "libpipewire-module-echo-cancel ignores capture.props and playback.props in configuration file" (it connected to the default devices whatever target was set) — [pipewire#2939](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/2939). A user reported that `node.passive`/`node.pause-on-idle` had no effect — [pipewire#3412](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/3412). These may be fixed in newer 1.x versions.
- WirePlumber picks defaults by `priority.session` (a higher value makes a device the default), and a stream with no target follows the default node. — [WirePlumber policy configuration](https://stefanu21.pages.freedesktop.org/wireplumber/daemon/configuration/policy.html)
- Library versions:
  - WebRTC AudioProcessing v2.0 (standalone libwebrtc APM, rebased to WebRTC M131) was released January 2025 — [announcement](https://lists.freedesktop.org/archives/pulseaudio-discuss/2025-January/032491.html).
  - PulseAudio's module-echo-cancel gained AEC3 support in 2023 — [pulseaudio-commits Aug 2023](https://lists.freedesktop.org/archives/pulseaudio-commits/2023-August/006372.html).
  - Older distro packages such as libwebrtc-audio-processing 0.3 predate AEC3 (from a third-party project note, low authority) — [wz-phone audio_linux_aec.rs](https://git.manko.yoga/manawenuz/wz-phone/raw/commit/6f434152859a259db8be0c4360aea26f7f8b40a6/crates/wzp-client/src/audio_linux_aec.rs).

### Inferences
- **Pros:** zero-copy inside the graph, mic and reference processed in the same graph cycle, WebRTC AEC3 quality, no Python DSP. **Cons:** global config file or module load (visible nodes that WirePlumber may make default, which is the old hijack problem), no measured latency or drift handling across devices, `monitor.mode` captures *all* apps on the default sink, flatpak apps can't load server modules into the user's daemon, and behaviour varies with the webrtc-audio-processing version the distro built against (0.x vs 1.x vs 2.x).
- **A private, app-scoped instance (not verified, test first):** PipeWire modules can generally be loaded into a *client* context (the way `pw-loopback` hosts module-loopback). Voxa could call `pw_context_load_module(ctx, "libpipewire-module-echo-cancel", args, NULL)` in its own process, so the nodes exist only while Voxa runs and disappear when it exits. Settings to try:
  - Pin `capture.props = { target.object = <mic> }` and `playback.props = { target.object = <current default sink> }`.
  - Set `node.dont-reconnect = true` and `node.passive = true`.
  - Set a low `priority.session` on the source/sink props so WirePlumber never picks them as default.
  - Give the nodes Voxa-specific names and descriptions.

  Issue #2939 suggests checking that the targets are actually honoured in the installed version. This route still needs the PipeWire native socket, which flatpak doesn't grant by default (see section 5).
- Since Voxa plays its own TTS, the cleanest PipeWire-native layout is non-monitor mode with Voxa playing TTS into the private echo-cancel **sink**. The reference is then exactly Voxa's audio (no other apps). This mirrors GStreamer's webrtcechoprobe layout.

### Gaps
- I couldn't read the PipeWire GitLab issues or any statement from Wim Taymans or Arun Raghavan about recommended EC latency or cross-device use (Anubis block). No asymptotic.io post on PipeWire AEC turned up.
- I found no primary source on how GNOME, KDE, Discord, OBS or EasyEffects use the module. Discord and browsers run WebRTC APM in-process. EasyEffects has its own echo-canceller plugin. These are from memory and not verified here.
- I couldn't confirm whether `priority.session` set in `source.props` reliably keeps an echo-cancel source from becoming default in WirePlumber 0.5.

## 2. Own TTS samples as the reference vs monitor capture; exact playback timestamps; drift; adaptive resampling

### Takeaway
Using Voxa's own TTS buffer as the reference gives a clean, app-only reference. Its accuracy depends on knowing when each sample is actually heard, which PipeWire exposes via `pw_stream_get_time_n()` (`pw_time.delay` + `queued` + `buffered`) and PulseAudio via `pa_stream_get_latency`/timing info. Even with timestamps, mic and speaker clocks drift when they are different devices. Use a delay estimator (WebRTC AEC3 has a matched-filter delay estimator and a clock-drift detector) or a measured-rate resampler.

### Cited Findings
- `pw_stream_get_time_n()` is real-time safe and returns a fresh snapshot. `pw_stream_get_time` has been deprecated since 0.3.50. — [stream.h docs](https://docs.pipewire.org/stream_8h_source.html)
- `pw_time` fields:
  - `now`: time in ns of the snapshot, usually updated every graph cycle.
  - `rate`: units of ticks and delay, usually 1/samplerate.
  - `ticks`: monotonically increasing driver-time position.
  - `delay`: "the time it will take for the next output sample of the stream to be presented by the playback device". It excludes queued buffers, is positive for input and negative for output in some versions, and includes filters and hardware.
  - `queued`: bytes in queued `pw_buffer`s.
  - `buffered`: extra frames held in the resampler (since 0.3.50).
  - `queued_buffers` and `avail_buffers`.
  - `size`: samples requested by the resampler for the current quantum (since 1.1.0).

  — [pw_time docs](https://docs.pipewire.org/structpw__time.html); [search summary of 1.x stream.h](https://docs.pipewire.org/devel/stream_8h_source.html)
- Official extrapolation formulas:
  ```c
  uint64_t now = pw_stream_get_nsec(stream);
  int64_t diff = now - pw_time.now;
  int64_t elapsed = (pw_time.rate.denom * diff) / (pw_time.rate.num * SPA_NSEC_PER_SEC);
  // ms until the first sample of a newly queued buffer is played:
  (pw_time.buffered * 1000 / stream.samplerate) +
  (pw_time.queued * 1000 / app.rate) +
  ((pw_time.delay - elapsed) * 1000 * pw_time.rate.num / pw_time.rate.denom)
  ```
  — [pw_time docs](https://docs.pipewire.org/structpw__time.html)
- Known inaccuracy: when pavucontrol is open, its monitor streams report zero latency, and the delay reported to other clients is halved (it averages max+min). — [pipewire#3191](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/3191)
- PulseAudio API (also served by pipewire-pulse):
  - `pa_stream_get_latency` is based on `pa_stream_get_time`. For a monitoring stream it "can be negative, i.e. the captured samples are not yet played".
  - `pa_stream_get_time` increases monotonically and needs timing-info updates.
  - `PA_STREAM_INTERPOLATE_TIMING` "favours 'smooth' time graphs over accurate ones".
  - `pa_stream_update_timing_info` requests an update. `write_index` is updated on each `pa_stream_write()`.

  — [PulseAudio stream.h doxygen](https://freedesktop.org/software/pulseaudio/doxygen/stream_8h.html)
- WebRTC AEC3 has its own delay handling. `EchoPathDelayEstimator` runs a matched filter on downsampled capture against the render buffer and feeds a `ClockdriftDetector`. That detector flags "probable" or "verified" drift from a history of delay steps, and `RenderDelayController::HasClockdrift()` exposes it. — [echo_path_delay_estimator.cc](https://webrtc.googlesource.com/src/+/408cb4bf30b3fdffcb8dfef54e295c0f8ac48743/modules/audio_processing/aec3/echo_path_delay_estimator.cc); [clockdrift_detector.h](https://webrtc.googlesource.com/src/+/53a7bb255dc00f7ef3f782ce99a3fcc2f4338726/modules/audio_processing/aec3/clockdrift_detector.h); [render_delay_controller.cc](https://webrtc.googlesource.com/src/+/4e615d590aa0bbcb5482be0d32ec7404da671f07/modules/audio_processing/aec3/render_delay_controller.cc)
- AEC3 `Delay` config defaults: `default_delay=5` (blocks), `down_sampling_factor=4`, `num_filters=5`, `delay_headroom_samples=32`, `hysteresis_limit_blocks=1`, `fixed_capture_delay_samples=0`, `delay_selection_thresholds={initial 5, converged 20}`, `use_external_delay_estimator=false`, `detect_pre_echo=true`. Refined and coarse filter `length_blocks=13` (initial 12). — [echo_canceller3_config.h](https://webrtc.googlesource.com/src/+/main/api/audio/echo_canceller3_config.h)
- GStreamer's webrtcechoprobe does timestamp alignment with running time. Each playback buffer's PTS is converted to running time. On a capture chunk it computes `play_time = prev_pts + sample offset + pipeline latency` and `diff = rec_time - play_time` in ms. It then pads with silence or drops samples so the 10 ms reference chunk matches the nominal `delay` (upstream latency). If timestamps are missing, it assumes "perfect delay". — [gstwebrtcechoprobe.cpp (GitHub mirror)](https://raw.githubusercontent.com/GStreamer/gstreamer/main/subprojects/gst-plugins-bad/ext/webrtcdsp/gstwebrtcechoprobe.cpp)

### Inferences
- **Recommended Voxa design:**
  1. Keep the TTS PCM Voxa generates.
  2. When each TTS buffer is queued, record `t_play = now + latency_from_pw_time` (or `pa_stream_get_latency`) on the monotonic clock.
  3. For each mic buffer, record `t_capture = now - capture_delay` (`pw_time.delay` on the input stream is positive).
  4. Build the reference by indexing the TTS history at `t_capture`.
  5. Feed 10 ms frames to AEC3 via `ProcessReverseStream` (render) and `ProcessStream` (capture). Pass the residual coarse offset with `set_stream_delay_ms`, or leave it at 0 and let AEC3's estimator lock.

  This is what webrtcechoprobe does inside GStreamer.
- Reference from own samples vs monitor:
  - The own-sample reference has no other-app contamination and no monitor jitter, and it exists even before the sink plays. But it can't see post-sink effects (EasyEffects EQ, volume, OS mixing). AEC3 adapts linear gain and EQ, so this is mostly fine, but non-linear sink effects reduce cancellation.
  - The monitor reference includes those effects but also every other app's audio.
- **Drift:** with separate devices (USB mic + HDA speaker) expect tens of ppm of drift, so a 30-minute session accumulates tens of ms. AEC3's estimator tracks slow drift (the ClockdriftDetector exists for this). For long sessions, re-estimate the offset per utterance (TTS turns are short) instead of building an adaptive resampler. Inside one PipeWire graph both streams already run at the driver rate, and PipeWire's adaptive resampling handles followers. The remaining drift is between the driver and the follower device's real clock, which PipeWire absorbs in its ALSA follower DLL.

### Gaps
- I didn't find an authoritative ppm figure for typical USB vs HDA clock mismatch.
- I didn't get an explicit statement of AEC3's maximum trackable delay. With 5 filters of 32 blocks of 64 samples at 4x decimation (memory, not verified), it is roughly 0.5 s or more at 16 kHz. Treat this as unverified, and check `matched_filter_*` config defaults before relying on it for A2DP.

## 3. Monitor-source latency/jitter, quantum, real-time scheduling

### Takeaway
A monitor capture is delivered one graph cycle (quantum) at a time, so jitter is on the order of one quantum (1024/48000 ≈ 21 ms by default in the echo-cancel example). Its latency relative to the actual speaker output is the sink's device latency, which PulseAudio reports as a negative latency on monitor streams. Smaller quanta reduce jitter at the cost of CPU and xruns.

### Cited Findings
- The echo-cancel example uses `node.latency = 1024/48000`, and the module snaps node latency to multiples of the 10 ms AEC block. — [PipeWire docs](https://docs.pipewire.org/page_module_echo_cancel.html); [module-echo-cancel.c](https://raw.githubusercontent.com/PipeWire/pipewire/master/src/modules/module-echo-cancel.c)
- On a monitor stream PulseAudio latency can be negative ("captured samples are not yet played"). `pa_stream_set_monitor_stream()` limits a monitor record stream to "a very specific sink input of the sink" and must be called before `pa_stream_connect_record()`. — [PulseAudio stream.h](https://freedesktop.org/software/pulseaudio/doxygen/stream_8h.html)
- The Arch wiki suggests `node.force-quantum` when echo-cancelled voice crackles. — [Arch wiki mirror](https://wikipedia.jakami.de/content/archlinux_en_all_maxi_2026-04/PipeWire/Examples)
- The `pw_time.delay` value for monitors is unreliable when pavucontrol runs (#3191). — [pipewire#3191](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/3191)

### Inferences
- **Key fix for Voxa's current design:** if Voxa keeps a monitor reference, record only its own sink input. Use `pa_stream_set_monitor_stream(stream, tts_sink_input_index)` via libpulse, or with PipeWire native a capture stream with `target.object` = Voxa's own playback node and `stream.capture.sink`. The second is unverified from memory; test it. This removes other apps' audio from the reference without changing any routing.
- Ask for a modest stream latency (e.g. `PULSE_LATENCY_MSEC=20` / `node.latency=480/48000`) for Voxa's own streams only. Never force the global quantum.
- PipeWire's data thread uses RT scheduling via RTKit and the module-rt. In a flatpak, Voxa's Python thread won't be RT, so do AEC on chunks and treat timestamp jitter statistically (median offset), not per-sample.

### Gaps
- I found no measured monitor-jitter numbers for PipeWire 1.x, and no source for RTKit availability inside flatpak.

## 4. Bluetooth: A2DP latency, HFP/mSBC, profile switching, detecting headphones

### Takeaway
A2DP adds large, codec-dependent buffering, and HFP/HSP is a bidirectional low-quality mono profile where headsets often do their own AEC. WirePlumber can auto-switch to the headset profile when an input stream opens. Voxa should detect the BT profile and adapt (widen the AEC delay range on A2DP, rely on headset AEC on HFP) and must not switch profiles itself.

### Cited Findings
- WirePlumber BT roles: `hfp_hf`, `hfp_ag`, `a2dp_sink`, `a2dp_source`, etc. `bluez5.enable-msbc` (wideband HFP) and `bluez5.enable-sbc-xq` exist, and both are on by default in recent versions unless a quirk disables them. There is a node latency-offset property: larger values add constant latency but reduce transport jitter. — [WirePlumber Bluetooth configuration](https://pipewire.pages.freedesktop.org/wireplumber/daemon/configuration/bluetooth.html)
- WirePlumber auto-switches between A2DP and HSP/HFP when an input stream appears. Disable it with `wpctl settings --save bluetooth.autoswitch-to-headset-profile false`. Users report the switch to handsfree drops quality. — [Arch wiki PipeWire](https://wiki.archlinux.org/title/PipeWire) (via search snippet; page blocked for direct fetch)
- A2DP isn't real-time telephony and tolerates more buffering. — [Argenox: BlueZ integration with ALSA](https://argenox.com/library/bluetooth-classic/bluez-integration-with-alsa-and-linux-audio-a-deep-dive)

### Inferences
- **Detection without changing anything:** read the default sink's properties (PulseAudio `pactl list sinks` / pulsectl `server_info().default_sink_name` → `sink.proplist`; PipeWire `api.bluez5.profile`, `api.bluez5.codec`, `device.form_factor` = headset/headphone, `device.bus` = bluetooth, active port type "headphones"). Treat it as **no acoustic echo path** (relax AEC, keep barge-in sensitive) when:
  - the active port is headphones or a headset (wired jack or BT headset in A2DP with a separate laptop mic), or
  - the form factor is headphone/headset.

  Note that a BT headset's speaker leaks little into a laptop mic. Property names are from memory and need checking against `pw-dump` on target hardware.
- On A2DP, the playback delay (150-300+ ms) can exceed the AEC's default delay range and changes on reconnect. Use the measured `pw_time.delay` / `pa_stream_get_latency` for coarse alignment, then let AEC3 fine-tune.
- On HFP/mSBC the headset mic and speaker share one device and usually have on-device AEC/NR. Voxa's canceller can run lightly or be bypassed. Voxa opening the BT mic may itself trigger the autoswitch, so prefer the laptop mic when the user's output is A2DP, unless the user picked the headset mic.

### Gaps
- I didn't fetch measured A2DP latency figures per codec (SBC/AAC/aptX/LDAC). The 150-300 ms figure comes from the brief, not a cited source here.
- I couldn't verify how widely headsets do on-board AEC in HFP. It is vendor-specific and there is no public source.

## 5. Python access: pulsectl, pipewire bindings, sounddevice/PortAudio, GStreamer webrtcdsp, flatpak

### Takeaway
The most direct off-the-shelf match for Voxa is GStreamer `webrtcdsp` + `webrtcechoprobe` (via PyGObject), with TTS playback and mic capture in one pipeline. It does running-time alignment automatically and needs no system routing changes. In a flatpak, `--socket=pulseaudio` is the supported audio route. The native PipeWire socket isn't exposed by default, which rules out loading PipeWire modules or using native PipeWire APIs unless extra filesystem access is granted.

### Cited Findings
- `webrtcdsp`:
  - Pads: S16LE interleaved or F32LE non-interleaved, rates 48000/32000/16000/8000 only (no 44.1 kHz), any channel count. AEC needs a `webrtcechoprobe` on the far-end path before the speaker sink.
  - The probe is found by name (`probe` property, default `webrtcechoprobe0`). It must be in the same top-level pipeline, created before the DSP sink pad activates, and at the same rate (channels may differ).
  - Properties: `echo-cancel` (true), `echo-suppression-level`, `delay-agnostic` (false), `extended-filter`, `noise-suppression`(+level), `gain-control` (true, adaptive-digital), `high-pass-filter`, `voice-detection`, etc.
  - Layout:
    ```
    gst-launch-1.0 far-end-src ! audio/x-raw,rate=48000 ! webrtcechoprobe ! pulsesink \
                   pulsesrc ! audio/x-raw,rate=48000 ! webrtcdsp ! far-end-sink
    ```
  — [GStreamer webrtcdsp docs](https://gstreamer.freedesktop.org/documentation/webrtcdsp/webrtcdsp.html)
- The probe aligns using running-time PTS + the pipeline LATENCY event + an upstream latency query. See section 2. — [gstwebrtcechoprobe.cpp](https://raw.githubusercontent.com/GStreamer/gstreamer/main/subprojects/gst-plugins-bad/ext/webrtcdsp/gstwebrtcechoprobe.cpp)
- Collabora introduced the GStreamer echo canceller in 2016. — [Collabora blog](https://www.collabora.com/about-us/blog/2016/07/08/gstreamer-echo-canceller/)
- Flatpak:
  - PipeWire's portal doc says the PipeWire daemon runs outside the sandbox and its socket is not exposed inside. Sandboxed apps get PipeWire via portals such as camera — [PipeWire portal docs](https://docs.pipewire.org/page_portal.html).
  - Apps are expected to use the PulseAudio socket, and Flatpak doesn't grant the PipeWire socket — [WebKit bug 239682](https://bugs.webkit.org/show_bug.cgi?id=239682).
  - There is no `--socket=pipewire` — [Arch BBS](https://bbs.archlinux.org/viewtopic.php?id=268949).
  - Some apps instead get it via `--filesystem=xdg-run/pipewire-0` — [Flathub Discourse](https://discourse.flathub.org/t/what-is-the-difference-if-the-pipewire-socket-is-read-only/11951).
  - GNOME Software treats PipeWire runtime-dir access as unsandboxed device access — [gnome-software MR 1797](https://gitlab.gnome.org/GNOME/gnome-software/-/merge_requests/1797).
  - There is no microphone portal yet — [Mozilla bug 1726218](https://bugzilla.mozilla.org/show_bug.cgi?id=1726218); [xdg-desktop-portal discussion 1142](https://54ab3db1-c909-41ca-90c6-1ee199732964.proxy.brevy.com/flatpak/xdg-desktop-portal/discussions/1142).
- PipeWire's `libpipewire-module-access` decides client permissions, and flatpak clients are restricted — [man libpipewire-module-access](https://man.archlinux.org/man/extra/pipewire/libpipewire-module-access.7.en).

### Inferences
- **pulsectl** (ctypes binding for libpulse) works over the flatpak pulseaudio socket. It is fine for introspection: default sink/source, sink-input index of Voxa's TTS stream, port types, BT properties, and change events via `event_listen`. It doesn't do streaming. For timestamps on Voxa's own playback, use libpulse streams directly (via ctypes) or GStreamer `pulsesink`, which handles timing.
- **sounddevice/PortAudio:** on PipeWire this goes through ALSA → pipewire-alsa or pulse plugin. `stream.latency` and per-callback `time.outputBufferDacTime` / `inputBufferAdcTime` give usable timestamps on the same clock. That is enough for coarse alignment of an own-sample reference. It can't target a monitor of a single sink input.
- **pipewire python bindings:** no mature official Python binding exists (only C, and Rust `pipewire-rs`), and they need the native socket. Not recommended for a flatpak.
- **Ranking for Voxa:**
  1. GStreamer pipeline: TTS appsrc → `webrtcechoprobe` → `pulsesink`, and `pulsesrc` → `webrtcdsp` → appsink to ASR. Works in a flatpak with `--socket=pulseaudio`, changes no routing, and the reference is exactly Voxa's audio.
  2. Own WebRTC APM (python bindings to webrtc-audio-processing) with own-TTS reference, timestamped via libpulse latency.
  3. Monitor capture restricted with `pa_stream_set_monitor_stream`.

  Avoid loading PipeWire or PA echo-cancel modules into the user's daemon.

### Gaps
- I didn't verify current Python bindings for webrtc-audio-processing 1.x/2.x on PyPI, or whether the GNOME/Freedesktop flatpak runtime ships `gst-plugins-bad` with webrtcdsp built. Check the `org.freedesktop.Platform` 24.08/25.08 manifests.

## 6. Known pitfalls: monitor captures other apps, sample-rate mismatch, device switches

### Takeaway
The default-sink monitor mixes in every app. Fix it by monitoring only Voxa's sink input, or by using its own samples. Rate mismatches matter because webrtcdsp/AEC run only at 8/16/32/48 kHz, so 44.1 kHz TTS must be resampled to the AEC rate. Device switches change latency and clock, so re-sync (reset alignment and AEC state) on every default-device or profile change event.

### Cited Findings
- `monitor.mode` captures "from the monitor ports of the default sink", i.e. everything played there. — [PipeWire docs](https://docs.pipewire.org/page_module_echo_cancel.html)
- `pa_stream_set_monitor_stream` restricts a monitor to one sink input. — [PulseAudio stream.h](https://freedesktop.org/software/pulseaudio/doxygen/stream_8h.html)
- webrtcdsp accepts only 48000/32000/16000/8000 Hz, and the probe must match the DSP rate. — [GStreamer webrtcdsp docs](https://gstreamer.freedesktop.org/documentation/webrtcdsp/webrtcdsp.html)
- The PipeWire AEC backend needs 10 ms-multiple buffers, and the module default is 48 kHz F32P. — [aec-webrtc.cpp](https://raw.githubusercontent.com/PipeWire/pipewire/master/spa/plugins/aec/aec-webrtc.cpp); [module-echo-cancel.c](https://raw.githubusercontent.com/PipeWire/pipewire/master/src/modules/module-echo-cancel.c)
- The module ignored configured targets and followed defaults in some versions. — [pipewire#2939](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/2939)
- BT autoswitch changes the profile when a mic opens. — [WirePlumber BT config](https://pipewire.pages.freedesktop.org/wireplumber/daemon/configuration/bluetooth.html); [Arch wiki PipeWire](https://wiki.archlinux.org/title/PipeWire)

### Inferences
- Run AEC at 48 kHz (or 16 kHz to match faster-whisper and save CPU). Resample TTS output (often 22.05/24/44.1 kHz) once with a good resampler before both playback and reference. That way the reference and what is played are identical sample streams.
- Subscribe to server events (pulsectl `event_listen` for `server`/`sink`/`source` change, or `pa_context_subscribe`). On a default sink/source change or BT profile change: flush the reference history, reset the AEC (create a new APM instance), re-query latency, and re-detect headphones.
- Don't trust latency values while pavucontrol is open (#3191). Verify alignment with a cross-correlation check on the first TTS utterance after any change.

### Gaps
- I found no source on how often users hit mid-session device switches with Voxa-like apps. Needs telemetry or testing on the user's hardware matrix (HDA, USB, BT).
