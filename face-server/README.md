# Voxa High face server

The neural talking face behind Voxa's **High** facial quality. It animates a
portrait from Voxa's speech audio with [Ditto](https://github.com/antgroup/ditto-talkinghead)
and streams JPEG frames back to Voxa, paced against the real audio playback clock.

Voxa starts it on the host as `~/voxa-neural/run-face-server.sh --steps 5 --max-fps 12`
(see `face_server_command()` in `voxa/window.py`).

## Layout

| File | Purpose |
| --- | --- |
| `face_server.py` | The server: Ditto pipeline, GPU render path, playback-gated frame sender (runs sandboxed) |
| `face_relay.py` | Forwards `127.0.0.1:8765` to the server's Unix socket (runs outside the sandbox) |
| `run-face-server.sh`, `stop-face-server.sh` | Start/stop the relay and server |
| `run-sandboxed-py.sh`, `run-sandboxed.sh` | bubblewrap sandbox: no network, home hidden |
| `render_locked.py`, `after-render.sh` | Offline clip rendering for face packs |
| `probe.sh`, `live_probe.py`, `sync_probe.py`, `sync_measure.py` | Speed and lip-sync measurements |
| `setup.sh` | Creates the Python environment and downloads the Ditto checkpoints |
| `ditto/stream_pipeline_offline.patch` | Voxa's change to Ditto (small queues, playback gate, frame sampling), against `ditto/UPSTREAM_COMMIT` |
| `install.sh` | Symlinks these files into `~/voxa-neural`, fetches Ditto and applies the patch |

Not in Git (large and rebuildable): `venv/` (~6.8 GB), Ditto checkpoints (~2 GB),
`out/`, and the source PNGs in `inputs*/` used by `render_locked.py`.

## Install or restore

```sh
face-server/install.sh     # links the scripts, clones Ditto, applies the patch
~/voxa-neural/setup.sh     # only if venv/ or the checkpoints are missing
```

After `install.sh`, `~/voxa-neural/*.py` and `*.sh` are symlinks into this folder, so
edit them here and commit.

## GPU use

Warp and decode are nearly all of the per-frame GPU time. `face_server.accelerate()`
runs both in fp16 with the portrait features kept on the GPU, hands the warped
features straight to the decoder, and compiles both networks with `torch.compile`
(compiled kernels cached in `out/torch-cache`): about 22 ms a frame instead of 32 ms.
`--no-compile` skips compilation; any failure falls back to Ditto's own path, and the
server log says which mode is in use (`rendering: fp16+compiled`).
