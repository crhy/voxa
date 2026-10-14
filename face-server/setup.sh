#!/bin/bash
# Proof-of-quality setup for a neural talking face (Ditto, Apache-2.0). CPU-only run: the GPU is busy.
cd /home/rhy/voxa-neural || exit 1
[ -d venv ] || /home/rhy/voxa-characters/analysis-venv/bin/python -m venv venv
venv/bin/pip -q install torch torchvision torchaudio librosa tqdm filetype imageio opencv_python_headless scikit-image cython imageio-ffmpeg colored onnxruntime "huggingface_hub" 2>&1 | tail -3
venv/bin/python - <<'P'
from huggingface_hub import snapshot_download
p = snapshot_download("digital-avatar/ditto-talkinghead", local_dir="ditto-talkinghead/checkpoints",
                      allow_patterns=["ditto_cfg/*pytorch*", "ditto_pytorch/**"])
print("checkpoints at", p)
P
du -sh venv ditto-talkinghead/checkpoints; echo SETUP DONE
