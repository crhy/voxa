#!/bin/bash
# Runs the Ditto proof with no network and with the home folder hidden: only this project is visible
# (code and checkpoints read-only, out/ writable) plus the Python runtime it was installed with.
N=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# A downloaded face server brings its own Python in python/; a developer setup has a venv/.
if [ -x "$N/python/bin/python3" ]; then PY=$N/python/bin/python3; PYBASE=$N/python
else PY=$N/venv/bin/python; PYBASE=$(sed -n 's/^home = //p' "$N/venv/pyvenv.cfg" 2>/dev/null); PYBASE=${PYBASE%/bin}; fi
exec bwrap --ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /tmp --tmpfs /home --tmpfs /media --tmpfs /root \
  --ro-bind $N $N --bind $N/out $N/out --ro-bind "$PYBASE" "$PYBASE" \
  --unshare-net --unshare-pid --unshare-ipc --die-with-parent --new-session \
  --setenv HOME /tmp --setenv PYTHONDONTWRITEBYTECODE 1 --chdir $N/ditto-talkinghead \
  nice -n 10 "$PY" inference.py --data_root ./checkpoints/ditto_pytorch \
  --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl \
  --audio_path "$1" --source_path "$2" --output_path "$3"
