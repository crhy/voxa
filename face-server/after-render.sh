#!/bin/bash
cd /home/rhy/voxa-neural
./run-sandboxed-py.sh $PWD/render_locked.py > out/locked/render.log 2>&1
for f in out/locked/*.mp4.tmp.mp4; do mv "$f" "${f%.tmp.mp4}"; done
echo "RENDER END $(date +%T)" >> out/locked/render.log
setsid nohup /home/rhy/.local/bin/strata-opencode-starter --model coder-iq1_m --server-only > out/strata-restart.log 2>&1 < /dev/null &
