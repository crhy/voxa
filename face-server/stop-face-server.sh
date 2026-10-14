#!/bin/bash
# Stops the Voxa face server and its relay.
for q in $(pgrep -f 'neural/face_serve[r]'; pgrep -f 'face_rela[y]'; pgrep -f 'run-face-serve[r]'); do kill "$q" 2>/dev/null; done
sleep 2
pgrep -f 'neural/face_serve[r]' >/dev/null && echo "still running" || echo "stopped"
