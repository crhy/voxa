#!/bin/bash
# Stops the Voxa face server and its relay.
for q in $(pgrep -f 'face_serve[r].py'; pgrep -f 'face_rela[y]'; pgrep -f 'run-face-serve[r]'); do kill "$q" 2>/dev/null; done
sleep 2
pgrep -f 'face_serve[r].py' >/dev/null && echo "still running" || echo "stopped"
