#!/bin/bash
# Start the real server with the given number of steps, run three probes, stop it.
cd /home/rhy/voxa-neural
stopall() { for q in $(pgrep -f 'neural/face_serve[r]'; pgrep -f 'face_rela[y]'; pgrep -f 'run-face-serve[r]'); do kill $q 2>/dev/null; done; sleep 2; }
stopall
(setsid nohup ./run-face-server.sh --steps $1 > out/face-server.log 2>&1 < /dev/null &)
n=0; until grep -q 'face server ready' out/face-server.log 2>/dev/null || [ $n -ge 40 ]; do sleep 2; n=$((n+1)); done
echo "== steps $1"
venv/bin/python live_probe.py aoife 6 2>&1 | tail -1
venv/bin/python live_probe.py aoife 6 2>&1 | tail -1
venv/bin/python live_probe.py seamus 3 2>&1 | tail -1
grep 'setup\|failed\|Error' out/face-server.log | tail -3
[ "$2" = keep ] || stopall
