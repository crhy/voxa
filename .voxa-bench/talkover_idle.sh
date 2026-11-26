#!/bin/bash
# task v11d step 4 remaining: VoxaTest --talk-over (5 cases) + 2-min idle face-state
# check. Kept inside one detached process so the live instance is not killed by the
# caller's shell (the 22-char runner used the same setsid-survives pattern).
set -u
export XDG_DATA_HOME=/home/rhy/.local/share
REPO=/home/rhy/strata-fleet/repos/voxa
CFG=/home/rhy/.var/app/io.github.crhy.voxa/config/voxa/config.json
LOG=/home/rhy/.var/app/io.github.crhy.voxa/.local/state/voxa/barge.log
cd "$REPO" || exit 1
inst_count() { pgrep -fc 'python3 /app/lib/voxa/voxa.py' 2>/dev/null; }
kill_all() {
  flatpak kill io.github.crhy.voxa 2>/dev/null
  for i in $(seq 8); do [ "$(inst_count)" -eq 0 ] && return 0; sleep 3; done
  pkill -9 'voxa.p[y]'; pkill -f 'flatpak run --user i[o]'
  for i in $(seq 12); do [ "$(inst_count)" -eq 0 ] && return 0; sleep 2; done
  echo "  ERROR: instances survived kill_all ($(inst_count))"
}
wait_load() {
  for i in $(seq 12); do
    local load=$(cut -d" " -f1 /proc/loadavg)
    if awk -v l="$load" "BEGIN{exit !(l<3.0)}" && ! pgrep -f "pytes[t]|flatpak-builde[r]" >/dev/null; then return 0; fi
    echo "  busy (load $load), waiting"; sleep 20
  done; return 1
}
# keep the grading backend up
if ! pgrep -f 'ollama serve' >/dev/null; then
  setsid bash -c "ollama serve > '$REPO/.voxa-bench/ollama.log' 2>&1 </dev/null &"
fi
for i in $(seq 20); do curl -s 127.0.0.1:11434 >/dev/null && break; sleep 2; done

kill_all
python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); c["character_id"]="aoife"; c["tts_voice"]="en-IE-EmilyNeural"
json.dump(c,open(sys.argv[1],"w"),ensure_ascii=False)
PY
wait_load || { echo "== ABORT busy"; exit 1; }

start_app() {
  setsid flatpak run --user io.github.crhy.voxa >/dev/null 2>&1 &
  for i in $(seq 40); do sleep 3; gapplication action io.github.crhy.voxa start-listening >/dev/null 2>&1 && break; done
}
n=0
for try in 1 2 3 4; do
  kill_all
  start_app
  sleep 8
  n=$(inst_count)
  # require exactly one instance. A listening app logs NOTHING while truly idle
  # (barge.log lines appear only during a reply window), so no log-delta gate here.
  if [ "$n" -eq 1 ]; then echo "  single live instance ok"; break; fi
  echo "  restart $try: instances=$n logdelta=$((a-b))"
done
[ "$(inst_count)" -eq 1 ] || { echo "== ABORT: never exactly one live instance"; kill_all; exit 1; }

echo "==== TALK-OVER ===="
.voxa-bench/venv-live/bin/python -m voxatest run --cases .voxa-bench/talkover5.json --talk-over \
  --reports .voxa-bench/voxatest-reports --json-report .voxa-bench/talkover-report.json 2>&1 | tail -25
grep -o '"obeyed"[^,}]*' .voxa-bench/talkover-report.json 2>/dev/null; tail -8 .voxa-bench/talkover-report.json 2>/dev/null

echo "==== IDLE 120s (face-state changes must be 0; false interrupts 0) ===="
off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
sleep 120
tail -c +"$((off + 1))" "$LOG" > .voxa-bench/idle-window.log 2>/dev/null
echo "face-state changes: $(grep -c 'face-state' .voxa-bench/idle-window.log)"
echo "INTERRUPT lines:   $(grep -cE '\bINTERRUPT\b' .voxa-bench/idle-window.log)"
echo "self-commands:     $(grep -cE 'speak-start' .voxa-bench/idle-window.log)"

# leave the config as the gate expects
python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); c["character_id"]="aoife"; c["tts_voice"]="en-IE-EmilyNeural"
json.dump(c,open(sys.argv[1],"w"),ensure_ascii=False)
PY
flatpak kill io.github.crhy.voxa 2>/dev/null
echo ALLDONE_TALKOVER
