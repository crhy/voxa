#!/bin/bash
# task v11e: live wake-word / talk-over / idle check on one of the two mics.
# Usage: v11e_live.sh <A|B> <wake|talkover|idle>
# A = Galyimage Live camera (Voxa's configured mic), B = Generic Blue Microphones.
# Pattern copied from talkover_idle.sh (v11d): one detached process, kills all
# instances, requires EXACTLY ONE live instance, waits for a quiet machine.
set -u
export XDG_DATA_HOME=/home/rhy/.local/share
REPO=/home/rhy/strata-fleet/repos/voxa
CFG=/home/rhy/.var/app/io.github.crhy.voxa/config/voxa/config.json
LOG=/home/rhy/.var/app/io.github.crhy.voxa/.local/state/voxa/barge.log
BENCH=$REPO/.voxa-bench
MIC=${1:-A}
PHASE=${2:-wake}
SRC_A=alsa_input.usb-Image__Galyimage_Live_camera_HU12345679876543-02.analog-stereo
SRC_B=alsa_input.usb-Generic_Blue_Microphones_2111BAB03T18-00.analog-stereo
if [ "$MIC" = A ]; then
  SERIAL='device.serial:Image+_Galyimage_Live_camera_HU12345679876543'
  NAME='Galyimage Live camera Analog Stereo'
  SRC=$SRC_A
else
  SERIAL='device.serial:Generic_Blue_Microphones_2111BAB03T18'
  NAME='Blue Microphones Analog Stereo'
  SRC=$SRC_B
fi
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
  for i in $(seq 24); do
    local load=$(cut -d" " -f1 /proc/loadavg)
    if awk -v l="$load" "BEGIN{exit !(l<3.0)}" && ! pgrep -f "pytes[t]" >/dev/null && ! pgrep -x "flatpak-builder" >/dev/null; then return 0; fi
    echo "  busy (load $load), waiting"; sleep 20
  done; return 1
}
if ! pgrep -f 'ollama serve' >/dev/null; then
  setsid bash -c "ollama serve > '$BENCH/ollama.log' 2>&1 </dev/null &"
fi
for i in $(seq 20); do curl -s 127.0.0.1:11434 >/dev/null && break; sleep 2; done

# --- mic switch (the way a user does it: Voxa's config, with Voxa closed) ---
kill_all
python3 - "$CFG" "$MIC" "$SERIAL" "$NAME" <<'PY'
import json,sys
cfg,mic,serial,name=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
c=json.load(open(cfg))
c["microphone_id"]=serial; c["microphone_name"]=name
c["character_id"]="aoife"; c["tts_voice"]="en-IE-EmilyNeural"
json.dump(c,open(cfg,"w"),ensure_ascii=False)
print(f"  config mic set to {mic}: {serial}")
PY
wait_load || { echo "== ABORT busy"; exit 1; }

if [ "$PHASE" = measure ]; then
  echo "==== ARRIVAL MEASUREMENT (mic $MIC, Voxa closed) ===="
  .voxa-bench/venv-live/bin/python .voxa-bench/measure_arrival.py --source "$SRC"
  echo ALLDONE_MEASURE
  exit 0
fi

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
  if [ "$n" -eq 1 ]; then echo "  single live instance ok"; break; fi
  echo "  restart $try: instances=$n"
done
[ "$(inst_count)" -eq 1 ] || { echo "== ABORT: never exactly one live instance"; kill_all; exit 1; }

logoff=$(stat -c %s "$LOG" 2>/dev/null || echo 0)

if [ "$PHASE" = wake ]; then
  .voxa-bench/venv-live/bin/python -m voxatest run --cases .voxa-bench/wake20.json --max-silent 0 \
    --reports .voxa-bench/voxatest-reports --json-report .voxa-bench/wake-report-$MIC.json 2>&1 | tail -12
  echo "---- wake report summary (mic $MIC) ----"
  .voxa-bench/venv-live/bin/python - ".voxa-bench/wake-report-$MIC.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
c=d.get("comprehension",{})
lc=c.get("layer_counts",{})
total=d["summary"]["total"]; nh=lc.get("not_heard",0)
print(f"cases={total} not_heard={nh} WAKE_HITS={total-nh} of {total}")
for row in d.get("results",[]):
    print(f"  {row.get('case_id')} passed={row.get('passed')} transcript={str(row.get('transcript'))[:70]!r}")
PY
fi

if [ "$PHASE" = talkover ]; then
  echo "==== TALK-OVER 5 (mic $MIC) ===="
  .voxa-bench/venv-live/bin/python -m voxatest run --cases .voxa-bench/talkover5.json --talk-over \
    --reports .voxa-bench/voxatest-reports --json-report .voxa-bench/talkover-report-$MIC.json 2>&1 | tail -12
  echo "---- talkover summary (mic $MIC) ----"
  .voxa-bench/venv-live/bin/python - ".voxa-bench/talkover-report-$MIC.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
c=d.get("comprehension",{})
lc=c.get("layer_counts",{})
total=d["summary"]["total"]; nh=lc.get("not_heard",0)
print(f"cases={total} not_heard={nh} WAKE_HITS={total-nh} of {total}")
obeyed=0
for row in d.get("results",[]):
    h=row.get("hearing") or {}
    yes=bool(h.get("yielded")); passed=bool(row.get("passed"))
    obeyed+=1 if (yes and passed) else 0
    print(f"  {row.get('case_id')} yielded={yes} answered={passed} stop={h.get('stop_latency_ms')} OBEYED={yes and passed}")
print(f"OBEYED {obeyed} of {total}")
PY
fi

if [ "$PHASE" = idle ]; then
  echo "==== IDLE 120s (mic $MIC): face-state and INTERRUPT changes must be 0 ===="
  sleep 120
fi

# collect barge.log delta for this phase
tail -c +"$((logoff + 1))" "$LOG" > ".voxa-bench/barge-$MIC-$PHASE.log" 2>/dev/null
echo "---- barge.log delta (mic $MIC, $PHASE) ----"
echo "lines=$(wc -l < .voxa-bench/barge-$MIC-$PHASE.log) no-wake=$(grep -c 'no-wake' .voxa-bench/barge-$MIC-$PHASE.log) INTERRUPT=$(grep -cE '\bINTERRUPT\b' .voxa-bench/barge-$MIC-$PHASE.log) face-state=$(grep -c 'face-state' .voxa-bench/barge-$MIC-$PHASE.log) self-commands=$(grep -c 'speak-start' .voxa-bench/barge-$MIC-$PHASE.log)"
grep -o "no-wake heard='[^']*'" .voxa-bench/barge-$MIC-$PHASE.log | head -30
if [ "$PHASE" = idle ]; then
  echo "idle face-state: $(grep -c 'face-state' .voxa-bench/barge-$MIC-$PHASE.log)"
  echo "idle INTERRUPT:  $(grep -cE '\bINTERRUPT\b' .voxa-bench/barge-$MIC-$PHASE.log)"
fi

# leave the config the way the gate wants at the very end of a run
python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); c["character_id"]="aoife"; c["tts_voice"]="en-IE-EmilyNeural"
json.dump(c,open(sys.argv[1],"w"),ensure_ascii=False)
PY
flatpak kill io.github.crhy.voxa 2>/dev/null
echo ALLDONE_$PHASE
