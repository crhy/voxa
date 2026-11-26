#!/bin/bash
# re-run camille + greta only (the two rc=1 chars from the v11l run) to see if the
# reference-silent / 0-of-0 deviations are transient.
set -u
export XDG_DATA_HOME=/home/rhy/.local/share
REPO=/home/rhy/strata-fleet/repos/voxa
CFG=/home/rhy/.var/app/io.github.crhy.voxa/config/voxa/config.json
LOG=/home/rhy/.var/app/io.github.crhy.voxa/.local/state/voxa/barge.log
LOGDIR=$REPO/.voxa-bench/live22
TEXTS=$REPO/.voxa-bench/langtexts
cd "$REPO" || exit 1
inst_count() { pgrep -fc 'python3 /app/lib/voxa/voxa.py' 2>/dev/null; }
wait_load() {
  for i in $(seq 12); do
    local load=$(cut -d" " -f1 /proc/loadavg)
    if awk -v l="$load" "BEGIN{exit !(l<3.0)}" && ! pgrep -f "pytes[t]|flatpak-builde[r]" >/dev/null; then
      return 0
    fi
    echo "  busy (load $load), waiting"; sleep 20
  done; return 1
}
kill_all() {
  flatpak kill io.github.crhy.voxa 2>/dev/null
  for i in $(seq 5); do [ "$(inst_count)" -eq 0 ] && return 0; sleep 3; done
  pkill -9 'voxa.p[y]'; pkill -f 'flatpak run --user i[o]'
  for i in $(seq 10); do [ "$(inst_count)" -eq 0 ] && return 0; sleep 2; done
}
run_one() {
  local id="$1" voice="$2" lang="$3"
  kill_all
  wait_load || { echo "== $id ABORT: busy"; return; }
  python3 - "$CFG" "$id" "$voice" <<'PY'
import json, sys
cfg, cid, voice = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cfg)); c["character_id"] = cid; c["tts_voice"] = voice
json.dump(c, open(cfg, "w"), ensure_ascii=False)
PY
  setsid flatpak run --user io.github.crhy.voxa >/dev/null 2>&1 &
  for i in $(seq 40); do sleep 3; gapplication action io.github.crhy.voxa start-listening >/dev/null 2>&1 && break; done
  local n ok=0
  for try in 1 2 3; do
    n=$(inst_count); [ "$n" -eq 1 ] && { ok=1; break; }
    echo "  reset $try: $n instances"; kill_all
    setsid flatpak run --user io.github.crhy.voxa >/dev/null 2>&1 &
    for i in $(seq 40); do sleep 3; gapplication action io.github.crhy.voxa start-listening >/dev/null 2>&1 && break; done
  done
  [ "$ok" -eq 1 ] || { echo "== $id ABORT: never one instance"; kill_all; return; }
  local off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  local tf=""; [ "$lang" != "en" ] && tf="--text-file $TEXTS/$lang.txt"
  /Voxa/.venv/bin/python3 tools/live_reference_check.py --replies 3 --voice "$id" $tf > "$LOGDIR/$id.out" 2>&1
  echo "== $id rc=$? $(tail -3 "$LOGDIR/$id.out" | tr '\n' ' ')"
}
run_one camille "fr-FR-DeniseNeural" "fr"
run_one greta "de-DE-KatjaNeural" "de"
flatpak kill io.github.crhy.voxa 2>/dev/null
echo RERUN_DONE
