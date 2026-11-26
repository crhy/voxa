#!/bin/bash
# task v11d step 4 (v11e: instrumented): live_reference_check for all 22 AVATARS, each in its
# own language. Hard invariants: EXACTLY ONE Voxa instance per reply, and enough free RAM.
set -u
export XDG_DATA_HOME=/home/rhy/.local/share
REPO=/home/rhy/strata-fleet/repos/voxa
CFG=/home/rhy/.var/app/io.github.crhy.voxa/config/voxa/config.json
LOG=/home/rhy/.var/app/io.github.crhy.voxa/.local/state/voxa/barge.log
LOGDIR=$REPO/.voxa-bench/live22
TEXTS=$REPO/.voxa-bench/langtexts
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

inst_count() { pgrep -fc 'python3 /app/lib/voxa/voxa.py' 2>/dev/null; }
inst_pids() { pgrep -f 'python3 /app/lib/voxa/voxa.py' 2>/dev/null; }
# NOTE: never count via `flatpak ps` - Voxa spawns glycin image loaders via
# `flatpak-spawn --sandbox`, and the portal attributes those sandbox processes to the
# app ID too, so `flatpak ps | grep voxa` reports phantom instances (found 19:38,
# this task: every character ABORTed on "2 instances").

wait_free_ram() {
  # never below ~12 G available; bounded 4 min
  for i in $(seq 8); do
    local avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    [ "$avail" -gt 12000000 ] && return 0
    echo "  low RAM ($((avail/1048576)) G avail), waiting"
    sleep 30
  done
  return 1
}

wait_load() {
  # rhY 21:20 rule: a live check on a loaded machine is meaningless (the hearing worker
  # starves). load < 3.0, and no pytest/flatpak-builder running; bounded 4 min.
  for i in $(seq 8); do
    local load=$(cut -d" " -f1 /proc/loadavg)
    if awk -v l="$load" "BEGIN{exit !(l<3.0)}" \
         && ! pgrep -f "pytes[t]|flatpak-builde[r]" >/dev/null; then
      return 0
    fi
    echo "  machine busy (load $load, or pytest/build running), waiting"
    sleep 30
  done
  return 1
}

kill_all() {
  # flatpak kill sends SIGINT (the app's clean shutdown); the sandbox main process
  # is "python3 /app/lib/voxa/voxa.py" on the host, so SIGKILL it if SIGINT lags.
  flatpak kill io.github.crhy.voxa 2>/dev/null
  for i in $(seq 5); do [ "$(inst_count)" -eq 0 ] && return 0; sleep 3; done
  pkill -9 'voxa.p[y]'
  pkill -f 'flatpak run --user i[o]'
  for i in $(seq 10); do [ "$(inst_count)" -eq 0 ] && return 0; sleep 2; done
  echo "  ERROR: instances survived kill_all"
}

run_one() {
  local id="$1" voice="$2" lang="$3"
  kill_all
  [ "$(inst_count)" -eq 0 ] || echo "  ERROR: $(inst_count) instances survived the kill"
  wait_free_ram || { echo "== $id ABORT: low RAM"; return; }
  wait_load || { echo "== $id ABORT: machine busy"; return; }
  python3 - "$CFG" "$id" "$voice" <<'PY'
import json, sys
cfg, cid, voice = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cfg))
c["character_id"] = cid
c["tts_voice"] = voice
json.dump(c, open(cfg, "w"), ensure_ascii=False)
PY
  setsid flatpak run --user io.github.crhy.voxa >/dev/null 2>&1 &
  for i in $(seq 40); do sleep 3; gapplication action io.github.crhy.voxa start-listening >/dev/null 2>&1 && break; done
  local n ok=0
  for try in 1 2 3; do
    n=$(inst_count)
    if [ "$n" -eq 1 ]; then ok=1; break; fi
    echo "  reset $try: $n Voxa instances before check, restarting"
    kill_all
    setsid flatpak run --user io.github.crhy.voxa >/dev/null 2>&1 &
    for i in $(seq 40); do sleep 3; gapplication action io.github.crhy.voxa start-listening >/dev/null 2>&1 && break; done
  done
  [ "$ok" -eq 1 ] || { echo "== $id ABORT: never exactly one instance"; kill_all; return; }
  echo "  instances-before-check=1"
  local off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  local tf=""
  [ "$lang" != "en" ] && tf="--text-file $TEXTS/$lang.txt"
  /Voxa/.venv/bin/python3 tools/live_reference_check.py --replies 1 --voice "$id" $tf \
      > "$LOGDIR/$id.out" 2>&1
  local rc=$?
  tail -c +"$((off + 1))" "$LOG" 2>/dev/null > "$LOGDIR/$id.logtail"
  grep -c "echo-health" "$LOGDIR/$id.logtail" > "$LOGDIR/$id.echo.txt"
  echo "== $id ($voice, $lang) rc=$rc instances-after=$(inst_count) $(awk '/^MemAvailable:/{printf "avail=%dG", $2/1048576}' /proc/meminfo)"
  cat "$LOGDIR/$id.out"
  echo "   echo-health lines: $(cat "$LOGDIR/$id.echo.txt") $(tail -1 "$LOGDIR/$id.logtail" | grep -o 'echo-health.*' | cut -c1-160)"
}

declare -A CH=(
  [jack]="en-US-GuyNeural en" [grace]="en-US-AriaNeural en"
  [seamus]="en-IE-ConnorNeural en" [aoife]="en-IE-EmilyNeural en"
  [oliver]="en-GB-RyanNeural en" [charlotte]="en-GB-SoniaNeural en"
  [juan]="es-MX-JorgeNeural es" [valentina]="es-MX-DaliaNeural es"
  [etienne]="fr-FR-HenriNeural fr" [camille]="fr-FR-DeniseNeural fr"
  [klaus]="de-DE-ConradNeural de" [greta]="de-DE-KatjaNeural de"
  [hiroshi]="ja-JP-KeitaNeural ja" [sakura]="ja-JP-NanamiNeural ja"
  [omar]="ar-SA-HamedNeural ar" [layla]="ar-SA-ZariyahNeural ar"
  [arjun]="en-IN-PrabhatNeural en" [priya]="en-IN-NeerjaNeural en"
  [koda]="en-US-ChristopherNeural en" [aiyana]="en-US-JennyNeural en"
  [chidi]="en-NG-AbeoNeural en" [amara]="en-NG-EzinneNeural en"
)
for id in jack grace seamus aoife oliver charlotte juan valentina etienne camille klaus greta hiroshi sakura omar layla arjun priya koda aiyana chidi amara; do
  run_one "$id" ${CH[$id]}
done
flatpak kill io.github.crhy.voxa 2>/dev/null
echo ALLDONE
