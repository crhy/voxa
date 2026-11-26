#!/bin/bash
# task v11d step 4: live_reference_check for all 22 AVATARS, each in its own language.
# based on v11c_live22.sh; scripts and texts now live in the repo (.voxa-bench).
set -u
# The OpenCode shell points XDG_DATA_HOME at a stale copy: every flatpak/gapplication
# call here must see the REAL user dir (notes-v11c).
export XDG_DATA_HOME=/home/rhy/.local/share
REPO=/home/rhy/strata-fleet/repos/voxa
CFG=/home/rhy/.var/app/io.github.crhy.voxa/config/voxa/config.json
LOG=/home/rhy/.var/app/io.github.crhy.voxa/.local/state/voxa/barge.log
LOGDIR=$REPO/.voxa-bench/live22
TEXTS=$REPO/.voxa-bench/langtexts
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

run_one() {
  local id="$1" voice="$2" lang="$3"
  flatpak kill io.github.crhy.voxa 2>/dev/null
  for i in $(seq 10); do flatpak ps 2>/dev/null | grep -q io.github.crhy.voxa || break; sleep 2; done
  python3 - "$CFG" "$id" "$voice" <<'PY'
import json, sys
cfg, cid, voice = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cfg))
c["character_id"] = cid
c["tts_voice"] = voice
json.dump(c, open(cfg, "w"), ensure_ascii=False)
PY
  # Launch the app ourselves and wait until its actions answer (12 s is not enough on
  # this machine after a cold start: the model loads slowly on CPU).
  setsid flatpak run --user io.github.crhy.voxa > /dev/null 2>&1 &
  for i in $(seq 30); do sleep 2; gapplication action io.github.crhy.voxa start-listening >/dev/null 2>&1 && break; done
  local off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  local tf=""
  [ "$lang" != "en" ] && tf="--text-file $TEXTS/$lang.txt"
  /Voxa/.venv/bin/python3 tools/live_reference_check.py --replies 1 --voice "$id" $tf \
      > "$LOGDIR/$id.out" 2>&1
  local rc=$?
  tail -c +"$((off + 1))" "$LOG" 2>/dev/null | grep -c "echo delay:" > "$LOGDIR/$id.echo.txt"
  echo "== $id ($voice, $lang) rc=$rc"
  cat "$LOGDIR/$id.out"
  echo "   echo-delay-lines: $(cat "$LOGDIR/$id.echo.txt")"
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
