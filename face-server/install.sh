#!/bin/bash
# Puts this folder's face server where Voxa looks for it (~/voxa-neural) and
# applies Voxa's change to Ditto.  The scripts are symlinked, not copied, so
# the running server is always this repository's code and edits land in Git.
#
# Large, rebuildable parts are not in the repository: the Python environment
# and model checkpoints (setup.sh downloads them), the Ditto source (cloned
# here at the pinned commit) and generated output (out/).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
N=/home/rhy/voxa-neural   # every script in this folder uses this path
FILES=(face_server.py face_relay.py render_locked.py live_probe.py sync_probe.py sync_measure.py
       run-face-server.sh run-sandboxed-py.sh run-sandboxed.sh stop-face-server.sh setup.sh after-render.sh probe.sh)

mkdir -p "$N/out" "$N/portraits" "$N/backups"
backup="$N/backups/replaced-$(date +%Y%m%d-%H%M%S)"
for name in "${FILES[@]}"; do
  target="$N/$name"
  if [ -e "$target" ] && [ ! -L "$target" ]; then
    mkdir -p "$backup"
    mv "$target" "$backup/"
  fi
  ln -sfn "$HERE/$name" "$target"
done
[ -d "$backup" ] && echo "previous copies kept in $backup"

# The server's portraits are Voxa's own still portraits (same 768x768 crop).
cp -n "$REPO"/voxa/ui/assets/portraits/*.jpg "$N/portraits/"

DITTO="$N/ditto-talkinghead"
commit=$(cat "$HERE/ditto/UPSTREAM_COMMIT")
if [ ! -d "$DITTO/.git" ]; then
  git clone https://github.com/antgroup/ditto-talkinghead "$DITTO"
  git -C "$DITTO" checkout -q "$commit"
fi
patch="$HERE/ditto/stream_pipeline_offline.patch"
if git -C "$DITTO" apply --reverse --check "$patch" 2>/dev/null; then
  echo "Ditto patch already applied"
elif git -C "$DITTO" apply --check "$patch" 2>/dev/null; then
  git -C "$DITTO" apply "$patch"
  echo "Ditto patch applied"
else
  echo "Ditto's stream_pipeline_offline.patch does not apply cleanly; compare $DITTO with $patch" >&2
  exit 1
fi

if [ ! -x "$N/venv/bin/python" ] || [ ! -d "$DITTO/checkpoints/ditto_pytorch" ]; then
  echo "The Python environment or checkpoints are missing: run $N/setup.sh"
fi
echo "face server installed in $N"
