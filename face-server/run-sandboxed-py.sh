#!/bin/bash
# Runs Ditto inside bubblewrap when it is installed.  On systems without that
# optional package, use only this project's dedicated Python environment.
N=/home/rhy/voxa-neural; PYBASE=/home/rhy/.local/share/uv/python

# bubblewrap is preferred, but it is an optional host package.  Previously a
# missing bwrap made the High face server exit immediately; Voxa then silently
# displayed Medium, making both quality settings look identical.  Keep High
# functional with the same dedicated venv when bubblewrap is unavailable.
if command -v bwrap >/dev/null 2>&1; then
  exec bwrap --ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /tmp --tmpfs /home --tmpfs /media --tmpfs /root \
    --ro-bind "$N" "$N" --bind "$N/out" "$N/out" --ro-bind "$PYBASE" "$PYBASE" \
    --unshare-net --unshare-pid --unshare-ipc --die-with-parent --new-session \
    --setenv HOME /tmp --setenv PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --setenv PYTHONDONTWRITEBYTECODE 1 --chdir "$N/ditto-talkinghead" \
    nice -n 10 "$N/venv/bin/python" "$@"
fi

cd "$N/ditto-talkinghead" || exit 1
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 10 "$N/venv/bin/python" "$@"
