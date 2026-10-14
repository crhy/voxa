#!/bin/bash
# Starts the Voxa face server (sandboxed: no network, home hidden) and its loopback relay.
# Usage: run-face-server.sh [--fake] [--steps N] [--port 8765]
# The caller's Python settings must not leak into the server's own environment.
unset PYTHONPATH PYTHONHOME
N=/home/rhy/voxa-neural; SOCK=$N/out/face.sock; PORT=8765; ARGS=()
while [ $# -gt 0 ]; do case "$1" in --port) PORT=$2; shift 2;; *) ARGS+=("$1"); shift;; esac; done
# One server at a time: a second start (two clicks on High) must not fight over the socket and port.
if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; then
  echo "face server already running on port $PORT"; exit 0
fi
python3 $N/face_relay.py $SOCK $PORT &
RELAY=$!
trap 'kill $RELAY 2>/dev/null' EXIT
$N/run-sandboxed-py.sh $N/face_server.py $SOCK "${ARGS[@]}"
