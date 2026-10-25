#!/bin/bash
# Builds the High face server bundle that Voxa downloads (voxa/faceinstall.py, #73): a standalone Python with the
# server's packages, Ditto at the pinned commit with Voxa's patch, its checkpoints, the scripts and the portraits,
# packed as one .tar.xz split into parts under GitHub's 2 GB per-file limit, plus the manifest Voxa reads.
#
# The two InsightFace models in Ditto's checkpoints are licensed for non-commercial research, so they are not put
# in the bundle: the manifest lists them as "external" files that Voxa fetches from Ditto's own Hugging Face repo.
#
# Usage: build-bundle.sh --python DIR --site-packages DIR --ditto DIR --stage DIR --out DIR [--version N]
#   --python         a python-build-standalone CPython 3.12 install (for example one that uv manages)
#   --site-packages  the site-packages of a working face server environment, copied as-is (same Python version)
#   --ditto          a Ditto checkout at ditto/UPSTREAM_COMMIT with ditto/stream_pipeline_offline.patch applied,
#                    with checkpoints/ditto_cfg and checkpoints/ditto_pytorch
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
VERSION=1
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHON=$2; shift 2;; --site-packages) SITE=$2; shift 2;; --ditto) DITTO=$2; shift 2;;
    --stage) STAGE=$2; shift 2;; --out) OUT=$2; shift 2;; --version) VERSION=$2; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
: "${PYTHON:?--python}" "${SITE:?--site-packages}" "${DITTO:?--ditto}" "${STAGE:?--stage}" "${OUT:?--out}"
NAME=face-server-$VERSION
BASE_URL=https://github.com/crhy/voxa/releases/download/$NAME
HF=https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/e4a2f60328ee7c32af585ac4b3cce299e4c8e254
AUX=ditto_pytorch/aux_models
# path in the bundle, Hugging Face path, bytes, sha256
EXTERNAL=(
  "ditto-talkinghead/checkpoints/$AUX/det_10g.onnx|$AUX/det_10g.onnx|16923827|5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
  "ditto-talkinghead/checkpoints/$AUX/2d106det.onnx|$AUX/2d106det.onnx|5030888|f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"
)

# The inputs must be what the release says they are.
[ "$(git -C "$DITTO" rev-parse HEAD)" = "$(cat "$HERE/ditto/UPSTREAM_COMMIT")" ] || { echo "Ditto is not at UPSTREAM_COMMIT" >&2; exit 1; }
git -C "$DITTO" apply --reverse --check "$HERE/ditto/stream_pipeline_offline.patch" || { echo "Ditto patch not applied" >&2; exit 1; }
PYV=$("$PYTHON/bin/python3" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
[ -d "$SITE" ] && [ "$(basename "$(dirname "$SITE")")" = "python$PYV" ] || { echo "site-packages is not for Python $PYV" >&2; exit 1; }

echo "== staging in $STAGE"
rm -rf "$STAGE"; mkdir -p "$STAGE" "$OUT"
cp -a "$PYTHON" "$STAGE/python"
rm -f "$STAGE/python/lib/python$PYV/EXTERNALLY-MANAGED"
cp -a "$SITE/." "$STAGE/python/lib/python$PYV/site-packages/"
mkdir -p "$STAGE/ditto-talkinghead"
tar -C "$DITTO" --exclude=.git --exclude=checkpoints/.cache --exclude='__pycache__' \
    --exclude="checkpoints/$AUX/det_10g.onnx" --exclude="checkpoints/$AUX/2d106det.onnx" \
    --exclude='checkpoints/ditto_onnx' --exclude='checkpoints/ditto_trt*' -cf - . | tar -C "$STAGE/ditto-talkinghead" -xf -
for f in face_server.py face_relay.py run-face-server.sh run-sandboxed-py.sh stop-face-server.sh; do
  cp "$HERE/$f" "$STAGE/$f"
done
chmod +x "$STAGE"/*.sh
mkdir -p "$STAGE/portraits" "$STAGE/out"
cp "$REPO"/voxa/ui/assets/portraits/*.jpg "$STAGE/portraits/"
cp "$DITTO/LICENSE" "$STAGE/LICENSE-ditto"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
if grep -rl --include='*.sh' --include='*.py' -e /home/rhy "$STAGE"/*.sh "$STAGE"/*.py; then
  echo "a script still has a developer path" >&2; exit 1
fi

# Smoke test: the bundled Python imports the server's packages from inside the stage.
( cd "$STAGE/ditto-talkinghead" && env -i PATH=/usr/bin:/bin "$STAGE/python/bin/python3" -c \
    'import torch, numpy, cv2, librosa, onnxruntime; print("imports ok, torch", torch.__version__)' )

echo "== packing"
rm -f "$OUT/$NAME".tar.xz.part*
tar -C "$STAGE" -cf - . | xz -T0 -6 | split -b 1900M -d -a 2 - "$OUT/$NAME.tar.xz.part"

echo "== manifest"
python3 - "$OUT" "$NAME" "$VERSION" "$BASE_URL" "$HF" "$(du -sb "$STAGE" | cut -f1)" "${EXTERNAL[@]}" <<'PY'
import hashlib, json, os, sys
out, name, version, base, hf, staged = sys.argv[1:7]
parts = []
for part in sorted(p for p in os.listdir(out) if p.startswith(name + ".tar.xz.part")):
    digest = hashlib.sha256()
    with open(os.path.join(out, part), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    parts.append({"name": part, "url": f"{base}/{part}", "bytes": os.path.getsize(os.path.join(out, part)),
                  "sha256": digest.hexdigest()})
external = []
for item in sys.argv[7:]:
    path, hf_path, size, sha = item.split("|")
    external.append({"path": path, "url": f"{hf}/{hf_path}", "bytes": int(size), "sha256": sha})
extra = sum(e["bytes"] for e in external)
manifest = {"version": version, "download_bytes": sum(p["bytes"] for p in parts) + extra,
            "install_bytes": int(staged) + extra, "parts": parts, "external": external}
with open(os.path.join(out, f"{name}.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(f"{len(parts)} parts, download {manifest['download_bytes'] / 1e9:.2f} GB, install {manifest['install_bytes'] / 1e9:.2f} GB")
PY
ls -l "$OUT"
