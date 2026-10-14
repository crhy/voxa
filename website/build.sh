#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="$repo_root/website/dist"

mkdir -p "$output/docs/screenshots"
cp "$repo_root/website/index.html" "$repo_root/website/site.css" "$repo_root/website/site.js" "$output/"
cp "$repo_root/logo.png" "$output/logo.png"
cp "$repo_root/docs/screenshots/voxa-and-voxatest-passed.png" "$output/docs/screenshots/voxa-and-voxatest-passed.png"
cp "$repo_root/docs/screenshots/voxa-and-voxatest-ready.png" "$output/docs/screenshots/voxa-and-voxatest-ready.png"
