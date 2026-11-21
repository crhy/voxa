"""Install the High face server bundle on the host (GitHub issue #73, step 2).

Voxa runs in a Flatpak that can only READ ~/.local/share/voxa/face-server, so all
downloading and unpacking runs on the host through one bash script.

Since GitHub issue #117 the release may also publish a .torrent next to the parts. The script then
tries the release's web seeds for the same file, and a file already on disk is reused only when it
hashes to the published checksum. The sha256sum checks before and after every download stay the same
either way, so a peer or a web seed can only ever make the download faster, never different.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from voxa.agent.host import host_command
from voxa.torrentsource import TorrentMeta, read_torrent, total_length, web_seed_urls_for

MANIFEST_URL = "https://github.com/crhy/voxa/releases/download/face-server-1/face-server-1.json"
MIN_VRAM_GB = 6.0

_SHA_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")

# usage: bash -c INSTALL_SCRIPT bash <version> <part count> <name> <url> <bytes> <sha256>
#        [web seed url]... ... <path> <url> <bytes> <sha256> [web seed url]...
# A web seed url (GitHub issue #117) is any further https:// token in that entry's own block:
# curl is handed the release URL and the web seeds for the same output file. With no torrent
# published the argv is exactly what it was before, and the sha256sum checks below stay as they
# are either way, so a peer or a web seed can only ever make the download faster, never different.
INSTALL_SCRIPT = r'''
set -u
# Cancelling stops the download too, not only this script.
trap 'kill $(jobs -p) 2>/dev/null; exit 1' TERM INT HUP
FH=$HOME/.local/share/voxa/face-server
DL=$FH.download
NEW=$FH.new
version=$1
nparts=$2
shift 2
declare -a p_name p_url p_bytes p_sha p_extra
for ((i=0; i<nparts; i++)); do
  p_name[$i]=$1; p_url[$i]=$2; p_bytes[$i]=$3; p_sha[$i]=$4
  shift 4
  extra=()
  while [ "$#" -gt 0 ] && [ "${1#https://}" != "$1" ]; do extra+=("$1"); shift; done
  p_extra[$i]="${extra[*]}"
done
total=0
for ((i=0; i<nparts; i++)); do total=$((total + p_bytes[i])); done
mkdir -p "$DL"
for ((i=0; i<nparts; i++)); do
  name=${p_name[i]}; url=${p_url[i]}; sha=${p_sha[i]}
  f="$DL/$name"
  if [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ]; then
    continue
  fi
  # The web seeds are tried for the same file; the checks below stay as they are, so a peer or
  # a web seed can only make the download faster, never different.
  read -r -a extra <<< "${p_extra[i]}"
  curl -fsSL -C - -o "$f" "$url" ${extra[@]+"${extra[@]}"} &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    got=0
    for ((j=0; j<nparts; j++)); do
      pf="$DL/${p_name[j]}"
      if [ -f "$pf" ]; then got=$((got + $(stat -c %s "$pf"))); fi
    done
    echo "PROGRESS $got $total"
  done
  wait "$pid" || { echo "ERROR download $name"; exit 1; }
  if [ "$(sha256sum "$f" | cut -d' ' -f1)" != "$sha" ]; then
    rm -f "$f"; echo "ERROR checksum $name"; exit 1
  fi
done
rm -rf "$NEW"; mkdir -p "$NEW"
echo UNPACKING
cat "${p_name[@]/#/$DL/}" | tar -xJ -C "$NEW" || { echo "ERROR unpack"; exit 1; }
while [ "$#" -gt 0 ]; do
  epath=$1; eurl=$2; ebytes=$3; esha=$4
  shift 4
  extra=()
  while [ "$#" -gt 0 ] && [ "${1#https://}" != "$1" ]; do extra+=("$1"); shift; done
  mkdir -p "$NEW/$(dirname "$epath")"
  curl -fsSL -o "$NEW/$epath" "$eurl" ${extra[@]+"${extra[@]}"} || { echo "ERROR download $epath"; exit 1; }
  if [ "$(sha256sum "$NEW/$epath" | cut -d' ' -f1)" != "$esha" ]; then
    echo "ERROR checksum $epath"; exit 1
  fi
done
mkdir -p "$NEW/out"
printf '%s\n' "$version" > "$NEW/BUNDLE_VERSION"
rm -rf "$FH"
mv "$NEW" "$FH"
rm -rf "$DL"
echo DONE
'''


@dataclass(frozen=True)
class Part:
    name: str
    url: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    version: str
    download_bytes: int
    install_bytes: int
    parts: tuple[Part, ...]
    external: tuple[Part, ...] = ()
    torrent: str = ""


def _check_torrent_url(value: object) -> str:
    """The https URL of a .torrent published with the release; "" when the release publishes none."""
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or not value.startswith("https://") or not value.endswith(".torrent"):
        raise ValueError(f"bad torrent: {value}")
    return value


def _check_torrent_length(meta: TorrentMeta | None, manifest: Manifest) -> None:
    """Refuse a .torrent whose declared length disagrees with the parts it would feed.

    A torrent for a different build of the bundle would otherwise feed the wrong bytes to the
    web seed step, and only the final checksum would notice.
    """
    if meta is None:
        return
    declared = total_length(meta)
    if declared != sum(part.bytes for part in manifest.parts):
        raise ValueError(
            f"torrent length {declared} disagrees with the manifest's parts "
            f"({sum(part.bytes for part in manifest.parts)})"
        )


def _check_part(item: dict, *, external: bool) -> Part:
    name = item.get("path") if external else item.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("part missing name/path")
    if external:
        if name.startswith("/") or ".." in name.split("/") or "://" in name:
            raise ValueError(f"bad external path: {name}")
    else:
        if "/" in name or name.startswith("."):
            raise ValueError(f"bad part name: {name}")
    url = item.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError(f"bad url: {url}")
    sha = item.get("sha256")
    if not isinstance(sha, str) or not _SHA_RE.match(sha):
        raise ValueError(f"bad sha256: {sha}")
    nbytes = item.get("bytes")
    if not isinstance(nbytes, int):
        raise ValueError(f"bad bytes: {nbytes}")
    return Part(name=name, url=url, bytes=nbytes, sha256=sha)


def parse_manifest(text: str, torrent: TorrentMeta | None = None) -> Manifest:
    """Parse a release manifest.

    Pass the metadata read from the release's .torrent as *torrent* to have its declared length
    cross-checked against the parts it would feed.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"bad JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("manifest is not an object")
    raw_parts = data.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError("no parts")
    parts = tuple(_check_part(p, external=False) for p in raw_parts)
    raw_ext = data.get("external", [])
    if not isinstance(raw_ext, list):
        raise ValueError("external is not a list")
    external = tuple(_check_part(e, external=True) for e in raw_ext)
    torrent_url = _check_torrent_url(data.get("torrent"))
    manifest = Manifest(
        version=str(data.get("version", "")),
        download_bytes=int(data.get("download_bytes", 0)),
        install_bytes=int(data.get("install_bytes", 0)),
        parts=parts,
        external=external,
        torrent=torrent_url,
    )
    _check_torrent_length(torrent, manifest)
    return manifest


def fetch_manifest(timeout: float = 15.0) -> Manifest:
    with urllib.request.urlopen(MANIFEST_URL, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
    return parse_manifest(text)


def gpu_problem() -> str:
    from voxa.hardware import _nvidia_vram_gb

    vram = _nvidia_vram_gb()
    if vram is not None and vram >= MIN_VRAM_GB:
        return ""
    return "The High face needs an NVIDIA graphics card with at least 6 GB of video memory."


def install_command(manifest: Manifest, torrent: TorrentMeta | None = None) -> list[str]:
    """The host command that installs the bundle.

    With no *torrent* the argv is exactly what it has always been. Pass the metadata read from the
    .torrent published with the release and each part gains its web seeds as extra curl URLs; the
    checksum pre-check and post-check in the script are untouched, so a web seed can only ever
    make the download faster, never different.
    """
    args = [manifest.version, str(len(manifest.parts))]
    for p in manifest.parts:
        args += [p.name, p.url, str(p.bytes), p.sha256, *web_seed_urls_for(torrent, p)]
    for e in manifest.external:
        args += [e.name, e.url, str(e.bytes), e.sha256, *web_seed_urls_for(torrent, e)]
    return host_command(["bash", "-c", INSTALL_SCRIPT, "bash", *args])


def fetch_torrent(manifest: Manifest, timeout: float = 15.0) -> TorrentMeta | None:
    """Read the .torrent published with the release; None when the release publishes none.

    A torrent that cannot be read raises TorrentError, and the caller keeps the plain release
    download: an unreadable torrent must never change what is installed, only how fast it arrives.
    """
    if not manifest.torrent:
        return None
    with urllib.request.urlopen(manifest.torrent, timeout=timeout) as resp:
        data = resp.read()
    meta = read_torrent(data)
    _check_torrent_length(meta, manifest)
    return meta


def home_free_bytes() -> int | None:
    """Free bytes on the host's home file system (where the face server goes); None when unknown."""
    try:
        result = subprocess.run(
            host_command(["sh", "-c", 'df --output=avail -B1 "$HOME" | tail -n 1']),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def install(
    manifest: Manifest,
    on_progress: Callable[[float, str], None],
    cancel: threading.Event,
    popen=subprocess.Popen,
    free_bytes: Callable[[], int | None] | None = None,
    torrent: TorrentMeta | None = None,
) -> tuple[bool, str]:
    # The parts and the unpacked server are on disk together for a moment.
    needed = manifest.download_bytes + manifest.install_bytes
    free = (free_bytes or home_free_bytes)()
    if free is not None and free < needed:
        return (
            False,
            f"The High face needs about {needed / 1e9:.0f} GB of free disk space, and there is {free / 1e9:.0f} GB.",
        )
    proc = popen(
        install_command(manifest, torrent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    for line in proc.stdout:
        if cancel.is_set():
            proc.terminate()
            return (False, "Cancelled.")
        line = line.strip()
        if line.startswith("PROGRESS"):
            fields = line.split()
            if len(fields) == 3:
                a = float(fields[1])
                b = float(fields[2])
                if b > 0:
                    pct = int(100 * a / b)
                    on_progress(0.95 * a / b, f"Downloading the High face… {pct}%")
        elif line == "UNPACKING":
            on_progress(0.96, "Unpacking the High face…")
        elif line == "DONE":
            on_progress(1.0, "High face installed.")
            return (True, "")
        elif line.startswith("ERROR"):
            proc.wait()
            return (False, _error_sentence(line))
    proc.wait()
    return (False, "The High face download stopped.")


def _error_sentence(line: str) -> str:
    fields = line.split()
    kind = fields[1] if len(fields) > 1 else ""
    target = " ".join(fields[2:]) if len(fields) > 2 else ""
    if kind == "download":
        return f"The High face download of {target} failed."
    if kind == "checksum":
        return f"The High face file {target} failed its checksum."
    if kind == "unpack":
        return "The High face download could not be unpacked."
    return "The High face download failed."


def remove_command() -> list[str]:
    return host_command(
        [
            "bash",
            "-c",
            'rm -rf "$HOME/.local/share/voxa/face-server" '
            '"$HOME/.local/share/voxa/face-server.download" '
            '"$HOME/.local/share/voxa/face-server.new"',
        ]
    )


def gb(nbytes: int) -> str:
    """Format a byte count as gigabytes with one decimal place."""
    return f"{nbytes / (1024 ** 3):.1f}"


def offer_text(manifest: Manifest) -> tuple[str, str]:
    """Heading and body for the install confirmation dialog."""
    heading = "Install the High face?"
    body = (
        "The High face is a full neural talking face. It is a one-time download of about "
        f"{gb(manifest.download_bytes)} GB and uses about {gb(manifest.install_bytes)} GB of disk space. "
        "It needs an NVIDIA graphics card, and uses about 5 GB of its memory while it runs."
    )
    return heading, body
