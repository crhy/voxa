"""Find, check and download the latest Spaced Linux ISO image.

The ISO is a release asset of crhy/spaced, not a model, so it gets its own fetcher
here instead of a ``smallbrain.ModelFile``; the download behaviour mirrors
``voxa.smallbrain.download`` (a ``.part`` file, a ``Range`` resume, chunked reads,
a sha256 check, and the same error semantics).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from voxa.agent.filematch import STANDARD_NAMES, folder_key, resolve_folder
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.sysupdate import version_key

log = logging.getLogger("voxa.agent.isoimage")

BLOCK = 1024 * 1024
DOWNLOAD_TIMEOUT = 30
ISO_TIMEOUT = 3600.0
# No Spaced Linux ISO is smaller than this, so free space can be checked honestly
# before anything is fetched over the network.
MIN_ISO_BYTES = 1024 * 1024 * 1024
DEFAULT_ARCH = "amd64"
_GB = 1e9


class DownloadError(Exception):
    """A download that cannot be trusted or completed."""


class DownloadCancelled(Exception):
    """The user stopped a download; the partial file is kept for resuming."""


def _loads(text_or_object):
    """Parsed JSON, or None when the text is not JSON (an object passes through)."""
    if isinstance(text_or_object, (dict, list)):
        return text_or_object
    try:
        return json.loads(text_or_object)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _releases(release_json) -> list[dict]:
    """Release objects in a GitHub releases payload (one release counts as a list)."""
    data = _loads(release_json)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _assets(release: dict) -> list[dict]:
    return [item for item in (release.get("assets") or []) if isinstance(item, dict)]


def _iso_asset(release: dict, arch: str) -> dict | None:
    """The release's ISO asset for `arch`, or None.

    Spaced names its deliverables like `spaced-linux-11.26.1-amd64.iso`: the release
    line, then an architecture suffix. A release carrying a single ISO answers for
    whichever architecture was asked for.
    """
    arch = (arch or DEFAULT_ARCH).strip().casefold()
    for asset in _assets(release):
        if str(asset.get("name") or "").casefold().endswith(f"-{arch}.iso"):
            return asset
    isos = [asset for asset in _assets(release) if str(asset.get("name") or "").casefold().endswith(".iso")]
    return isos[0] if len(isos) == 1 else None


def _digest_in_body(body: str, name: str) -> str:
    """A sha256 the release states for `name` in its notes, or "".

    Only a digest the release actually publishes counts; an empty string here means
    "no checksum published", never "verified".
    """
    tokens = re.findall(r"\b[0-9a-fA-F]{64}\b", body)
    if not tokens:
        return ""
    for line in body.splitlines():
        if name.casefold() in line.casefold():
            found = re.search(r"\b[0-9a-fA-F]{64}\b", line)
            if found:
                return found.group(0).casefold()
    return tokens[0].casefold() if len(tokens) == 1 else ""


def pick_iso(release_json, arch: str = DEFAULT_ARCH) -> tuple[str, str, int, str]:
    """Return (name, url, size, sha256) for the newest release's ISO for `arch`.

    Newest is by ``voxa.agent.sysupdate.version_key``. ("", "", 0, "") when the payload
    holds no release or no ISO for that architecture. The digest is filled in only when
    the release publishes one in its notes; the `.sha256` asset itself is read by the
    tool, because its text is not in the release JSON.
    """
    best: tuple[tuple, dict, dict] | None = None
    for release in _releases(release_json):
        asset = _iso_asset(release, arch)
        if asset is None:
            continue
        key = version_key(str(release.get("tag_name") or ""))
        if best is None or key > best[0]:
            best = (key, release, asset)
    if best is None:
        return ("", "", 0, "")
    release, asset = best[1], best[2]
    name = Path(str(asset.get("name") or "")).name
    url = str(asset.get("browser_download_url") or asset.get("url") or "")
    try:
        size = int(asset.get("size"))
    except (TypeError, ValueError):
        size = 0
    return (name, url, max(size, 0), _digest_in_body(str(release.get("body") or ""), name))


def _checksum_asset_url(release_json, name: str) -> str:
    """URL of a published `<iso>.sha256` asset, or ""."""
    wanted = f"{name}.sha256".casefold()
    for release in _releases(release_json):
        for asset in _assets(release):
            if str(asset.get("name") or "").casefold() == wanted:
                return str(asset.get("browser_download_url") or asset.get("url") or "")
    return ""


def _published_digest(text: str) -> str:
    """The first sha256 in a checksum file's text, or ""."""
    found = re.search(r"\b[0-9a-fA-F]{64}\b", text)
    return found.group(0).casefold() if found else ""


def _fetch_release() -> str:
    """The latest Spaced release JSON, through Spaced Update's own fetcher."""
    from voxa.agent.tools.sysupdate import LATEST_URL, fetch

    return fetch(LATEST_URL)


def _fetch_text(url: str) -> str:
    """A small text file from the release (a checksum asset), same fetcher as above."""
    from voxa.agent.tools.sysupdate import fetch

    return fetch(url)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(
    url: str,
    dest: Path,
    expected_sha256: str = "",
    on_progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    opener=urllib.request.urlopen,
    total: int = 0,
) -> Path:
    """Download `url` into `dest`, resuming a partial file and checking the digest.

    `expected_sha256` empty means the release published no checksum: the file is kept
    and reported as unverified, never as verified. A cancelled download keeps the
    `.part` file so the next call resumes with a `Range` header.
    """
    cancel = cancel if cancel is not None else threading.Event()
    report = on_progress or (lambda done, total_bytes: None)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if cancel.is_set():
        raise DownloadCancelled()

    known_total = max(int(total or 0), 0)
    resume_from = part.stat().st_size if part.is_file() else 0
    if known_total and resume_from >= known_total:
        if resume_from == known_total and (
            not expected_sha256 or _sha256(part) == expected_sha256
        ):
            if cancel.is_set():
                raise DownloadCancelled()
            part.replace(dest)
            report(known_total, known_total)
            return dest
        part.unlink(missing_ok=True)
        resume_from = 0

    headers = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)

    try:
        response = opener(request, timeout=DOWNLOAD_TIMEOUT)
    except (urllib.error.URLError, OSError) as exc:
        raise DownloadError("The download failed; please try again.") from exc

    with response:
        status = getattr(response, "status", 200)
        if resume_from and status != 206:
            resume_from = 0
        done = resume_from
        last_report = 0.0
        mode = "ab" if resume_from else "wb"
        try:
            with part.open(mode) as handle:
                while True:
                    if cancel.is_set():
                        raise DownloadCancelled()
                    chunk = response.read(BLOCK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if cancel.is_set():
                        raise DownloadCancelled()
                    now = time.monotonic()
                    if now - last_report >= 0.5:
                        report(done, max(done, known_total))
                        last_report = now
            report(done, max(done, known_total))
        except DownloadCancelled:
            raise
        except OSError as exc:
            raise DownloadError("The download failed; please try again.") from exc

    size = part.stat().st_size
    if known_total and size != known_total:
        part.unlink(missing_ok=True)
        raise DownloadError("The download was damaged; please try again.")
    if expected_sha256 and _sha256(part) != expected_sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"The download of {dest.name} did not match the checksum the release publishes; "
            "only that partial file was removed."
        )
    part.replace(dest)
    report(size, max(size, known_total))
    return dest


def free_bytes(path: str | Path) -> int | None:
    """Free bytes on the file system holding `path`; None when unknown.

    Shaped after ``voxa.faceinstall.home_free_bytes``: df only reads, so it runs
    even when actions are simulated.
    """
    try:
        result = subprocess.run(
            host_command(
                ["sh", "-c", f"df --output=avail -B1 {shlex.quote(str(path))} | tail -n 1"]
            ),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def folder_path(spoken: str) -> str:
    """Absolute directory for a spoken folder name or path; "" when unknown.

    Empty means the user's Downloads folder.
    """
    spoken = (spoken or "").strip()
    home = str(Path.home())
    if not spoken:
        return str(Path(home) / STANDARD_NAMES["DOWNLOAD"])
    if spoken.startswith(("/", ".", "~")):
        return str(Path(spoken).expanduser().resolve())
    key = folder_key(spoken)
    if key is None:
        return ""
    if key == "":
        return home
    reported = ""
    try:
        reported = subprocess.run(
            host_command(["xdg-user-dir", key]),
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        reported = ""
    return resolve_folder(key, home, reported)


def _nearest_existing(path: str) -> str:
    """The closest directory that exists above `path`, for the free-space check."""
    directory = Path(path)
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    return str(directory)


def _gb(value: float | int) -> str:
    return f"{value / _GB:.1f}"


def _log_progress(done: int, total: int) -> None:
    log.info("ISO download: %s of %s bytes", done, total)


def download_iso(
    args: dict[str, str],
    opener=urllib.request.urlopen,
    cancel: threading.Event | None = None,
) -> ToolResult:
    """Find the newest Spaced Linux ISO, check the free space, then download it."""
    arch = (args.get("arch") or DEFAULT_ARCH).strip()
    folder = (args.get("folder") or "").strip()
    destination = folder_path(folder)
    if not destination:
        return ToolResult.failure(f"I do not know the folder {folder}.")

    # Free space is checked before anything is fetched over the network.
    free = free_bytes(_nearest_existing(destination))
    if free is not None and free < MIN_ISO_BYTES:
        return ToolResult.failure(
            f"A Spaced Linux ISO needs at least {_gb(MIN_ISO_BYTES)} GB of free disk space, "
            f"and there is {_gb(free)} GB."
        )

    try:
        release_json = _fetch_release()
    except Exception:
        log.exception("download_iso: the release lookup failed")
        return ToolResult.failure("I could not reach GitHub to find the latest Spaced Linux release.")

    if not _releases(release_json):
        return ToolResult.failure("I could not find a Spaced Linux release on GitHub.")

    name, url, size, digest = pick_iso(release_json, arch)
    if not name:
        return ToolResult.failure(f"The latest Spaced Linux release publishes no ISO image for {arch}.")
    if not url:
        return ToolResult.failure(f"I found the {name} ISO but no address to download it from.")

    if not digest:
        checksum_url = _checksum_asset_url(release_json, name)
        if checksum_url:
            try:
                digest = _published_digest(_fetch_text(checksum_url))
            except Exception:
                log.exception("download_iso: the published checksum could not be read")
                return ToolResult.failure(
                    f"The release publishes a checksum for {name} that I could not read, "
                    "so I will not pretend the download is verified."
                )

    if size and free is not None and free < size:
        return ToolResult.failure(
            f"The {name} download needs about {_gb(size)} GB of free disk space, and there is {_gb(free)} GB."
        )

    dest = Path(destination) / name
    try:
        path = fetch_file(url, dest, digest, _log_progress, cancel, opener=opener, total=size)
    except DownloadCancelled:
        return ToolResult.failure(
            f"The download of {name} was cancelled. The partial file is kept, so asking again "
            "resumes where it stopped."
        )
    except DownloadError as exc:
        return ToolResult.failure(str(exc))
    except Exception:
        log.exception("download_iso failed")
        return ToolResult.failure(f"The download of {name} did not finish.")

    saved = path.stat().st_size
    checked = (
        "I checked it against the checksum the release publishes."
        if digest
        else "The release publishes no checksum, so I could not verify the download."
    )
    return ToolResult.success(
        f"{name} is saved at {path}, {_gb(saved)} GB. {checked} "
        "Next: write it to a flash drive with the desktop ISO writer."
    )


def iso_tools() -> list[Tool]:
    return [
        Tool(
            name="download_iso",
            description=(
                "Find the newest Spaced Linux ISO on GitHub, check there is room for it, "
                "and download it to a folder, checking the release's checksum when it publishes one."
            ),
            parameters={"arch": "which architecture's image", "folder": "where to put it"},
            risk=RiskLevel.REVERSIBLE,
            handler=download_iso,
            timeout_seconds=ISO_TIMEOUT,
        ),
    ]
