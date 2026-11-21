#!/usr/bin/env python3
"""Print the SHA-256 pin for the current Ollama install script.

``voxa/installer.py`` refuses to hand ``https://ollama.com/install.sh`` to a root
shell unless the script's digest appears in ``PINNED_SCRIPT_DIGESTS``. This tool
fetches the script once, hashes the exact bytes it received, and prints the line
to paste into that dictionary. It is a maintainer tool: nothing in the app calls
it, and the digest it prints is only a candidate until a human has read the script.

    python tools/pin_ollama_installer.py            # digest + paste-ready line
    python tools/pin_ollama_installer.py --quiet    # paste-ready line only

Exits non-zero with a plain message when the network is unavailable, so an offline
checkout never produces a digest that looks trustworthy.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxa.installer import INSTALL_SCRIPT_URL, script_digest  # noqa: E402

TIMEOUT_SECONDS = 30.0
_VERSION_RE = re.compile(r"v?(\d+\.\d+\.\d+[A-Za-z0-9.-]*)")


def _fetch(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _release_note(script: str) -> str:
    """Best-effort label for the pin; a human still has to review the script."""
    named = re.search(r"OLLAMA_INSTALL_VERSION[=:]\s*['\"]?v?(\d+\.\d+\.\d+[A-Za-z0-9.-]*)", script)
    if named:
        return f"Ollama {named.group(1)} install.sh"
    found = _VERSION_RE.search(script)
    if found:
        return f"Ollama install.sh, mentions {found.group(1)}"
    return "Ollama install.sh (release not stated in the script)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=INSTALL_SCRIPT_URL, help="script to hash (default: the official installer)")
    parser.add_argument("--quiet", action="store_true", help="print only the paste-ready line")
    args = parser.parse_args()

    try:
        script = _fetch(args.url, TIMEOUT_SECONDS)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            f"Could not download {args.url}: {exc}\n"
            "No digest is printed: pinning a script you have not fetched would be guesswork. "
            "Re-run this tool with network access, then review the script before pasting the line.",
            file=sys.stderr,
        )
        return 1

    digest = script_digest(script)
    line = f'    "{digest}": "{_release_note(script)}",'
    if args.quiet:
        print(line)
        return 0
    print(f"SHA-256 of {args.url} ({len(script.encode('utf-8'))} bytes): {digest}")
    print("\nPaste into PINNED_SCRIPT_DIGESTS in voxa/installer.py after reviewing the script:\n")
    print("PINNED_SCRIPT_DIGESTS: dict[str, str] = {")
    print(line)
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
