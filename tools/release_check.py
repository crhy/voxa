#!/usr/bin/env python3
"""Single source of truth for Voxa's release metadata.

``voxa.APP_VERSION`` (voxa/__init__.py) is the one place the version is written.
Everything else is derived from it or must agree with it:

* pyproject.toml declares the version as dynamic, read from ``voxa.APP_VERSION``
* the About dialog imports ``APP_VERSION``
* the newest ``<release>`` in the AppStream metainfo must equal it
* a release tag must be ``v<APP_VERSION>``
* the Flatpak bundle is named ``Voxa-<version>-<arch>.flatpak``

Standard library only, so it runs anywhere CI does before anything is installed:

    python tools/release_check.py                    # validate everything
    python tools/release_check.py --tag v0.1.1       # also validate a release tag
    python tools/release_check.py --print artifact   # print the bundle file name
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCH = "x86_64"
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def app_version(root: Path = ROOT) -> str:
    """The version declared in voxa/__init__.py, read without importing the package."""
    text = (root / "voxa" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"[ \t]*(?:#.*)?$', text, re.MULTILINE)
    if match is None:
        raise ValueError("voxa/__init__.py does not define APP_VERSION = \"x.y.z\"")
    return match.group(1)


def metainfo_latest_release(root: Path = ROOT) -> str | None:
    """The version of the first <release> in the AppStream metainfo (newest first)."""
    tree = ET.parse(root / "io.github.crhy.voxa.metainfo.xml")
    release = tree.getroot().find("./releases/release")
    return release.attrib.get("version") if release is not None else None


def artifact_name(version: str, arch: str = DEFAULT_ARCH) -> str:
    """The versioned Flatpak bundle file name, e.g. Voxa-0.6.0-x86_64.flatpak."""
    return f"Voxa-{version}-{arch}.flatpak"


def check(root: Path = ROOT, tag: str | None = None) -> list[str]:
    """Return a list of human-readable problems; empty means the metadata is consistent."""
    problems: list[str] = []
    try:
        version = app_version(root)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    if not _VERSION_RE.match(version):
        problems.append(f"APP_VERSION {version!r} is not a plain x.y.z version")

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    if "version" in project:
        problems.append(
            "pyproject.toml has a static [project] version; declare it dynamic and read "
            "voxa.APP_VERSION instead so there is only one place to edit"
        )
    else:
        if "version" not in project.get("dynamic", []):
            problems.append('pyproject.toml must list "version" in [project] dynamic')
        attr = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {})
        if attr.get("attr") != "voxa.APP_VERSION":
            problems.append('[tool.setuptools.dynamic] version must be {attr = "voxa.APP_VERSION"}')

    latest = metainfo_latest_release(root)
    if latest != version:
        problems.append(f"AppStream metainfo's newest release is {latest!r} but APP_VERSION is {version!r}")

    if tag is not None and tag != f"v{version}":
        problems.append(f"release tag {tag!r} does not match APP_VERSION (expected v{version})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", help="a git tag (such as v0.1.1) that must match the version")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--print", dest="show", choices=("version", "artifact"), help="print a value and exit")
    args = parser.parse_args(argv)

    problems = check(tag=args.tag)
    if problems:
        for problem in problems:
            print(f"release metadata: {problem}", file=sys.stderr)
        return 1
    if args.show == "version":
        print(app_version())
    elif args.show == "artifact":
        print(artifact_name(app_version(), args.arch))
    else:
        print(f"release metadata OK: version {app_version()}, bundle {artifact_name(app_version(), args.arch)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
