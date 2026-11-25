"""Voxa application package."""

import os

# NumPy 2 madvise()s its big temporaries to huge pages; khugepaged on this machine
# stalls on compaction (compact_fail 111k of 137k stalls, measured 2026-11-25) while
# the local model pins tens of GB, and DTLN per-block work burned 20x kernel time to
# user time - the same work that runs while Voxa is LIVE. Opt out before any voxa
# module imports numpy (this package __init__ runs first, by Python import order).
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

APP_ID = "io.github.crhy.voxa"
APP_NAME = "Voxa"
APP_VERSION = "0.1.9"  # the one place the version is written; see tools/release_check.py
__version__ = APP_VERSION
