---
name: voxa-verify-before-root
description: Voxa never hands a downloaded Ollama installer to a root shell unless its SHA-256 is on a reviewed pin list; the privileged host policy belongs in Spaced, the user flow and the refusal belong in Voxa.
metadata:
    pinned: false
---

GitHub issue crhy/voxa#132: `voxa/installer.py` fetched `https://ollama.com/install.sh`
and fed the text straight into `pkexec sh -s` (through `flatpak-spawn --host` in the
Flatpak). HTTPS authenticates the connection, not the content: whatever ollama.com
serves that second is what gets root.

The rule now: **no network response is passed directly to a root shell.**
`script_digest()` hashes the bytes already in memory and `verify_install_script()`
matches that digest against `PINNED_SCRIPT_DIGESTS` in `voxa/installer.py` *before*
`subprocess.Popen` starts anything. An unpinned script raises `InstallerError`, which
`voxa/window.py` shows in the install dialog: the message names the digest and gives
the two honest routes — review the script and pin it, or install Ollama yourself from
a terminal. `PINNED_SCRIPT_DIGESTS` is deliberately **empty** until a human has read
the script it pins; `python tools/pin_ollama_installer.py` prints the digest of the
script currently served plus a paste-ready line, and refuses to print anything when
the network is unavailable, so no digest is ever guessed. Until rhY pins one, the
install button explains itself instead of running unverified bytes — that is the
intended state, not a regression.

The boundary with Spaced (why crhy/spaced#312 was not merged as written, and why
Voxa does not call a helper that does not exist yet — `grep -rn -i ollama` in the
Spaced checkout matches only two triage docs):

- **Spaced owns the privileged host policy**: the narrowly scoped versioned host
  helper, install *and* update, architecture/GPU asset selection, safe extraction,
  rollback.
- **Voxa owns the user flow and capability detection**: which helper exists, how
  progress and errors are shown, the cancel button, and — today — the refusal.
  Voxa must not invent a helper path it cannot test, and must not invoke a generic
  privileged shell.

Still open after this slice: the versioned Spaced host helper, the tested update
path for existing installs, architecture/GPU asset selection, partial-install
rollback, and the remaining two test cases (host invocation against the real helper,
rollback). Related: [[voxa-flatpak-host-access-approved]],
[[voxa-spec-lives-in-github-issues]].
