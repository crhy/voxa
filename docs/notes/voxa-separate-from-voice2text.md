---
name: voxa-separate-from-voice2text
description: Voxa and Voice2Text AI are two separate apps that must look nothing alike; a mixup between them already happened once.
metadata:
    pinned: false
---

rhY treats Voxa (`io.github.crhy.voxa`, repo `crhy/voxa`, source in `/Voxa`) and
Voice2Text AI (`io.github.crhy.voice2textai`, repo `crhy/Voice2Text-AI`, source
in `~/Voice2Text-AI`) as two totally separate products. In rhY's words, they
"should be totally separate, and look nothing alike."

Why it matters: on 2026-09-18 rhY launched a freshly built Voxa flatpak, saw the
old Voice2Text AI interface, and thought Voxa had regressed. Both apps were still
installed side by side (the old one system-wide) and had shared the same
microphone-and-waveform icon and a similar name, so they were easy to confuse.
The Voxa icon has since been replaced by the V badge.

So when working on either app: keep names, icons, window titles, config paths and
branding clearly distinct, never let one launch or overwrite the other, and don't
touch the other app's installs without asking. Voxa keeps a migration path for old
Voice2Text config files, but that is compatibility only, not shared identity.
