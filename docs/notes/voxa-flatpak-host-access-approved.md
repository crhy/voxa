---
name: voxa-flatpak-host-access-approved
description: rhY explicitly approved broad host access for the Voxa flatpak (start/stop local AI servers) and wants Voxa to reach Brave tabs, LibreOffice and everything else via permissioned tools.
metadata:
    pinned: false
---

The Voxa roadmap (GitHub issue #6, phase 27) says to avoid gratuitously broad
Flatpak host access. On 2026-09-19 rhY was told that a sandboxed app can only
launch programs on the host through the `org.freedesktop.Flatpak` talk permission
(`flatpak-spawn --host`), and was asked whether to grant it so Voxa can launch and
stop llama-server and Ollama itself. rhY answered: "Yes, let's grant it broad host
access."

So for Voxa it is approved to use `--talk-name=org.freedesktop.Flatpak` and
read-only host filesystem access in the manifest. Keep it visible rather than
hidden: document the permissions in the README and the manifest comments, and make
the app say plainly when it is starting or stopping a host process.

rhY then widened the goal: "We want this app to have access to brave tabs, libre
office, everything." Build that through the roadmap's structured tools, visible
tasks and permission/approval model (milestones D and E) rather than letting model
output run shell commands directly. Model output is still untrusted planning input,
and risky actions still need an understandable approval step.
