"""voxatest: an end-to-end harness that exercises Voxa the way a human user does.

It speaks a prompt out loud, listens on a microphone for Voxa's spoken reply,
transcribes it, and asks a local model to grade the reply. It imports Voxa's
own GTK-free modules (speech, transcription, audio, dictation, model clients)
rather than reimplementing them.
"""

from __future__ import annotations

APP_NAME = "voxatest"
