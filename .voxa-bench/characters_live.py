"""issue #72 v11b step 11: run the alone-check for EVERY character in
voxa/ui/avatars.py AVATARS, each speaking its own language.

Per character: set character_id/tts_voice/language in the installed profile, restart
Voxa, start listening, read one localized reply aloud, and let
tools/live_reference_check.py print its three lines. Results land in
.voxa-bench/characters_live.json; aoife/en-IE-EmilyNeural/en is restored at the end.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app/lib/voxa" if False else str(Path(__file__).resolve().parents[1]))
from voxa.ui.avatars import AVATARS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = Path.home() / ".var/app/io.github.crhy.voxa/config/voxa/config.json"
OUT = REPO / ".voxa-bench/characters_live.json"
import os
ENV = {**os.environ, "XDG_DATA_HOME": str(Path.home() / ".local/share")}

TEXTS = {
    "en": "Hello there. How are you doing today? I hope you are having a good day.",
    "es": "Hola. ¿Cómo estás hoy? Espero que estés teniendo un buen día.",
    "fr": "Bonjour. Comment vas-tu aujourd'hui ? J'espère que tu passes une bonne journée.",
    "de": "Hallo. Wie geht es dir heute? Ich hoffe, dir geht es heute richtig gut.",
    "ja": "こんにちは。今日はいかがですか。よければお返事をください。",
    "ar": "مرحبا. كيف حالك اليوم؟ أتمنى أن تكون بصحة جيدة.",
}
LANG = {
    "en-US": "en", "en-IE": "en", "en-GB": "en", "en-IN": "en", "en-NG": "en",
    "es-MX": "es", "fr-FR": "fr", "de-DE": "de", "ja-JP": "ja", "ar-SA": "ar",
}


def _flatpak(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["flatpak", *args], env=ENV, capture_output=True, text=True, timeout=timeout)


def _running() -> bool:
    return "io.github.crhy.voxa" in _flatpak("ps").stdout


def _set_config(character_id: str, voice: str, language: str) -> None:
    cfg = json.loads(CONFIG.read_text())
    cfg["character_id"] = character_id
    cfg["tts_voice"] = voice
    cfg["language"] = language
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"  config: {character_id} {voice} lang={language}", flush=True)


def _restart_app() -> None:
    if _running():
        _flatpak("kill", "io.github.crhy.voxa")
        time.sleep(3)
    app = subprocess.Popen(
        ["flatpak", "run", "--user", "io.github.crhy.voxa"],
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if _running():
            break
        time.sleep(1)
    else:
        raise RuntimeError("Voxa did not start")
    time.sleep(12)  # whisper model load + first sink
    _flatpak("run", "--user", "--command=gapplication", "io.github.crhy.voxa",
             "action", "io.github.crhy.voxa", "start-listening", timeout=120)


def main() -> int:
    results = []
    try:
        for avatar in AVATARS:
            language = LANG[avatar.locale]
            text = TEXTS[language]
            text_file = REPO / f".voxa-bench/text_{avatar.id}.txt"
            text_file.write_text(text, encoding="utf-8")
            _set_config(avatar.id, avatar.voice, language)
            _restart_app()
            run = subprocess.run(
                [sys.executable, str(REPO / "tools/live_reference_check.py"),
                 "--replies", "1", "--per-reply-timeout", "40",
                 "--voice", avatar.id, "--text-file", str(text_file)],
                env={**ENV, "PYTHONUNBUFFERED": "1"}, capture_output=True, text=True, timeout=240)
            lines = run.stdout.splitlines()
            got = {k: next((ln for ln in lines if ln.startswith(k)), "")
                   for k in ("reference silent while speaking:",
                             "alignment healthy while speaking:",
                             "false interrupts while she speaks alone:")}
            print(" ", avatar.id, " | ".join(got.values()), flush=True)
            if run.returncode != 0:
                print("    exit", run.returncode, "stderr:", run.stderr.strip().splitlines()[-3:], flush=True)
            results.append({"character": avatar.id, "voice": avatar.voice,
                            "locale": avatar.locale, "language": language,
                            "exit": run.returncode, "lines": lines, "stderr": run.stderr})
            OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    finally:
        _set_config("aoife", "en-IE-EmilyNeural", "en")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
