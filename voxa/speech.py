from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import edge_tts
except ImportError:  # The offline fallback still works in source-only installs.
    edge_tts = None

Gst: Any = None  # Initialized lazily so importing this module needs no GStreamer or PyGObject.
GLib: Any = None

NATURAL_VOICE_ATTEMPTS = 2
NATURAL_VOICE_RETRY_SECONDS = 0.35
LOGGER = logging.getLogger(__name__)

# Speech engines may pronounce Markdown emphasis and list markers as
# "asterisk".  Leave the displayed reply untouched, but remove those marks
# at the common speech boundary so streamed, prepared, and fallback voices
# all receive the same speakable text.  An optional backslash removes the
# Markdown escape along with the mark rather than making Voxa say "backslash".
_ASTERISK_MARKS = re.compile(r"\\?[*\u2217\uff0a]")


def _speakable_text(text: str) -> str:
    return _ASTERISK_MARKS.sub("", text)


def _download_google_voice(text: str, path: str) -> None:
    """Download natural MP3 speech without pulling in gTTS's conflicting CLI dependencies."""
    chunks = textwrap.wrap(text, width=180, break_long_words=True, replace_whitespace=True)
    if not chunks:
        raise RuntimeError("No text was provided for speech")
    headers = {"User-Agent": "Mozilla/5.0"}
    with open(path, "wb") as audio:
        for chunk in chunks:
            query = urlencode({"ie": "UTF-8", "client": "tw-ob", "tl": "en", "q": chunk})
            request = Request(f"https://translate.google.com/translate_tts?{query}", headers=headers)
            with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS host
                audio.write(response.read())


def _word_timing_option() -> dict:
    """Ask Edge TTS for per-WORD timings. Newer versions send only sentence timings unless asked, which left
    the face without the word times its lip sync is built on."""
    try:
        import inspect

        if edge_tts is not None and "boundary" in inspect.signature(edge_tts.Communicate.__init__).parameters:
            return {"boundary": "WordBoundary"}
    except (TypeError, ValueError):
        pass
    return {}


_WORD_TIMING = _word_timing_option()


def _run_quietly(coro):
    """Run *coro* on a fresh event loop, closing async generators before the loop is closed."""
    loop = asyncio.new_event_loop()
    try:
        try:
            value = loop.run_until_complete(coro)
        except BaseException:
            loop.run_until_complete(loop.shutdown_asyncgens())
            raise
        loop.run_until_complete(loop.shutdown_asyncgens())
        return value
    finally:
        loop.close()


def _ui_once(callback, *args) -> None:
    """Run ``callback`` once on the GTK thread at default priority.

    Idle priority is not enough: a window that is animating can keep idle callbacks waiting, which delayed
    the start of speech and the Speaking/Ready captions.
    """

    def run() -> bool:
        callback(*args)
        return False

    if GLib is None:
        # GLib is loaded lazily with GStreamer; a path that has not touched GStreamer yet (fetching the voice
        # to a file) must not find it missing.
        _ensure_gstreamer()
    GLib.idle_add(run, priority=GLib.PRIORITY_DEFAULT)


def _ensure_gstreamer() -> Any:
    global Gst, GLib
    if Gst is None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib as _GLib
        from gi.repository import Gst as _Gst

        _Gst.init(None)
        Gst = _Gst
        GLib = _GLib
    return Gst


class Prepared:
    """One utterance whose audio is being (or has been) fetched ahead of time."""

    def __init__(self, text: str, rate: int, voice: str) -> None:
        self.text, self.rate, self.voice = text, rate, voice
        self.path = ""
        self.words: list = []
        self.error = ""
        self.extra = None  # whatever on_ready attached (the neural face's frame buffer)
        self.ready = threading.Event()
        self.cancelled = threading.Event()

    def discard(self) -> None:
        """Not going to be played: stop fetching and remove the file."""
        self.cancelled.set()
        if self.path:
            with contextlib.suppress(OSError):
                os.unlink(self.path)


class SpeechService:
    """Stream natural speech, with optional online and eSpeak NG fallbacks."""

    def __init__(
        self,
        *,
        allow_online_fallback: bool = True,
        allow_offline_fallback: bool = True,
    ) -> None:
        self.allow_online_fallback = allow_online_fallback
        self.allow_offline_fallback = allow_offline_fallback
        self.pipeline: Any | None = None
        self.appsrc: Any | None = None
        self.temp_path: str | None = None
        self.cancel_event = threading.Event()
        self._on_started = None
        self._on_done = None
        self._on_error = None
        self._on_words = None
        self._started_at = None
        self._started_emitted = False
        self._finished = False
        self._before_play = None

    def speak(
        self,
        text: str,
        rate: int,
        voice: str,
        *,
        on_started,
        on_done,
        on_error,
        on_words=None,
        before_play=None,
    ) -> None:
        text = _speakable_text(text)
        self.stop()
        cancel_event = threading.Event()
        self.cancel_event = cancel_event
        self._on_started = on_started
        self._on_done = on_done
        self._on_error = on_error
        self._on_words = on_words
        self._before_play = before_play
        self._started_at = None
        self._started_emitted = False
        self._finished = False

        if edge_tts is not None:
            try:
                if self._before_play is not None:
                    # Something (the neural face) needs the whole sentence before it is heard: fetch the
                    # voice to a file first, hand it over, then play the file.
                    self._start_natural_file(text, rate, voice, cancel_event)
                else:
                    self._start_natural_stream(text, rate, voice, cancel_event)
                return
            except Exception as exc:  # noqa: BLE001 - multimedia boundary
                natural_error = str(exc)
        else:
            natural_error = "Edge TTS is unavailable"

        self._start_fallback_worker(text, rate, cancel_event, natural_error)

    def prepare(self, text: str, rate: int, voice: str, on_ready=None) -> Prepared:
        """Fetch the voice for ``text`` in the background, without touching what is playing now.

        ``on_ready(item)`` runs in the worker thread once the audio file exists (used to start the neural face
        on it early). Play the result with :meth:`speak_prepared`.
        """
        text = _speakable_text(text)
        item = Prepared(text=text, rate=rate, voice=voice)

        def worker() -> None:
            try:
                if edge_tts is None:
                    raise RuntimeError("Edge TTS is unavailable")
                for attempt in range(NATURAL_VOICE_ATTEMPTS):
                    try:
                        item.path, item.words = _run_quietly(
                            self._fetch_to_file(text, rate, voice, item.cancelled.is_set)
                        )
                        item.error = ""
                        if on_ready is not None and item.path and not item.cancelled.is_set():
                            try:
                                on_ready(item)
                            except Exception:  # noqa: BLE001 - preparation extras must never lose the audio
                                LOGGER.warning("Prepared-speech callback failed", exc_info=True)
                        break
                    except Exception as exc:  # noqa: BLE001 - network/service boundary
                        item.error = str(exc)
                        if item.cancelled.is_set() or attempt + 1 == NATURAL_VOICE_ATTEMPTS:
                            break
                        LOGGER.info("Natural voice preparation failed; retrying: %s", exc)
                        time.sleep(NATURAL_VOICE_RETRY_SECONDS)
            except Exception as exc:  # noqa: BLE001 - missing dependency boundary
                item.error = str(exc)
            finally:
                item.ready.set()

        threading.Thread(target=worker, name="natural-speech-prepare", daemon=True).start()
        return item

    def speak_prepared(self, item: Prepared, *, on_started, on_done, on_error, on_words=None, before_play=None) -> None:
        """Play audio fetched by :meth:`prepare` (waiting for it if it is not ready yet)."""
        self.stop()
        cancel_event = threading.Event()
        self.cancel_event = cancel_event
        self._on_started, self._on_done, self._on_error = on_started, on_done, on_error
        self._on_words, self._before_play = on_words, None
        self._started_at = None
        self._started_emitted = False
        self._finished = False

        def worker() -> None:
            item.ready.wait(30)
            if cancel_event.is_set() or cancel_event is not self.cancel_event:
                item.discard()
                return
            if item.error or not item.path:
                _ui_once(self._begin_offline_fallback, item.text, item.rate, cancel_event, item.error or "no audio")
                return
            if on_words is not None and item.words:
                on_words(item.words)
            if before_play is not None:
                try:
                    before_play(item)
                except Exception:  # noqa: BLE001 - callback must not prevent speech
                    LOGGER.warning("Prepared-speech playback callback failed", exc_info=True)
            _ui_once(self._play_file, item.path, cancel_event)

        threading.Thread(target=worker, name="natural-speech-prepared", daemon=True).start()

    async def _fetch_to_file(self, text: str, rate: int, voice: str, should_stop) -> tuple[str, list]:
        """Download one utterance to a temporary MP3. Returns (path, word timings)."""
        percent = max(-50, min(50, round(((rate - 180) / 120) * 50)))
        communicate = edge_tts.Communicate(text, voice or "en-US-AriaNeural", rate=f"{percent:+d}%", **_WORD_TIMING)
        words: list[tuple[str, float, float]] = []
        with tempfile.NamedTemporaryFile(prefix="voxa-speech-", suffix=".mp3", delete=False) as handle:
            path = handle.name
            wrote = False
            stream = communicate.stream()
            try:
                async for message in stream:
                    if should_stop():
                        break
                    kind = message.get("type")
                    if kind == "WordBoundary":
                        words.append(
                            (
                                message.get("text", ""),
                                message.get("offset", 0) / 10_000_000,
                                message.get("duration", 0) / 10_000_000,
                            )
                        )
                    elif kind == "audio" and message.get("data"):
                        handle.write(message["data"])
                        wrote = True
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(path)
                raise
            finally:
                with contextlib.suppress(Exception):
                    await stream.aclose()
        if not wrote or should_stop():
            with contextlib.suppress(OSError):
                os.unlink(path)
            if should_stop():
                return "", []
            raise RuntimeError("The speech service returned no audio.")
        return path, words

    def _start_natural_file(self, text: str, rate: int, voice: str, cancel_event: threading.Event) -> None:
        threading.Thread(
            target=self._natural_file_worker,
            args=(text, rate, voice, cancel_event),
            name="natural-speech-file",
            daemon=True,
        ).start()

    def _natural_file_worker(self, text: str, rate: int, voice: str, cancel_event: threading.Event) -> None:
        path = ""
        for attempt in range(NATURAL_VOICE_ATTEMPTS):
            try:
                path = _run_quietly(self._fetch_natural_audio(text, rate, voice, cancel_event))
                break
            except Exception as exc:  # noqa: BLE001 - network/service boundary
                if cancel_event.is_set() or cancel_event is not self.cancel_event:
                    return
                if attempt + 1 == NATURAL_VOICE_ATTEMPTS:
                    _ui_once(self._begin_offline_fallback, text, rate, cancel_event, str(exc))
                    return
                LOGGER.info("Natural voice download failed; retrying: %s", exc)
                time.sleep(NATURAL_VOICE_RETRY_SECONDS)
        if not path or cancel_event.is_set() or cancel_event is not self.cancel_event:
            if path:
                with contextlib.suppress(OSError):
                    os.unlink(path)
            return
        if self._before_play is not None:
            try:
                self._before_play(path)
            except Exception:  # noqa: BLE001 - callback must not prevent speech
                pass
        _ui_once(self._play_file, path, cancel_event)

    async def _fetch_natural_audio(self, text: str, rate: int, voice: str, cancel_event: threading.Event) -> str:
        """Download the whole utterance to a temporary MP3; word timings are reported as they arrive."""
        percent = max(-50, min(50, round(((rate - 180) / 120) * 50)))
        communicate = edge_tts.Communicate(text, voice or "en-US-AriaNeural", rate=f"{percent:+d}%", **_WORD_TIMING)
        words: list[tuple[str, float, float]] = []
        with tempfile.NamedTemporaryFile(prefix="voxa-speech-", suffix=".mp3", delete=False) as handle:
            path = handle.name
            wrote = False
            stream = communicate.stream()
            try:
                async for message in stream:
                    if cancel_event.is_set() or cancel_event is not self.cancel_event:
                        break
                    kind = message.get("type")
                    if kind == "WordBoundary":
                        words.append(
                            (
                                message.get("text", ""),
                                message.get("offset", 0) / 10_000_000,
                                message.get("duration", 0) / 10_000_000,
                            )
                        )
                    elif kind == "audio" and message.get("data"):
                        handle.write(message["data"])
                        wrote = True
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(path)
                raise
            finally:
                with contextlib.suppress(Exception):
                    await stream.aclose()
        if not wrote:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise RuntimeError("The speech service returned no audio.")
        if words and self._on_words is not None:
            self._on_words(words)
        return path

    def _start_natural_stream(
        self,
        text: str,
        rate: int,
        voice: str,
        cancel_event: threading.Event,
    ) -> None:
        _ensure_gstreamer()
        pipeline = Gst.parse_launch(
            "appsrc name=source format=bytes block=true max-bytes=1048576 ! "
            "queue max-size-bytes=2097152 ! "
            "decodebin ! audioconvert ! audioresample ! autoaudiosink"
        )
        appsrc = pipeline.get_by_name("source")
        if appsrc is None:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer appsrc is unavailable.")

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self._on_eos, cancel_event)
        bus.connect("message::error", self._on_error_message, cancel_event)
        self.pipeline = pipeline
        self.appsrc = appsrc
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            self._finish_media()
            raise RuntimeError("The streaming speech pipeline could not start.")

        threading.Thread(
            target=self._natural_worker,
            args=(text, rate, voice, appsrc, cancel_event),
            name="natural-speech-stream",
            daemon=True,
        ).start()

    def _natural_worker(
        self,
        text: str,
        rate: int,
        voice: str,
        appsrc,
        cancel_event: threading.Event,
    ) -> None:
        _ensure_gstreamer()
        received_audio = threading.Event()
        for attempt in range(NATURAL_VOICE_ATTEMPTS):
            try:
                _run_quietly(
                    self._push_natural_audio(
                        text,
                        rate,
                        voice,
                        appsrc,
                        cancel_event,
                        lambda: self._mark_audio_received(cancel_event, received_audio),
                    )
                )
                return
            except Exception as exc:  # noqa: BLE001 - network/service boundary
                if cancel_event.is_set() or cancel_event is not self.cancel_event:
                    return
                if received_audio.is_set():
                    _ui_once(
                        self._emit_error_for,
                        f"Natural voice was interrupted: {exc}",
                        cancel_event,
                    )
                    return
                if attempt + 1 == NATURAL_VOICE_ATTEMPTS:
                    _ui_once(
                        self._begin_offline_fallback,
                        text,
                        rate,
                        cancel_event,
                        str(exc),
                    )
                    return
                LOGGER.info("Natural voice stream failed; retrying: %s", exc)
                time.sleep(NATURAL_VOICE_RETRY_SECONDS)

    async def _push_natural_audio(
        self,
        text: str,
        rate: int,
        voice: str,
        appsrc,
        cancel_event: threading.Event,
        on_first_audio,
    ) -> None:
        _ensure_gstreamer()
        if edge_tts is None:
            raise RuntimeError("Edge TTS is unavailable")
        percent = max(-50, min(50, round(((rate - 180) / 120) * 50)))
        communicate = edge_tts.Communicate(
            text,
            voice or "en-US-AriaNeural",
            rate=f"{percent:+d}%",
            **_WORD_TIMING,
        )
        received_audio = False
        cancelled = False
        stream = communicate.stream()
        try:
            async for message in stream:
                if cancel_event.is_set() or cancel_event is not self.cancel_event:
                    cancelled = True
                    break
                if message.get("type") == "WordBoundary":
                    if self._on_words is not None:
                        word = message.get("text", "")
                        start_s = message.get("offset", 0) / 10_000_000
                        duration_s = message.get("duration", 0) / 10_000_000
                        self._on_words([(word, start_s, duration_s)])
                    continue
                if message.get("type") != "audio":
                    continue
                data = message.get("data", b"")
                if not data:
                    continue
                if not received_audio:
                    received_audio = True
                    on_first_audio()
                buffer = Gst.Buffer.new_allocate(None, len(data), None)
                buffer.fill(0, data)
                flow = appsrc.emit("push-buffer", buffer)
                if flow != Gst.FlowReturn.OK:
                    if cancel_event.is_set():
                        cancelled = True
                        break
                    raise RuntimeError(f"GStreamer rejected speech audio ({flow.value_nick}).")
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

        if cancelled:
            return
        if not received_audio:
            raise RuntimeError("The speech service returned no audio.")
        if self._is_current(cancel_event) and appsrc is self.appsrc:
            appsrc.emit("end-of-stream")

    def _mark_audio_received(
        self,
        cancel_event: threading.Event,
        received_audio: threading.Event,
    ) -> None:
        received_audio.set()
        _ui_once(self._emit_started_for, cancel_event)

    def _begin_offline_fallback(
        self,
        text: str,
        rate: int,
        cancel_event: threading.Event,
        natural_error: str,
    ) -> bool:
        _ensure_gstreamer()
        if not self._is_current(cancel_event):
            return False
        self._finish_media()
        self._start_fallback_worker(text, rate, cancel_event, natural_error)
        return False

    def _start_fallback_worker(
        self,
        text: str,
        rate: int,
        cancel_event: threading.Event,
        natural_error: str,
    ) -> None:
        threading.Thread(
            target=self._synthesize_fallback,
            args=(text, rate, cancel_event, natural_error),
            name="backup-speech-synthesis",
            daemon=True,
        ).start()

    def _synthesize_fallback(
        self,
        text: str,
        rate: int,
        cancel_event: threading.Event,
        natural_error: str,
    ) -> None:
        """Try a natural Google voice before considering the robotic offline voice."""
        fallback_error = f"Edge TTS failed ({natural_error})"
        if self.allow_online_fallback:
            path = ""
            try:
                with tempfile.NamedTemporaryFile(
                    prefix="voxa-google-", suffix=".mp3", delete=False
                ) as handle:
                    path = handle.name
                _download_google_voice(text, path)
                if cancel_event.is_set() or cancel_event is not self.cancel_event:
                    with contextlib.suppress(OSError):
                        os.unlink(path)
                    return
                if not os.path.getsize(path):
                    raise RuntimeError("Google TTS returned no audio")
                if self._before_play is not None:
                    try:
                        self._before_play(path)
                    except Exception:  # noqa: BLE001 - callback must not prevent speech
                        pass
                _ui_once(self._play_file, path, cancel_event)
                return
            except Exception as exc:  # noqa: BLE001 - network/service boundary
                if path:
                    with contextlib.suppress(OSError):
                        os.unlink(path)
                fallback_error += f"; Google TTS failed ({exc})"

        if not self.allow_offline_fallback:
            message = f"Natural test voice unavailable; acoustic test was not run. {fallback_error}"
            _ui_once(self._emit_error_for, message, cancel_event)
            return

        LOGGER.warning("Natural voices unavailable; using eSpeak NG fallback: %s", fallback_error)
        self._synthesize_offline(text, rate, cancel_event, fallback_error)

    def _synthesize_offline(
        self,
        text: str,
        rate: int,
        cancel_event: threading.Event,
        natural_error: str,
    ) -> None:
        _ensure_gstreamer()
        try:
            result = subprocess.run(
                ["espeak-ng", "--stdout", "-s", str(rate), text],
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
            )
            if cancel_event.is_set():
                return
            with tempfile.NamedTemporaryFile(
                prefix="voxa-",
                suffix=".wav",
                delete=False,
            ) as handle:
                handle.write(result.stdout)
                path = handle.name
            if self._before_play is not None:
                try:
                    self._before_play(path)
                except Exception:  # noqa: BLE001 - callback must not prevent speech
                    pass
            _ui_once(self._play_file, path, cancel_event)
        except FileNotFoundError:
            detail = f"Natural voice failed ({natural_error}); eSpeak NG is not installed."
            _ui_once(self._emit_error_for, detail, cancel_event)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace")[:200]
            message = f"Natural voice failed ({natural_error}); offline speech failed: {detail}"
            _ui_once(self._emit_error_for, message, cancel_event)
        except subprocess.TimeoutExpired:
            detail = "espeak-ng timed out after 30 seconds"
            message = f"Natural voice failed ({natural_error}); offline speech failed: {detail}"
            _ui_once(self._emit_error_for, message, cancel_event)

    def _play_file(self, path: str, cancel_event: threading.Event) -> bool:
        _ensure_gstreamer()
        if not self._is_current(cancel_event):
            with contextlib.suppress(OSError):
                os.unlink(path)
            return False
        self.temp_path = path
        pipeline = Gst.ElementFactory.make("playbin")
        if pipeline is None:
            with contextlib.suppress(OSError):
                os.unlink(path)
            self._emit_error_for("GStreamer playback is unavailable.", cancel_event)
            return False
        pipeline.set_property("uri", Path(path).as_uri())
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self._on_eos, cancel_event)
        bus.connect("message::error", self._on_error_message, cancel_event)
        self.pipeline = pipeline
        pipeline.set_state(Gst.State.PLAYING)
        self._emit_started_for(cancel_event)
        return False

    def _is_current(self, cancel_event: threading.Event) -> bool:
        return cancel_event is self.cancel_event and not cancel_event.is_set()

    def _emit_started_for(self, cancel_event: threading.Event) -> bool:
        if not self._is_current(cancel_event) or self._started_emitted or self._finished:
            return False
        self._started_emitted = True
        self._started_at = time.monotonic()
        if self._on_started is not None:
            self._on_started()
        return False

    def position(self) -> float:
        """Seconds of audio actually played in the current utterance, for the face renderer.

        ``0.0`` until sound is really coming out and again once the utterance has finished, so a mouth
        driven by this clock is shut before and after the voice. The pipeline's own position is used when
        it can be queried (it starts counting only when playback starts); the wall clock since
        :meth:`on_started` is the fallback for players that cannot report one.
        """
        if self._finished or not self._started_emitted:
            return 0.0
        pipeline = self.pipeline
        if pipeline is not None:
            try:
                ok, nanoseconds = pipeline.query_position(Gst.Format.TIME)
                # appsrc pipelines can report a successful TIME query with a
                # permanent zero position even while their audio sink is
                # playing.  Treat that as unknown so the lip-sync clock keeps
                # advancing from the actual playback-start callback.
                if ok and nanoseconds > 0:
                    return nanoseconds / 1_000_000_000
            except Exception:  # noqa: BLE001 - a clock that cannot answer falls back to the wall clock
                pass
        if self._started_at is not None:
            return time.monotonic() - self._started_at
        return 0.0

    def _on_eos(self, _bus, _message, cancel_event: threading.Event) -> None:
        if not self._is_current(cancel_event):
            return
        self._finished = True
        self._finish_media()
        if self._on_done is not None:
            self._on_done()

    def _on_error_message(self, _bus, message, cancel_event: threading.Event) -> None:
        if not self._is_current(cancel_event):
            return
        error, _debug = message.parse_error()
        self._emit_error_for(error.message, cancel_event)

    def _emit_error_for(self, message: str, cancel_event: threading.Event) -> bool:
        if not self._is_current(cancel_event):
            return False
        self._finished = True
        self._finish_media()
        if self._on_error is not None:
            self._on_error(message)
        return False

    def stop(self) -> None:
        self.cancel_event.set()
        self._finish_media()

    def _finish_media(self) -> None:
        if Gst is None:
            return
        appsrc, self.appsrc = self.appsrc, None
        if appsrc is not None:
            with contextlib.suppress(Exception):
                appsrc.emit("end-of-stream")
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        path, self.temp_path = self.temp_path, None
        if path:
            with contextlib.suppress(OSError):
                os.unlink(path)
