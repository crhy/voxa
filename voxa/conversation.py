from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from difflib import SequenceMatcher

from .dictation import segment_stream
from .endpoint import is_complete, is_followup
from .transcription import WhisperService
from .vocabulary import build_hint, refresh_app_names


def strip_wake_word(text: str, wake_word: str) -> str | None:
    """Return the text with the wake word removed, or ``None`` if absent.

    Matching is a case-insensitive substring search, so "Hey Computer, what
    time is it" and "Computer" both match a wake word of "computer". The
    returned remainder has leading punctuation left over from the wake word
    trimmed off.
    """
    wake = wake_word.casefold().strip()
    if not wake:
        return None
    lowered = text.casefold()
    index = lowered.find(wake)
    if index == -1:
        return _fuzzy_strip(text, wake)
    remainder = text[:index] + text[index + len(wake) :]
    return remainder.strip(" ,.!?—-\t\n")


def _fuzzy_strip(text: str, wake: str) -> str | None:
    """Match a wake word the tiny Whisper model spelled slightly differently.

    Only the start of the utterance is checked (wake words come first), and a
    word or word pair must be very close to the wake word, so ordinary speech
    is not mistaken for it.
    """
    words = list(re.finditer(r"[\w']+", text))
    for count in (1, 2):
        window = words[:count]
        if len(window) < count:
            continue
        candidate = "".join(m.group().casefold() for m in window)
        if len(candidate) >= 3 and SequenceMatcher(None, candidate, wake).ratio() >= 0.65:
            return text[window[-1].end() :].strip(" ,.!?—-\t\n")
    return None


# Utterances that end a conversation turn, matched exactly (whitespace and
# punctuation normalized) so a normal question can never be mistaken for one.
# "cancel" abandons the current question and returns to listening for the
# wake word; "goodbye" ends conversation mode altogether.
_CANCEL_PHRASES = frozenset(
    {
        "cancel",
        "cancelled",
        "never mind",
        "nevermind",
        "forget it",
    }
)
_STOP_PHRASES = frozenset({"stop", "stop talking", "stop it", "stop dictation", "stop listening"})
_GOODBYE_PHRASES = frozenset(
    {
        "go offline",
        "go off line",
        "offline",
        "turn off",
        "shut down",
        "go to sleep",
        "goodbye",
        "bye",
        "goodbye for now",
        "see you",
        "see ya",
        "that's all",
        "thats all",
        "done",
    }
)


def detect_exit_phrase(text: str) -> str | None:
    """Return "cancel", "stop" or "goodbye" if the utterance ends the conversation.

    Comparison is exact after casefolding and stripping punctuation, so
    "Never mind." and "Goodbye!" match while "never mind that" does not.
    """
    normalized = " ".join(re.sub(r"[^a-z0-9'\s]", " ", text.casefold()).split())
    if normalized in _GOODBYE_PHRASES:
        return "goodbye"
    if normalized in _STOP_PHRASES:
        return "stop"
    if normalized in _CANCEL_PHRASES:
        return "cancel"
    return None


class ConversationHistory:
    """The system prompt plus a bounded list of user/assistant turns.

    The system prompt is stored separately from the turns so it survives the
    turn cap: a single deque holding it would evict it once the conversation
    grows past the limit.
    """

    def __init__(self, system_prompt: str, max_turns: int) -> None:
        self.system_prompt = system_prompt
        self._turns: deque[dict[str, str]] = deque(maxlen=max_turns)

    def add_user(self, content: str) -> None:
        self._turns.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._turns.append({"role": "assistant", "content": content})

    def drop_last(self) -> None:
        if self._turns:
            self._turns.pop()

    def clear(self) -> None:
        self._turns.clear()

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, *self._turns]

    def __bool__(self) -> bool:
        return bool(self._turns)


class ConversationController:
    """Listens for a wake word, then captures the next utterance as a prompt.

    Reuses the same speech-pause segmentation as :class:`DictationController`
    (via :func:`segment_stream`), so no dedicated wake-word model is required
    — a small, always-loaded Whisper model transcribes each utterance heard
    while waiting, and the result is checked for the configured wake word.
    Once woken, the caller's real (possibly much larger) model transcribes
    the actual command, since accuracy matters there but not during idle
    listening. A dedicated low-latency wake-word engine could replace the
    wake phase later without changing the caller contract (``feed``/
    ``start``/``stop`` plus the four callbacks).
    """

    PROMPT_TIMEOUT_SECONDS = 8.0

    def __init__(
        self,
        *,
        wake_whisper: WhisperService,
        prompt_whisper: WhisperService,
        language: str,
        wake_word: str,
        threshold: int,
        silence_ms: int,
        max_segment_seconds: float,
        early_silence_ms: int = 300,
        early_final_pass: bool = False,
        on_woken: Callable[[], None],
        on_prompt: Callable[[str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        on_exit: Callable[[str], None] | None = None,
        on_timing: Callable[[str], None] | None = None,
    ) -> None:
        # The wake phase runs constantly in the background, so it uses a small,
        # dedicated model (see WAKE_WHISPER_MODEL in window.py) instead of
        # whatever (possibly much larger) model the user picked for real
        # transcription — that model is only needed once actually woken.
        self.wake_whisper = wake_whisper
        self.prompt_whisper = prompt_whisper
        self.language = language
        self.wake_word = wake_word
        self.threshold = threshold
        self.silence_seconds = silence_ms / 1000.0
        self.early_silence_seconds = early_silence_ms / 1000.0
        self.early_final_pass = early_final_pass
        self.max_segment_seconds = max_segment_seconds
        self.on_woken = on_woken
        self.on_prompt = on_prompt
        self.on_status = on_status
        self.on_error = on_error
        self.on_exit = on_exit or (lambda _kind: None)
        self.on_timing = on_timing or (lambda _stage: None)
        self.queue: queue.Queue[tuple[bytes, float] | None] = queue.Queue(maxsize=80)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._muted = threading.Event()
        # An utterance currently heard becomes a prompt without requiring the
        # wake word first (set by the caller after a barge-in, so the user can
        # just keep talking after interrupting a spoken reply).
        self.waiting_for_prompt = False
        # Dictation mode: every utterance is a prompt, no wake word needed, until
        # the user says "stop dictating".
        self.keep_prompt = False
        # Quick transcript captured at the early pause, reused by _run when the
        # utterance ended early and early_final_pass is False.
        self._early_text: str | None = None
        # Vocabulary hint for prompt transcription, built off the listening
        # thread at start so it never blocks the mic (see _build_prompt_hint).
        self._prompt_hint = ""
        # Monotonic deadline for the follow-up window: while it is in the future
        # an utterance without the wake word may still be delivered as a prompt
        # (see open_followup). None means no window is open.
        self.followup_deadline: float | None = None

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    def start(self) -> None:
        self.stop_event.clear()
        self._muted.clear()
        self.waiting_for_prompt = False
        self.thread = threading.Thread(target=self._run, name="conversation-worker", daemon=True)
        self.thread.start()
        threading.Thread(target=self._build_prompt_hint, name="prompt-hint-builder", daemon=True).start()

    def _build_prompt_hint(self) -> None:
        try:
            self._prompt_hint = build_hint(refresh_app_names(), self.wake_word)
        except Exception as exc:  # noqa: BLE001 - hint boundary, never fatal
            logging.debug("prompt hint build failed: %s", exc)

    def feed(self, pcm: bytes, level: float) -> None:
        # Dropped while muted so the assistant's own spoken reply, played
        # through the speakers, is never picked back up as a new utterance.
        if self.stop_event.is_set() or self._muted.is_set():
            return
        try:
            self.queue.put_nowait((pcm, level))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((pcm, level))
            except queue.Full:
                pass

    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()

    def arm_prompt(self) -> None:
        """Skip the wake word: the next utterance becomes a prompt directly.

        Called by the UI after the user barge-in interrupts a spoken reply,
        so they can keep talking without repeating the wake word.
        """
        self.waiting_for_prompt = True

    def hold_prompt(self) -> None:
        """Keep listening for prompts: every utterance is one, wake word or not."""
        self.keep_prompt = True
        self.waiting_for_prompt = True
        self.followup_deadline = None

    def release_prompt(self) -> None:
        """Go back to wake-word-gated listening."""
        self.keep_prompt = False
        self.waiting_for_prompt = False

    def open_followup(self, seconds: float) -> None:
        """Open a window where follow-ups without the wake word are accepted.

        A non-positive ``seconds`` disables the window. The deadline is cleared
        by the first accepted follow-up, by :meth:`stop`, and by dictation mode.
        """
        if seconds <= 0:
            self.followup_deadline = None
        else:
            self.followup_deadline = time.monotonic() + seconds

    def _followup_open(self) -> bool:
        return self.followup_deadline is not None and time.monotonic() < self.followup_deadline

    def stop(self) -> None:
        self.stop_event.set()
        self.waiting_for_prompt = False
        self.followup_deadline = None
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def _next_segment(self, idle_timeout_seconds: float | None) -> bytes | None:
        self._early_text = None
        early = not self.keep_prompt
        for segment in segment_stream(
            self.queue,
            self.stop_event,
            threshold=self.threshold,
            silence_seconds=self.silence_seconds,
            max_segment_seconds=self.max_segment_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            early_silence_seconds=self.early_silence_seconds if early else None,
            early_check=self._early_check if early else None,
        ):
            return segment
        return None

    def _early_check(self, segment: bytes) -> bool:
        """Quick look at the audio so far with the small wake model.

        Never raises: an exception here must not break listening, so it is
        logged at debug level and reported as "not complete".
        """
        if self.stop_event.is_set():
            return False
        hint = f"{self.wake_word.strip().capitalize()}."
        try:
            text = self.wake_whisper.transcribe(segment, self.language, hint)
        except Exception as exc:  # noqa: BLE001 - early-check boundary
            logging.debug("early completion check failed: %s", exc)
            return False
        if is_complete(text, self.wake_word):
            self._early_text = text
            return True
        return False

    def _transcribe(self, segment: bytes, whisper: WhisperService, hint: str = "") -> str:
        if self.stop_event.is_set():
            return ""
        try:
            self.on_status("Transcribing…")
            text = whisper.transcribe(segment, self.language, hint) if hint else whisper.transcribe(segment, self.language)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            if not self.stop_event.is_set():
                self.on_error(str(exc))
            return ""
        # A stop pressed while the transcription was in flight must not
        # deliver its result to the UI.
        return "" if self.stop_event.is_set() else text

    def _run(self) -> None:
        while not self.stop_event.is_set():
            keep = self.keep_prompt
            # Dictation mode never times out: the user keeps talking until they
            # say "stop dictating".
            idle_timeout = None if keep else (self.PROMPT_TIMEOUT_SECONDS if self.waiting_for_prompt else None)
            segment = self._next_segment(idle_timeout)
            # Re-read after the segment: a barge-in can arm the prompt state
            # while the user is mid-utterance, and that utterance must be the
            # prompt, not a wake-word candidate.
            waiting_for_prompt = self.waiting_for_prompt or self.keep_prompt
            if segment is None:
                if waiting_for_prompt and not self.stop_event.is_set():
                    self.on_status(f"Didn't catch that — say “{self.wake_word}” again.")
                self.waiting_for_prompt = False
                continue

            self.on_timing("speech_end")

            if not self.early_final_pass and self._early_text is not None:
                text = self._early_text
            else:
                whisper = self.prompt_whisper if waiting_for_prompt else self.wake_whisper
                text = self._transcribe(
                    segment,
                    whisper,
                    hint=self._prompt_hint if waiting_for_prompt else f"{self.wake_word.strip().capitalize()}.",
                )
            if not text:
                continue
            self.on_timing("transcribed")
            if self.stop_event.is_set():
                break

            if not self.keep_prompt:
                exit_kind = detect_exit_phrase(text)
                if exit_kind is not None:
                    # The user is ending things: abandon any in-flight prompt and
                    # either go back to waiting for the wake word (cancel) or
                    # leave conversation mode entirely (goodbye — the caller stops
                    # us when it hears that).
                    self.waiting_for_prompt = False
                    if not self.stop_event.is_set():
                        self.on_exit(exit_kind)
                    if exit_kind == "goodbye":
                        break
                    continue

            if not waiting_for_prompt:
                remainder = strip_wake_word(text, self.wake_word)
                if remainder is None:
                    # No wake word: accept it as a follow-up only while the
                    # window is open and it is clearly addressed to Voxa (at
                    # least two words and a routed call, continuation or
                    # question). The first accepted follow-up clears the window.
                    if (
                        self._followup_open()
                        and len(text.split()) >= 2
                        and is_followup(text)
                    ):
                        self.followup_deadline = None
                        if self.stop_event.is_set():
                            break
                        self.on_prompt(text)
                    continue
                if self.stop_event.is_set():
                    break
                self.on_woken()
                if remainder:
                    self.on_prompt(remainder)
                else:
                    self.on_status("Listening for your request…")
                    self.waiting_for_prompt = True
            else:
                if self.stop_event.is_set():
                    break
                self.on_prompt(text)
                # While dictating the next utterance is a prompt too; otherwise
                # we go back to waiting for the wake word.
                self.waiting_for_prompt = self.keep_prompt
