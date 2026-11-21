from __future__ import annotations

import contextlib
import logging
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from difflib import SequenceMatcher

from . import spellings
from .agent.intents import is_start_dictation
from .capture import CaptureRing, Frame
from .dictation import DICTATION_CHUNK_SECONDS, segment_stream
from .endpoint import EndpointPolicy, SmartTurn, ends_dangling, is_complete, is_followup, smart_turn_model_path
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
        if len(candidate) < 3:
            continue
        matched = SequenceMatcher(None, candidate, wake).ratio() >= 0.65
        if not matched and candidate[0] in _SOFT_ONSETS and wake[0] in _SOFT_ONSETS:
            # Said in one breath with the command, the first consonant is what Whisper gets wrong
            # ("Box open Pluma", "Laksa, what's the weather" for "Voxa"). Soft first consonants are
            # treated as the same sound, and "ks"/"cks" as "x".
            sounded = _sound(wake[0] + candidate[1:])
            matched = SequenceMatcher(None, sounded, _sound(wake)).ratio() >= 0.65
        if matched:
            end = window[-1].end()
            trailing = words[count] if len(words) > count else None
            if (
                trailing is not None
                and trailing.group().casefold() in ("a", "ah", "uh")
                and wake.endswith("a")
                and not candidate.endswith("a")
            ):
                end = trailing.end()  # "Locks a open …": the wake word's last syllable written apart
            return text[end:].strip(" ,.!?—-\t\n")
    return None


_SOFT_ONSETS = frozenset("bvflwp")


def _sound(word: str) -> str:
    return re.sub(r"cks|ks|cs", "x", word)


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

_POLITE_OFFLINE_PHRASES = frozenset(
    {
        "can you go offline",
        "could you go offline",
        "would you go offline",
        "will you go offline",
        "go offline now",
        "turn yourself off",
        "shut yourself down",
    }
)


def detect_exit_phrase(text: str, wake_word: str = "") -> str | None:
    """Return "cancel", "stop" or "goodbye" if the utterance ends the conversation.

    A leading configured wake word is ignored, because the natural one-breath
    command is "Voxa, go offline". Comparison remains exact after that and
    after punctuation/politeness normalization, so a sentence which merely
    mentions going offline cannot trip the kill switch.
    """
    if wake_word:
        addressed = re.sub(r"^\s*(?:hey|ok|okay)\b[\s,]*", "", text, flags=re.IGNORECASE)
        remainder = strip_wake_word(addressed, wake_word)
        if remainder is not None:
            text = remainder
    normalized = " ".join(re.sub(r"[^a-z0-9'\s]", " ", text.casefold()).split())
    if normalized.startswith("please "):
        normalized = normalized.removeprefix("please ")
    if normalized in _GOODBYE_PHRASES or normalized in _POLITE_OFFLINE_PHRASES:
        return "goodbye"
    if normalized in _STOP_PHRASES:
        return "stop"
    if normalized in _CANCEL_PHRASES:
        return "cancel"
    return None


def detect_addressed_exit_phrase(text: str, wake_word: str) -> str | None:
    """Detect an exit command only when the utterance explicitly addresses Voxa.

    This stricter form is used while Voxa is speaking. Requiring her wake word
    prevents an echoed fragment of her own reply from accidentally taking the
    assistant offline, while still making "Voxa, go offline" a kill switch.
    """
    addressed = re.sub(r"^\s*(?:hey|ok|okay)\b[\s,]*", "", text, flags=re.IGNORECASE)
    remainder = strip_wake_word(addressed, wake_word)
    if remainder is None:
        return None
    return detect_exit_phrase(remainder)


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

    PROMPT_TIMEOUT_SECONDS = 12.0  # longer than the follow-up window, so silence never cuts that short

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
        voice_detector: bool = True,
        smart_turn_model: str = "",
        endpoint_patience: str = "default",
        on_woken: Callable[[], None],
        on_prompt: Callable[[str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        on_exit: Callable[[str], None] | None = None,
        on_timing: Callable[[str], None] | None = None,
        on_transcribe_timing: Callable[[str, float, float], None] | None = None,
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
        # Streaming voice detector: when it loads it replaces the loudness rule in
        # segment_stream, so background music no longer blocks pause detection.
        # Created once per session; a detector that raises falls back to the loudness
        # rule for the rest of the session (see _voice_check).
        self._voice_detector: VoiceDetector | None = None
        self._voice_fell_back = False
        if voice_detector:
            try:
                from .voicedetect import VoiceDetector

                self._voice_detector = VoiceDetector()
            except Exception as exc:  # noqa: BLE001 - optional detector
                logging.debug("voice detector unavailable: %s", exc)
                self._voice_detector = None
        # Roadmap task 9: the acoustic endpoint decision. The weights ship with the
        # flatpak (or the bench data root); without them the pause rules stay as
        # they were. The session is shared process-wide, inference stays off the
        # UI thread (this controller's worker thread segments).
        self._endpoint_policy = EndpointPolicy(endpoint_patience)
        self._smart_turn: SmartTurn | None = None
        model_path = smart_turn_model_path(smart_turn_model)
        if model_path:
            engine = SmartTurn.shared(model_path)
            if engine.available:
                self._smart_turn = engine
            else:
                logging.debug("smart turn model unavailable: %s", engine.error)
        self.on_woken = on_woken
        self.on_prompt = on_prompt
        self.on_status = on_status
        self.on_error = on_error
        self.on_exit = on_exit or (lambda _kind: None)
        self.on_timing = on_timing or (lambda _stage: None)
        # Optional hook: (model name, audio seconds, processing seconds) after every
        # real prompt transcription, so the UI can watch the true dictation speed.
        self.on_transcribe_timing = on_transcribe_timing
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
        # Set when _early_check already fired on_woken for a bare wake word at the
        # early pause, so _run does not acknowledge the same wake word twice.
        self._woken_early = False
        # Vocabulary hint for prompt transcription, built off the listening
        # thread at start so it never blocks the mic (see _build_prompt_hint).
        self._prompt_hint = ""
        # Monotonic deadline for the follow-up window: while it is in the future
        # an utterance without the wake word may still be delivered as a prompt
        # (see open_followup). None means no window is open.
        self.followup_deadline: float | None = None
        # The raw utterance most recently heard (before wake-word stripping), so
        # the caller can still see the wake word for pause/resume phrasing.
        self.last_heard: str = ""
        # Bounded timestamped capture of the microphone with pinning (voxa/capture.py):
        # every frame gets a sequence id, speech onset pins the span, and reading it never
        # consumes it. This replaces the old five-second ``_barge_buf``, which ``candidate_text()``,
        # ``mute()`` and ``unmute()`` all destroyed, so the words that proved an interruption
        # never reached the request (handoff 5.1).
        self.capture = CaptureRing(clock=time.monotonic)
        # The transcript of the pinned span last inspected, with the span it came from, so the
        # same audio is never transcribed twice while nothing new has arrived.
        self._candidate_cache: tuple[int, int, str] | None = None
        # Transcript of the captured span before her own words were stripped, so the caller
        # can see what the echo filter removed (the barge-in trace and the bench keep it).
        self.last_candidate_raw = ""
        # Word timings of that same candidate pass, when the transcription service reports them.
        self.last_candidate_words: list[dict] = []
        # The candidate's own words, so promoting the span can carry them into the turn once.
        self._candidate_text = ""
        # What the capture path currently knows about the source of the microphone (a
        # voxa.attribution.SourceDecision, or None when there is no reference to compare with).
        # The own-speech filter reads it, so a span the audio attributes to a second source is
        # never rewritten by a text match.
        self.audio_evidence = None
        # The own-speech filter's text-overlap diagnostic for the last transcript (traced, never
        # used to delete anything on its own).
        self.last_filter_features: dict = {}
        # The turn in progress that began with a promoted span: its stable transcript prefix
        # and how many bytes of audio that prefix already accounts for.
        self._turn_prefix_text = ""
        self._turn_prefix_bytes = 0
        self._turn_prefix_words: list[dict] = []
        # The prefix belongs to the turn in progress: the UI thread promotes while the worker
        # thread takes, so the handover is guarded.
        self._turn_lock = threading.Lock()
        self.on_ignored = None
        # Optional echo filter (see voxa/ownspeech.py): set by the window so a
        # transcript that is (part of) what Voxa was saying is dropped, not fed
        # to the wake check, the prompt or a barge-in candidate. None = off.
        self.own_speech = None

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
            self._prompt_hint = build_hint(refresh_app_names(), self.wake_word, people=spellings.names())
        except Exception as exc:  # noqa: BLE001 - hint boundary, never fatal
            logging.debug("prompt hint build failed: %s", exc)

    def feed(self, pcm: bytes, level: float, seq: int | None = None, *,
             raw: bytes | None = None) -> None:
        """Capture one frame, then queue it only when the microphone is open.

        Capture runs in EVERY state (handoff 6.5: AEC/capture keep running across turn
        states): while muted the frame still gets a sequence id and stays in the ring, so
        the audio of an interruption survives the decision about what to do with it. Only
        the QUEUE is gated: her own spoken reply is never picked back up as an utterance
        until the caller promotes the pinned span into one.
        """
        if self.stop_event.is_set():
            return None
        frame = self.capture.add(pcm, level, seq=seq, raw=raw)
        if self._muted.is_set():
            return frame
        if self.capture.was_consumed(frame.seq):
            # Already handed to the user turn by promote_pinned(): appending it twice
            # would make the user's words reach the request twice.
            return frame
        self._enqueue(frame.pcm, frame.level)
        return frame

    def _enqueue(self, pcm: bytes, level: float) -> None:
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
        """Stop queueing the microphone. Capture and any pinned span are untouched."""
        self._muted.set()

    def unmute(self) -> None:
        """Let the microphone through again. A pinned span is never dropped here:
        the audio that proved the interruption is still owed to the user turn.
        """
        self._muted.clear()

    def note_onset(self, seq: int | None = None, *, level_floor: float | None = None) -> bool:
        """Speech onset: pin the span, from the onset back over the run of speech it belongs
        to, plus :data:`voxa.capture.PRE_ROLL_SECONDS` before it.

        Called by the UI and the bench when the barge-in gate opens. Pinning is what stops a
        rolling trim from overwriting an utterance that is still being judged.
        """
        return self.capture.pin(seq, level_floor=level_floor)

    def note_rejected(self) -> None:
        """A candidate the classifier did not accept: its audio stays for the evidence window.

        Rejection releases the PIN after that window; it never deletes the frames.
        """
        self.capture.note_rejected()

    def promote_pinned(self) -> int:
        """Make the pinned span the beginning of the user turn, exactly once.

        The frames keep their sequence ids and are marked consumed, so the frames that arrive
        after the interruption are appended as the continuation of the SAME utterance instead
        of the span being transcribed again or the same audio queued twice. Returns the number
        of frames promoted (0 when nothing is pinned).
        """
        frames = self.capture.promote()
        for frame in frames:
            self._enqueue(frame.pcm, frame.level)
        return len(frames)

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
        self._woken_early = False
        early = not self.keep_prompt

        # Read on every chunk, not once here: this loop is already collecting the
        # next utterance when the UI switches dictation on, and that first dictated
        # utterance used to run to the full 20 s command limit before anything was
        # typed.  Dictation must apply to the utterance in progress straight away.
        def max_segment_seconds() -> float:
            if self.keep_prompt:
                return min(self.max_segment_seconds, DICTATION_CHUNK_SECONDS)
            return self.max_segment_seconds

        def early_check(segment: bytes) -> bool:
            # Commands only: a dictated sentence is never a wake-word command.
            return not self.keep_prompt and self._early_check(segment)

        for segment in segment_stream(
            self.queue,
            self.stop_event,
            threshold=self.threshold,
            silence_seconds=self.silence_seconds,
            max_segment_seconds=max_segment_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            early_silence_seconds=self.early_silence_seconds if early else None,
            early_check=early_check if early else None,
            is_voice=self._voice_check if self._voice_detector is not None else None,
            endpoint_probe=self._endpoint_probe if self._smart_turn is not None else None,
            endpoint_policy=self._endpoint_policy,
        ):
            return segment
        return None

    def _endpoint_probe(self, segment: bytes) -> "bool | None":
        """Roadmap task 9: is the WHOLE turn heard so far acoustically finished?

        Smart Turn v3.1's own verdict decides; a transcript ending on a
        conjunction, filler or preposition (report 186) vetoes a "complete"
        answer, because a first valid command match is not completeness. The
        small always-loaded model does the text check, exactly like
        :meth:`_early_check`.
        """
        if self._smart_turn is None:
            return None
        complete = self._smart_turn.is_complete(segment)
        if complete is not True:
            return complete
        try:
            text = self.wake_whisper.transcribe(segment, self.language,
                                                f"{self.wake_word.strip().capitalize()}.")
        except Exception as exc:  # noqa: BLE001 - a probe must not lose the turn
            logging.debug("endpoint probe transcript failed: %s", exc)
            return True
        return not ends_dangling(text)

    def _voice_check(self, pcm: bytes, level: float) -> bool:
        """Detector verdict for one chunk; falls back to the loudness rule on error.

        Once the detector has raised, the fallback stays for the rest of the
        session and the failure is logged once.
        """
        if self._voice_detector is None or self._voice_fell_back:
            return level >= self.threshold
        try:
            return self._voice_detector.feed(pcm)
        except Exception as exc:  # noqa: BLE001 - detector boundary
            self._voice_fell_back = True
            logging.debug("voice detector failed; using the loudness rule: %s", exc)
            return level >= self.threshold

    def _filter_own_speech(self, text: str, *, evidence=None, confirmed_user: bool = False) -> str | None:
        """Strip echoes of Voxa's own voice from a transcript.

        Returns the text to use, or ``None`` when the whole utterance is her own voice (drop it).
        Never raises: a filter bug must not lose the user's speech, so the original text is
        returned and the failure logged.

        ``evidence`` is the attribution decision for that span and ``confirmed_user`` marks an
        utterance already established as the user's: either one of them stops a text match from
        rewriting anything (handoff 5.3 - the user may repeat Voxa's words on purpose). The
        overlap is still measured and kept in :attr:`last_filter_features` as a diagnostic.
        """
        if self.own_speech is None or not self.own_speech.active():
            return text
        try:
            ev = evidence if evidence is not None else self.audio_evidence
            self.last_filter_features = self.own_speech.features(text, protect=[self.wake_word])
            if self.own_speech.is_echo(text, protect=[self.wake_word],
                                        evidence=ev, confirmed_user=confirmed_user):
                logging.debug("ignored her own words: %s", text)
                return None
            stripped, _share = self.own_speech.strip(text, protect=[self.wake_word],
                                                     evidence=ev, confirmed_user=confirmed_user)
            return stripped
        except Exception as exc:  # noqa: BLE001 - echo filter must never lose speech
            logging.debug("own-speech filter failed: %s", exc)
            return text

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
        text = self._filter_own_speech(text)
        if text is None:
            return False
        if is_complete(text, self.wake_word):
            self._early_text = text
            return True
        # A bare wake word (exact, not fuzzy) heard at the early pause: acknowledge now so the
        # user can keep talking, instead of waiting for the full silence to end the utterance.
        normalized = " ".join(re.sub(r"[^a-z0-9'\s]", " ", text.casefold()).split())
        if normalized == self.wake_word.strip().casefold():
            self._woken_early = True
            self.on_woken()
            self._early_text = text
            return True
        return False

    def _transcribe(self, segment: bytes, whisper: WhisperService, hint: str = "") -> str:
        if self.stop_event.is_set():
            return ""
        try:
            self.on_status("Transcribing…")
            prompt_model = whisper is self.prompt_whisper
            start = time.perf_counter() if prompt_model else 0.0
            text = whisper.transcribe(segment, self.language, hint) if hint else whisper.transcribe(segment, self.language)
            if prompt_model:
                self._report_timing(whisper, len(segment) / 32000.0, time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            if not self.stop_event.is_set():
                self.on_error(str(exc))
            return ""
        # A stop pressed while the transcription was in flight must not
        # deliver its result to the UI.
        return "" if self.stop_event.is_set() else text

    def _wake_hint(self) -> str:
        return f"{self.wake_word.strip().capitalize()}."

    def _refine_wake_utterance(
        self,
        segment: bytes,
        wake_text: str,
        wake_remainder: str | None,
    ) -> tuple[str, str | None]:
        """Verify and refine a one-breath wake word plus command.

        The tiny always-on model is quick, but it can omit a short word at the
        start of connected speech (notably ``Voxa``).  Give the configured
        prompt model the same audio once.  A missed wake word is accepted only
        when that second transcript contains it, so background conversation
        does not become a command.  Once the quick model has already confirmed
        the wake word, the second transcript may safely supply just the more
        accurate command text.
        """
        if self.prompt_whisper is self.wake_whisper:
            return wake_text, wake_remainder

        hint = self._prompt_hint or f"{self.wake_word.strip().capitalize()}."
        refined = self._transcribe(segment, self.prompt_whisper, hint=hint)
        refined = self._filter_own_speech(refined) if refined else None
        if not refined:
            return wake_text, wake_remainder

        refined_remainder = strip_wake_word(refined, self.wake_word)
        if wake_remainder is None:
            # Neither model found the wake word: preserve the original text so
            # follow-up handling and missed-wake diagnostics remain unchanged.
            if refined_remainder is None and hint != self._wake_hint():
                # The vocabulary hint itself can pull the wake word towards a listed name
                # ("Bazaar, what windows…"). One more pass with only the wake word as the hint.
                plain = self._transcribe(segment, self.prompt_whisper, hint=self._wake_hint())
                plain = self._filter_own_speech(plain) if plain else None
                if plain:
                    plain_remainder = strip_wake_word(plain, self.wake_word)
                    if plain_remainder is not None:
                        return plain, plain_remainder
            if refined_remainder is None:
                return wake_text, wake_remainder
            return refined, refined_remainder

        # The fast model already proved that this was addressed to Voxa.  Some
        # larger-model transcripts omit the wake word but retain a better
        # command, so in that case the complete refined text is the command.
        if refined_remainder is None:
            return refined, refined
        if refined_remainder:
            return refined, refined_remainder
        return wake_text, wake_remainder

    def _report_timing(self, whisper: WhisperService, audio_seconds: float, processing_seconds: float) -> None:
        """Hand one real prompt transcription's speed to the UI guard; a raising hook is swallowed."""
        if self.on_transcribe_timing is None:
            return
        try:
            self.on_transcribe_timing(whisper.model_name, audio_seconds, processing_seconds)
        except Exception:  # noqa: BLE001 - optional hook boundary
            pass

    def candidate_text(self) -> str:
        """Transcribe a SNAPSHOT of the captured span, as a prompt would be.

        Called by the UI when the level gate opens, to check the words before her reply is
        stopped. Reading is non-destructive: the span stays in the ring, keeps its sequence
        ids, and becomes the request when the interruption is accepted (handoff 5.1: the
        words that proved the interruption must not be consumed by the check). The snapshot is
        the part of the span not yet handed to a user turn, so a second candidate judges only
        the audio that arrived since the first was accepted. "" when nothing was captured; the
        same span is not transcribed twice while no new frame arrived.
        """
        start, end = self.capture.span_bounds()
        segment = self.capture.snapshot()
        if not segment:
            self.last_candidate_raw = ""
            self.last_candidate_words = []
            return ""
        cached = self._candidate_cache
        if cached is not None and cached[0] == start and cached[1] == end:
            return cached[2]
        raw, words = self._transcribe_span(segment)
        self.last_candidate_raw = raw or ""
        self.last_candidate_words = words
        # The candidate is judged on the WHOLE transcript: her own words are not stripped out of
        # it first, because the stripping is a decision about a SOURCE and the source is exactly
        # what is still being decided (handoff 5.3: the user may repeat Voxa's words, and a
        # repetition stripped here can never be attributed afterwards).
        text = raw or ""
        self._candidate_cache = (start, end, text)
        self._candidate_text = text
        return text

    def _transcribe_span(self, segment: bytes) -> tuple[str, list[dict]]:
        """One ASR pass over a captured span, keeping word timings when the service reports them.

        The candidate pass and the turn prefix want the same audio transcribed once, with its
        words; a service that only returns text (the live ``WhisperService`` today) simply yields
        no timings, and nothing else changes.
        """
        hook = getattr(self.prompt_whisper, "transcribe_words", None)
        if hook is not None:
            try:
                found = hook(segment, self._prompt_hint)
                return str(found.get("text") or ""), list(found.get("words") or [])
            except Exception as exc:  # noqa: BLE001 - word-timed pass is optional, text is not
                logging.debug("word-timed pass failed: %s", exc)
        return self._transcribe(segment, self.prompt_whisper, hint=self._prompt_hint), []

    def promote_pinned(self, *, enqueue: bool = True) -> list[Frame]:
        """Make the pinned span the beginning of the user turn, exactly once.

        The frames keep their sequence ids and are marked consumed, so the frames that arrive
        after the interruption are appended as the continuation of the SAME utterance instead
        of the span being transcribed again or the same audio queued twice. The words already
        recognised from exactly that audio become the turn's stable transcript prefix (handoff
        6.4: a stable prefix and a revisable suffix, never a second command from partial ASR),
        and only the audio after the span is transcribed from now on.

        ``enqueue=False`` is for a caller that drives the speech-pause rules itself (the hearing
        bench) and wants the frames back to feed them onward instead of putting them on the
        worker queue. Returns the promoted frames, oldest first, ids intact.
        """
        frames = self.capture.promote()
        if not frames:
            return frames
        for frame in frames:
            if enqueue:
                self._enqueue(frame.pcm, frame.level)
        with self._turn_lock:
            self._turn_prefix_bytes += sum(len(frame.pcm) for frame in frames)
            self._turn_prefix_text = f"{self._turn_prefix_text} {self._candidate_text}".strip()
            self._turn_prefix_words = [*self._turn_prefix_words, *self.last_candidate_words]
        return frames

    def take_turn_prefix(self) -> tuple[str, int, list[dict]]:
        """Take the stable prefix of the turn in progress (empty when the turn started normally).

        Text, the number of audio bytes it already accounts for, and its word timings when the
        transcription service reported them. Taken once: the same prefix never heads two turns.
        """
        with self._turn_lock:
            prefix = (self._turn_prefix_text, self._turn_prefix_bytes, self._turn_prefix_words)
            self._turn_prefix_text = ""
            self._turn_prefix_bytes = 0
            self._turn_prefix_words = []
        return prefix

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

            prefix_text, prefix_bytes, _prefix_words = self.take_turn_prefix()
            prefixed = bool(prefix_text)
            if prefixed:
                # The turn began with audio promoted out of the capture ring: its words were
                # already recognised once, from exactly that audio, at the moment the interruption
                # was accepted. Only the audio that arrived AFTER the span is transcribed here,
                # so the user's words reach the request once - not twice, and not not at all
                # (handoff 6.4: a stable transcript prefix and a revisable suffix).
                tail = segment[prefix_bytes:] if len(segment) >= prefix_bytes else b""
                whisper = self.prompt_whisper if waiting_for_prompt else self.wake_whisper
                tail_raw = self._transcribe(
                    tail,
                    whisper,
                    hint=self._prompt_hint if waiting_for_prompt else f"{self.wake_word.strip().capitalize()}.",
                ) if tail else ""
                tail_text = (self._filter_own_speech(tail_raw, confirmed_user=True)
                             or "") if tail_raw else ""
                text = f"{prefix_text} {tail_text}".strip()
            elif not self.early_final_pass and self._early_text is not None:
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
            self.last_heard = text
            if not prefixed:
                filtered = self._filter_own_speech(text)
                if filtered is None:
                    continue
                text = filtered
                if not text:
                    continue
            if self.stop_event.is_set():
                break

            if not self.keep_prompt:
                exit_kind = detect_exit_phrase(text, self.wake_word)
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
                # The fast model has already proved this request is addressed to
                # Voxa. Acknowledge it before the slower accuracy pass so the UI
                # reacts immediately instead of appearing stuck in transcription.
                if remainder is not None and not self._woken_early:
                    self._woken_early = True
                    self.on_woken()
                # A command spoken immediately after the wake word is one audio
                # segment.  Verify a tiny-model miss, or refine the command it
                # found, using the accurate prompt model without asking the user
                # to pause and repeat themselves.
                # A destination-qualified dictation request is already
                # unambiguous. Avoid a second, slower transcription pass so
                # "Voxa, start dictation in Pluma" reacts like bare dictation.
                if remainder is None or (remainder and not is_start_dictation(remainder)):
                    text, remainder = self._refine_wake_utterance(segment, text, remainder)
                    self.last_heard = text
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
                    elif self.on_ignored is not None:
                        # Heard, but no wake word found: reported so a missed "Voxa, …" can be explained.
                        with contextlib.suppress(Exception):
                            self.on_ignored(text, segment)
                    continue
                if self.stop_event.is_set():
                    break
                if not self._woken_early:
                    self._woken_early = True
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
