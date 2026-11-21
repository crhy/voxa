"""B2: the file-driven runner that pushes bench clips through Voxa's real hearing pipeline.

Roadmap B2: read a clip's ``mic.wav`` and ``ref.wav``, cut them into 20 ms frames and walk them
through the same stages the live app uses, in the same order:

    reference ring + canceller  ``voxa/audio.py:88`` ``mix_chunk`` with ``EchoCanceller``; the delay
                                 is estimated the way ``AudioCapture._on_sample`` /
                                 ``_maybe_estimate_delay`` (audio.py:418 / 468) do it
    reference mask              ``voxa/refmask.py:14`` — a lab stage, NOT in the live capture path
    voice detector              the controller's own ``_voice_check`` -> ``voxa/voicedetect.py:43``
    conversation feed           ``voxa/conversation.py:331`` ``feed`` (mute/unmute, the 5 s barge buffer)
    barge-in decision           ``voxa/bargectl.py``, the class ``voxa/window.py:3135`` now calls
    own-speech filter           ``voxa/ownspeech.py``
    segmentation + endpoint     ``voxa/dictation.py`` ``SpeechSegmenter``, ``voxa/endpoint.py:86``
    faster-whisper              pinned: ``temperature=0``, one fixed beam,
                                 ``condition_on_previous_text=False``, ``word_timestamps=True``,
                                 Voxa's default model (``voxa/config.py:22`` ``whisper_model="base"``)

A simulated clock advances 20 ms per frame, so a clip runs faster than real time and nothing sleeps.
Per clip the runner writes ``.voxa-bench/<suite>/<clip>/events.jsonl``: one ``frame`` event per frame
(frame sequence id, sample span, VAD verdict and probability, level, echo removed in dB, the
microphone and cleaned levels in dB, mask gain, current delay estimate) plus ``suspect``, ``candidate``,
``barge-in-confirm``, ``tts-stop``, ``pause``, ``endpoint``, ``transcript`` and ``request`` (the text that
would reach the assistant, with word times and ``no_speech_prob``).

This task MEASURES; it does not fix. The product pipeline is untouched except the two factoring
steps that let both paths share one decision (``voxa/bargectl.py`` and ``SpeechSegmenter``). Roadmap
B3 hooks the metrics in: ``run`` hands each clip's events to ``voxa/bench/report.py``, which computes
every metric the brief names with a bootstrap CI and writes ``report.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from ..anc import MIN_REFERENCE_POWER, EchoCanceller, estimate_delay
from ..audio import AudioCapture, ReferenceRing, mix_chunk, webrtc_available
from ..bargectl import BargeInController
from ..onset import dbfs
from ..attribution import EchoPathBaseline, SourceEvidence, attribute
from ..capture import Frame
from ..config import Settings
from ..conversation import ConversationController, detect_exit_phrase, strip_wake_word
from ..dictation import SpeechSegmenter
from ..endpoint import is_complete, is_followup
from ..ownspeech import OwnSpeech
from ..refmask import ReferenceMask
from . import metrics as metrics_module
from . import report as report_module
from .render import DEFAULT_DATA_ROOT, SR, read_wav

CHUNK_MS = 20
CHUNK_SAMPLES = SR * CHUNK_MS // 1000  # 320 samples per 20 ms frame
REFERENCE_SECONDS = 3  # voxa/audio.py:32, the ring the live capture keeps
SPEAKING_POWER = 1e-6  # voxa/anc.py:21 SILENT_REFERENCE_POWER: the speakers are on above this
QUIET_FRAMES = 10  # 200 ms of a quiet reference ends the reply; word gaps inside it are shorter
DEFAULT_BEAM_SIZE = 1
VAD_PARAMETERS = {"min_silence_duration_ms": 350}  # voxa/transcription.py:177, the app's own VAD setting


def pcm16(samples: np.ndarray) -> bytes:
    """Float samples as the int16 little-endian PCM the pipeline moves (same rounding as mix_chunk)."""
    x = np.clip(np.rint(np.asarray(samples, dtype=np.float64) * 32767.0), -32768, 32767)
    return x.astype("<i2").tobytes()


def erle_removed_db(mic, cleaned) -> float:
    """What the canceller took away in one frame (``voxa/anc.py:8`` ``erle_db`` semantics)."""
    mic_ms = float(np.mean(np.asarray(mic, dtype=np.float64) ** 2))
    if mic_ms < 1e-12:
        return 0.0
    cleaned_ms = max(float(np.mean(np.asarray(cleaned, dtype=np.float64) ** 2)), 1e-20)
    return 10.0 * float(np.log10(mic_ms / cleaned_ms))


def level_db(samples: np.ndarray) -> float:
    """One frame's RMS level in dB, the way the metrics compare what the pipeline did to it.

    B3 measures what the echo stages do to the USER's own frames (near-end attenuation), which
    needs both sides of the stage: the level the microphone delivered and the level the ASR stage
    received. A silent frame floors at -160 dB instead of going to minus infinity.
    """
    rms_value = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))
    return round(20.0 * float(np.log10(max(rms_value, 1e-8))), 3)


class PinnedASR:
    """faster-whisper pinned the way the bench requires, with word times and no_speech_prob kept.

    The live :meth:`voxa.transcription.WhisperService.transcribe` throws the word times and
    ``no_speech_prob`` away, so the bench calls the model itself with the pinned parameters and
    keeps everything else the app already sets (``vad_filter=True`` with the same
    ``vad_parameters``, the wake hint as ``initial_prompt``). It exposes the same
    ``transcribe(pcm, language, hint)`` shape the
    :class:`voxa.conversation.ConversationController` expects. The controller is handed this one
    object as both its wake and its prompt model: one model, one beam, no VAD filter of its own
    (the Silero detector in the pipeline decides where speech is).
    """

    def __init__(self, model_name: str = "", device: str = "cpu", compute_type: str = "int8") -> None:
        from faster_whisper import WhisperModel

        settings = Settings().normalized()
        self.model_name = model_name or settings.whisper_model
        try:
            threads = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            threads = os.cpu_count() or 4
        self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type,
                                  cpu_threads=max(1, min(8, threads)))
        self.language = settings.language
        self.calls = 0
        self.audio_seconds = 0.0
        self.processing_seconds = 0.0

    def transcribe_words(self, pcm: bytes, hint: str = "") -> dict:
        """One pinned pass: text, word times (relative to the segment) and no_speech_prob."""
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        started = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=self.language or None,
            beam_size=DEFAULT_BEAM_SIZE,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(VAD_PARAMETERS),
            initial_prompt=hint or None,
        )
        found = [segment for segment in segments]
        took = time.perf_counter() - started
        self.calls += 1
        self.audio_seconds += len(audio) / SR
        self.processing_seconds += took
        words = [
            {
                "word": word.word.strip(),
                "onset_s": round(float(word.start), 3),
                "offset_s": round(float(word.end), 3),
            }
            for segment in found
            for word in (segment.words or [])
        ]
        text = " ".join(segment.text.strip() for segment in found if segment.text.strip()).strip()
        probs = [float(segment.no_speech_prob) for segment in found]
        return {
            "text": text,
            "words": words,
            "no_speech_prob": round(max(probs), 4) if probs else None,
            "audio_seconds": round(len(audio) / SR, 3),
            "asr_seconds": round(took, 3),
            "segments": len(found),
        }

    def transcribe(self, pcm_s16le: bytes, language: str = "en", hint: str = "") -> str:
        """The call shape ``ConversationController.candidate_text`` uses on the barge window."""
        return self.transcribe_words(pcm_s16le, hint)["text"]


class ClipRunner:
    """One clip through the pipeline; the simulated clock is the only clock there is."""

    def __init__(
        self,
        *,
        asr: PinnedASR,
        echo_mode: str = "off",
        wake_word: str = "voxa",
        settings: Settings | None = None,
    ) -> None:
        self.asr = asr
        self.echo_mode = echo_mode
        self.settings = settings or Settings().normalized()
        self.wake_word = wake_word
        self.events: list[dict] = []
        self._now = 0.0
        self._frame = -1
        # Echo stages, created only for the mode asked for.
        self.ref_ring: ReferenceRing | None = None
        self.canceller: EchoCanceller | None = None
        self.mask: ReferenceMask | None = None
        self.degraded: str | None = None
        self._mic_counter = 0
        self._delay_done = True
        self._delay_mic: list[np.ndarray] = []
        self._delay_ref: list[np.ndarray] = []
        self._delay_samples = 0
        # The cleaned stream of the clip, kept so the runner can transcript what its pipeline
        # delivered to the ASR stage as a whole (what `voxa.bench agree` measures).
        self._clean_stream: list[bytes] = []
        # Conversation state: the real controller, driven by this runner instead of by its thread.
        self.conversation: ConversationController | None = None
        self.own_speech = OwnSpeech(clock=self._clock)
        self.barge = BargeInController(clock=self._clock)
        self.segmenter: SpeechSegmenter | None = None
        # Frame sequence ids already handed to the speech-pause rules (dedupe by frame id).
        self._segmented: set[int] = set()
        # Attribution (handoff 6.3): the echo path's own recent residual level, and the evidence
        # record built for the frame being judged.
        self.echo_baseline = EchoPathBaseline()
        self.last_evidence: SourceEvidence | None = None
        # The reference power of the last 400 ms, so an echo still in flight counts as a live
        # reference even after the speakers went quiet (the delay is 100-300 ms in these clips).
        self._ref_recent: deque[float] = deque(maxlen=20)
        # What the clip produced.
        self.requests: list[str] = []
        self.turns: list[dict] = []
        self.counts = {"suspect": 0, "barge_in": 0, "own_voice": 0, "ignored": 0, "exit": 0,
                       "turns": 0, "hold": 0}
        self.barge_heard: list[str] = []
        self.transcript_texts: list[str] = []
        self.speaking = False
        # When the reply being read out started, so a yield can report the position of her
        # playback at the moment the floor was taken, and whether that floor was already taken.
        self._speak_started_at = 0.0
        self.barge_yielded = False
        # The reply whose floor was taken. The reference stream keeps playing a reply the app
        # stopped, and restarting the barge evidence for it would interrupt the same user words
        # twice - the grace period used to hide that, and task A3 removed it.
        self._spoken_text = ""
        self._interrupted_text: str | None = None
        self._quiet_frames = 0
        # The follow-up window the app opens after every reply, kept on the simulated clock
        # (ConversationController.open_followup stamps a wall-clock deadline, which a bench
        # running faster than real time cannot use).
        self._followup_deadline: float | None = None

    def _clock(self) -> float:
        return self._now

    def emit(self, event: str, **fields) -> None:
        self.events.append({
            "event": event,
            "frame": self._frame,
            "t_s": round(self._now, 3),
            "samples": [self._frame * CHUNK_SAMPLES, (self._frame + 1) * CHUNK_SAMPLES],
            **fields,
        })

    # --- echo stages ------------------------------------------------------------

    def _prepare_echo_stages(self) -> None:
        if self.echo_mode == "voxa":
            self.ref_ring = ReferenceRing(REFERENCE_SECONDS * SR)
            self.canceller = EchoCanceller()
            self._delay_done = False
        elif self.echo_mode == "webrtc":
            if not webrtc_available():
                # The live app falls back to a plain capture pipeline in exactly this case
                # (voxa/audio.py:519 set_echo_mode), and says so; the bench records it the same way
                # instead of pretending the cancellation happened.
                self.degraded = "webrtc unavailable (no GStreamer runtime here): running with echo off"
                self.echo_mode = "off"
            else:
                self.mask = ReferenceMask()
        if self.echo_mode == "voxa":
            self.mask = ReferenceMask()

    def _estimate_delay(self, mic: np.ndarray, ref: np.ndarray) -> None:
        """Mirror of ``AudioCapture._maybe_estimate_delay`` (audio.py:468) on the simulated clock."""
        if self._delay_done or self.ref_ring is None or ref.size == 0:
            return
        if float(np.mean(ref * ref)) < 1e-6:
            return
        self._delay_mic.append(mic)
        self._delay_ref.append(ref)
        collected = sum(chunk.size for chunk in self._delay_mic)
        if collected < REFERENCE_SECONDS * SR:
            return
        self._delay_done = True
        delay = int(estimate_delay(np.concatenate(self._delay_mic), np.concatenate(self._delay_ref)))
        if delay > 0:
            self.canceller = EchoCanceller(delay=delay)
        self._delay_samples = delay
        self._delay_mic, self._delay_ref = [], []
        self.emit("delay-estimate", delay_samples=delay, delay_ms=round(delay / SR * 1000, 1),
                  collected_s=round(collected / SR, 2))

    def _clean_frame(self, mic_bytes: bytes, ref: np.ndarray) -> tuple[bytes, float, float, float, float]:
        """Cleaned chunk for this frame, the dB the canceller removed, the mask's gain in dB, and
        the microphone and cleaned levels in dB (B3's near-end attenuation reads the last two).

        ``ref`` is the reference for this frame in the scale the live capture path uses (-1..1,
        see audio.py:411); the microphone travels as the int16 bytes the pipeline moves.
        """
        gain_db = 0.0
        mic = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float32)
        mic_db = level_db(mic)
        if self.canceller is not None and self.ref_ring is not None:
            self._estimate_delay(mic, ref)
            mic_bytes = mix_chunk(mic_bytes, self.ref_ring, self._mic_counter, self.canceller, 0)
        self._mic_counter += len(mic_bytes) // 2
        cleaned = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float32)
        erle = erle_removed_db(mic, cleaned) if self.canceller is not None else 0.0
        if self.mask is not None:
            # KNOWN INSTRUMENT DEFECT, left exactly as B2/B5 recorded it (f139 measured what fixing
            # it would do; see tasks/notes-f139-*.md "the instrument" section). ReferenceMask.process
            # is a spectral gain: it returns samples in the scale it was given, and its dB semantics
            # (refmask.py `last_user_db = 10*log10(power)`, `user_present(mask, 3.0)`) only make sense
            # on normalised -1..1 samples. B2 feeds it int16-scale microphone samples together with
            # float32 -1..1 reference samples, then clips the result to -1..1 before pcm16() multiplies
            # it back by 32767, so EVERY non-silent frame of echo mode `voxa` lands at ±32767: a
            # saturated square wave, and a mask whose band model sees a microphone 32768x louder than
            # the reference it is comparing with. That is why the mode reports erle 0 dB, a microphone
            # level pinned at 1.0, zero suspects, zero barge-ins and a NULL turn_retention_rate
            # (gates.toml records those as null in the default mode).
            #
            # f139 tried the fix (normalise in, pcm16 back out) and measured the whole default-mode
            # column: user_recall_nst 0.6091 -> 0.9455, nearend_atten_dt_db 8.637 -> 3.638,
            # user_word_deletion_rate 0.3161 -> 0.2720, protected_word_keep_rate 0.2273 -> 0.5455,
            # turn_retention_rate NULL -> 1.0 - and at the same time echo_leak_word_rate 0.1652 ->
            # 0.2174, cpu_core_share 0.7211 -> 0.7888 and every self-interruption metric 0.0 ->
            # non-zero, because those zeros meant "nothing ever triggers", not "nothing goes wrong".
            # Seven ratchet lines in voxa/bench/gates.toml would go red for EVERY later task until the
            # echo stages (roadmap tasks 2 and 4: AEC3, residual mask) earn them back, and the brief
            # forbids touching a threshold. So the fix belongs with those tasks, which re-record the
            # baseline; milestone A and A2 measure the barge-in path in echo mode `off`, where it runs.
            masked = np.clip(np.asarray(self.mask.process(cleaned.astype(np.float64),
                                                          ref.astype(np.float64)), dtype=np.float64), -1.0, 1.0)
            if masked.size:
                gain = float(np.mean(np.abs(masked))) / max(float(np.mean(np.abs(cleaned))), 1e-9)
                gain_db = 20.0 * float(np.log10(max(gain, 1e-9)))
                mic_bytes = pcm16(masked)
        self._clean_stream.append(mic_bytes)
        return mic_bytes, erle, gain_db, mic_db, level_db(np.frombuffer(mic_bytes, dtype="<i2"))

    # --- the barge-in candidate -------------------------------------------------

    def _evidence(self, level01: float, ref_power: float) -> SourceEvidence:
        """Handoff 6.3's evidence streams for this frame, from what this pipeline can measure.

        Stream 1 (echo consistency): the reference at the echo's moment and how much of the
        microphone it explains. Stream 2 (independent speech): the residual over the echo path's
        OWN recent baseline, plus the VAD probability in the CLEANED signal. Stream 3 (the words)
        is added by ``bargectl`` from the candidate transcript. A field this stage cannot measure
        stays ``None``, which pushes the decision towards "uncertain" rather than towards a guess.
        """
        self._ref_recent.append(ref_power)
        residual = (self.mask.user_energy_db() if self.mask is not None else
                    (self.canceller.residual_power_db if self.canceller is not None else None))
        margin = self.echo_baseline.note(residual) if residual is not None else None
        explained = (self.mask.echo_explained_db() if self.mask is not None
                     else (self.canceller.echo_explained_db if self.canceller is not None else None))
        detector = self.conversation._voice_detector if self.conversation is not None else None
        # A detector that has never run has no opinion; 0.0 would be read as "the cleaned signal
        # is not speech-like", which is a guess, not a measurement.
        measured = detector is not None and getattr(detector, "probs", None)
        prob = float(detector.last_prob) if measured else None
        evidence = SourceEvidence(
            # An echo still travelling counts as a live reference: the speakers went quiet 100-300
            # ms ago and the microphone is still hearing it.
            reference_active=bool(max(self._ref_recent) > MIN_REFERENCE_POWER),
            reference_power=float(ref_power),
            echo_explained_db=explained,
            residual_db=residual,
            residual_margin_db=margin,
            vad_prob=prob,
            alignment_healthy=bool(self._delay_samples > 0
                                   or max(self._ref_recent) <= MIN_REFERENCE_POWER),
            mic_level=float(level01),
            own_level=float(self.barge.gate.baseline),
            # level_ratio is left to bargectl: it is the one that knows the level of the sound
            # that OPENED the gate, not of the frame that happens to do the judging.
        )
        self.last_evidence = evidence
        return evidence

    def _candidate(self) -> tuple[str, str]:
        """The words the muted microphone captured, as ``window._barge_candidate`` reads them.

        ``candidate_text`` (conversation.py:585) transcribes the rolling window and consumes it.
        """
        heard = self.conversation.candidate_text()
        return heard, self.conversation.last_candidate_raw

    # --- one finished utterance -------------------------------------------------

    def _finish_utterance(self, segment: bytes, frame: int) -> None:
        """The decision ``ConversationController._run`` makes once a segment is complete.

        A turn that began with a promoted span keeps its stable prefix: the prefix's words were
        recognised once, from exactly the promoted audio, and only the audio that arrived after
        the span is transcribed here (the same rule the live controller applies).
        """
        self.emit("pause", segment_seconds=round(len(segment) / 32000.0, 3))
        waiting = self.conversation.waiting_for_prompt or self.conversation.keep_prompt
        hint = f"{self.wake_word.strip().capitalize()}." if not waiting else ""
        prefix_text, prefix_bytes, prefix_words = self.conversation.take_turn_prefix()
        if prefix_text:
            tail = segment[prefix_bytes:] if len(segment) >= prefix_bytes else b""
            heard = (self.asr.transcribe_words(tail, hint) if tail
                     else {"text": "", "words": [], "no_speech_prob": None,
                           "audio_seconds": 0.0, "asr_seconds": 0.0, "segments": 0})
            heard["text"] = f"{prefix_text} {heard['text']}".strip()
            heard["words"] = [*prefix_words, *heard["words"]]
            heard["audio_seconds"] = round(len(segment) / 32000.0, 3)
            heard["prefixed"] = True
        else:
            heard = self.asr.transcribe_words(segment, hint)
        self.emit("transcript", **heard)
        self.transcript_texts.append(heard["text"])
        # The own-speech filter is gated on the AUDIO evidence for this span, and a turn that
        # began with a promoted span is confirmed user: its words are never rewritten by a text
        # match (voxa/attribution.py, handoff 5.3).
        text = self.conversation._filter_own_speech(heard["text"], evidence=self.last_evidence,
                                                    confirmed_user=bool(prefix_text))
        if text is None:
            self.emit("endpoint", text="", complete=False, dropped="own-speech")
            return
        if not text:
            return
        if not prefix_text and self.last_evidence is not None and self.last_evidence.reference_active:
            # A whole utterance taken while the speakers were still live: it may only become a
            # request when the audio supports a second source (shared brief: no automatic
            # commands from unqualified audio during playback). When it does, the words fall
            # through to the request rules below - they are not ignored for being qualified.
            from voxa.attribution import USER as SOURCE_USER
            if attribute(self.last_evidence).source != SOURCE_USER:
                self.counts["ignored"] += 1
                self.emit("ignored", text=text, reason="unqualified")
                return
        keep_prompt = self.conversation.keep_prompt
        exit_kind = detect_exit_phrase(text, self.wake_word) if not keep_prompt else None
        followup_open = bool(self._followup_deadline is not None
                             and self._now < self._followup_deadline)
        self.emit(
            "endpoint",
            text=text,
            complete=is_complete(text, self.wake_word),
            wake_found=strip_wake_word(text, self.wake_word) is not None,
            exit=exit_kind,
            followup_open=followup_open,
            waiting=waiting,
            no_speech_prob=heard["no_speech_prob"],
            words=len(heard["words"]),
        )
        if exit_kind is not None:
            self.counts["exit"] += 1
            self.emit("exit", kind=exit_kind, text=text)
            if exit_kind == "cancel":
                self.conversation.waiting_for_prompt = False
                self._followup_deadline = None
            return
        if not keep_prompt and not waiting:
            remainder = strip_wake_word(text, self.wake_word)
            if remainder is None:
                if followup_open and len(text.split()) >= 2 and is_followup(text):
                    self._followup_deadline = None
                    self._deliver(text, frame, kind="followup")
                else:
                    self.counts["ignored"] += 1
                    self.emit("ignored", text=text)
                return
            if remainder:
                self._deliver(remainder, frame, kind="wake", words=heard["words"],
                              no_speech_prob=heard["no_speech_prob"])
            else:
                self.conversation.waiting_for_prompt = True
            return
        self._deliver(text, frame, kind="prompt" if keep_prompt else "armed",
                      words=heard["words"], no_speech_prob=heard["no_speech_prob"])
        self.conversation.waiting_for_prompt = keep_prompt

    def _deliver(self, text: str, frame: int, *, kind: str, words: list[dict] | None = None,
                 no_speech_prob: float | None = None) -> None:
        """The request that would reach the assistant (what ``on_prompt`` carries in the app)."""
        if not text:
            return
        self.requests.append(text)
        self.counts["turns"] += 1
        self.turns.append({"text": text, "kind": kind, "frame": frame})
        self.emit("request", kind=kind, text=text, words=words or [],
                  no_speech_prob=no_speech_prob, request_id=len(self.requests) - 1)

    # --- the barge-in effects ---------------------------------------------------

    def _segment_frame(self, frame: Frame) -> bytes | None:
        """Feed one captured frame to the speech-pause rules, EXACTLY ONCE per sequence id.

        The promoted span and the frames that arrive after an interruption overlap in time;
        the frame id is what makes the continuation an append instead of a duplicate.
        """
        if frame.seq in self._segmented:
            return None
        self._segmented.add(frame.seq)
        return self.segmenter.step(frame.pcm, frame.level, frame.t_s)

    def _apply_barge(self, outcome) -> None:
        if outcome.action == "yield":
            # The fast half of the decision (task A3): the floor is yielded on the ACOUSTIC
            # onset, before a single word has been transcribed. The reference stream keeps
            # playing - that is the echo the microphone still hears, and what the next frames
            # compare against - while the app's own playback stops at the position it had
            # reached. What happens to her WORDS is decided later, by the transcript.
            self.emit("tts-stop", reason="barge-in-onset",
                      audible_ms=round((self._now - self._speak_started_at) * 1000.0, 1),
                      onset_audio_ms=round(outcome.onset_audio_ms, 1),
                      evidence_ms=round(outcome.evidence_ms, 1),
                      margin_db=round(outcome.margin_db, 2))
            self.barge_yielded = True
            self.conversation.unmute()
            self.conversation.arm_prompt()
            return
        if outcome.action == "offline":
            self.counts["exit"] += 1
            self.emit("exit", kind="goodbye", source="barge-in", text=outcome.heard)
            return
        if outcome.action != "interrupt":
            return
        self.counts["barge_in"] += 1
        self.barge_heard.append(outcome.heard)
        self._interrupted_text = self._spoken_text
        self.emit("barge-in-confirm", heard=outcome.heard, raw=outcome.raw,
                  level=round(outcome.level, 4), listening_seconds=round(outcome.listening_seconds, 3))
        # The effects voxa/window.py performs on its INTERRUPT branch. There is no duck/pause of
        # Voxa's own stream in the current code, so no "duck" event is emitted: that is a finding,
        # not an omission. When the onset already yielded the floor, the playback stopped there:
        # a second tts-stop for the same reply would count the same cessation twice.
        if not self.barge_yielded:
            self.emit("tts-stop", heard=outcome.heard)
        self.own_speech.finished()
        # The effects voxa/window.py performs on its INTERRUPT branch, plus milestone A's
        # promotion: the pinned span becomes the beginning of the user turn (once), and the
        # microphone that opens next continues the same utterance. The live controller queues
        # those frames for its worker thread (ConversationController.promote_pinned); this
        # runner drives the same frames through its own SpeechSegmenter instead.
        promoted = self.conversation.promote_pinned(enqueue=False)
        self.emit("capture-promote", frames=[item.seq for item in promoted],
                  span_seconds=round(self.conversation.capture.span_seconds, 3),
                  pin_from=self.conversation.capture.pin_from)
        for item in promoted:
            self._segment_frame(item)
        self.conversation.unmute()
        self.conversation.arm_prompt()
        self.conversation.waiting_for_prompt = True
        self.speaking = False

    # --- the frame loop ---------------------------------------------------------

    def run(self, mic: np.ndarray, ref: np.ndarray, truth: dict) -> dict:
        self._prepare_echo_stages()
        wake = self.wake_word
        self.conversation = ConversationController(
            wake_whisper=self.asr,
            prompt_whisper=self.asr,
            language=self.settings.language,
            wake_word=wake,
            threshold=self.settings.voice_threshold,
            silence_ms=self.settings.silence_ms,
            max_segment_seconds=self.settings.max_segment_seconds,
            early_silence_ms=self.settings.early_silence_ms,
            voice_detector=True,
            on_woken=lambda: None,
            on_prompt=lambda text: None,
            on_status=lambda text: None,
            on_error=lambda text: None,
        )
        self.conversation.own_speech = self.own_speech
        # The capture ring timestamps its frames with the runner's simulated clock, so a
        # clip replayed faster than real time keeps real ordering and real durations.
        self.conversation.capture.clock = self._clock
        self.segmenter = SpeechSegmenter(
            threshold=self.settings.voice_threshold,
            silence_seconds=self.settings.silence_ms / 1000.0,
            max_segment_seconds=self.settings.max_segment_seconds,
            early_silence_seconds=self.settings.early_silence_ms / 1000.0,
            early_check=self._early_check,
            is_voice=self.conversation._voice_check,
        )
        frames = (max(len(mic), len(ref)) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
        playback_text = (truth.get("playback") or {}).get("text") or ""
        erle_values: list[float] = []
        mask_gains: list[float] = []
        asr_calls_at_start, asr_audio_at_start = self.asr.calls, self.asr.audio_seconds
        asr_processing_at_start = self.asr.processing_seconds
        started = time.perf_counter()
        if self.degraded:
            self.emit("echo-mode", mode=self.echo_mode, requested="webrtc", available=False,
                      degraded=self.degraded)
        for frame in range(frames):
            self._frame = frame
            self._now = (frame + 1) * (CHUNK_MS / 1000.0)
            lo, hi = frame * CHUNK_SAMPLES, (frame + 1) * CHUNK_SAMPLES
            mic_chunk_i16 = mic[lo:hi]
            ref_chunk = ref[lo:hi]  # float32 -1..1, exactly what the live reference ring holds (audio.py:411)
            if mic_chunk_i16.size == 0:
                break
            ref_power = float(np.mean(ref_chunk * ref_chunk)) if ref_chunk.size else 0.0
            if self.ref_ring is not None:
                self.ref_ring.append(ref_chunk)
            # 1. Does this frame belong to a reply Voxa is reading out? The live app is told
            #    when playback starts and finishes; the reference stream is the bench's equivalent,
            #    with a 200 ms hold because a reply dips between words.
            if ref_power > SPEAKING_POWER and not self.speaking:
                self._speak_started(playback_text)
            elif self.speaking:
                self._quiet_frames = self._quiet_frames + 1 if ref_power <= SPEAKING_POWER else 0
                if self._quiet_frames >= QUIET_FRAMES:
                    self._speak_finished()
            # 2. The microphone frame through the echo stages, then the level the app would see.
            mic_bytes = np.asarray(mic_chunk_i16, dtype="<i2").tobytes()
            cleaned_bytes, erle, gain_db, mic_db, clean_db = self._clean_frame(mic_bytes, ref_chunk)
            erle_values.append(erle)
            mask_gains.append(gain_db)
            level = AudioCapture._rms(cleaned_bytes)
            evidence = self._evidence(min(level, 4000.0) / 4000.0, ref_power)
            captured = self.conversation.feed(cleaned_bytes, level, seq=frame, raw=mic_bytes)
            # While the mic is muted the live queue receives nothing either (conversation.py:feed),
            # so no utterance is segmented out of her own playback. The CAPTURE ring keeps every
            # frame in both states, which is what lets an accepted interruption keep its words.
            completed = (self._segment_frame(captured)
                         if captured is not None and not self.conversation.muted else None)
            detector = self.conversation._voice_detector
            prob = float(detector.last_prob) if detector is not None else 0.0
            self.emit(
                "frame",
                vad=bool(prob >= detector.threshold) if detector is not None else None,
                vad_prob=round(prob, 3),
                voiced=bool(completed is not None),
                level=round(level, 1),
                erle_db=round(erle, 2),
                mask_gain_db=round(gain_db, 2),
                mic_level_db=mic_db,
                clean_level_db=clean_db,
                delay_samples=self._delay_samples,
                ref_power=round(ref_power, 8),
                echo_explained_db=(None if evidence.echo_explained_db is None
                                   else round(evidence.echo_explained_db, 2)),
                residual_margin_db=(None if evidence.residual_margin_db is None
                                    else round(evidence.residual_margin_db, 2)),
                speaking=bool(self.barge.speaking_since > 0.0),
                muted=bool(self.conversation.muted),
            )
            outcome = self.barge.observe(
                min(level, 4000.0) / 4000.0,
                speaking=self.barge.speaking_since > 0.0,
                paused=False,
                muted=bool(self.conversation.muted),
                wake_word=wake,
                candidate=self._candidate,
                evidence=evidence,
                # The onset rule's inputs (voxa/onset.py): the microphone level of this frame and
                # the loudest the speaker path could have put in it at the same moment, BOTH in
                # dBFS (the microphone arrives as an int16 RMS, the reference as a float -1..1),
                # the echo's travel time - which is what makes the echo's moment knowable at all -
                # and how much audio this frame stands for.
                mic_db=dbfs(mic_db),
                reference_db=10.0 * math.log10(ref_power) if ref_power > 1e-12 else -140.0,
                delay_ms=(self.canceller.delay_samples / SR * 1000.0
                          if self.canceller is not None and self.canceller.delay_samples > 0 else None),
                dt_ms=CHUNK_MS,
            )
            if outcome.onset:
                # Speech onset: pin the captured span (pre-roll included) before the words are
                # judged, exactly as voxa/window.py does on the same observation.
                self.conversation.note_onset(seq=frame,
                                             level_floor=self.barge.onset_level_floor())
                self.emit("capture-pin", onset_frame=frame, pin_from=self.conversation.capture.pin_from,
                          span_seconds=round(self.conversation.capture.span_seconds, 3))
            if outcome.gate_opened and outcome.reason == "listening" and self.barge.pending_since == self._now:
                self.counts["suspect"] += 1
                self.emit("suspect", level=round(outcome.level, 4),
                          baseline=round(float(self.barge.gate.baseline), 4),
                          settled=bool(self.barge.gate.settled))
            if outcome.heard or outcome.raw:
                self.emit("candidate", heard=outcome.heard, raw=outcome.raw,
                          source=outcome.source,
                          own_voice=bool(outcome.action == "own-voice"))
                if outcome.action == "own-voice":
                    self.counts["own_voice"] += 1
                    # Rejected, not deleted: the span survives its evidence window.
                    self.conversation.note_rejected()
                elif outcome.action == "hold":
                    # Undecided: the pin stays open, nothing is stripped, no command runs from it.
                    self.counts["hold"] += 1
            self._apply_barge(outcome)
            if completed is not None:
                self._finish_utterance(completed, frame)
        if self.speaking:
            self._speak_finished()
        pending = self.segmenter.pending()
        if pending:
            self._finish_utterance(pending, self._frame)
        # One more pass over the whole cleaned stream of the clip: the runner's own transcript of
        # what its pipeline delivered to the ASR stage, independent of where the speech-pause
        # rules cut the utterances. ``voxa.bench agree`` compares this with a direct transcribe
        # of the mic WAV, which is what proves the instrument (20 ms chunking, int16 handling,
        # pinned parameters, the echo stages) rather than the product's turn-taking.
        stream = b"".join(self._clean_stream)
        if stream:
            clip_heard = self.asr.transcribe_words(stream, "")
            self.emit("clip-transcript", text=clip_heard["text"], words=clip_heard["words"],
                      no_speech_prob=clip_heard["no_speech_prob"],
                      audio_seconds=clip_heard["audio_seconds"], asr_seconds=clip_heard["asr_seconds"])
        wall = time.perf_counter() - started
        audio_seconds = frames * CHUNK_SAMPLES / SR
        return {
            "id": truth["id"],
            "scenario": truth["scenario"],
            "expect": truth.get("expect"),
            "overlap_position": truth.get("overlap_position"),
            "pair": truth.get("pair"),
            "seconds": round(audio_seconds, 3),
            "frames": frames,
            "wall_seconds": round(wall, 3),
            "realtime_factor": round(audio_seconds / wall, 3) if wall > 0 else None,
            "echo_mode": self.echo_mode,
            "degraded": self.degraded,
            "delay_estimated_samples": self._delay_samples,
            "erle_median_db": _median(erle_values),
            "mask_gain_max_db": round(max(mask_gains), 2) if mask_gains else None,
            "requests": self.requests,
            "turns": self.turns,
            "counts": self.counts,
            "barge_heard": self.barge_heard,
            "transcript_texts": self.transcript_texts,
            "asr_calls": self.asr.calls - asr_calls_at_start,
            "asr_audio_seconds": round(self.asr.audio_seconds - asr_audio_at_start, 3),
            "asr_processing_seconds": round(self.asr.processing_seconds - asr_processing_at_start, 3),
        }

    def _speak_started(self, text: str) -> None:
        """Voxa starts playing: the mic is muted and the onset evidence restarts (window.py:3004-3010).

        There is no grace period any more (task A3): what restarts is the evidence - the gate's
        baseline and the onset detector's run of unexplained sound - so the first frame of the
        reply is already judged, which is where an interruption that starts together with her
        speech lives.
        """
        self.speaking = True
        self._quiet_frames = 0
        self._speak_started_at = self._now
        self.barge_yielded = False
        self.own_speech.started(text)
        self._spoken_text = text
        if text != self._interrupted_text:
            self.barge.reply_started(text)
        self.conversation.mute()
        self.emit("tts-start", text=text[:80])

    def _speak_finished(self) -> None:
        """Playback ended: unmute and let the echo tail finish (window.py:_on_conversation_speech_done).

        The live method also arms the next utterance as a prompt and opens the follow-up window,
        so a person answering straight after a reply does not have to say the wake word again.
        """
        self.speaking = False
        self._quiet_frames = 0
        self.own_speech.finished()
        self.conversation.unmute()
        if self.settings.followup_seconds > 0:
            self.conversation.arm_prompt()
            self._followup_deadline = self._now + self.settings.followup_seconds
        self.emit("tts-end")
    def _early_check(self, segment: bytes) -> bool:
        """``segment_stream``'s early hook: is what has been heard already a whole command?"""
        heard = self.asr.transcribe_words(segment, f"{self.wake_word.strip().capitalize()}.")
        text = self.conversation._filter_own_speech(heard["text"])
        if text is None:
            return False
        return bool(is_complete(text, self.wake_word))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.median(np.asarray(values, dtype=np.float64))), 3)


def read_wav_int16(path: Path) -> np.ndarray:
    """A clip's WAV as the int16 samples it holds.

    The runner must not re-encode the microphone: a float round-trip changes samples by a unit
    and would make the transcript comparison measure the bench instead of the pipeline.
    """
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != SR:
            raise ValueError("only 16-bit mono 16 kHz WAVs are supported")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")


def load_clip(clip_dir: Path) -> dict:
    """Read one rendered clip: truth plus the microphone (int16), the reference (float) and, for
    roadmap task 1's render-timeline instrument, the three components the renderer mixed."""
    truth_path = clip_dir / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"{truth_path} missing: render the suite first")
    arrays = {"mic": read_wav_int16(clip_dir / "mic.wav"), "ref": read_wav(clip_dir / "ref.wav")}
    for name in ("s", "d", "n"):
        path = clip_dir / f"{name}.wav"
        if path.exists():
            arrays[name] = read_wav(path)
    return {"truth": json.loads(truth_path.read_text()), **arrays}


def build_report(suite: str, seed: int, echo_mode: str, results: list[dict],
                 facts: list[dict], audio_seconds: float, wall: float, model: str) -> dict:
    """The report envelope the bench fixes: suite, echo_mode, pass, every metric with its CI.

    Roadmap B3 moved the real work into :func:`voxa.bench.report.build_report`, which computes
    every metric the shared brief names over the clip facts (``voxa/bench/metrics.py``), bootstraps
    a 95 % CI from 1000 resamples and breaks the numbers down per scenario. This is the one call
    site, kept so ``run`` reads as: run the clips, then report them.
    """
    from .report import build_report as _build

    return _build(suite, seed, echo_mode, results, facts, audio_seconds, wall, model)


def check_gates(report: dict) -> list[str]:
    """Evaluate ``voxa/bench/gates.toml`` (roadmap B3 creates it with no thresholds, B5 records them)."""
    from . import gates

    return gates.evaluate(gates.GATES_PATH, report)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voxa.bench run",
        description="push bench clips through Voxa's real hearing and barge-in pipeline")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=0, help="the render's seed; clips are read, not built")
    parser.add_argument("--echo-mode", choices=("voxa", "webrtc", "aec3", "off"), default="voxa")
    parser.add_argument("--out", type=Path, default=None,
                        help="report path (default .voxa-bench/report-<suite>.json)")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("VOXA_BENCH_DATA", str(DEFAULT_DATA_ROOT))))
    parser.add_argument("--model", default="", help="faster-whisper model (default: Voxa's, config.py:22)")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N clips (debugging)")
    parser.add_argument("--no-gates", action="store_true", help="skip the gates.toml checks (roadmap B5)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.echo_mode == "aec3":
        print("echo mode aec3 is roadmap task 2 (an AEC3 backend in voxa/anc.py); it does not exist yet.",
              file=sys.stderr)
        return 2
    report = run_clips(args.suite, args.seed, args.echo_mode, args.data_root, args.limit,
                       args.model, args.quiet)
    out_path = args.out or Path(f".voxa-bench/report-{args.suite}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        audio = report["run"]["audio_seconds"]
        wall = report["run"]["wall_seconds"]
        print(f"wrote {out_path}: {report['run']['clips']} clips, {audio:.1f} s of audio in "
              f"{wall:.1f} s wall ({audio / wall:.2f}x audio time), "
              f"{report['run']['turns']} turns, {report['run']['barge_ins']} barge-ins")
        print("\n".join(report_module.format_metrics(report)))
    if not args.no_gates:
        failures = check_gates(report)
        for line in failures:
            print(f"gate: {line}", file=sys.stderr)
        if failures:
            return 1
    return 0


def render_timeline_measure(loaded: dict, echo_mode: str) -> dict:
    """Roadmap task 1: the same clip through the simulated output sink (``voxa/bench/sinksim``).

    The frozen B1 clips have no output sink, no capture latency and no second application, which are
    exactly the things the render timeline exists to get right, so the instrument wraps the clip's
    own components around a sink of known latency and jitter and measures the product's
    :class:`voxa.timeline.RenderTimeline` against it. It runs in the modes that use Voxa's own
    canceller; in ``off`` there is no canceller to align, and the metrics say so instead of reporting
    a number they cannot defend.
    """
    if echo_mode != "voxa":
        return {"measurable": False,
                "reason": f"echo mode {echo_mode} does not run Voxa's own canceller to align"}
    try:
        from . import sinksim

        return sinksim.measure_clip(loaded, loaded["truth"])
    except Exception as exc:  # noqa: BLE001 - the instrument must never lose the clip's own result
        return {"measurable": False, "reason": f"render instrument failed: {exc}"}


def run_clips(suite: str, seed: int, echo_mode: str, data_root: Path, limit: int | None,
              model: str, quiet: bool = False) -> dict:
    """Run every clip of a suite, write the per-clip events, and return the report."""
    suite_dir = data_root / suite
    if not suite_dir.is_dir():
        raise SystemExit(f"{suite_dir} does not exist: run python -m voxa.bench render --suite {suite} first")
    clip_dirs = sorted(path for path in suite_dir.iterdir() if (path / "truth.json").exists())
    if limit:
        clip_dirs = clip_dirs[:limit]
    asr = PinnedASR(model_name=model)
    wake_word = Settings().normalized().wake_word
    events_root = Path(".voxa-bench") / suite
    started = time.perf_counter()
    audio_seconds = 0.0
    results: list[dict] = []
    facts: list[dict] = []
    for index, clip_dir in enumerate(clip_dirs):
        loaded = load_clip(clip_dir)
        runner = ClipRunner(asr=asr, echo_mode=echo_mode, wake_word=wake_word)
        summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])
        summary["render"] = render_timeline_measure(loaded, echo_mode)
        clip_out = events_root / summary["id"]
        clip_out.mkdir(parents=True, exist_ok=True)
        (clip_out / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in runner.events)
        )
        results.append(summary)
        facts.append(metrics_module.clip_facts(loaded["truth"], runner.events, summary))
        audio_seconds += summary["seconds"]
        if not quiet:
            print(f"ran {index + 1}/{len(clip_dirs)} {summary['id']} {summary['seconds']:6.2f}s audio "
                  f"{summary['wall_seconds']:6.2f}s wall turns {len(summary['requests'])} "
                  f"barge {summary['counts']['barge_in']}", flush=True)
    return build_report(suite, seed, echo_mode, results, facts, audio_seconds,
                        time.perf_counter() - started, model)


if __name__ == "__main__":
    raise SystemExit(run())
