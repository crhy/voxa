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

from ..align import ExactAligner, ExactDelayLock
from ..anc import (MIN_REFERENCE_POWER, Aec3Canceller, DelayTracker, EchoCanceller,
                   aec3_available, estimate_delay)
from ..audio import AudioCapture, ReferenceRing, mix_chunk, webrtc_available
from ..bargectl import BargeInController
from ..onset import OnsetDetector, dbfs
from ..echo import OUTPUT_SPEAKERS, output_tuning
from ..attribution import EchoPathBaseline, SourceEvidence, attribute
from ..capture import Frame
from ..config import Settings
from ..conversation import ConversationController, detect_exit_phrase, strip_wake_word
from ..dictation import SpeechSegmenter
from ..endpoint import (EndpointPolicy, SmartTurn, arbitrate_endpoint, ends_dangling,
                        smart_turn_model_path)
from ..endpoint import is_complete, is_followup
from ..ownspeech import OwnSpeech
from ..refmask import LockedResidualMask, ReferenceMask, ResidualMask
from ..voicedetect import SPEECH_PLAYBACK_THRESHOLD, DoubleTalkGate, VoiceDetector
from . import metrics as metrics_module
from . import modes
from . import report as report_module
from .render import DEFAULT_DATA_ROOT, SR, read_wav

# Roadmap task 4 (f131): the residual suppressor on the canceller's output. The defaults are the
# task's rule (6 dB protect, TTS + 300 ms tail); the sweep of the over-subtraction and the protect
# margin lives in tasks/notes-f131-voxa-hear-t4-residual-mask.md and chose the shipped values.
MASK_OVER = 0.05  # base suppression gain on a residual band
MASK_PROTECT_DB = 9.0  # output exceeds the predicted residual by this much = the user, keep it
MASK_OVER_RAMP = 0.5  # extra dB of suppression per dB of ERLE shortfall
MASK_ERLE_TARGET_DB = 20.0  # a linear canceller removes 15-25 dB; below this it left more echo
MASK_TAIL_S = 0.3  # active during TTS plus this tail

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
        mask_params: dict | None = None,
        dt_params: dict | None = None,
        word_window_ms: float = 400.0,
        data_root: str = "",
    ) -> None:
        self.asr = asr
        self.echo_mode = echo_mode
        self._mask_params = dict(mask_params or {})
        self._dt_params = dict(dt_params or {})
        # Roadmap task 7 (f175): the +- window (ms) a heard word may sit from one of her played
        # words and still be a time-matched echo candidate. Swept 200/400/600 in the notes; 400
        # is the shipped default (the edge-tts boundaries are estimates, so the slack absorbs
        # synthesis jitter - the delay itself is exact, f168's lock).
        self._word_window_ms = float(word_window_ms)
        self._playback_words: list[dict] = []
        self._joint_history: list[tuple[float, bool]] = []
        self._timeline_fed = False
        self.settings = settings or Settings().normalized()
        self.wake_word = wake_word
        # The live ConversationController nudges Whisper with a machine vocabulary
        # (``voxa/vocabulary.build_hint``: command verbs + installed app names + Spaced
        # products + taught names), built once in ``_build_prompt_hint``. The bench must feed
        # the SAME hint or it under-tests what the live product's ASR actually sees
        # (f146: voxatest scored "Devuan"→"Devon" because the bench hint was only "Voxa.").
        # Never fatal: an empty/absent list falls back to the wake-word-only hint the bench used
        # before, so a clip with no machine vocabulary runs exactly the pipeline it ran before.
        self._vocabulary_hint = self._build_vocabulary_hint()
        # Roadmap task 9: the same acoustic endpoint decision the live app makes,
        # from the weights under the bench data root.
        model_path = smart_turn_model_path(self.settings.smart_turn_model, data_root)
        self._smart_turn = SmartTurn.shared(model_path) if model_path else None
        self._endpoint_policy = EndpointPolicy(self.settings.endpoint_patience)
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
        # CPU seconds spent inside the echo stages (canceller + mask) for this clip: roadmap task 2's
        # "<= 10 % of one core" is about the canceller, and cpu_core_share measures the ASR stage.
        self._echo_cpu = 0.0
        self._delay_mic: list[np.ndarray] = []
        self._delay_ref: list[np.ndarray] = []
        self._delay_samples = 0
        self._delay_frac = 0.0  # Hearing 2: the fractional locked delay (the read offset)
        self._aligner: ExactAligner | None = None  # Hearing 2: exact lock, mode voxa
        self._delay_lock: ExactDelayLock | None = None  # f168: the shipping lock (mode voxa)
        # f164: speex-locked / aec3-locked take Voxa's locked bulk delay as a reference-ring offset
        # (their own delay search then only refines the room tail); voxa-locked uses set_offset instead.
        self._ref_offset_lock = False
        self._locked_ref_offset = 0
        # f169: the voxa-residual mask is handed the locked delay frame by frame (set_delay).
        self._mask_sees_lock = False
        # f172: is the delay the canceller/mask read a PROVEN lock (never the drifting
        # DelayTracker estimate)? The joint gate reads coherence only when it is.
        self._lock_proven = False
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
        # Roadmap task 5: the joint double-talk gate over the cleaned stream, and a Silero run
        # that SEES the muted microphone: the conversation's own detector never runs while a
        # reply plays, which left vad_prob "not measured" for every playback frame. The runner
        # feeds the cleaned bytes of every frame to both.
        # Roadmap task 10: the output class the clip plays through, its epoch counter, and the
        # recovery window an epoch opens. Defaults are today's shipped thresholds, so a clip with
        # no device switch (every frozen clip) runs exactly the pipeline it ran before.
        self.output_class = OUTPUT_SPEAKERS
        self.tuning = output_tuning(self.output_class)
        self.epoch = 0
        self._recovery_until_s = 0.0
        self._device_switch: dict | None = None
        self.held_commands = 0
        self.dt_gate = DoubleTalkGate(**self._dt_params)
        self._clean_vad: VoiceDetector | None = None  # lazy: it loads the Silero model
        self._clean_prob: float | None = None
        self._mask_margin: float | None = None
        self._dt_suspect = False
        self._dt_measurable = False
        self._dt_echo = False
        # Roadmap task 9 re-plan (f181): the promote_wait. While the joint suspect / turn state
        # / arriving-speech verdicts are open and a never-completed pinned span is still owed,
        # the events that CLOSE a turn (endpoint/request/exit/ignored) are held - the assembly
        # cut waits the open suspect out; the closure is never silently dropped.
        self._promote_defer = False
        self._promote_held_events: list[tuple[str, dict]] = []
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

    def _build_vocabulary_hint(self) -> str:
        """The live app's Whisper vocabulary hint, or "" when it cannot be built here.

        Mirrors ``ConversationController._build_prompt_hint`` (voxa/conversation.py:366): the same
        ``voxa/vocabulary.build_hint`` sentence over the machine's command verbs, installed app names
        and taught spellings. Never fatal — an empty hint leaves the bench on the wake-word-only
        hint it used before, so a clip with no machine vocabulary runs exactly the pipeline it ran before.
        """
        try:
            from ..spellings import names as taught_names
            from ..vocabulary import build_hint, refresh_app_names

            return build_hint(refresh_app_names(), self.wake_word, people=taught_names())
        except Exception:  # noqa: BLE001 - a broken probe must not change what the clip runs
            return ""

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
        if self.echo_mode in modes.HEARING2_MODES:
            self._prepare_hearing2_stages()
            return
        if self.echo_mode == "aec3" and not aec3_available():
            # Same rule as the live app (voxa/audio.py:set_echo_mode): say so instead of pretending
            # the cancellation happened, and keep the mode name honest for the report.
            self.degraded = "aec3 unavailable (pywebrtc-audio not importable): running Voxa's NLMS"
            self.echo_mode = "voxa"
        if self.echo_mode in ("voxa", "aec3"):
            self.ref_ring = ReferenceRing(REFERENCE_SECONDS * SR)
            if self.echo_mode == "aec3":
                # The bench measures the canceller plus the protection below, not AEC3's own
                # power-ratio guard: that one mistakes the tail of these rooms (RT60 0.2-0.8 s) for
                # a speaker, and the near-end mask owns protection in this mode instead.
                self.canceller = Aec3Canceller(double_talk_guard=False)
            else:
                # Roadmap task 3 strata: these rooms decay over hundreds of milliseconds, so the
                # filter spans 256 ms (taps) and the step runs fast enough to track the +-200 ppm
                # read-point movers inside one clip (mu).
                self.canceller = EchoCanceller(taps=4096, mu=0.7)
                # Hearing 2 (task h02), shipped into production by f168: Voxa's own render
                # timeline makes the echo delay KNOWN, so it is LOCKED, not re-estimated every
                # 0.5 s. The bench now measures the SAME shipping path AudioCapture runs:
                # ExactDelayLock (voxa/align.py) re-points the canceller's reference from the
                # proven lock, and only while the lock cannot prove itself does the honest
                # DelayTracker estimate keep it pointed (visible degraded, never raw audio).
                self._delay_lock = ExactDelayLock(sample_rate=SR, canceller=self.canceller)
                self._aligner = self._delay_lock.aligner
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
            # Roadmap task 4 (f131): the whole point of the mode is now the CHAIN - her NLMS
            # canceller plus the residual mask on its output, the same rule the aec3 mode runs.
            # (Before this task the mask was fed at the wrong scale and the stream it emitted was
            # the documented instrument defect; see tasks/notes-f139-voxa-hear-a2-attribute-by-audio.md.
            # Task 4 is the task the brief says owns the fix, and it is fixed here: the residual
            # mask is fed the cleaned float stream and the -1..1 reference, and it gives back
            # float samples that go to pcm16 the normal way.)
            # f171 (shipping decision on f169's measurement): the mode runs the SAME residual
            # stage the live capture path now runs - LockedResidualMask on the locked output,
            # handed that lock frame by frame - so this row stays production's number.
            self.mask = self._exact_residual_mask()
            self._mask_sees_lock = True
        elif self.echo_mode == "aec3":
            # f131 (roadmap task 4): ReferenceMask on the AEC output with the aligned reference.
            # The f129 binary protect/floor switch is gone: the residual mask IS the per-band
            # protection now (predicted residual from the output-vs-reference ratio, held at the
            # room's decay for the tail, over-subtraction that rises when the running ERLE is low,
            # and a band whose output exceeds the predicted residual by >= 6 dB stays untouched).
            self.mask = self._residual_mask()

    def _prepare_hearing2_stages(self) -> None:
        """Hearing 2 (task h01): the candidate arms, behind the same canceller interface.

        A candidate whose dependency is not on this machine is never substituted: the mode keeps
        its name, says why in ``degraded``, and runs with no linear stage at all. Quoting that as
        a cancellation number would lie, so :meth:`voxa.bench.modes.availability` is recorded in
        the report and docs/hearing/frontier.md marks the arm unmeasured.
        """
        self.ref_ring = ReferenceRing(REFERENCE_SECONDS * SR)
        self._delay_done = False
        probe = modes.availability()
        if self.echo_mode == "hybrid":
            # The best linear stage this machine has, then the best residual stage it has: AEC3
            # through the installed wheel, or Voxa's own NLMS, and f131's residual suppressor on
            # its output. (The neural residual stage of the task's spec needs DTLN weights, which
            # this host cannot fetch; that is recorded, not hidden.)
            if aec3_available():
                self.canceller = Aec3Canceller(double_talk_guard=False)
                self._hybrid_linear = "aec3"
            else:
                self.canceller = EchoCanceller(taps=4096, mu=0.7)
                self._delay_tracker = DelayTracker(sample_rate=SR)
                self._hybrid_linear = "voxa"
            self.mask = self._residual_mask()
            return
        if self.echo_mode in modes.LOCKED_MODES or self.echo_mode in modes.RESIDUAL_MODES:
            # f164 (Hearing 2): every linear arm given Voxa's LOCKED render-timeline delay. The
            # ExactAligner (h02/f163) locks the bulk delay from the render clock and is handed to
            # the arm per frame, so the arm's own delay search only refines the room tail. An arm
            # whose base dependency is missing keeps its name and says why (run() already exits 2).
            self.mode_probe = probe[self.echo_mode]
            if not probe[self.echo_mode]["available"]:
                self.degraded = f"{self.echo_mode} unavailable: {probe[self.echo_mode]['reason']}"
                return
            self._aligner = ExactAligner(sample_rate=SR)
            if self.echo_mode in ("voxa-locked", "voxa-residual"):
                # NLMS reference read at the locked fractional offset (set_offset); span 4096 taps.
                self.canceller = EchoCanceller(taps=4096, mu=0.7)
                if self.echo_mode == "voxa-residual":
                    # f169 (roadmap task 4 on exact-delay audio): the residual stage runs on
                    # THIS locked canceller's own output and is handed the same lock, so its
                    # band ratios, its held tail and its 6 dB near-end rule all see the pair
                    # the lock proved. The row differs from ``voxa-locked`` by the stage alone.
                    self.mask = self._exact_residual_mask()
                    self._mask_sees_lock = True
                else:
                    self.mask = self._residual_mask()
            elif self.echo_mode == "aec3-locked":
                # AEC3 wheel fed that same offset by re-timing its reference at the ring (the wheel
                # has no delay setter); its estimator then only refines the room tail.
                self.canceller = Aec3Canceller(double_talk_guard=False)
                self.mask = self._residual_mask()
                self._ref_offset_lock = True
            else:  # speex-locked: reference re-timed OUTSIDE the MDF (its delay_ctl is absent here).
                self.canceller = modes.make_linear("speex", sample_rate=SR)
                self._speex_settings = getattr(self.canceller, "settings", None)
                self._ref_offset_lock = True
            return
        ok, reason = bool(probe[self.echo_mode]["available"]), str(probe[self.echo_mode]["reason"])
        self.mode_probe = probe[self.echo_mode]
        if not ok:
            self.degraded = f"{self.echo_mode} unavailable: {reason}"
            return
        self.canceller = modes.make_linear(self.echo_mode, sample_rate=SR)
        self._speex_settings = getattr(self.canceller, "settings", None)

    def _residual_mask(self) -> ResidualMask:
        """f131: the residual suppressor shared by both modes (the chain: canceller + mask). The
        defaults are the task's; the sweep axes override them via mask_params (run() flags)."""
        params: dict = {
            "over": MASK_OVER,
            "protect_db": MASK_PROTECT_DB,
            "over_ramp": MASK_OVER_RAMP,
            "erle_target_db": MASK_ERLE_TARGET_DB,
            "tail_s": MASK_TAIL_S,
        }
        params.update(self._mask_params)
        return ResidualMask(rate=SR, **params)

    def _exact_residual_mask(self) -> LockedResidualMask:
        """f169 (roadmap task 4 on exact-delay audio): the residual stage of the
        ``voxa-residual`` arm. Its defaults ARE the task's rule (protect_db 6.0; see
        voxa/refmask.LockedResidualMask); the mask sweep axes still override them through
        mask_params (run() flags), and the locked delay arrives per frame via set_delay."""
        return LockedResidualMask(rate=SR, **dict(self._mask_params))

    # --- roadmap task 10: device, headphone and Bluetooth changes as explicit epochs ----

    def _note_output(self, echo_truth: dict) -> None:
        """Read what the clip plays through, and whether it switches mid-clip (task 10).

        Frozen clips carry none of these keys and run exactly the pipeline they ran before. A
        stratum (tools/device_strata_lab.py) or a future clip pins the output class and, when the
        device changes, the instant and the new class - the same facts the live watcher reads from
        ``pactl``/``pw-dump`` (device.bus, device.form_factor, port type).
        """
        self.output_class = echo_truth.get("output_class") or OUTPUT_SPEAKERS
        self.tuning = output_tuning(self.output_class)
        self._device_switch = echo_truth.get("device_switch")
        self._recovery_until_s = 0.0

    def _apply_output_tuning(self) -> None:
        """Hand the task-5 thresholds to this output (report line 189, relaxation for headphones).

        The joint gate is rebuilt from the chain's sweep params (the ``--gate-*`` run() axes),
        so the sweep survives; a non-speakers output then overrides the table's values on top.
        SPEAKERS are today's shipped constants, so the frozen set (all speakers) is untouched.
        """
        tuning = self.tuning
        params = dict(self._dt_params)
        if self.output_class != OUTPUT_SPEAKERS:
            params.update(speech_threshold=tuning.speech_threshold, joint_ms=tuning.joint_ms,
                          hold_ms=tuning.hold_ms, margin_db=tuning.residual_margin_db,
                          coherence_echo=tuning.coherence_echo, erle_drop_db=tuning.erle_drop_db)
        self.dt_gate = DoubleTalkGate(**params)
        # The onset rule is the live barge-in's residual margin: an epoch replaces it wholesale,
        # because its run of unexplained sound started counting a different acoustic path.
        self.barge.onset = OnsetDetector(margin_db=tuning.onset_margin_db,
                                         evidence_ms=tuning.onset_evidence_ms)

    def _begin_epoch(self, output_class: str, *, reason: str = "device-switch") -> None:
        """One explicit epoch on the simulated clock: flush the reference, recreate the canceller,
        re-measure the delay, retune the thresholds, open the recovery window (handoff 6.2)."""
        self.epoch += 1
        self.output_class = output_class or self.output_class
        self.tuning = output_tuning(self.output_class)
        self._apply_output_tuning()
        if self.ref_ring is not None:
            self.ref_ring = ReferenceRing(REFERENCE_SECONDS * SR)
        self._delay_mic, self._delay_ref = [], []
        self._delay_samples = 0
        self._delay_done = False
        self._delay_frac = 0.0
        self._lock_proven = False
        self._locked_ref_offset = 0
        self._delay_tracker = (DelayTracker(sample_rate=SR)
                               if self.echo_mode == "voxa" else None)
        if self.canceller is not None:
            release = getattr(self.canceller, "close", None)
            if release is not None:
                release()
            self.canceller = self._new_canceller()
            if self._delay_lock is not None and self.canceller is not None:
                # The chain's own delay machinery: the new path gets a fresh exact lock
                # (aligner + fallback estimate), exactly as the live begin_epoch resets
                # ``AudioCapture._delay_lock``; the residual mask starts ungifted too.
                self._delay_lock = ExactDelayLock(sample_rate=SR, canceller=self.canceller)
                self._aligner = self._delay_lock.aligner
                if self._mask_sees_lock and self.mask is not None:
                    self.mask = self._exact_residual_mask()
        self._recovery_until_s = self._now + self.tuning.recovery_s
        self.emit("device-epoch", output_class=self.output_class, epoch=self.epoch,
                  reason=reason, delay_search_ms=self.tuning.delay_search_ms,
                  residual_margin_db=self.tuning.residual_margin_db,
                  onset_margin_db=self.tuning.onset_margin_db,
                  recovery_s=self.tuning.recovery_s)

    def _recovering(self) -> bool:
        return self._now < self._recovery_until_s

    def _switch_due(self, frame: int) -> str | None:
        """The epoch a mid-clip device switch asks for, once, at its own frame."""
        switch = self._device_switch
        if not switch or switch.get("fired"):
            return None
        at = float(switch.get("at_s") or 0.0)
        if self._now + (CHUNK_MS / 1000.0) <= at:
            return None
        switch["fired"] = True
        return str(switch.get("output_class") or self.output_class)

    def _new_canceller(self) -> EchoCanceller | None:
        """The canceller this mode/epoch needs (the bench mode decides, as _prepare_echo_stages does)."""
        if self.echo_mode == "aec3":
            return Aec3Canceller(double_talk_guard=False,
                                 stream_delay_ms=int(self.tuning.aec3_seed_ms))
        if self.echo_mode == "voxa":
            return EchoCanceller(taps=4096, mu=0.7)
        return None

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
        # Task 10: the search window the OUTPUT allows bounds the estimate. A plain output's
        # 300 ms is today's bound; Bluetooth widens it to 400 ms for the codec buffering on
        # top of the room (report line 189), so a 300 ms bulk delay is inside the search.
        search_samples = int(round(self.tuning.delay_search_ms / 1000.0 * SR))
        delay = int(estimate_delay(np.concatenate(self._delay_mic), np.concatenate(self._delay_ref),
                                 max_delay=max(4800, search_samples)))
        if delay > 0 and isinstance(self.canceller, Aec3Canceller):
            # AEC3 owns the delay search, but it starts blind: hand it the measured end-to-end
            # delay so its filter and its estimator lock within the clip instead of across it.
            self.canceller.close()
            self.canceller = Aec3Canceller(
                stream_delay_ms=max(1, round(delay / SR * 1000)), double_talk_guard=False)
        elif delay > 0 and getattr(self.canceller, "delay_ctl_supported", False):
            self.canceller.set_delay(delay)
        elif delay > 0 and isinstance(self.canceller, EchoCanceller):
            self.canceller = EchoCanceller(taps=4096, mu=0.7, delay=delay)
        self._delay_samples = delay
        self._delay_mic, self._delay_ref = [], []
        self.emit("delay-estimate", delay_samples=delay, delay_ms=round(delay / SR * 1000, 1),
                  collected_s=round(collected / SR, 2))

    def _observe_echo_delay(self, mic01: np.ndarray, ref: np.ndarray) -> None:
        """Hearing 2 (task h02): lock the echo delay from the render timeline, refine it sub-sample
        only around the prediction, and hold it on the drift line (the fractional-delay resampler).
        The canceller keeps its filter: its reference is read at the locked fractional offset every
        frame - never re-estimated per sentence, never rebuilt.

        f168: mode ``voxa`` runs the SHIPPING lock (:class:`voxa.align.ExactDelayLock`): the lock
        itself - proven or falling back to the estimate - re-points the canceller, exactly as
        ``AudioCapture._maybe_estimate_delay`` does, so the ``voxa`` row of the frontier is
        production's number and not a private arm. The ``*-locked`` comparison arms (f164) keep
        the bare :class:`ExactAligner` handed to them per frame."""
        if self._aligner is None or self.ref_ring is None or ref.size == 0:
            return
        if self._delay_lock is not None:
            applied = self._delay_lock.observe(self._now, mic01, ref)
            locked = self._delay_lock.delay_frac
            if locked is None:
                return
            # f172: "the delay the canceller/mask were handed" is only a MEASURED delay while
            # the lock itself re-pointed them. While the shipping DelayTracker estimate holds
            # the reference (visible degraded), the joint gate does not read coherence at all:
            # the estimate is exactly the 1.5-2.5 ms drift that made f132's coherence half
            # meaningless. A "hold" (no new update) keeps the last proven state: the lock's
            # drift line and the canceller's anchors have not moved.
            if applied == "lock":
                self._lock_proven = True
            elif applied == "estimate":
                self._lock_proven = False
            self._delay_frac = float(locked)
            coarse = int(round(locked))
            if self._mask_sees_lock and self.mask is not None and applied == "lock":
                # f171: the mode runs the SHIPPING chain, so the mask is handed the delay the
                # lock just drove the canceller at - and never a delay the canceller is not
                # using: while the estimate holds the reference (visible degraded) the mask
                # keeps the last delay it was measured, exactly as voxa/audio.py does.
                self.mask.set_delay(float(locked))
            if self._delay_samples != coarse:
                self._delay_samples = coarse
                self.emit("delay-estimate", delay_samples=coarse,
                          delay_ms=round(coarse / SR * 1000.0, 1),
                          kind="epoch" if self._aligner.epochs else "lock",
                          drift_ppm=round(self._aligner.drift_ppm, 1),
                          offset_samples=round(locked, 2),
                          degraded=bool(self._delay_lock.degraded),
                          confidence=round(self._aligner.confidence, 3))
            return
        self._aligner.observe(self._now, mic01, ref)
        locked = self._aligner.locked_delay
        if locked is None:
            return
        self._lock_proven = bool(self._aligner.refined)
        self._delay_frac = float(locked)
        coarse = int(round(locked))
        self._apply_locked_delay(coarse, locked)
        if self._mask_sees_lock and self.mask is not None:
            # f169: the residual mask reads its reference at the SAME locked delay the
            # canceller is driven at - until the lock exists the mask estimates its own lag
            # (the pre-f169 behaviour), never a half-known one.
            self.mask.set_delay(locked)
        if self._delay_samples != coarse:
            self._delay_samples = coarse
            self.emit("delay-estimate", delay_samples=coarse,
                      delay_ms=round(coarse / SR * 1000.0, 1),
                      kind="epoch" if self._aligner.epochs else "lock",
                      drift_ppm=round(self._aligner.drift_ppm, 1),
                       offset_samples=round(locked, 2),
                       confidence=round(self._aligner.confidence, 3))

    def _apply_locked_delay(self, coarse: int, frac: float) -> None:
        """Hearing 2 (f164): hand the arm's own delay search the bulk delay Voxa already knows, so
        it only refines the room tail. Each arm takes it the way ITS library reaches:

        * ``voxa-locked`` (``EchoCanceller``) reads its reference at the locked FRACTIONAL offset
          (``set_offset`` is the drift resampler; the filter never rebuilds);
        * ``speex-locked`` / ``aec3-locked`` have no delay setter this build exposes, so the locked
          bulk delay is applied by re-timing the reference the canceller is fed - a ring offset in
          :func:`voxa.audio.mix_chunk`, where the reference history lives.
        """
        if self._ref_offset_lock:
            self._locked_ref_offset = max(0, coarse)
            return
        c = self.canceller
        if c is not None and hasattr(c, "set_offset"):
            c.set_offset(coarse, 0.0, frac, 0.0, frac)

    def _clean_frame(self, mic_bytes: bytes, ref: np.ndarray) -> tuple[bytes, float, float, float, float]:
        """Cleaned chunk for this frame, the dB the canceller removed, the mask's gain in dB, and
        the microphone and cleaned levels in dB (B3's near-end attenuation reads the last two).

        ``ref`` is the reference for this frame in the scale the live capture path uses (-1..1,
        see audio.py:411); the microphone travels as the int16 bytes the pipeline moves.
        """
        gain_db = 0.0
        cpu_at = time.process_time()
        mic = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float32)
        mic_db = level_db(mic)
        if self.canceller is not None and self.ref_ring is not None:
            if self._aligner is not None:
                # Hearing 2: continuous exact lock of delay + clock drift, on the -1..1 scale the
                # live capture path uses.
                self._observe_echo_delay(mic / 32768.0, ref)
            else:
                self._estimate_delay(mic, ref)
            mic_bytes = mix_chunk(mic_bytes, self.ref_ring, self._mic_counter, self.canceller,
                                  self._locked_ref_offset)
        self._mic_counter += len(mic_bytes) // 2
        cleaned = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float32)
        erle = erle_removed_db(mic, cleaned) if self.canceller is not None else 0.0
        if self.mask is not None and self.canceller is not None:
            # f131 (roadmap task 4): the residual suppressor runs ON THE CANCELLER'S OUTPUT, with
            # the aligned reference, at the scale both the mask and its dB semantics mean
            # (-1..1 on both sides - the instrument-defect branch below is what this replaces).
            mic_f = mic.astype(np.float64) / 32768.0
            cleaned_f = cleaned.astype(np.float64) / 32768.0
            # The running ERLE of the LINEAR canceller: when it removed little (cheap speakers,
            # high volume, a tail still ringing), the mask subtracts deeper.
            self.mask.set_running_erle(erle)
            out = np.clip(self.mask.process(cleaned_f, ref.astype(np.float64)), -1.0, 1.0)
            if out.size:
                # The canceller chain (canceller + residual mask) is what the gate measures; the
                # mask's suppression is part of removing the echo, so erle is measured against
                # THIS output, not the canceller alone. (research/voxa-hearing: the residual mask
                # is layer 2 of AEC3; f129 recorded the same rule for the aec3 arm.)
                final = (out * 32768.0).astype(np.float32)
                erle = erle_removed_db(mic, final)
                gain = (float(np.mean(np.abs(out)))
                        / max(float(np.mean(np.abs(mic_f))), 1e-9))
                gain_db = 20.0 * float(np.log10(max(gain, 1e-9)))
                mic_bytes = pcm16(out)
        elif self.mask is not None:
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
        self._echo_cpu += time.process_time() - cpu_at
        self._clean_stream.append(mic_bytes)
        self._note_joint_evidence(mic_bytes, ref, erle, raw_mic=mic.astype(np.float64) / 32768.0)
        return mic_bytes, erle, gain_db, mic_db, level_db(np.frombuffer(mic_bytes, dtype="<i2"))

    def _note_joint_evidence(self, cleaned_bytes: bytes, ref: np.ndarray, erle_db: float,
                             *, raw_mic: np.ndarray) -> None:
        """Roadmap task 5: feed the joint double-talk gate one frame of the CLEANED stream.

        Silero runs on the cleaned bytes every frame - muted or not - so the probability the
        evidence carries during playback is a measurement, not a stale 0.0. The gate holds all
        three audio streams before it calls the frame suspect; while it does, the canceller's OWN
        double-talk control is extended through the hold (voxa/anc.py ``note_double_talk``), never
        a blanket freeze. The coherence half of the gate reads the RAW microphone (``raw_mic``):
        the echo test asks whether the microphone is still the delayed reference, and the cleaned
        stream has already lost the echo it would have to match.
        """
        if self._clean_vad is None:
            self._clean_vad = VoiceDetector(threshold=SPEECH_PLAYBACK_THRESHOLD)
        self._clean_vad.feed(cleaned_bytes)
        prob = float(self._clean_vad.last_prob) if self._clean_vad.probs else None
        ref_power = float(np.mean(np.asarray(ref, dtype=np.float64) ** 2))
        margin = (float(self.mask.residual_margin_db)
                  if self.mask is not None and np.isfinite(self.mask.residual_margin_db)
                  else None)
        self._mask_margin = margin
        # f172: the gate's margin half is the mask's PER-BAND output-vs-prediction ratio (the
        # median over the bands it predicts), not the frame sums: the sums saturate on the
        # near-silent predictions of unlearned bands, which is why f132's her-echo frames
        # reported "+6 dB above the predicted residual" constantly.
        gate_margin = (float(self.mask.band_margin_db)
                       if self.mask is not None and np.isfinite(self.mask.band_margin_db)
                       else None)
        # f172 (re-plan of parked f132): the coherence half of the gate reads at the LOCKED
        # delay - the delay_samples the canceller and the residual mask are actually driven at -
        # never the canceller's own drifting DelayTracker estimate. f132's blocker was that this
        # gate read a delay that drifted 1.5-2.5 ms, so coherence was low even on pure-echo
        # frames and the joint verdict could not tell "her echo" from "a second source". f168
        # shipped the exact render-timeline lock (delay error 0.0727 ms) and f171 ships
        # LockedResidualMask on that output, so the lock exists: use it, and treat "no lock yet"
        # as alignment unhealthy (the joint gate stays honest, the caller degrades).
        delay = self._delay_samples if self._delay_samples > 0 else (
            self.canceller.delay_samples if self.canceller is not None else 0)
        alignment_healthy = bool(self._lock_proven
                                 or getattr(self.canceller, "delay_managed_internally", False))
        self._clean_prob = prob
        self._dt_suspect = self.dt_gate.observe(
            dt_ms=CHUNK_MS, prob=prob, mic=raw_mic,
            ref=np.asarray(ref, dtype=np.float64),
            delay_samples=delay if delay > 0 else None,
            residual_over_prediction_db=gate_margin, erle_db=erle_db,
            # An echo still in flight counts as a live reference (the same 400 ms rule the
            # evidence uses): the reference ring keeps holding the delay's audio after the
            # speakers go quiet between words.
            reference_active=bool(max(list(self._ref_recent) + [ref_power]) > MIN_REFERENCE_POWER),
            alignment_healthy=alignment_healthy)
        self._dt_measurable = bool(self.dt_gate.conditions.get("measurable"))
        self._dt_echo = self.dt_gate.echo_consistent
        if self._dt_suspect and self.canceller is not None:
            self.canceller.note_double_talk(2.0 * CHUNK_MS)

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
        # Task 5: the runner's own Silero on the cleaned stream outranks the conversation's,
        # which never runs on a muted microphone.
        prob = (self._clean_prob if self._clean_prob is not None
                else (float(detector.last_prob) if measured else None))
        evidence = SourceEvidence(
            # An echo still travelling counts as a live reference: the speakers went quiet 100-300
            # ms ago and the microphone is still hearing it.
            reference_active=bool(max(self._ref_recent) > MIN_REFERENCE_POWER),
            reference_power=float(ref_power),
            echo_explained_db=explained,
            residual_db=residual,
            residual_margin_db=margin,
            # Task 5: the mask's own margin (output over ITS predicted residual) and the joint
            # gate's voice-only verdict, measured this frame.
            residual_over_prediction_db=(
                float(self.mask.residual_margin_db)
                if self.mask is not None and np.isfinite(self.mask.residual_margin_db)
                else None),
            # None when THIS gate could not measure (the caller falls back to the level gate
            # and says so - the degraded path, not a guess of "no second source").
            joint_suspect=(self._dt_suspect if (self._dt_measurable or self._dt_suspect)
                           else None),
            joint_echo=(self._dt_echo if (self._dt_measurable or self._dt_echo) else None),
            vad_prob=prob,
            alignment_healthy=bool(self._delay_samples > 0
                                   or getattr(self.canceller, "delay_managed_internally", False)
                                   or max(self._ref_recent) <= MIN_REFERENCE_POWER),
            mic_level=float(level01),
            own_level=float(self.barge.gate.baseline),
            # level_ratio is left to bargectl: it is the one that knows the level of the sound
            # that OPENED the gate, not of the frame that happens to do the judging.
        )
        # Roadmap task 7 (f175): mirror the LIVE path (voxa/window.py::_hearing_evidence) exactly:
        # the word-timed overlap is ONE capped weight on the SAME evidence the joint verdicts
        # ride, computed from the last candidate the pipeline transcribed (its words carry the
        # capture ring's own arrival anchor). The degraded path is unchanged: no candidate words
        # or no played timeline -> no share, exactly as live.
        if self._playback_words and not self._timeline_fed and self._delay_frac:
            # Feed the timeline as soon as the LOCKED delay exists - before any candidate needs
            # it - so the evidence weight sees spans in microphone time from the first frame.
            self._feed_playback_timeline()
        if self.conversation is not None and self.own_speech.played_words():
            words = getattr(self.conversation, "last_candidate_words", None)
            base = getattr(self.conversation, "last_candidate_span_start", None)
            if words and base is not None:
                timed = self.own_speech.timed_overlap(
                    [(w["word"], base + float(w["onset_s"]),
                      base + float(w["offset_s"])) for w in words],
                    window_ms=self._word_window_ms,
                )
                evidence.word_timed_overlap = timed["share"]
        self.last_evidence = evidence
        return evidence

    def _candidate(self) -> tuple[str, str]:
        """The words the muted microphone captured, as ``window._barge_candidate`` reads them.

        ``candidate_text`` (conversation.py:585) transcribes the rolling window and consumes it.
        """
        heard = self.conversation.candidate_text()
        # Roadmap task 7 (f175): the fresh candidate also refreshes the word-timed weight on
        # the evidence bargectl is about to judge (the same object this frame's _evidence
        # returned): a word that lands while her speakers were audible reading THAT word is
        # echo EVIDENCE - one capped weight, never a decision, and joint None keeps the
        # degraded path exactly as it was.
        if self.last_evidence is not None and self.own_speech.played_words():
            base = self.conversation.last_candidate_span_start
            words = self.conversation.last_candidate_words
            if words and base is not None:
                timed = self.own_speech.timed_overlap(
                    [(w["word"], base + float(w["onset_s"]),
                      base + float(w["offset_s"])) for w in words],
                    window_ms=self._word_window_ms,
                )
                self.last_evidence.word_timed_overlap = timed["share"]
        return heard, self.conversation.last_candidate_raw

    def _feed_playback_timeline(self) -> None:
        """Feed the word-timed playback timeline once per clip (roadmap task 7, f175).

        Fed at the first moment BOTH sides are ready, not at speak-start: the truth's playback
        words are edge-tts boundaries on the REFERENCE stream, and only the locked delay f168
        shipped can shift them into the microphone's time - the time the recogniser's word times
        live in. By the time an utterance finishes the lock has proven itself on every shipping
        clip; while it has not, 0.0 is the shift the pipeline itself is reading (degraded).
        """
        if self._timeline_fed or not self._playback_words:
            return
        shift = (float(self._delay_frac) / SR) if self._delay_frac else 0.0
        self.own_speech.note_played_boundaries(
            [(w["word"], float(w["onset_s"]) + shift,
              max(0.0, float(w["offset_s"]) - float(w["onset_s"])))
             for w in self._playback_words],
            generation=self._spoken_text,
        )
        self._timeline_fed = True

    def _joint_closed_at(self, onset: float) -> bool:
        """Was the joint gate CLOSED at the moment this word was heard (echo was the only
        explanation)? The clip's per-frame gate history answers it; nearest frame wins; an
        unmeasurable gate is never closed (the drop rule needs a measured mirror, f173)."""
        best_gap: float | None = None
        closed = False
        for t, verdict in self._joint_history:
            gap = abs(t - float(onset))
            if best_gap is None or gap < best_gap:
                best_gap, closed = gap, verdict
        return bool(closed)

    # --- one finished utterance -------------------------------------------------

    def _finish_utterance(self, segment: bytes, frame: int, *, promote: bool = True) -> None:
        """The decision ``ConversationController._run`` makes once a segment is complete.

        A turn that began with a promoted span keeps its stable prefix: the prefix's words were
        recognised once, from exactly the promoted audio, and only the audio that arrived after
        the span is transcribed here (the same rule the live controller applies).

        ``promote=False`` is the f176 candidate-window completion: a pinned span that the barge-in
        decision did NOT accept (HOLD, an echo-measured onset that blocked the yield, a candidate
        whose 900 ms window never completed) is still carried to the speech-pause rules and read by
        the recogniser, so the pinned span reaches a segment ``transcript`` - the audio is owed to
        the record (handoff 6.4: an accepted overlap becomes the request exactly once; never a
        silent drop). But no command is run from audio this weak: the method returns after the
        transcript with an ``endpoint`` marked ``dropped="hold"``, never a request/exit/ignored.
        """
        # Roadmap task 9 re-plan (f181): the promote_wait on the assembly path. When a
        # never-completed pinned span is still owed AND a wait signal is open (the joint
        # double-talk suspect f173, the turn state f174, gated speech still arriving), the
        # turn must not be CLOSED here: an open-floor segment's pause floor ran its silence
        # clock while the mic was muted and the user's continuing speech sat unpromoted in
        # the ring - handing that assembly over mid-speech closes a turn while speech is
        # arriving (f178's two premature cuts). The audio, the transcript, the prefix
        # bookkeeping and the controller state run exactly as before; only the events that
        # close the turn wait, to the suspect's close or the hard cap (the clip end, where
        # the f176 completion promotes the span at last). Never a silent drop, never a
        # re-promotion: the probe (f178's arbitrator) is untouched.
        self._promote_defer = bool(promote
                                   and self.conversation.capture.promote_waiting()
                                   and self._promote_wait_active())
        self.emit("pause", segment_seconds=round(len(segment) / 32000.0, 3))
        waiting = self.conversation.waiting_for_prompt or self.conversation.keep_prompt
        hint = (self._vocabulary_hint or f"{self.wake_word.strip().capitalize()}.") if not waiting else ""
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
        # Roadmap task 7 (f175): the word-timed overlap. The recogniser's word times are
        # segment-relative, so shift them onto the clip's clock; the played timeline is in the
        # microphone's time because the locked delay f168 shipped shifts it there. The share is
        # ONE capped evidence weight (voxa/attribution.py); the only words this path may delete
        # are ones that time-match one of her played words at a moment the joint gate CLOSED on
        # echo - the drop rule f175 ships, handoff 5.2/5.3 respected.
        self._feed_playback_timeline()
        timed_words: list[tuple[str, float, float]] = []
        if heard.get("words"):
            seg_start = self._now - len(segment) / SR
            timed_words = [(w["word"], seg_start + float(w["onset_s"]),
                            seg_start + float(w["offset_s"]))
                           for w in heard["words"]]
        if timed_words and self.own_speech.played_words():
            timed = self.own_speech.timed_overlap(timed_words, window_ms=self._word_window_ms)
            if self.last_evidence is not None:
                self.last_evidence.word_timed_overlap = timed["share"]
        if not promote:
            # f176: the span was transcribed (its words reach the segment transcript and the
            # record) but the barge-in decision did not accept it - HOLD, never a silent drop and
            # never a command from weak audio (handoff 6.5: no automatic commands from unqualified
            # audio during playback). The own-speech filter never runs here: it only ever DELETES,
            # and deleting would undo the completion this branch exists to make.
            self.emit("endpoint", text=heard["text"], complete=False, dropped="hold")
            return
        # The own-speech filter is gated on the AUDIO evidence for this span, and a turn that
        # began with a promoted span is confirmed user: its words are never rewritten by a text
        # match (voxa/attribution.py, handoff 5.3).
        text = self.conversation._filter_own_speech(
            heard["text"], evidence=self.last_evidence, confirmed_user=bool(prefix_text),
            heard_with_times=timed_words or None,
            joint_closed_at=self._joint_closed_at if timed_words else None)
        if text is None:
            self._turn_event("endpoint", text="", complete=False, dropped="own-speech")
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
                self._turn_event("ignored", text=text, reason="unqualified")
                return
        keep_prompt = self.conversation.keep_prompt
        exit_kind = detect_exit_phrase(text, self.wake_word) if not keep_prompt else None
        followup_open = bool(self._followup_deadline is not None
                             and self._now < self._followup_deadline)
        self._turn_event(
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
            self._turn_event("exit", kind=exit_kind, text=text)
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
                    self._turn_event("ignored", text=text)
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

    def _unqualified_audio(self) -> bool:
        """True when the pipeline cannot explain the microphone: no delay, or an echo model that
        has not converged on this path yet (roadmap task 10's recovery rule)."""
        aligned = (self._delay_samples > 0
                   or getattr(self.canceller, "delay_managed_internally", False))
        return not aligned

    def _deliver(self, text: str, frame: int, *, kind: str, words: list[dict] | None = None,
                 no_speech_prob: float | None = None) -> None:
        """The request that would reach the assistant (what ``on_prompt`` carries in the app)."""
        if not text:
            return
        # Roadmap task 10: inside an epoch's recovery window no automatic command may come from
        # audio the canceller cannot qualify yet. The audio itself is KEPT (the capture ring and
        # the cleaned stream hold it); only the automatic decision is held off, and it says so.
        if kind == "request" and self._recovering() and self._unqualified_audio():
            self.held_commands += 1
            self._turn_event("command-held", text=text, frame=frame, epoch=self.epoch,
                      reason="recovery-window-unqualified-audio")
            return
        self.requests.append(text)
        self.counts["turns"] += 1
        self.turns.append({"text": text, "kind": kind, "frame": frame})
        self._turn_event("request", kind=kind, text=text, words=words or [],
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

    def _finish_unpromoted_pinned_span(self) -> None:
        """f176: carry a pinned span that was never promoted to a segment transcript.

        The candidate window (``BARGE_IN_LISTEN_MS`` 900 ms) can end with the span still pinned
        and never judged - an echo-measured onset that blocked the yield, a HOLD, or a candidate
        that simply never collected 900 ms before the clip ended. Those spans were captured while
        the microphone was MUTED, so the frame loop never fed them to the speech-pause rules
        (``runner.py:1097``): they reach the whole-stream ``clip-transcript`` but never a segment
        ``transcript``, and the protected words they carry are lost before the segment transcript
        (the 8/22 f175 named). Finish them the way handoff 6.4 says: the pinned span is transcribed
        exactly once (its words reach the record and the metric), stays HOLD - never a command from
        weak audio and never a silent drop.
        """
        capture = self.conversation.capture
        if capture.pin_from is None:
            return  # nothing was ever pinned: no span owed to the record
        if any(event.get("event") == "transcript" for event in self.events):
            return  # the normal path already transcribed this clip's user audio: completing it
            # again would transcribe the accepted words twice (and skew the WER/endpoint metrics).
        # Only a span the AUDIO attributes to a SECOND source is owed a HOLD transcript: a span
        # the joint gate measured as Voxa's own echo is her voice, and transcribing it would put
        # her words into the record (echo leak) - exactly what the design rules forbid.
        from voxa.attribution import ECHO as SOURCE_ECHO, attribute
        if self.last_evidence is not None and attribute(self.last_evidence).source == SOURCE_ECHO:
            return
        # Hand the never-promoted pinned span to the record exactly once via the ONE promotion
        # primitive (voxa/capture.py::CaptureRing.promote_pinned, handoff 6.4): its ids are marked
        # consumed so the same words are never transcribed or commanded twice, then transcribe what
        # is actually still owed to the speech-pause rules (frames the muted loop never segmented).
        # f181: reaching this IS the hard cap - the wait passes force=True, so it never turns the
        # HOLD completion into a silent drop.
        span = [frame for frame in capture.promote_pinned(force=True) if frame.seq not in self._segmented]
        if not span:
            return
        for frame in span:
            self._segmented.add(frame.seq)
            self.segmenter.step(frame.pcm, frame.level, frame.t_s)
        segment = self.segmenter.pending()
        if segment:
            self._finish_utterance(segment, self._frame, promote=False)

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
        # runner drives the same frames through its own SpeechSegmenter instead. f181: this
        # is the ACCEPTED interruption - the decision is made, so the promote_wait does not
        # hold it (the f176 contract: the accepted span becomes the turn's beginning now).
        # The wait applies to the never-completed span, not to an accepted decision.
        promoted = self.conversation.promote_pinned(enqueue=False, force=True)
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
        # Roadmap task 10: what the clip plays through, and any mid-clip device switch. With the
        # frozen clips this is the default output and no switch: the pipeline is unchanged.
        self._note_output(truth.get("echo") or {})
        self._apply_output_tuning()
        # Task 5: a new clip starts the joint gate, its Silero run and the suspect flag over.
        self.dt_gate.reset()
        if self._clean_vad is not None:
            self._clean_vad.reset()
        self._clean_prob = None
        self._dt_suspect = False
        self._dt_measurable = False
        self._dt_echo = False
        self._promote_defer = False
        self._promote_held_events = []
        # Roadmap task 7 (f175): a new clip starts the word-timed playback timeline over - it is
        # evidence about THIS reply's words, and a stale timeline would time-match one clip's
        # heard word to another clip's played word.
        self.own_speech = OwnSpeech(clock=self._clock)
        self._playback_words = list((truth.get("playback") or {}).get("words") or [])
        self._joint_history = []
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
            endpoint_probe=self._endpoint_probe,
            endpoint_policy=self._endpoint_policy,
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
            # 0b. A mid-clip device/headphone/Bluetooth switch is an explicit epoch (task 10).
            wanted = self._switch_due(frame)
            if wanted is not None:
                self._begin_epoch(wanted)
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
            # The live app still STEPS the segmenter while muted (the queue goes empty and
            # segment_stream steps with nothing), so silence keeps counting and the roadmap
            # task 9 endpoint probe keeps running across her playback; without the same step
            # here the user's turn flushed at clip end instead of at its endpoint floor.
            completed = (self._segment_frame(captured)
                         if captured is not None and not self.conversation.muted else None)
            detector = self.conversation._voice_detector
            prob = float(detector.last_prob) if detector is not None else 0.0
            # Roadmap task 7 (f175): the word-timed drop rule asks whether the joint gate was
            # CLOSED (echo was the only explanation) at the moment a heard word landed - so the
            # gate's per-frame verdict is kept for the clip, not just emitted.
            self._joint_history.append((self._now, bool(self.dt_gate.echo_consistent)))
            self.emit(
                "frame",
                vad=bool(prob >= detector.threshold) if detector is not None else None,
                vad_prob=round(prob, 3),
                clean_vad_prob=(None if self._clean_prob is None else round(self._clean_prob, 3)),
                suspect=bool(self._dt_suspect),
                joint_conditions=self.dt_gate.conditions or None,
                joint_echo=bool(self.dt_gate.echo_consistent),
                joint_coherence=(None if self.dt_gate.coherence is None
                                 else round(float(self.dt_gate.coherence), 3)),
                joint_run_ms=round(self.dt_gate.joint_run_ms, 1),
                mask_margin_db=(None if self._mask_margin is None
                                else round(self._mask_margin, 2)),
                band_margin_db=(None if self.dt_gate is None or self.mask is None or
                                not np.isfinite(self.mask.band_margin_db)
                                else round(float(self.mask.band_margin_db), 2)),
                voiced=bool(completed is not None),
                level=round(level, 1),
                erle_db=round(erle, 2),
                mask_gain_db=round(gain_db, 2),
                mic_level_db=mic_db,
                clean_level_db=clean_db,
                delay_samples=self._delay_samples,
                delay_frac=round(self._delay_frac, 4),
                delay_confident=bool(self._aligner is not None and self._aligner.refined),
                output_class=self.output_class,
                epoch=self.epoch,
                recovering=self._recovering(),
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
                # judged, exactly as voxa/window.py does on the same observation. f177: the
                # barge-in turn starts at the suspect onset (the run of the user's OWN sound the
                # onset rule confirmed), NOT the late confirming frame, so the first word of an
                # interruption that began with her playback survives - over the CLEANED stream,
                # never the raw ring (f133 measured raw-ring cold-start recovery re-introduce echo).
                self.conversation.note_onset(seq=frame,
                                             level_floor=self.barge.onset_level_floor(),
                                             onset_audio_ms=outcome.onset_audio_ms)
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
            # f181: the suspect's close. With the wait signals quiet again (or the span no
            # longer owed), the held turn closures run; while the wait holds, they wait.
            if self._promote_held_events and not (
                    self.conversation.capture.promote_waiting()
                    and self._promote_wait_active()):
                self._flush_promote_held()
        if self.speaking:
            self._speak_finished()
        # Give the echo-path claim back before the next clip builds its own canceller: exactly one
        # AEC owner at a time, and the bench runs one ClipRunner per clip.
        release = getattr(self.canceller, "close", None)
        if release is not None:
            release()
        pending = self.segmenter.pending()
        if pending:
            self._finish_utterance(pending, self._frame)
        # f176: the pinned span that the candidate window never promoted (never completed its
        # 900 ms, an echo-measured onset that blocked the yield, or a HOLD) is still owed a segment
        # transcript - transcribe it once, HOLD, before the whole-stream pass.
        self._finish_unpromoted_pinned_span()
        # f181 hard cap: the clip ended with the wait still open (the suspect never closed
        # on the simulated clock) - the held turn closures are still owed to the record, so
        # they run here, once, and the f176 completion above has already handed the pinned
        # span over at last. The close is never silently dropped.
        self._flush_promote_held()
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
            "echo_processing_seconds": round(self._echo_cpu, 4),
            "degraded": self.degraded,
            "delay_estimated_samples": self._delay_samples,
            "output_class": self.output_class,
            "device_epochs": self.epoch,
            "device_switch": bool(self._device_switch),
            "held_commands": self.held_commands,
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

    def _endpoint_probe(self, segment: bytes) -> bool | None:
        """Roadmap task 9: Smart Turn v3.1 over the WHOLE turn heard so far, then
        the f178 endpoint ARBITRATOR.

        The model's own threshold (0.5) decides; a transcript ending on a
        conjunction, filler or preposition (report 186) vetoes even a "complete"
        verdict, because the first valid command match is NOT completeness. On
        top of that, the model's "complete" verdict is gated on the shipped joint
        double-talk suspect (f173) and the still-arriving-speech / turn state: a
        short pause is NOT cut when a joint suspect is open or more gated speech
        is arriving (the arbitrator WAITS it out to the 2.0 s incomplete floor,
        capped). The model probability is never retuned (f136 measured that wall).
        No weights, no verdict: the pause rule the app already had.
        """
        if self._smart_turn is None:
            return None
        prob = self._smart_turn.probability(segment)
        complete = prob is not None and prob > 0.5
        if not complete:
            self.emit("endpoint-probe", complete=False, prob=None if prob is None else round(prob, 3))
            return False if prob is not None else None
        heard = self.asr.transcribe_words(segment, f"{self.wake_word.strip().capitalize()}.")
        dangling = ends_dangling(heard["text"])
        model_verdict = not dangling
        # f178: the shipped joint double-talk verdict the frame already measured.
        joint_suspect = (self._dt_suspect if (self._dt_measurable or self._dt_suspect)
                         else None)
        more_speech_arriving = self._more_speech_arriving()
        verdict = arbitrate_endpoint(model_verdict, joint_suspect=joint_suspect,
                                     more_speech_arriving=more_speech_arriving)
        self.emit("endpoint-probe", complete=verdict, prob=round(prob, 3),
                  text=heard["text"], dangling=dangling,
                  model_complete=model_verdict, joint_suspect=joint_suspect,
                  more_speech_arriving=more_speech_arriving)
        return verdict

    def _more_speech_arriving(self) -> bool:
        """The f178 "gated speech is still arriving" signal for the arbitrator: the
        cleaned-signal Silero is still on (hysteresis open), so the VAD pause is a
        breath inside a run of speech, not the end of the turn."""
        detector = self.conversation._voice_detector
        if detector is None:
            return False
        prob = getattr(detector, "last_prob", None)
        return prob is not None and float(prob) >= detector.threshold

    # --- the f181 promote_wait ---------------------------------------------------

    def _promote_wait_active(self) -> bool:
        """The shipped verdicts that make :meth:`voxa.capture.CaptureRing.promote_wait` hold
        a never-completed pinned span off the user turn: the joint double-talk suspect
        (f173; a degraded/INDETERMINATE None adds NO veto, f136's rule), the turn state
        (f174 - this runner never ducks, so False) and the still-arriving gated speech
        signal the f178 arbitrator already reads. No new inference: the same three numbers.
        """
        joint_suspect = (self._dt_suspect if (self._dt_measurable or self._dt_suspect)
                         else None)
        return self.conversation.capture.promote_wait(
            joint_suspect=joint_suspect, turn_suspect=False,
            more_speech_arriving=self._more_speech_arriving())

    def _turn_event(self, name: str, **fields) -> None:
        """One event of a turn closure (endpoint/request/exit/ignored): held while the
        promote_wait defers this completion, emitted unchanged at the close/cap."""
        if self._promote_defer:
            self._promote_held_events.append((name, fields))
        else:
            self.emit(name, **fields)

    def _flush_promote_held(self) -> None:
        """The suspect's close, or the hard cap: the held turn closures run in order.
        Never silently dropped - this is the last line before the clip's record is final.
        """
        held, self._promote_held_events = self._promote_held_events, []
        for name, fields in held:
            self.emit(name, **fields)


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


def arg_parser() -> argparse.ArgumentParser:
    """The bench ``run`` command line, built separately so the tests can inspect its choices."""
    parser = argparse.ArgumentParser(
        prog="python -m voxa.bench run",
        description="push bench clips through Voxa's real hearing and barge-in pipeline")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=0, help="the render's seed; clips are read, not built")
    parser.add_argument("--echo-mode",
                        choices=("voxa", "webrtc", "aec3", "off") + modes.HEARING2_MODES,
                        default="voxa")
    parser.add_argument("--out", type=Path, default=None,
                        help="report path (default .voxa-bench/report-<suite>.json)")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("VOXA_BENCH_DATA", str(DEFAULT_DATA_ROOT))))
    parser.add_argument("--model", default="", help="faster-whisper model (default: Voxa's, config.py:22)")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N clips (debugging)")
    parser.add_argument("--no-gates", action="store_true", help="skip the gates.toml checks (roadmap B5)")
    # f131 (roadmap task 4): the residual mask's axes, for the sweep. Unset means the task's
    # defaults (MASK_* above).
    parser.add_argument("--mask-over", type=float, default=None,
                        help="residual mask: base over-subtraction (default %(default)s means MASK_OVER)")
    parser.add_argument("--mask-protect-db", type=float, default=None,
                        help="residual mask: protection margin in dB (default: MASK_PROTECT_DB)")
    parser.add_argument("--mask-ramp", type=float, default=None,
                        help="residual mask: over-subtraction ramp (default: MASK_OVER_RAMP)")
    parser.add_argument("--mask-erle-target-db", type=float, default=None,
                        help="residual mask: the ERLE the linear stage is assumed to give (default: MASK_ERLE_TARGET_DB)")
    parser.add_argument("--mask-tail-s", type=float, default=None,
                        help="residual mask: tail window ceiling in s (default: MASK_TAIL_S)")
    # f172: the joint double-talk gate's thresholds, for the sweep. Unset means the task's
    # defaults (SPEECH_PLAYBACK_THRESHOLD / RESIDUAL_MARGIN_DB / COHERENCE_ECHO / ERLE_DROP_DB).
    parser.add_argument("--gate-speech-threshold", type=float, default=None,
                        help="joint gate: cleaned-signal Silero threshold while TTS plays")
    parser.add_argument("--gate-margin-db", type=float, default=None,
                        help="joint gate: per-band output over the predicted residual (dB)")
    parser.add_argument("--gate-coherence", type=float, default=None,
                        help="joint gate: coherence at/above this is echo-consistency")
    parser.add_argument("--gate-erle-drop-db", type=float, default=None,
                        help="joint gate: running ERLE drop that counts as double talk")
    # f175 (roadmap task 7): the +- window (ms) of the word-timed own-speech evidence; swept
    # 200/400/600 in the notes. Unset means the shipped default (ownspeech.WORD_WINDOW_MS).
    parser.add_argument("--word-window-ms", type=float, default=400.0,
                        help="word-timed evidence: +- window in ms for a heard word to match a "
                             "played word (default %(default)s)")
    parser.add_argument("--quiet", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = arg_parser().parse_args(argv)
    if args.echo_mode == "aec3" and not aec3_available():
        print("echo mode aec3 needs pywebrtc-audio in this interpreter; "
              "pip install pywebrtc-audio or use --echo-mode voxa.", file=sys.stderr)
        return 2
    probe = modes.availability()
    if args.echo_mode in modes.HEARING2_MODES and not probe[args.echo_mode]["available"]:
        # An arm that cannot run here is never quietly replaced by another one: the numbers a
        # run like that produces are not that arm's numbers. Hearing 2 records the reason.
        print(f"echo mode {args.echo_mode} cannot run here: {probe[args.echo_mode]['reason']}",
              file=sys.stderr)
        return 2
    mask_params: dict = {}
    if args.mask_over is not None:
        mask_params["over"] = args.mask_over
    if args.mask_protect_db is not None:
        mask_params["protect_db"] = args.mask_protect_db
    if args.mask_ramp is not None:
        mask_params["over_ramp"] = args.mask_ramp
    if args.mask_erle_target_db is not None:
        mask_params["erle_target_db"] = args.mask_erle_target_db
    if args.mask_tail_s is not None:
        mask_params["tail_s"] = args.mask_tail_s
    dt_params: dict = {}
    if args.gate_speech_threshold is not None:
        dt_params["speech_threshold"] = args.gate_speech_threshold
    if args.gate_margin_db is not None:
        dt_params["margin_db"] = args.gate_margin_db
    if args.gate_coherence is not None:
        dt_params["coherence_echo"] = args.gate_coherence
    if args.gate_erle_drop_db is not None:
        dt_params["erle_drop_db"] = args.gate_erle_drop_db
    report = run_clips(args.suite, args.seed, args.echo_mode, args.data_root, args.limit,
                       args.model, args.quiet, mask_params, dt_params, args.word_window_ms)
    # Hearing 2 (h01): what each candidate arm really is on this machine, next to the numbers.
    report["run"]["echo_mode_probe"] = dict(probe)
    report["run"]["hybrid_linear"] = ("aec3" if aec3_available() else "voxa")
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
              model: str, quiet: bool = False, mask_params: dict | None = None,
              dt_params: dict | None = None,
              word_window_ms: float | None = None) -> dict:
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
        runner = ClipRunner(asr=asr, echo_mode=echo_mode, wake_word=wake_word,
                            mask_params=mask_params, dt_params=dt_params,
                            word_window_ms=(400.0 if word_window_ms is None else word_window_ms),
                            data_root=str(data_root))
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
    report = build_report(suite, seed, echo_mode, results, facts, audio_seconds,
                          time.perf_counter() - started, model)
    if mask_params:
        # f131: the sweep must be able to cite which knobs each row used.
        report["run"]["mask_params"] = dict(mask_params)
    if dt_params:
        # f172: the joint gate sweep cites its thresholds too.
        report["run"]["gate_params"] = dict(dt_params)
    # Roadmap task 2's CPU gate is about the echo stages, not the recogniser: cpu_core_share divides
    # ASR seconds by wall seconds and can never show what a canceller costs. This is the number the
    # "<= 10 % of one core" gate is actually about, measured on the same clips.
    echo_cpu = sum(float(item.get("echo_processing_seconds") or 0.0) for item in results)
    wall = float(report["run"].get("wall_seconds") or 0.0)
    report["run"]["echo_cpu_seconds"] = round(echo_cpu, 3)
    report["run"]["echo_cpu_core_share"] = round(echo_cpu / wall, 5) if wall > 0 else None
    return report


if __name__ == "__main__":
    raise SystemExit(run())
