"""Decide whether a spoken utterance is finished, so Voxa can stop waiting.

Used at the early pause: Voxa takes a quick look at what it has heard so far
with the small always-loaded model and asks :func:`is_complete` whether that
is already a whole command. Questions and ordinary speech are left to wait the
full pause, because people routinely pause mid-sentence.

Roadmap task 9 adds the acoustic half of the same decision: :class:`SmartTurn`
runs pipecat's Smart Turn v3.1 (BSD-2, an 8 MB int8 ONNX classifier on the
Whisper-Tiny encoder) on the WHOLE turn after 250 ms of VAD silence, and
:class:`EndpointPolicy` turns its verdict plus :func:`is_complete` and
:func:`ends_dangling` into the turn-ending timings the handoff 6.6 table
("endpoint decisions") and the report (lines 66-77, 185-187) ask for:
complete -> end at 500 ms total silence (the minimum silence floor),
incomplete or dangling -> wait up to 2.0 s and re-run at every new pause,
with a hard cap of 3 s (5 s in dictation). A "patience" setting (like Siri's
Pause Time) scales those waits. The model runs on OMP_NUM_THREADS=1 in the
caller's worker thread, never on the UI thread.
"""

from __future__ import annotations

import logging
import os
import re
import threading

from voxa.agent.hearing import normalize
from voxa.agent.intents import is_compound, is_start_dictation, route

__all__ = ["is_complete", "is_followup", "EndpointPolicy", "SmartTurn", "ends_dangling",
           "patience_factor", "smart_turn_model_path", "arbitrate_endpoint"]

# A final word from this set promises more is coming ("turn off the", "play
# some"), so the utterance is not finished even if a tool already matched.
_TRAILING_PROMISE = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "for",
        "of",
        "on",
        "in",
        "at",
        "with",
        "and",
        "or",
        "my",
        "some",
        "please",
        "then",
        "um",
        "uh",
    }
)

# Whole utterances that end things on their own.
_STOP_PHRASES = frozenset({"stop", "cancel", "never mind", "go offline"})

# Play commands often go on with "on/in <app>" ("play UFO ... on my Audacious player"), so they wait for
# the full pause instead of finishing at the early one (#111).
_WAIT_FOR_FULL_PAUSE = frozenset({"play_video", "play_youtube", "play_music", "play_latest", "play_on_app"})

# Words and short phrases that clearly continue a previous exchange, so an
# utterance without the wake word is still addressed to Voxa.
_CONTINUATION_WORDS = frozenset(
    {
        "and",
        "also",
        "then",
        "now",
        "next",
        "again",
        "no",
        "not",
        "actually",
        "instead",
        "another",
        "more",
        "less",
        "louder",
        "quieter",
        "yes",
        "yeah",
        "okay",
    }
)
_CONTINUATION_PHRASES = frozenset({"what about", "how about", "make it", "the other"})

# Leading words that open a question.
_QUESTION_STARTS = frozenset(
    {"who", "what", "when", "where", "why", "how", "is", "are", "can", "could", "do", "does", "will"}
)


def _bare(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9'\s]", " ", text.casefold()).split())


def is_complete(text: str, wake_word: str = "voxa") -> bool:
    """True when the utterance is a finished command, stop phrase or dictation start."""
    if not text or not text.strip():
        return False

    wake = _bare(wake_word)
    if wake and _bare(text) == wake:
        return False

    if is_compound(text):
        return False

    normalized = normalize(text)
    if not normalized:
        return False

    if is_start_dictation(text):
        return True

    if _bare(normalized) in _STOP_PHRASES:
        return True

    call = route(text)
    if call is None:
        return False
    if call.tool in _WAIT_FOR_FULL_PAUSE:
        return False
    if not all(value.strip() for value in call.args.values()):
        return False

    words = re.findall(r"[a-z']+", normalized.casefold())
    if words and words[-1] in _TRAILING_PROMISE:
        return False
    return True


def is_followup(text: str) -> bool:
    """True when an utterance without the wake word still looks addressed to Voxa.

    A routed tool call, a continuation word/phrase, or a question all count;
    anything else is treated as ordinary speech not meant for Voxa.
    """
    if route(text) is not None:
        return True

    bare = _bare(text)
    if not bare:
        return False

    words = bare.split()
    if words[0] in _CONTINUATION_WORDS:
        return True
    if " ".join(words[:2]) in _CONTINUATION_PHRASES:
        return True
    if words[0] in _QUESTION_STARTS:
        return True
    if text.strip().endswith("?"):
        return True
    return False


# ---------------------------------------------------------------------------
# Roadmap task 9: the acoustic endpoint decision (Smart Turn v3.1).
# ---------------------------------------------------------------------------

# The timings from the handoff 6.6 table and the research report (185-187):
# a Smart Turn probe at 250 ms of VAD silence decides how long the pause has
# to be before the turn ends.
ENDPOINT_TRIGGER_MS = 250
ENDPOINT_COMPLETE_MS = 500
ENDPOINT_INCOMPLETE_WAIT_MS = 2000
ENDPOINT_CAP_MS = 3000
DICTATION_ENDPOINT_CAP_MS = 5000

# Smart Turn's own published decision threshold (pipecat's inference.py:
# probability > 0.5 means complete).
SMART_TURN_THRESHOLD = 0.5

SMART_TURN_MODEL_FILE = "smart-turn-v3.1-cpu.onnx"
SMART_TURN_MODEL_ENV = "VOXA_SMART_TURN_MODEL"

# A patience setting, like Siri's Pause Time: the waits a person gets before
# Voxa takes the floor. It scales the probe trigger and every pause floor.
PATIENCE_FACTORS = {"default": 1.0, "longer": 1.5, "longest": 2.0}


def patience_factor(name: str) -> float:
    return PATIENCE_FACTORS.get((name or "default").strip().casefold(), 1.0)


# Words that dangle: a sentence ending with one of them has not ended
# ("...came so", "passed the", "you proposed to"). The report (186): Smart
# Turn incomplete OR a text ending with a conjunction, filler or preposition
# waits. This is deliberately a CLOSED list of word classes; a full stop after
# a preposition ("I'll take it from.") is what the model and the router are for.
_DANGLING_WORDS = frozenset(
    {
        # conjunctions
        "and", "or", "but", "so", "nor", "yet",
        # fillers
        "um", "uh", "er", "eh", "hmm", "like", "well",
        # prepositions
        "to", "of", "in", "on", "at", "by", "for", "with", "from", "into",
        "over", "under", "before", "after", "between", "through", "during",
        "against", "upon", "onto",
        # determiners/pronouns that open something bigger
        "a", "an", "the", "my", "your", "our", "their", "its", "his", "her",
        "some", "this", "that", "these", "those", "which", "what",
    }
)


def ends_dangling(text: str) -> bool:
    """True when the text's last word promises more is coming."""
    words = re.findall(r"[a-z']+", (text or "").casefold())
    return bool(words) and words[-1] in _DANGLING_WORDS


class EndpointPolicy:
    """The turn-ending timings of one pause rule, scaled by the patience setting.

    The bench and the live app read the same numbers: the probe trigger, the
    floor once a probe said the turn is complete (the minimum silence floor),
    how long an incomplete turn waits for the speaker to resume, and the hard
    cap that ends the silence no matter what.
    """

    def __init__(self, patience: str = "default", *, dictation: bool = False) -> None:
        factor = patience_factor(patience)
        self.patience = patience if patience in PATIENCE_FACTORS else "default"
        self.factor = factor
        self.trigger_seconds = ENDPOINT_TRIGGER_MS / 1000.0 * factor
        self.complete_seconds = ENDPOINT_COMPLETE_MS / 1000.0 * factor
        self.incomplete_wait_seconds = ENDPOINT_INCOMPLETE_WAIT_MS / 1000.0 * factor
        self.cap_seconds = (DICTATION_ENDPOINT_CAP_MS if dictation else ENDPOINT_CAP_MS) / 1000.0 * factor

    def incomplete_floor_seconds(self, dangling: bool = False) -> float:
        """How long an incomplete turn stays open before the cap cuts it."""
        return self.cap_seconds if dangling else self.incomplete_wait_seconds


def smart_turn_model_path(explicit: str = "", data_root: str = "") -> str:
    """Where the Smart Turn v3.1 ONNX lives, or "" when it is nowhere to be found.

    Order: the explicit setting, the environment, the bench data root (the
    offline run must not need the network), then a system-wide copy.
    """
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get(SMART_TURN_MODEL_ENV, "")
    if env:
        candidates.append(env)
    root = data_root or os.environ.get("VOXA_BENCH_DATA", "")
    if root:
        candidates.append(os.path.join(root, "models", SMART_TURN_MODEL_FILE))
    candidates.append(os.path.join("/app/share/voxa/models", SMART_TURN_MODEL_FILE))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


class SmartTurn:
    """Smart Turn v3.1 (an 8 MB int8 ONNX classifier on Whisper-Tiny) over raw audio.

    ``is_complete(segment_pcm)`` answers whether the WHOLE turn heard so far is
    acoustically finished, from the waveform alone. The ONNX graph wants the
    Whisper log-mel of the last 8 seconds, batch of 1, and its single output is
    already the probability of "complete" (pipecat's own inference.py).

    The session pins OMP_NUM_THREADS=1 / single-threaded execution: that is how
    pipecat measures the 9-57 ms CPU inference, and multi-threaded ONNX runs of
    this graph are slower, not faster. Callers keep it off the UI thread.
    A missing model, missing onnxruntime or broken weights give ``None`` from
    every call - the caller falls back to the pause rules it already had.
    """

    SR = 16000
    N_FFT = 400
    HOP = 160
    FRAMES = 800  # 8 s of Whisper log-mel, the model's window

    _lock = threading.Lock()
    _cache: dict[str, "SmartTurn"] = {}

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._session = None
        self._error: str | None = None
        self._load()

    # -- loading ---------------------------------------------------------------

    def _load(self) -> None:
        try:
            # Pipecat's own performance note: OMP_NUM_THREADS=1, OMP_WAIT_POLICY
            # PASSIVE. Set once per process; the session pins its threads too.
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(self.model_path, sess_options=options,
                                                 providers=["CPUExecutionProvider"])
            inputs = self._session.get_inputs()
            if not inputs or inputs[0].name != "input_features":
                self._error = f"unexpected input {[i.name for i in inputs]}"
                self._session = None
        except Exception as exc:  # noqa: BLE001 - a model boundary never raises into the pipeline
            self._error = str(exc)
            self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None

    @property
    def error(self) -> str | None:
        return self._error

    @classmethod
    def shared(cls, model_path: str) -> "SmartTurn":
        """One session per process and path: the weights are loaded once."""
        with cls._lock:
            engine = cls._cache.get(model_path)
            if engine is None:
                engine = cls(model_path)
                cls._cache[model_path] = engine
            return engine

    # -- features ----------------------------------------------------------------

    @staticmethod
    def _mel_filters() -> "object":
        import numpy as np

        n_mels, n_fft, sr = 80, SmartTurn.N_FFT, SmartTurn.SR

        def hz2mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel2hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        fft_bins = np.linspace(hz2mel(0.0), hz2mel(sr / 2.0), n_mels + 2)
        ff = mel2hz(fft_bins)
        bins = np.floor((n_fft + 1) * ff / sr).astype(int)
        weights = np.zeros((n_mels, 1 + n_fft // 2))
        for j in range(1, n_mels + 1):
            lo, center, up = bins[j - 1], bins[j], bins[j + 1]
            for k in range(1 + n_fft // 2):
                if lo <= k < center:
                    weights[j - 1, k] = (k - lo) / (center - lo)
                elif center <= k < up:
                    weights[j - 1, k] = (up - k) / (up - center)
        weights[1:-1] *= 2.0 / (ff[2] - ff[0])
        weights /= sr / n_fft / 2
        return weights

    _MEL = None
    _WINDOW = None

    def features(self, audio: "object") -> "object":
        """The [1, 80, 800] Whisper log-mel of the LAST 8 seconds, HF's protocol:
        peak-normalise, zero-pad at the end, 400-point hann windows every 160 samples."""
        import numpy as np

        if type(self)._MEL is None:
            type(self)._MEL = self._mel_filters()
            type(self)._WINDOW = np.hanning(self.N_FFT + 1)[: self.N_FFT]
        samples = self.N_FFT + (self.FRAMES - 1) * self.HOP
        if len(audio) > samples:
            audio = audio[-samples:]
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0.0:
            audio = audio * (0.95 / peak)
        audio = np.concatenate([audio, np.zeros(samples - len(audio) + 1)])
        frames = np.lib.stride_tricks.sliding_window_view(audio, self.N_FFT)[: self.FRAMES * self.HOP: self.HOP]
        if len(frames) < self.FRAMES:
            frames = np.concatenate([frames, np.zeros((self.FRAMES - len(frames), self.N_FFT))])
        spec = np.abs(np.fft.rfft(frames * type(self)._WINDOW, n=self.N_FFT, axis=-1)) ** 2
        mel = type(self)._MEL @ spec.T
        log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, float(log_mel.max()) - 80.0)
        return np.ascontiguousarray(((log_mel + 40.0) / 40.0), dtype=np.float32)[None]

    # -- the decision ------------------------------------------------------------

    def probability(self, segment_pcm: bytes) -> float | None:
        """P(the turn is acoustically complete) for int16 PCM bytes, or None."""
        if self._session is None:
            return None
        try:
            import numpy as np

            audio = np.frombuffer(segment_pcm, dtype="<i2").astype(np.float64) / 32768.0
            if not len(audio):
                return None
            output = self._session.run(None, {"input_features": self.features(audio)})
            return float(output[0][0][0])
        except Exception as exc:  # noqa: BLE001 - the endpoint never rides on a model failure
            logging.debug("smart turn inference failed: %s", exc)
            return None

    def is_complete(self, segment_pcm: bytes) -> bool | None:
        """True/False from the model, None when it has no answer."""
        probability = self.probability(segment_pcm)
        if probability is None:
            return None
        return probability > SMART_TURN_THRESHOLD


# ---------------------------------------------------------------------------
# Roadmap task 9 re-plan (f178): the endpoint ARBITRATOR.
# ---------------------------------------------------------------------------

def arbitrate_endpoint(model_verdict: "bool | None", *,
                       joint_suspect: "bool | None" = None,
                       turn_suspect: bool = False,
                       more_speech_arriving: bool = False) -> "bool | None":
    """Gate a Smart Turn "complete" verdict on the shipped joint double-talk
    suspect and the explicit turn state, WITHOUT touching the model's own
    probability.

    f136 measured that Smart Turn v3.1 alone cannot satisfy both the ep90 timing
    and the premature-cut-off target on the frozen set: no threshold point
    satisfies ep90<=1200 AND premature<=0.01 ("the two conditions pull opposite
    ways"). What it lacked was the joint gate f173 and the turn states f174 now
    ship. This arbitrator adds only a WAIT: the model's verdict is never
    retuned, only trusted less when the shipped evidence says MORE speech is
    still arriving.

    A model ``True`` ("this turn is finished") is not enough to end the turn on a
    short pause when the joint gate believes the speaker is still mid-utterance:
    such a pause is a possible breath, not the end of the turn. So when a joint
    suspect is OPEN, the turn is in a suspect/duck state, or more gated speech is
    still arriving (Silero hysteresis on / the capture span has not reached its
    trailing endpoint), the endpointer WAITS the pause out - the 2.0 s incomplete
    floor, capped at 3 s (5 s in dictation) - instead of firing complete. A quiet
    joint gate over a text-complete turn still fires normally.

    Honesty constraints, both inherited from f136:
      * a model with no verdict (``None``) is returned unchanged - the pure
        always-loaded-pause-rule path is untouched;
      * a DEGRADED joint verdict (``joint_suspect is None``, INDETERMINATE) adds
        no veto, so f136's pure-model behaviour is preserved on the degraded
        path; only a definite ``True`` suspect (or the turn/more-speech flags)
        can hold the turn open.

    Returns the verdict to feed the pause floor: ``True`` fires at the minimum
    silence floor, ``False`` waits the incomplete floor, ``None`` keeps the pause
    rule the app already had.
    """
    if model_verdict is not True:
        # Incomplete, or no model verdict: the arbitrator never ADDS a cut, so
        # this is exactly f136's behaviour.
        return model_verdict
    # The model says "finished". Trust it only while the shipped joint evidence
    # agrees nothing more is coming. A definite suspect (not the None/INDETERMINATE
    # case) is the only joint signal that can veto; the turn state and the
    # still-arriving-speech flag are the other shipped reasons to wait.
    if joint_suspect is True or turn_suspect or more_speech_arriving:
        return False
    return True
