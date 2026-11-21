"""B3: every hearing metric the shared brief names, computed from a clip's truth and the runner's events.

The inputs are files, never a sound device: ``truth.json`` (what the clip says: the user's word
onsets/offsets, Voxa's TTS word timings, SER/SNR, delay, drift, the expected turn decision) and the
``events.jsonl`` ``voxa/bench/runner.py`` wrote for the same clip (one ``frame`` event per 20 ms frame
plus ``suspect``, ``barge-in-confirm``, ``tts-stop``, ``pause``, ``endpoint``, ``transcript``,
``request``, ``delay-estimate`` and ``clip-transcript``).

Text is normalised the way the evaluation table requires - lower-case, punctuation dropped, numbers
spelled out - and compared with this module's own word-level Levenshtein (no jiwer, no new dependency):
:func:`edit_distance` for the distance, :func:`align` for one status per reference word
(``match`` / ``sub`` / ``del``) plus the insertion count, which is what WER, recall, deletion and
first-word truncation are all built from.

Every metric returns the samples it is computed from, so ``report.py`` can bootstrap a 95 % CI over
them (1000 resamples). A metric this stage cannot measure returns ``None`` with a reason, and the
report carries it as ``value: null, pass: true`` - never as a made-up number.

What each metric reads (the definitions in ``research/voxa-hearing/voxa-hearing-research.md`` lines
88-114 and ``handoff-full-duplex.md`` 8.3, mapped onto what B1 renders and B2 records):

* ERLE: the ``erle_db`` of far-end frames after 2 s of clips where only Voxa plays.
* near-end attenuation: what the pipeline does to the user's own frames while Voxa plays, in dB,
  normalised per clip against the mic level of the same frames (so SER, which differs between clips,
  cannot move the number).
* barge-in / turn metrics: the runner's events against the truth onsets.
* text metrics: the user's reference words against what the pipeline's ASR stage heard.

Unmeasurable at this stage, with the reason in ``UNMEASURED``: ``align_err_p90_ms`` (the live
reference-to-echo alignment is task 1), ``erle_music_loss_db`` (the frozen clip set has no music
playback), ``transport_drop_count`` (the offline runner has no queue; the live one is task A4).
"""

from __future__ import annotations

import re

import numpy as np

SR = 16000  # voxa/audio.py REFERENCE_RATE, the bench clip rate
CHUNK_MS = 20
SPEAKING_POWER = 1e-6  # voxa/anc.py SILENT_REFERENCE_POWER: the speakers are on above this

# The evaluation table's windows and thresholds-as-definitions.
ECHO_WINDOW_S = 1.5  # echo-leak words match TTS words played within +/- 1.5 s
ERLE_SETTLE_S = 2.0  # ERLE is scored on far-end frames after 2 s
ATTEMPT_MIN_WORDS = 2  # a barge-in attempt: >= 2 words ...
ATTEMPT_MIN_SECONDS = 0.5  # ... and >= 0.5 s of speech
MISSED_WINDOW_S = 1.0  # ... not ducking TTS within 1.0 s counts as missed
RECONVERGE_TARGET_DB = 15.0  # recovery = time until ERLE >= 15 dB again
DRIFT_MIN_PPM = 100.0  # drift_erle_loss_db compares clips at >= 100 ppm
BOOTSTRAP_N = 1000  # roadmap B3: 1000-sample bootstrap CIs
CI_LO, CI_HI = 2.5, 97.5  # the 95 % CI is the 2.5 % / 97.5 % pair of the resampled statistic

# Words that must never be lost because they are short, quiet or resemble Voxa's own voice
# (the design rules of the shared brief): an interrupt command, or the whole of a short answer.
PROTECTED_COMMAND_WORDS = frozenset({"stop", "cancel", "pause", "shut", "up", "wait"})


# --------------------------------------------------------------------------- text

def number_to_words(token: str) -> str:
    """Spell a number out, so "72" and "seventy two" are the same two words."""
    if token.startswith("-"):
        token = token[1:]
    if "." in token:  # "3.5" -> "three point five"
        head, _, tail = token.partition(".")
        out = number_to_words(head) if head.isdigit() else ""
        out = f"{out} point {number_to_words(tail)}" if tail.isdigit() else out
        return out
    if not token.isdigit():
        return token
    value = int(token)
    if value >= 1000:  # long numbers: read the digits
        return " ".join(_SMALL[int(digit)] for digit in token)
    return _below_thousand(value)


_SMALL = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen")
_TENS = ("", "one", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _below_thousand(value: int) -> str:
    if value < 20:
        return _SMALL[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] + (f" {_SMALL[ones]}" if ones else "")
    hundreds, rest = divmod(value, 100)
    out = f"{_SMALL[hundreds]} hundred"
    if rest:
        out = f"{out} {_below_thousand(rest)}"
    return out


def normalise(text: str) -> str:
    """Lower-case, drop punctuation and apostrophes, spell numbers out."""
    text = re.sub(r"['\u2019]", "", text.casefold())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(number_to_words(word) if word.isdigit() else word for word in text.split())


def word_list(text: str) -> list[str]:
    """The normalised words of a transcript or reference text."""
    return re.findall(r"[a-z0-9]+", normalise(text))


# --------------------------------------------------------------------------- alignment

def edit_distance(reference: list[str], heard: list[str]) -> int:
    """Word-level Levenshtein distance, pure Python (no jiwer)."""
    previous = list(range(len(heard) + 1))
    for index, first in enumerate(reference, 1):
        current = [index]
        for other_index, second in enumerate(heard, 1):
            current.append(min(previous[other_index] + 1,          # deletion
                               current[other_index - 1] + 1,       # insertion
                               previous[other_index - 1] + (first != second)))  # match/sub
        previous = current
    return previous[-1]


def align(reference: list[str], heard: list[str]) -> tuple[list[str], int]:
    """One status per reference word (``match`` / ``sub`` / ``del``) and the insertion count.

    The backtrace follows the same recurrence as :func:`edit_distance`, so the counts add up to
    ``(substitutions + deletions + insertions) / reference_words`` - the WER the table defines.
    """
    rows = len(reference) + 1
    columns = len(heard) + 1
    cost = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(columns):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, columns):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + (reference[i - 1] != heard[j - 1]))
    statuses: list[str] = []
    insertions = 0
    i, j = len(reference), len(heard)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != heard[j - 1]):
            statuses.append("match" if reference[i - 1] == heard[j - 1] else "sub")
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            statuses.append("del")
            i -= 1
        else:
            insertions += 1
            j -= 1
    statuses.reverse()
    return statuses, insertions


def wer(reference: list[str], heard: list[str]) -> float | None:
    """(substitutions + deletions + insertions) / reference words; None with no reference words."""
    if not reference:
        return None
    statuses, insertions = align(reference, heard)
    errors = sum(1 for status in statuses if status != "match") + insertions
    return errors / len(reference)


# --------------------------------------------------------------------------- samples

class Samples(dict):
    """What one metric is computed from: values, an optional weight per value, and the statistic.

    ``stat`` is the statistic applied to a resampled copy of the values; ``weights`` makes it a
    ratio (sum of values over the sum of their weights, which is how "per hour" and "per clip"
    rates keep their denominators), and ``minus`` subtracts a second group's statistic (the
    double-talk minus near-end-single-talk metrics).
    """

    def __init__(self, values: list[float], stat: str = "mean", weights: list[float] | None = None,
                 minus: list[float] | None = None, minus_stat: str = "mean") -> None:
        self["values"] = values
        self["stat"] = stat
        self["weights"] = weights
        self["minus"] = minus
        self["minus_stat"] = minus_stat


def _apply(values: np.ndarray, index: np.ndarray, stat: str) -> np.ndarray:
    """The statistic over every bootstrap resample at once."""
    rows = values[index]
    if stat == "mean":
        return rows.mean(axis=1)
    if stat == "median":
        return np.median(rows, axis=1)
    if stat == "sum":
        return rows.sum(axis=1)
    return np.percentile(rows, float(stat[1:]), axis=1)


def _point(values: list[float], stat: str, weights: list[float] | None = None,
           minus: list[float] | None = None, minus_stat: str = "mean") -> float | None:
    """The statistic on the samples themselves (not a resample): the metric's reported value."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return None
    if weights is not None:
        denominator = float(np.sum(weights))
        if not denominator:
            return None
        out = float(np.sum(data)) / denominator
    elif stat == "mean":
        out = float(data.mean())
    elif stat == "median":
        out = float(np.median(data))
    elif stat == "sum":
        out = float(data.sum())
    else:
        out = float(np.percentile(data, float(stat[1:])))
    if minus is not None:
        other = np.asarray(minus, dtype=np.float64)
        if other.size == 0:
            return None
        if minus_stat == "mean":
            out -= float(other.mean())
        elif minus_stat == "median":
            out -= float(np.median(other))
        elif minus_stat == "sum":
            out -= float(other.sum())
        else:
            out -= float(np.percentile(other, float(minus_stat[1:])))
    return out


def measure(samples: Samples, n: int = BOOTSTRAP_N, seed: int = 0) -> dict:
    """Value plus 95 % bootstrap CI for one metric, or ``{"value": None, "ci": [None, None]}``."""
    values = samples["values"]
    value = _point(values, samples["stat"], samples.get("weights"), samples.get("minus"),
                   samples.get("minus_stat", "mean"))
    if value is None:
        return {"value": None, "ci": [None, None], "samples": len(values)}
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, data.size, size=(n, data.size))
    if samples.get("weights") is not None:
        weight_data = np.asarray(samples["weights"], dtype=np.float64)
        denominator = weight_data[index].sum(axis=1)
        out = data[index].sum(axis=1) / np.where(denominator == 0, np.nan, denominator)
        out = out[~np.isnan(out)]
        if out.size == 0:
            return {"value": value, "ci": [None, None], "samples": len(values)}
    else:
        out = _apply(data, index, samples["stat"])
    if samples.get("minus") is not None:
        other = np.asarray(samples["minus"], dtype=np.float64)
        if other.size:
            out = out - _apply(other, rng.integers(0, other.size, size=(n, other.size)),
                               samples.get("minus_stat", "mean"))
    low, high = np.percentile(out, [CI_LO, CI_HI])
    return {"value": round(float(value), 6), "ci": [round(float(low), 6), round(float(high), 6)],
            "samples": len(values)}


# --------------------------------------------------------------------------- clip facts

def _events_named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event.get("event") == name]


def _word_times(words: list[dict], shift: float = 0.0) -> list[tuple[str, float, float]]:
    """Normalised (word, onset, offset) triples, shifted into clip time."""
    out = []
    for word in words:
        text = normalise(str(word.get("word", "")))
        if not text:
            continue
        onset = float(word.get("onset_s", 0.0)) + shift
        offset = float(word.get("offset_s", onset)) + shift
        for token in text.split():
            out.append((token, onset, offset))
    return out


def clip_facts(truth: dict, events: list[dict], summary: dict | None = None) -> dict:
    """Everything the metrics need about one clip, in one dict.

    Transcript word times are segment-relative in the runner's events, so they are shifted by
    ``t_s - segment_seconds`` of the ``pause`` event that ended the segment: every heard word then
    carries its position in clip time, which is what the +/- 1.5 s echo window and the latency
    metrics compare against the truth onsets.
    """
    playback = truth.get("playback")
    user = truth.get("user")
    heard: list[tuple[str, float, float]] = []
    segment_shift = 0.0
    for event in events:
        name = event.get("event")
        if name == "pause":
            segment_shift = float(event["t_s"]) - float(event.get("segment_seconds", 0.0))
        elif name == "transcript":
            heard.extend(_word_times(event.get("words") or [], segment_shift))
    stream = _events_named(events, "clip-transcript")
    frames = _events_named(events, "frame")
    delay = _events_named(events, "delay-estimate")
    echo = truth.get("echo") or {}
    return {
        "id": truth.get("id"),
        "scenario": truth.get("scenario"),
        "expect": truth.get("expect"),
        "pair": truth.get("pair"),
        "overlap_position": truth.get("overlap_position"),
        "echo_mode": (summary or {}).get("echo_mode"),
        "playback": None if playback is None else {
            "onset_s": float(playback.get("onset_s", 0.0)),
            "offset_s": float(playback.get("offset_s", 0.0)),
            "words": _word_times(playback.get("words") or []),
        },
        "user": None if user is None else {
            "onset_s": float(user.get("onset_s", 0.0)),
            "offset_s": float(user.get("offset_s", 0.0)),
            "words": _word_times(user.get("words") or []),
        },
        "echo": echo,
        "frames": frames,
        "heard": heard,
        "heard_text": " ".join(word for word, _on, _off in heard),
        "stream": _word_times((stream[-1].get("words") if stream else []) or []),
        "requests": _events_named(events, "request"),
        "endpoints": _events_named(events, "endpoint"),
        "stops": _events_named(events, "tts-stop"),
        "starts": _events_named(events, "tts-start"),
        "ends": _events_named(events, "tts-end"),
        "barge_ins": _events_named(events, "barge-in-confirm"),
        "suspects": _events_named(events, "suspect"),
        "pauses": _events_named(events, "pause"),
        "delay_est": delay[-1] if delay else None,
        "summary": summary or {},
    }


def _playback_seconds(fact: dict) -> float:
    playback = fact["playback"]
    return 0.0 if playback is None else max(0.0, playback["offset_s"] - playback["onset_s"])


def _only_voxa(fact: dict) -> bool:
    """A Voxa-only clip: she is playing and the user says nothing (fst, pair side voxa)."""
    return fact["playback"] is not None and fact["user"] is None


def _double_talk(fact: dict) -> bool:
    return fact["playback"] is not None and fact["user"] is not None


def _near_end_only(fact: dict) -> bool:
    return fact["playback"] is None and fact["user"] is not None


def _user_words(fact: dict) -> list[str]:
    user = fact["user"]
    return [] if user is None else [word for word, _on, _off in user["words"]]


def _heard_words(fact: dict) -> list[str]:
    return [word for word, _on, _off in fact["heard"]]


def _user_span(fact: dict) -> tuple[float, float]:
    user = fact["user"]
    if user is None:
        return 0.0, 0.0
    words = user["words"]
    if not words:
        return user["onset_s"], user["offset_s"]
    return min(word[1] for word in words), max(word[2] for word in words)


def _frames_in(fact: dict, start: float, end: float, only_while_playing: bool = False) -> list[dict]:
    """The 20 ms frames covering a span of clip time, optionally only while Voxa is playing."""
    out = []
    for frame in fact["frames"]:
        lo, hi = frame["samples"]
        onset, offset = lo / SR, hi / SR
        if offset <= start or onset >= end:
            continue
        if only_while_playing and float(frame.get("ref_power", 0.0)) <= SPEAKING_POWER:
            continue
        out.append(frame)
    return out


def _attenuation_db(fact: dict, start: float, end: float, only_while_playing: bool = True) -> float | None:
    """What the pipeline did to the user's own audio in a span, in dB (positive = attenuation).

    Per frame, the level the ASR stage received minus the level the microphone delivered; averaged
    over the frames of the span. Normalising every clip against its own input level is what keeps
    SER, which differs from clip to clip, out of the number.
    """
    values = []
    for frame in _frames_in(fact, start, end, only_while_playing):
        mic_db = frame.get("mic_level_db")
        clean_db = frame.get("clean_level_db")
        if mic_db is None or clean_db is None:
            continue
        values.append(float(mic_db) - float(clean_db))
    return None if not values else float(np.mean(values))


def _frame_ids(fact: dict) -> list[int]:
    return [int(frame["frame"]) for frame in fact["frames"]]


def _span_promoted_once(fact: dict, first: int, last: int, heard: str) -> float:
    """1.0 when the accepted overlap span reached the request exactly once, with no lost or
    duplicated frame id inside it (the handoff's turn-retention test)."""
    ids = _frame_ids(fact)
    if len(ids) != len(set(ids)):
        return 0.0  # a duplicated frame identifier
    if any(ids.count(index) != 1 for index in range(first, last + 1)):
        return 0.0  # a frame of the accepted span is missing or counted twice
    needle = normalise(heard)
    if not needle:
        return 0.0
    reached = 0
    for request in fact["requests"]:
        haystack = normalise(str(request.get("text", "")))
        if f" {needle} " in f" {haystack} ":
            reached += 1
    return 1.0 if reached == 1 else 0.0


# --------------------------------------------------------------------------- metrics
#
# Every builder takes the list of clip facts and returns the Samples its metric is computed
# from, or None when this stage cannot measure it. ``report.py`` turns a None into
# ``{"value": null, "ci": [null, null], "pass": true, "reason": ...}``.

def erle_fst_samples(facts: list[dict]) -> Samples:
    """ERLE on far-end-only frames after 2 s (``erle_fst_median_db``, ``erle_fst_p10_db``)."""
    values = [float(frame["erle_db"])
              for fact in facts if _only_voxa(fact)
              for frame in fact["frames"]
              if float(frame["t_s"]) >= ERLE_SETTLE_S and float(frame.get("ref_power", 0.0)) > SPEAKING_POWER]
    return Samples(values, "median")


def nearend_attenuation_samples(facts: list[dict]) -> Samples:
    """Near-end attenuation in double talk minus the same in near-end-only clips, dB."""
    dt, nst = [], []
    for fact in facts:
        if not fact["user"]:
            continue
        start, end = _user_span(fact)
        if _double_talk(fact):
            value = _attenuation_db(fact, start, end, only_while_playing=True)
            if value is not None:
                dt.append(value)
        elif _near_end_only(fact):
            value = _attenuation_db(fact, start, end, only_while_playing=False)
            if value is not None:
                nst.append(value)
    return Samples(dt, "mean", minus=nst, minus_stat="mean")


def fst_transcript_samples(facts: list[dict]) -> Samples:
    """TTS-only clips whose ASR stage produced at least one word."""
    values = [1.0 if _heard_words(fact) else 0.0 for fact in facts if _only_voxa(fact)]
    return Samples(values, "mean")


def echo_leak_samples(facts: list[dict]) -> Samples:
    """Played TTS words that come back as ASR words within the +/- 1.5 s window."""
    values: list[float] = []
    for fact in facts:
        playback = fact["playback"]
        if playback is None:
            continue
        for word, onset, offset in playback["words"]:
            leaked = any(match == word and onset - ECHO_WINDOW_S <= when <= offset + ECHO_WINDOW_S
                         for match, when, _off in fact["stream"])
            values.append(1.0 if leaked else 0.0)
    return Samples(values, "mean")


def false_bargein_samples(facts: list[dict]) -> Samples:
    """Barge-in triggers with no user onset, per hour of Voxa playback."""
    values, weights = [], []
    for fact in facts:
        seconds = _playback_seconds(fact)
        if seconds <= 0.0:
            continue
        user_onset, user_offset = _user_span(fact)
        count = 0
        for event in fact["barge_ins"]:
            when = float(event["t_s"])
            if fact["user"] and user_onset - 1.0 <= when <= user_offset + 1.0:
                continue  # a real user onset is in the listening window: not a false trigger
            count += 1
        values.append(float(count))
        weights.append(seconds / 3600.0)
    return Samples(values, "sum", weights=weights)


def _attempts(facts: list[dict]) -> list[dict]:
    """Barge-in attempts: the user speaks over Voxa with >= 2 words for >= 0.5 s."""
    out = []
    for fact in facts:
        if not _double_talk(fact):
            continue
        words = _user_words(fact)
        onset, offset = _user_span(fact)
        if len(words) >= ATTEMPT_MIN_WORDS and offset - onset >= ATTEMPT_MIN_SECONDS:
            out.append(fact)
    return out


def _first_stop_after(fact: dict, when: float, window: float | None = None) -> float | None:
    """The first playback cessation at or after ``when`` (a duck, a stop: the runner emits
    ``tts-stop``; a real duck event, once task 8 adds one, is the same measurement)."""
    best = None
    for event in fact["stops"]:
        stop_at = float(event["t_s"])
        if stop_at < when:
            continue
        if window is not None and stop_at > when + window:
            continue
        if best is None or stop_at < best:
            best = stop_at
    return best


def missed_bargein_samples(facts: list[dict]) -> Samples:
    """Attempts that did not duck Voxa's stream within 1.0 s of the user's onset."""
    values = []
    for fact in _attempts(facts):
        onset = _user_span(fact)[0]
        values.append(0.0 if _first_stop_after(fact, onset, MISSED_WINDOW_S) is not None else 1.0)
    return Samples(values, "mean")


def duck_latency_samples(facts: list[dict]) -> Samples:
    """t(duck/stop of Voxa's stream) - t(user onset), milliseconds, for the attempts."""
    values = []
    for fact in _attempts(facts):
        onset = _user_span(fact)[0]
        stop_at = _first_stop_after(fact, onset)
        if stop_at is not None:
            values.append((stop_at - onset) * 1000.0)
    return Samples(values, "p50")


def stop_latency_samples(facts: list[dict]) -> Samples:
    """The same onset-to-cessation interval, the handoff's stop latency (median and p95)."""
    values = duck_latency_samples(facts)["values"]
    return Samples(values, "p95")


def interrupt_stop_samples(facts: list[dict]) -> Samples:
    """Voxa-only responses her own playback stopped (the per-response self-interruption rate)."""
    values = [1.0 if fact["stops"] else 0.0 for fact in facts if _only_voxa(fact)]
    return Samples(values, "mean")


def backchannel_resume_samples(facts: list[dict]) -> Samples:
    """Back-channels that let her finish the reply instead of taking the floor."""
    values = []
    for fact in facts:
        if fact["expect"] != "backchannel" or not fact["user"]:
            continue
        onset, offset = _user_span(fact)
        values.append(0.0 if _first_stop_after(fact, onset, offset - onset + MISSED_WINDOW_S)
                      is not None else 1.0)
    return Samples(values, "mean")


def protected_word_samples(facts: list[dict]) -> Samples:
    """Every word of a short answer or of an interrupt command: kept by the pipeline or not."""
    values = []
    for fact in facts:
        words = _user_words(fact)
        if not words:
            continue
        command = any(word in PROTECTED_COMMAND_WORDS for word in words)
        if not (command or len(words) <= 2):
            continue
        heard = list(_heard_words(fact))
        for word in words:
            values.append(1.0 if word in heard else 0.0)
            if word in heard:
                heard.remove(word)
    return Samples(values, "mean")


def user_recall_samples(facts: list[dict]) -> Samples:
    """Reference words of near-end-only clips the pipeline heard (hits / reference words)."""
    values = []
    for fact in facts:
        if not _near_end_only(fact):
            continue
        statuses, _ins = align(_user_words(fact), _heard_words(fact))
        values.extend(1.0 if status == "match" else 0.0 for status in statuses)
    return Samples(values, "mean")


def user_word_deletion_samples(facts: list[dict]) -> Samples:
    """Words lost from known user speech, in any scenario (over-suppression, buffer loss)."""
    values = []
    for fact in facts:
        if not fact["user"]:
            continue
        statuses, _ins = align(_user_words(fact), _heard_words(fact))
        values.extend(1.0 if status == "del" else 0.0 for status in statuses)
    return Samples(values, "mean")


def dt_wer_penalty_samples(facts: list[dict]) -> Samples:
    """WER on double-talk clips minus WER on near-end-only clips, in percentage points.

    Both sides are measured against the clip's own user word labels with the same pinned
    recognizer, so the difference is what double talk costs. The matched no-playback recording
    the handoff asks for (the same utterance without playback) is not in the frozen clip set; the
    NST clips are different utterances, which is why this is a scenario difference, not a per-
    utterance one.
    """
    dt, nst = [], []
    for fact in facts:
        if not fact["user"]:
            continue
        score = wer(_user_words(fact), _heard_words(fact))
        if score is None:
            continue
        if _double_talk(fact):
            dt.append(score * 100.0)
        elif _near_end_only(fact):
            nst.append(score * 100.0)
    return Samples(dt, "mean", minus=nst, minus_stat="mean")


def _first_word_truncated(fact: dict) -> float:
    statuses, _ins = align(_user_words(fact), _heard_words(fact))
    if not statuses:
        return 1.0  # the whole utterance, first word with it, never reached the ASR stage
    return 0.0 if statuses[0] == "match" else 1.0


def first_word_trunc_cold_samples(facts: list[dict]) -> Samples:
    """First reference word deleted or substituted, cold start (the user starts as she finishes)."""
    values = [_first_word_truncated(fact) for fact in facts
              if fact["user"] and fact["overlap_position"] == "cold"]
    return Samples(values, "mean")


def first_word_trunc_bargein_samples(facts: list[dict]) -> Samples:
    """The same test on the utterances that arrived through a barge-in."""
    values = [_first_word_truncated(fact) for fact in facts
              if fact["user"] and fact["barge_ins"]]
    return Samples(values, "mean")


def _endpoint_rows(fact: dict) -> list[tuple[float, float]]:
    """(endpoint time, last user word end) for every endpoint on a clip with user speech."""
    if not fact["user"]:
        return []
    last_end = _user_span(fact)[1]
    return [(float(event["t_s"]), last_end) for event in fact["endpoints"]]


def premature_cutoff_samples(facts: list[dict]) -> Samples:
    """Endpoints that fire before the user's last word ends, per utterance."""
    values = [1.0 if when < last_end else 0.0 for fact in facts for when, last_end in _endpoint_rows(fact)]
    return Samples(values, "mean")


def endpoint_latency_samples(facts: list[dict]) -> Samples:
    """t(endpoint) - t(last word end), milliseconds (EP50 and EP90)."""
    values = [(when - last_end) * 1000.0 for fact in facts for when, last_end in _endpoint_rows(fact)]
    return Samples(values, "p50")


def delay_error_samples(facts: list[dict]) -> Samples:
    """|delay the canceller estimated - delay the renderer injected| in ms."""
    values = []
    for fact in facts:
        estimate = fact["delay_est"]
        truth_delay = (fact["echo"] or {}).get("delay_samples")
        if estimate is None or not truth_delay:
            continue
        values.append(abs(float(estimate["delay_samples"]) - float(truth_delay)) * 1000.0 / SR)
    return Samples(values, "mean")


def drift_erle_loss_samples(facts: list[dict]) -> Samples:
    """ERLE a drifting echo path loses against a steady one, in dB.

    The evaluation table states this as "ERLE at +-100 ppm within 3 dB of no-drift", so it compares
    two groups of Voxa-only clips rather than halves of one clip: far-end frames after 2 s of clips
    that drift by >= 100 ppm, against the same frames of clips that drift by less than that. A
    positive value is what the drift costs; the roadmap's target is <= 3 dB.
    """
    drifting, steady = [], []
    for fact in facts:
        if not _only_voxa(fact):
            continue
        echo = fact["echo"] or {}
        ppm = abs(float(echo.get("drift_ppm") or 0.0))
        group = drifting if ppm >= DRIFT_MIN_PPM else steady
        group.extend(float(frame["erle_db"])
                     for frame in fact["frames"]
                     if float(frame["t_s"]) >= ERLE_SETTLE_S
                     and float(frame.get("ref_power", 0.0)) > SPEAKING_POWER)
    return Samples(steady, "mean", minus=drifting, minus_stat="mean")


def reconverge_samples(facts: list[dict]) -> Samples:
    """Time until ERLE is back at 15 dB after an echo-path change (the delay re-estimate).

    A change that never reaches 15 dB inside the clip is not counted here; ``report.py`` puts
    those in ``run.reconverge_unrecovered`` so the latency mean cannot hide a failure to recover.
    """
    values = []
    for fact in facts:
        estimate = fact["delay_est"]
        if estimate is None:
            continue
        change_at = float(estimate["t_s"])
        recovered = [float(frame["t_s"]) for frame in fact["frames"]
                     if float(frame["t_s"]) > change_at and float(frame["erle_db"]) >= RECONVERGE_TARGET_DB]
        if recovered:
            values.append(min(recovered) - change_at)
    return Samples(values, "mean")


def cpu_core_share_samples(facts: list[dict]) -> Samples:
    """Cores the bench keeps busy: ASR processing seconds over the clip's wall seconds."""
    values = []
    for fact in facts:
        summary = fact["summary"]
        wall = float(summary.get("wall_seconds") or 0.0)
        asr = float(summary.get("asr_processing_seconds") or 0.0)
        if wall > 0.0:
            values.append(asr / wall)
    return Samples(values, "mean")


def self_command_samples(facts: list[dict]) -> Samples:
    """Voxa-only clips whose own playback created a user turn, request or command."""
    values = [1.0 if fact["requests"] else 0.0 for fact in facts if _only_voxa(fact)]
    return Samples(values, "mean")


def self_pause_samples(facts: list[dict]) -> Samples:
    """Every duck, pause or stop of Voxa's own playback on a Voxa-only clip, per playback hour.

    The runner emits ``tts-stop`` when the barge-in decision interrupts her reply; that is the
    playback cessation this metric counts, and a real duck event (roadmap task 8 adds one, provisional
    ones included, as the handoff demands) is counted by the same rule. A ``pause`` event is NOT
    counted: in the runner it marks a segment boundary in the microphone, not an output pause, and
    counting those would measure the segmentation instead of the self-interruption.
    """
    values, weights = [], []
    for fact in facts:
        seconds = _playback_seconds(fact)
        if seconds <= 0.0 or not _only_voxa(fact):
            continue
        values.append(float(len(fact["stops"])))
        weights.append(seconds / 3600.0)
    return Samples(values, "sum", weights=weights)


def turn_retention_samples(facts: list[dict]) -> Samples:
    """Accepted overlap spans promoted into the request exactly once, frame ids intact.

    The span of an accepted interruption runs from the frame the level gate opened (the ``suspect``
    event) to the frame that confirmed it: every frame identifier in that span must be present
    exactly once, and the words heard must reach the request once, not twice and not not at all.
    """
    values = []
    for fact in facts:
        for event in fact["barge_ins"]:
            last = int(event["frame"])
            first = last
            for suspect in fact["suspects"]:
                if int(suspect["frame"]) <= last:
                    first = int(suspect["frame"])  # the most recent gate opening before this
                else:
                    break
            values.append(_span_promoted_once(fact, first, last, str(event.get("heard", ""))))
    return Samples(values, "mean")


def pair_accuracy_samples(facts: list[dict]) -> Samples:
    """Matched pairs: the same words said by Voxa alone and by the user over Voxa.

    The pair counts only when BOTH sides are decided right: Voxa alone must not create a command
    (that side is a Voxa-only clip), and the user saying it over her must be kept or yielded as
    the clip's ``expect`` says. An always-stop or never-stop policy fails one side of every pair.
    """
    by_pair: dict[str, dict[str, bool]] = {}
    for fact in facts:
        pair = fact["pair"]
        if not pair:
            continue
        side = str(pair.get("side", ""))
        correct = False
        if side == "voxa":
            correct = not fact["requests"]
        elif side == "user":
            expect = fact["expect"]
            kept = bool(fact["requests"])
            stopped = bool(fact["stops"])
            if expect == "keep":
                correct = kept
            elif expect == "yield":
                correct = stopped and kept
            elif expect == "backchannel":
                correct = not stopped
            else:  # expect "none": the user's words must not become a command of their own
                correct = not kept
        if side:
            by_pair.setdefault(str(pair.get("id", "")), {})[side] = correct
    return Samples([1.0 if len(sides) == 2 and all(sides.values()) else 0.0
                    for sides in by_pair.values()], "mean")


def unmeasurable(name: str) -> dict:
    """The report shape for a metric this stage cannot measure: null, and it never fails."""
    return {"value": None, "ci": [None, None], "samples": 0, "reason": name}


# name -> (unit, comparison direction, builder). The names are exactly the ones the shared
# brief fixes; ``report.py`` and ``gates.toml`` both read this table, so a name cannot drift.
METRICS: tuple[tuple[str, str, str, object], ...] = (
    ("erle_fst_median_db", "dB", ">=", lambda facts: Samples(erle_fst_samples(facts)["values"], "median")),
    ("erle_fst_p10_db", "dB", ">=", lambda facts: Samples(erle_fst_samples(facts)["values"], "p10")),
    ("nearend_atten_dt_db", "dB", "<=", nearend_attenuation_samples),
    ("align_err_p90_ms", "ms", "<=", None),
    ("fst_transcript_rate", "fraction", "<=", fst_transcript_samples),
    ("echo_leak_word_rate", "fraction", "<=", echo_leak_samples),
    ("false_bargein_per_hour", "events/hour", "<=", false_bargein_samples),
    ("missed_bargein_rate", "fraction", "<=", missed_bargein_samples),
    ("duck_latency_p50_ms", "ms", "<=", duck_latency_samples),
    ("duck_latency_p90_ms", "ms", "<=", lambda facts: Samples(duck_latency_samples(facts)["values"], "p90")),
    ("interrupt_stop_rate", "fraction", "<=", interrupt_stop_samples),
    ("backchannel_resume_rate", "fraction", ">=", backchannel_resume_samples),
    ("protected_word_keep_rate", "fraction", ">=", protected_word_samples),
    ("user_recall_nst", "fraction", ">=", user_recall_samples),
    ("dt_wer_penalty_pts", "points", "<=", dt_wer_penalty_samples),
    ("first_word_trunc_cold", "fraction", "<=", first_word_trunc_cold_samples),
    ("first_word_trunc_bargein", "fraction", "<=", first_word_trunc_bargein_samples),
    ("premature_cutoff_rate", "fraction", "<=", premature_cutoff_samples),
    ("ep50_ms", "ms", "<=", endpoint_latency_samples),
    ("ep90_ms", "ms", "<=", lambda facts: Samples(endpoint_latency_samples(facts)["values"], "p90")),
    ("delay_est_err_ms", "ms", "<=", delay_error_samples),
    ("drift_erle_loss_db", "dB", "<=", drift_erle_loss_samples),
    ("reconverge_s", "s", "<=", reconverge_samples),
    ("cpu_core_share", "cores", "<=", cpu_core_share_samples),
    ("erle_music_loss_db", "dB", ">=", None),
    ("self_command_rate", "fraction", "<=", self_command_samples),
    ("self_pause_per_hour", "events/hour", "<=", self_pause_samples),
    ("turn_retention_rate", "fraction", ">=", turn_retention_samples),
    ("user_word_deletion_rate", "fraction", "<=", user_word_deletion_samples),
    ("pair_accuracy", "fraction", ">=", pair_accuracy_samples),
    ("stop_latency_p50_ms", "ms", "<=", lambda facts: Samples(stop_latency_samples(facts)["values"], "p50")),
    ("stop_latency_p95_ms", "ms", "<=", stop_latency_samples),
    ("transport_drop_count", "count", "<=", None),
)

METRIC_NAMES = tuple(name for name, _unit, _direction, _builder in METRICS)

UNMEASURED = {
    "align_err_p90_ms": "the live reference-to-echo alignment (audio.py's probe) is roadmap task 1",
    "erle_music_loss_db": "the frozen B1 clip set has no music playback, only Voxa's own replies",
    "transport_drop_count": "the offline runner has no queue to drop; the live appsinks are task A4",
}


def compute(facts: list[dict], *, seed: int = 0, n: int = BOOTSTRAP_N) -> dict[str, dict]:
    """Every metric's value and 95 % bootstrap CI, plus the per-metric sample count."""
    out: dict[str, dict] = {}
    for name, unit, direction, builder in METRICS:
        if builder is None:
            out[name] = {"value": None, "ci": [None, None], "samples": 0, "unit": unit,
                         "direction": direction, "reason": UNMEASURED[name]}
            continue
        samples = builder(facts)
        measured = measure(samples, n=n, seed=seed)
        measured["unit"] = unit
        measured["direction"] = direction
        out[name] = measured
    return out
