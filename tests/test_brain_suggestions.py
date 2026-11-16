"""voxa#130: the llama.cpp backend must offer hardware-ranked GGUF brains.

The Ollama catalog already ranked its tags against the machine; the GGUF ladder
had no ranking at all, so a llama.cpp user got an empty suggestion list. These
tests cover the pure layer: the pinned ladder, the headroom rule shared with
hardware.py, and the rule that a llama.cpp machine is never offered an Ollama
tag.
"""

from __future__ import annotations

import re

from voxa.hardware import (
    FACE_SERVER_VRAM_GB,
    MODEL_CATALOG,
    WHISPER_VRAM_GB,
    ModelSuggestion,
    _fits,
    reserved_gpu_gb,
)
from voxa.smallbrain import (
    BRAINS,
    LIGHTEST_BRAIN,
    SMALL_BRAIN,
    ModelFile,
    approx_gb,
    brain_suggestions,
    fit_note,
    fits_machine,
)

SHA_RE = re.compile(r"\A[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")

# Read off Hugging Face's own tree API for Qwen/Qwen2.5-1.5B-Instruct-GGUF.
MEDIUM_BYTES = 1117320736
MEDIUM_SHA = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
MEDIUM_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    "/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9"
    "/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)


def rung(bytes_size: int) -> ModelFile:
    """A ladder rung of exactly ``bytes_size`` bytes, for boundary tests."""
    return ModelFile("probe", "r/r", "0" * 40, "probe.gguf", bytes_size, "0" * 64)


def by_size(bytes_size: int) -> ModelFile:
    return next(model for model in BRAINS if model.bytes == bytes_size)


def test_ladder_sizes_strictly_increase() -> None:
    sizes = [model.bytes for model in BRAINS]
    assert all(smaller < bigger for smaller, bigger in zip(sizes, sizes[1:], strict=False))


def test_every_rung_is_pinned_with_a_real_size_and_a_distinct_sha256() -> None:
    for model in BRAINS:
        assert model.bytes > 0
        assert SHA_RE.match(model.sha256), model.label
        assert REVISION_RE.match(model.revision), model.label
        assert model.filename.endswith(".gguf")
    assert len({model.sha256 for model in BRAINS}) == len(BRAINS)
    assert len({model.filename for model in BRAINS}) == len(BRAINS)


def test_pinned_rungs_keep_the_digests_hugging_face_reported() -> None:
    # A changed digit here means a brain whose download fails its own checksum.
    assert by_size(MEDIUM_BYTES).sha256 == MEDIUM_SHA
    assert by_size(MEDIUM_BYTES).url == MEDIUM_URL


def test_suggestions_rank_the_largest_that_fits_first() -> None:
    # 6 GB clears the 2.55 GB rung (2.55 * 1.3 + 1 = 4.32) but not the 5.29 GB
    # one (7.88), so the large brain leads and the largest is not offered.
    assert brain_suggestions(6.0) == [
        by_size(2740937888),
        by_size(MEDIUM_BYTES),
        SMALL_BRAIN,
    ]


def test_suggestions_come_back_best_first_at_every_size() -> None:
    for available in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        sizes = [model.bytes for model in brain_suggestions(available)]
        assert sizes == sorted(sizes, reverse=True), available


def test_suggestions_never_exceed_the_limit() -> None:
    assert len(brain_suggestions(64.0)) == 3
    assert [model.bytes for model in brain_suggestions(64.0, limit=2)] == [
        model.bytes for model in brain_suggestions(64.0)[:2]
    ]
    assert len(brain_suggestions(64.0, limit=1)) == 1


def test_fits_machine_uses_the_same_headroom_rule_as_the_ollama_catalog() -> None:
    for model in BRAINS:
        twin = ModelSuggestion(model.label, approx_gb(model), "twin")
        for available in (0.5, 1.6, 2.4, 4.4, 8.0, 24.0):
            assert fits_machine(model, available) == _fits(available, twin)


def test_fits_machine_boundary_is_the_rule_not_a_round_number() -> None:
    model = rung(int(2.0 * 1024**3))
    threshold = approx_gb(model) * 1.3 + 1.0
    assert fits_machine(model, threshold)
    assert not fits_machine(model, threshold - 0.01)


def test_reserved_vram_for_the_live_face_pushes_a_rung_out() -> None:
    reserved = reserved_gpu_gb("live", True)
    assert reserved == FACE_SERVER_VRAM_GB + WHISPER_VRAM_GB
    roomy = [model.bytes for model in brain_suggestions(12.0)]
    assert roomy[0] == 5680522464
    kept_free = [model.bytes for model in brain_suggestions(12.0, reserved_gb=reserved)]
    assert 5680522464 not in kept_free
    assert kept_free == [model.bytes for model in brain_suggestions(12.0 - reserved)]


def test_reserved_vram_beyond_the_machine_still_offers_the_lightest_brain() -> None:
    assert brain_suggestions(4.0, reserved_gb=10.0) == [LIGHTEST_BRAIN]


def test_a_machine_with_nothing_fitting_still_gets_a_suggestion() -> None:
    assert brain_suggestions(0.0) == [LIGHTEST_BRAIN]
    assert brain_suggestions(0.1, reserved_gb=0.5) == [LIGHTEST_BRAIN]


def test_a_machine_without_a_graphics_card_is_still_offered_brains() -> None:
    # hardware.detect_available_model_memory_gb() falls back to system RAM,
    # conservatively 4 GB when it cannot even read /proc/meminfo.
    suggestions = brain_suggestions(4.0)
    assert suggestions
    assert LIGHTEST_BRAIN in suggestions
    assert all(fits_machine(model, 4.0) for model in suggestions)


def test_no_ollama_tag_is_ever_offered_to_the_llamacpp_backend() -> None:
    tags = {entry.name for entry in MODEL_CATALOG}
    for available in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0):
        for reserved in (0.0, reserved_gpu_gb("live", True)):
            suggestions = brain_suggestions(available, reserved_gb=reserved)
            assert suggestions, f"empty list at {available} GB"
            for model in suggestions:
                assert isinstance(model, ModelFile)
                assert model in BRAINS
                assert ":" not in model.label
                assert model.label not in tags
                assert model.filename.endswith(".gguf")


def test_only_brains_that_fit_are_returned() -> None:
    for available in (2.0, 4.0, 8.0, 16.0, 32.0):
        fitting = [model for model in BRAINS if fits_machine(model, available)]
        assert fitting, available
        assert brain_suggestions(available) == list(reversed(fitting))[:3]


def test_fit_note_marks_each_rung_for_the_brains_page() -> None:
    biggest = by_size(5680522464)
    assert fit_note(LIGHTEST_BRAIN, 8.0) == "Fits this machine"
    assert fit_note(biggest, 7.0) == "Too big for this machine (~7 GB)"
    # The High face and Whisper on the card make the same rung too big.
    assert fit_note(biggest, 12.0, reserved_gpu_gb("live", True)) == "Too big for this machine (~5 GB)"
