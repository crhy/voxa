"""Issue #72, task v08 step 5: the picker may never swap a saved model for models[0].

The live v07 run saved ``qwen3.5-0.8b-q4km`` (untagged) and the picker's tagged list
made it look gone: ``_apply_ollama_models`` replaced it with ``models[0]`` - the 11.7 GB
27B that way. An untagged saved name IS its ``:latest`` tag (Ollama's own default), and
window.py's fallback goes through ``voxa.ollama.installed_choice``.
"""

from voxa.ollama import installed_choice


def test_an_exact_hit_is_left_alone() -> None:
    assert installed_choice("qwen3.5-0.8b-q4km:latest", ["qwen3.5-0.8b-q4km:latest", "qwen2.5:7b"]) == \
        "qwen3.5-0.8b-q4km:latest"


def test_an_untagged_saved_name_matches_its_latest_tag() -> None:
    """"qwen3.5-0.8b-q4km" was never gone: ollama lists it as "...:latest"."""
    assert installed_choice("qwen3.5-0.8b-q4km", ["qwen3.5-0.8b-q4km:latest", "qwen2.5:7b"]) == \
        "qwen3.5-0.8b-q4km:latest"


def test_a_model_that_has_gone_is_not_matched() -> None:
    assert installed_choice("somebrain", ["otherbrain:latest"]) == ""
    assert installed_choice("", ["anything:latest"]) == ""


def test_a_tagged_saved_name_does_not_match_a_different_tag() -> None:
    """``x:q4km`` gone while ``x:latest`` remains is a real change: the caller must fall
    back, but only after guessing would end the ambiguity - no partial-name matching."""
    assert installed_choice("qwen3.5-0.8b-q4km:q4km", ["qwen3.5-0.8b-q4km:latest"]) == ""
