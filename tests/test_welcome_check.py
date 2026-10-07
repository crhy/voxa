from types import SimpleNamespace

from voxa.installer import ollama_installed
from voxa.welcome import probe, real_models


class Recorder:
    def __init__(self):
        self.installed_calls = 0
        self.list_calls = 0

    def installed(self):
        self.installed_calls += 1
        return True

    def list_models(self):
        self.list_calls += 1
        return []


def test_probe_other_backend_calls_nothing():
    rec = Recorder()
    assert probe("llamacpp", rec.installed, rec.list_models) == "ready"
    assert (rec.installed_calls, rec.list_calls) == (0, 0)


def test_probe_not_installed_skips_list():
    rec = Recorder()

    def not_installed():
        rec.installed_calls += 1
        return False

    def boom():
        rec.list_calls += 1
        raise OSError("not up yet")

    assert probe("ollama", not_installed, boom) == "install"
    assert rec.list_calls == 0


def test_probe_recovers_after_two_failures():
    rec = Recorder()
    results = [OSError("down"), OSError("down"), ["qwen2.5:0.5b"]]

    def list_models():
        rec.list_calls += 1
        outcome = results[rec.list_calls - 1]
        if isinstance(outcome, list):
            return outcome
        raise outcome

    rec.installed = lambda: True
    assert probe("ollama", rec.installed, list_models, sleep=lambda s: None) == "ready"
    assert rec.list_calls == 3


def test_probe_installed_no_models():
    rec = Recorder()
    assert probe("ollama", rec.installed, rec.list_models, sleep=lambda s: None) == "model"
    assert rec.list_calls == 1


def test_probe_never_answers_uses_exactly_attempts_calls():
    rec = Recorder()

    def boom():
        rec.list_calls += 1
        raise OSError("not up yet")

    assert probe("ollama", rec.installed, boom, attempts=3, sleep=lambda s: None) == "model"
    assert rec.list_calls == 3


def test_real_models_filters_status_sentences():
    models = ["qwen2.5:0.5b", "Strata is not running", "No models available - Ollama"]
    assert real_models(models) == ["qwen2.5:0.5b"]


def test_ollama_installed_present():
    assert ollama_installed(runner=lambda *args, **kwargs: SimpleNamespace(returncode=0))


def test_ollama_installed_absent():
    assert not ollama_installed(runner=lambda *args, **kwargs: SimpleNamespace(returncode=1))


def test_ollama_installed_runner_raises():
    def boom(*args, **kwargs):
        raise OSError("no subprocess")

    assert not ollama_installed(runner=boom)
