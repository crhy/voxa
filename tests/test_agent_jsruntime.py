from __future__ import annotations

from voxa.agent.jsruntime import find_runtime, missing_runtime_message, ytdlp_options


def test_path_order_prefers_deno():
    which = {"deno": "/usr/bin/deno", "node": "/usr/bin/node", "bun": "/usr/bin/bun"}.get
    assert find_runtime(which=which) == ("deno", "/usr/bin/deno")


def test_path_order_node_before_bun():
    which = {"node": "/usr/local/bin/node", "bun": "/usr/bin/bun"}.get
    assert find_runtime(which=which) == ("node", "/usr/local/bin/node")


def test_none_found_on_path():
    assert find_runtime(which=lambda _name: None) is None


def test_flatpak_with_host_node_uses_wrapper():
    assert find_runtime(in_flatpak=True, host_has=lambda: "node") == ("node", "/app/bin/voxa-host-node")


def test_flatpak_host_deno_is_not_wrapped():
    assert find_runtime(in_flatpak=True, host_has=lambda: "deno") is None


def test_flatpak_host_has_nothing():
    assert find_runtime(in_flatpak=True, host_has=lambda: None) is None


def test_caches_for_the_life_of_the_process():
    calls = {"n": 0}

    def host_has():
        calls["n"] += 1
        return "node"

    first = find_runtime(in_flatpak=True, host_has=host_has)
    second = find_runtime(in_flatpak=True, host_has=host_has)
    assert first == second == ("node", "/app/bin/voxa-host-node")
    assert calls["n"] == 1


def test_options_shape(monkeypatch):
    monkeypatch.setattr("voxa.agent.jsruntime.find_runtime", lambda: ("node", "/usr/bin/node"))
    assert ytdlp_options() == {"js_runtimes": {"node": {"path": "/usr/bin/node"}}}


def test_options_empty_when_none_found(monkeypatch):
    monkeypatch.setattr("voxa.agent.jsruntime.find_runtime", lambda: None)
    assert ytdlp_options() == {}


def test_missing_runtime_message():
    assert missing_runtime_message() == (
        "Playing YouTube directly needs Node.js or Deno installed. Install one, or I'll use the browser instead."
    )


def test_manifest_installs_the_wrapper(monkeypatch):
    from pathlib import Path

    import yaml

    text = Path(__file__).resolve().parents[1] / "io.github.crhy.voxa.yml"
    raw = text.read_text()
    assert "install -Dm755 packaging/flatpak/voxa-host-node /app/bin/voxa-host-node" in raw
    yaml.safe_load(raw)
