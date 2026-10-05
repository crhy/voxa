from __future__ import annotations

from voxa.agent.youtube import WALL_PHRASES, embed_url, play, player_state, video_id, wait_until_playing, watch_url

ID = "dQw4w9WgXcQ"


def _not_playing():
    return {"wall": False, "has_video": True, "playing": False, "time": 0.0, "error": ""}


def _playing():
    return {"wall": False, "has_video": True, "playing": True, "time": 1.0, "error": ""}


def _wall():
    return {"wall": True, "has_video": False, "playing": False, "time": 0.0, "error": ""}


class FakeSession:
    """A stand-in browser session that replays a scripted list of player states."""

    def __init__(self, states, start_play=True):
        self._states = list(states)
        self._start_play = start_play
        self._goto_urls: list[str] = []
        self._calls: list[tuple[str, dict]] = []
        self._player_evals = 0

    def goto(self, url, wait=15.0):
        self._goto_urls.append(url)

    def _call(self, method, params=None, timeout=None):
        self._calls.append((method, params or {}))
        return {}

    def evaluate(self, expression, timeout=10.0):
        if "has_video" in expression:
            self._player_evals += 1
            return self._states.pop(0) if self._states else _not_playing()
        return self._start_play


def test_video_id_bare_and_urls():
    assert video_id(ID) == ID
    assert video_id("  dQw4w9WgXcQ  ") == ID
    assert video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL") == ID
    assert video_id("https://youtu.be/dQw4w9WgXcQ?t=5") == ID
    assert video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == ID
    assert video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == ID
    assert video_id("https://www.youtube.com/v/dQw4w9WgXcQ") == ID


def test_video_id_rejects_bad():
    assert video_id("too-short") is None
    assert video_id("way_too_long_id_here") is None
    assert video_id("has spaces dQw4w9WgXcQ") is None
    assert video_id("") is None
    assert video_id("https://example.com/nothing") is None


def test_watch_and_embed_urls():
    assert watch_url(ID) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert embed_url(ID) == "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?autoplay=1"


def test_player_state_normalizes():
    session = FakeSession([{"wall": True, "playing": True}])
    state = player_state(session)
    assert state == {"wall": True, "has_video": False, "playing": True, "time": 0.0, "error": ""}


def test_player_state_handles_non_dict():
    session = FakeSession([None])
    state = player_state(session)
    assert state == {"wall": False, "has_video": False, "playing": False, "time": 0.0, "error": ""}


def test_wait_until_playing_stops_on_playing():
    session = FakeSession([_not_playing(), _not_playing(), _playing()])
    state = wait_until_playing(session, timeout=12.0, poll=0.5, sleep=lambda s: None)
    assert state["playing"]
    assert session._player_evals == 3


def test_wait_until_playing_stops_on_wall():
    session = FakeSession([_not_playing(), _wall()])
    state = wait_until_playing(session, timeout=12.0, poll=0.5, sleep=lambda s: None)
    assert state["wall"]
    assert not state["playing"]
    assert session._player_evals == 2


def test_wait_until_playing_times_out():
    session = FakeSession([_not_playing()])
    state = wait_until_playing(session, timeout=1.0, poll=0.5, sleep=lambda s: None)
    assert not state["playing"]
    assert not state["wall"]
    assert session._player_evals == 3


def test_play_watch_plays():
    session = FakeSession([_playing()])
    ok, how, url = play(ID, session)
    assert (ok, how, url) == (True, "watch", watch_url(ID))
    assert session._goto_urls == [watch_url(ID)]


def test_play_reload_recovers():
    session = FakeSession([_wall(), _playing()])
    ok, how, url = play(ID, session)
    assert (ok, how, url) == (True, "reload", watch_url(ID))
    assert session._goto_urls == [watch_url(ID), watch_url(ID)]


def test_play_embed_fallback():
    session = FakeSession([_wall(), _wall(), _playing()])
    ok, how, url = play(ID, session)
    assert (ok, how, url) == (True, "embed", embed_url(ID))
    assert session._goto_urls == [watch_url(ID), watch_url(ID), embed_url(ID)]


def test_play_gives_up():
    session = FakeSession([_wall(), _wall(), _wall()])
    ok, how, url = play(ID, session)
    assert (ok, how, url) == (False, "none", watch_url(ID))
    assert session._goto_urls == [watch_url(ID), watch_url(ID), embed_url(ID)]


def test_play_resolves_query_when_no_id():
    calls = []

    def resolve(query):
        calls.append(query)
        return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    session = FakeSession([_playing()])
    ok, how, url = play("lo-fi beats", session, resolve=resolve)
    assert (ok, how, url) == (True, "watch", watch_url(ID))
    assert calls == ["lo-fi beats"]


def test_play_no_resolve_no_id_gives_up():
    calls = []

    def resolve(query):
        calls.append(query)
        return "https://www.youtube.com/results?search_query=lo-fi"

    session = FakeSession([_wall()])
    ok, how, url = play("lo-fi beats", session, resolve=resolve)
    assert (ok, how, url) == (False, "none", "https://www.youtube.com/results?search_query=lo-fi")
    assert calls == ["lo-fi beats"]
    assert session._goto_urls == []


def test_wall_phrases_present():
    assert "ad blockers are not allowed" in WALL_PHRASES
    assert "it looks like you may be using an ad blocker" in WALL_PHRASES
