"""Play a YouTube video in the controlled browser, falling back when blocked."""

from __future__ import annotations

import json
import re
import time

WALL_PHRASES = (
    "ad blockers are not allowed",
    "ad blockers violate",
    "video player will be blocked",
    "allow youtube ads",
    "it looks like you may be using an ad blocker",
)

_ID = r"[A-Za-z0-9_-]{11}"
_BARE_ID_RE = re.compile(rf"^{_ID}$")
_WATCH_RE = re.compile(rf"[?&]v=({_ID})")
_SHORT_RE = re.compile(rf"youtu\.be/({_ID})")
_PATH_RE = re.compile(rf"/(?:embed|shorts|v)/({_ID})")

PLAYER_STATE_JS = """
(() => {
  const phrases = __PHRASES__;
  const body = document.body;
  const text = (body ? body.innerText : '').toLowerCase();
  let wall = false;
  for (const p of phrases) { if (text.includes(p)) { wall = true; break; } }
  const video = document.querySelector('video');
  const errEl = document.querySelector('.ytp-error');
  const error = errEl ? (errEl.innerText || '').trim() : '';
  return {
    wall: wall,
    has_video: video ? true : false,
    playing: (video && !video.paused && video.readyState >= 3 && video.currentTime > 0) ? true : false,
    time: video ? video.currentTime : 0,
    error: error
  };
})()
""".replace("__PHRASES__", json.dumps(list(WALL_PHRASES)))

START_PLAY_JS = """
(() => {
  const video = document.querySelector('video');
  if (video) { try { video.play(); } catch (e) {} }
  const nodes = document.querySelectorAll('button, [role=button]');
  for (const el of nodes) {
    if (el.hasAttribute('hidden') || el.hasAttribute('disabled')) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const label = ((el.getAttribute('aria-label') || '') + ' ' + (el.innerText || '')).trim().toLowerCase();
    if (label.includes('play')) { el.click(); break; }
  }
  return true;
})()
"""


def video_id(url_or_id: str) -> str | None:
    """Pull an 11-char video id out of a watch/short/embed/youtu.be URL or a bare id."""
    text = url_or_id.strip()
    if _BARE_ID_RE.match(text):
        return text
    for pattern in (_WATCH_RE, _SHORT_RE, _PATH_RE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def watch_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def embed_url(vid: str) -> str:
    return f"https://www.youtube-nocookie.com/embed/{vid}?autoplay=1"


def player_state(session) -> dict:
    """Read the player state from the page through one fixed snippet."""
    raw = session.evaluate(PLAYER_STATE_JS)
    if not isinstance(raw, dict):
        raw = {}
    return {
        "wall": bool(raw.get("wall")),
        "has_video": bool(raw.get("has_video")),
        "playing": bool(raw.get("playing")),
        "time": float(raw.get("time") or 0.0),
        "error": str(raw.get("error") or ""),
    }


def start_playback(session) -> None:
    """Start the video: click the play button / call play(), then press 'k' as a last resort."""
    session.evaluate(START_PLAY_JS)
    for event in ("rawKeyDown", "char", "keyUp"):
        params = {"type": event, "key": "k", "code": "KeyK",
                  "windowsVirtualKeyCode": 75, "nativeVirtualKeyCode": 75}
        if event == "char":
            params = {"type": "char", "text": "k"}
        session._call("Input.dispatchKeyEvent", params)


def wait_until_playing(session, timeout: float = 12.0, poll: float = 0.5, sleep=time.sleep) -> dict:
    """Poll player_state until playing or a wall shows; return the last state."""
    state = player_state(session)
    polls = int(timeout / poll) if poll > 0 else 0
    for _ in range(polls):
        if state["playing"] or state["wall"]:
            return state
        sleep(poll)
        state = player_state(session)
    return state


def play(query_or_url: str, session, resolve=None, log=None) -> tuple[bool, str, str]:
    """Try watch, reload, embed in turn; return (ok, how, url) for the first that plays."""
    if resolve is None:
        from voxa.agent.tools.browser import youtube_play_url as resolve

    vid = video_id(query_or_url)
    if vid is None:
        resolved = resolve(query_or_url)
        vid = video_id(resolved)
        if vid is None:
            return (False, "none", resolved)

    def attempt(label: str, target: str, start: bool) -> dict:
        session.goto(target)
        if start:
            start_playback(session)
        state = wait_until_playing(session)
        if log:
            log(label, state)
        return state

    url = watch_url(vid)
    state = attempt("watch", url, True)
    if state["playing"]:
        return (True, "watch", url)

    state = attempt("reload", url, False)
    if state["playing"]:
        return (True, "reload", url)

    embed = embed_url(vid)
    state = attempt("embed", embed, True)
    if state["playing"]:
        return (True, "embed", embed)

    return (False, "none", url)
