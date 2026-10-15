from __future__ import annotations

import urllib.error

import pytest

from voxa.agent.tools import recite

RAVEN = {"title": "The Raven", "author": "Edgar Allan Poe",
         "lines": ["Once upon a midnight dreary, while I pondered, weak and weary,", "Quoth the Raven “Nevermore.”"]}
LEE = {"title": "Annabel Lee", "author": "Edgar Allan Poe", "lines": ["It was many and many a year ago,"]}
OTHER = {"title": "Poem", "author": "Someone Poet", "lines": ["x"]}


@pytest.fixture
def library(monkeypatch):
    asked = []

    def fetch(path):
        asked.append(path)
        if path.startswith("/author/edgar%20allen%20poe/"):
            return []  # PoetryDB does not know the misspelling
        if path.startswith("/author/poe/"):
            return [RAVEN, LEE, OTHER]
        if path.startswith("/title/the%20raven/"):
            return [RAVEN, OTHER]
        return []

    monkeypatch.setattr(recite, "fetch", fetch)
    monkeypatch.setattr(recite.random, "choice", lambda poems: poems[0])
    return asked


def test_misspelled_poet_falls_back_to_the_surname(library):
    result = recite.recite_poem({"author": "edgar allen poe"})
    assert result.ok
    assert result.speech.startswith("The Raven, by Edgar Allan Poe. Once upon a midnight dreary")
    assert library == ["/author/edgar%20allen%20poe/title,author,lines", "/author/poe/title,author,lines"]


def test_whole_poem_is_spoken_with_pauses_between_lines(library):
    speech = recite.recite_poem({"title": "the raven", "author": "poe"}).speech
    assert speech.endswith("weak and weary, Quoth the Raven “Nevermore.”")


def test_other_poets_with_matching_names_are_filtered_out(library):
    poems = recite.find_poems("edgar allan poe")
    assert OTHER not in poems and len(poems) == 2


def test_nothing_found_and_no_poet(library):
    assert "could not find a poem by zzq" in recite.recite_poem({"author": "zzq"}).speech
    assert not recite.recite_poem({}).ok


def test_network_failure_is_explained(monkeypatch):
    def broken(path):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(recite, "fetch", broken)
    result = recite.recite_poem({"author": "poe"})
    assert not result.ok and "could not reach the poetry library" in result.speech
