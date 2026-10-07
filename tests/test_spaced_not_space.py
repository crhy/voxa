from voxa.agent.hearing import normalize
from voxa.vocabulary import build_hint


def test_spaced_product_names():
    assert "Spaced Linux" in normalize("update space linux")
    assert "Spaced Linux" in normalize("Update Space Lennox.")
    for heard in ("open space bazaar", "open spaced bizarre", "open spayed bazar"):
        assert "Spaced Bazaar" in normalize(heard)
    assert "Spaced Update" in normalize("open space update")
    assert "Spaced Store" in normalize("open the space store")
    assert normalize("update space").endswith("Spaced")


def test_ordinary_space_sentences_unchanged():
    for sentence in (
        "how much disk space do i have",
        "show me pictures of outer space",
        "play space invaders",
        "press space",
        "open space station simulator",
    ):
        assert normalize(sentence).lower() == sentence, normalize(sentence)


def test_hint_keeps_fixed_product_names():
    long_names = ["a" * 30 + str(i) for i in range(200)]
    for names in ([], long_names):
        hint = build_hint(names, "voxa")
        for product in ("Spaced Linux", "Spaced Update", "Spaced Bazaar"):
            assert product in hint
