from voxa.agent.intents import route


def test_image_search_phrasings():
    cases = {
        "look for sexy pictures": "sexy",
        "find me some cat photos": "cat",
        "show me vintage car images": "vintage car",
        "pull up more golden retriever pics": "golden retriever",
        "find me space wallpapers": "space wallpapers",
        "show me pictures of cats": "cats",
    }
    for text, query in cases.items():
        call = route(text)
        assert call is not None and call.tool == "image_search"
        assert call.args["query"] == query


def test_bare_image_of_phrase():
    call = route("pictures of the Eiffel tower")
    assert call is not None and call.tool == "image_search"
    assert "Eiffel tower" in call.args["query"]


def test_own_file_phrasings_are_not_image_search():
    for text in (
        "show me my pictures",
        "find my photos",
        "open my pictures",
        "take a picture",
        "what's in my pictures",
    ):
        call = route(text)
        assert call is None or call.tool != "image_search"
