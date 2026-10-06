from voxa import websearch

PAGE = """
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.steelers.com%2Fnews%2F&amp;rut=x">Steelers <b>News</b></a>
<a class="result__snippet" href="x">Latest &amp; greatest headlines</a>
<a class="result__a" href="https://example.com/two">Two</a>
"""


def test_parse_results_extracts_title_real_url_and_snippet() -> None:
    results = websearch.parse_results(PAGE)
    assert results[0] == websearch.SearchResult("Steelers News", "https://www.steelers.com/news/", "Latest & greatest headlines")
    assert results[1].url == "https://example.com/two" and results[1].snippet == ""


def test_search_query_only_for_current_or_explicit_questions() -> None:
    assert websearch.search_query_for("Search the web for Steelers schedule") == "Steelers schedule"
    assert websearch.search_query_for("what's the weather today?") == "what's the weather today"
    assert websearch.search_query_for("explain how a bubble sort works") is None
    assert websearch.search_query_for("") is None


def test_format_for_prompt_lists_numbered_sources() -> None:
    text = websearch.format_for_prompt("q", [websearch.SearchResult("T", "https://u", "S")])
    assert "1. T — S (https://u)" in text


def test_auto_mode_searches_named_things_and_fresh_facts() -> None:
    cases = [
        ("Tell me about Spaced Linux", True),
        ("Tell me about the Linux distribution, Spaced Linux", True),
        ("What is the current record of the Pittsburgh Steelers", True),
        ("Tell me about the musician Rye Thornton", True),
        ("Who is Ada Lovelace", True),
        ("What's the capital of Mongolia", True),
        ("Explain quantum computing", True),
        ("Describe the Mariana Trench", True),
        ("Do you know the capital of France", True),
        ("Have you heard of the band Tool", True),
        ("When was the Constitution signed", True),
        ("Where is the Grand Canyon", True),
        ("How tall is the Burj Khalifa", True),
        ("Which planet is closest to the Sun", True),
        ("What is the population of Tokyo", True),
        ("History of the Ottoman Empire", True),
        ("Info on red pandas", True),
        ("What is the weather like in Paris", True),
        ("What is the price of gold", True),
        ("Who won the World Series last year", True),
        ("What is the latest version of Python", True),
        ("How old are the pyramids of Giza", True),
        ("how are you", False),
        ("thank you", False),
        ("good morning", False),
        ("what can you do", False),
        ("what's your name", False),
        ("who are you", False),
        ("what is 12 times 9", False),
        ("what is love", False),
        ("write a poem about the sea", False),
        ("tell me a joke", False),
        ("tell me a story", False),
        ("explain how a bubble sort works", False),
        ("hello", False),
        ("nice one", False),
        ("which one do you want", False),
    ]
    for prompt, expected in cases:
        found = websearch.search_query_for(prompt) is not None
        assert found is expected, prompt


def test_followup_searches_with_previous_query() -> None:
    query = websearch.search_query_for(
        "How many wins and losses do they have", previous_query="Pittsburgh Steelers record"
    )
    assert query == "Pittsburgh Steelers record How many wins and losses do they have"


def test_never_mode_never_searches() -> None:
    assert websearch.search_query_for("Tell me about Spaced Linux", mode="never") is None
    assert websearch.search_query_for("what is the weather today?", mode="never") is None


def test_always_mode_searches_questions_and_requests() -> None:
    assert websearch.search_query_for("what is love", mode="always") == "what is love"
    assert websearch.search_query_for("hello", mode="always") is None


def test_query_cleaning_removes_request_words() -> None:
    query = websearch.search_query_for("Tell me about the Linux distribution, Spaced Linux")
    assert "Spaced Linux" in query
    assert not query.lower().startswith("tell me")


def test_query_is_capped_at_twelve_words() -> None:
    query = websearch.search_query_for(
        "Tell me about Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu Nu Xi"
    )
    assert len(query.split()) == 12


def test_format_for_prompt_forces_answers_from_results() -> None:
    text = websearch.format_for_prompt("q", [websearch.SearchResult("T", "https://u", "S")])
    assert "Answer ONLY from these search results." in text
    assert "Do not add facts from memory." in text
