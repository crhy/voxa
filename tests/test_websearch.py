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
