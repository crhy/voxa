from __future__ import annotations

from voxa.websearch import SearchResult, format_for_prompt, search_record_fields, source_line


def _result(title: str, url: str, snippet: str) -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet)


def test_format_for_prompt_order():
    results = [
        _result("Bitcoin price", "https://www.example.com/a", "x" * 400),
        _result("Second", "https://coins.site/b", "short"),
    ]
    block = format_for_prompt("what is the price", results)
    lines = block.split("\n")
    assert lines[0].startswith("Answer ONLY from these search results.")
    assert lines[1].startswith("1. Bitcoin price")
    assert lines[2].startswith("2. Second")
    assert lines[3] == "Today is 6 October 2026."
    assert lines[4] == "Question: what is the price"
    assert "x" * 301 not in block
    assert "x" * 300 in block


def test_format_for_prompt_caps_results():
    results = [_result(f"t{i}", "https://s.com", "s") for i in range(8)]
    block = format_for_prompt("q", results)
    assert block.count("\n1. ") + block.count("1. t0") >= 1
    numbered = [ln for ln in block.split("\n") if ln[:2].rstrip(".").isdigit() and ". " in ln]
    assert len(numbered) == 5


def test_source_line():
    assert source_line([]) == ""
    assert source_line([_result("t", "https://www.reuters.com/x", "s")]) == "Source: reuters.com"


def test_search_record_fields_search():
    fields = search_record_fields("bitcoin price", [_result("Bitcoin " + "y" * 100, "https://a.com", "s")], False)
    assert fields["route"] == "search"
    assert fields["args"] == {"query": "bitcoin price"}
    assert fields["detail"].startswith("1 results: Bitcoin ")
    assert len(fields["detail"].split(": ", 1)[1]) == 80


def test_search_record_fields_failed():
    fields = search_record_fields("q", [], True)
    assert fields["route"] == "model"
    assert fields["detail"] == "web search failed"


def test_search_record_fields_nothing():
    fields = search_record_fields("q", [], False)
    assert fields["route"] == "model"
    assert fields["detail"] == "web search found nothing"
