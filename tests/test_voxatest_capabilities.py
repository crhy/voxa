from voxatest.capabilities import Capability, format_html, format_markdown, load_capabilities, summary, tally


def test_the_hundred_capabilities_are_shipped():
    capabilities = load_capabilities()
    assert [capability.n for capability in capabilities] == list(range(1, 101))
    assert len({capability.category for capability in capabilities}) == 10


def test_status_comes_from_each_cases_recent_runs():
    working = Capability(81, "Computer", "Launch apps", "", cases={"a": [False] + [True] * 5, "b": [True]})
    partial = Capability(24, "Tasks", "Set reminders", "", cases={"r": [True, False]})
    locked = Capability(6, "Email", "Handle phone calls", "", needs_user="a phone line")
    planned = Capability(21, "Tasks", "To-do lists", "")
    assert (working.status, working.rate) == ("working", 1.0)       # the old failure has aged out
    assert (partial.status, partial.rate) == ("partial", 0.5)
    assert (locked.status, planned.status) == ("needs you", "planned")
    info = summary([working, partial, locked, planned])
    assert info["counts"] == {"working": 1, "partial": 1, "planned": 1, "needs you": 1}
    assert (info["passed"], info["runs"]) == (7, 8)


def test_reports_are_tallied_by_capability_number_or_known_case_id():
    capabilities = tally(load_capabilities(), [
        {"results": [{"case_id": "x", "capability": 81, "passed": True}, {"case_id": "desktop-01", "passed": False}]},
    ])
    launch = next(capability for capability in capabilities if capability.n == 81)
    assert launch.cases == {"x": [True], "desktop-01": [False]}


def test_markdown_and_html_carry_the_charts():
    capabilities = tally(load_capabilities(), [{"results": [{"case_id": "x", "capability": 81, "passed": True}]}])
    markdown = format_markdown(capabilities, "today")
    assert "pie showData" in markdown and "xychart-beta" in markdown and "| 81 | Launch and control applications | ✅ working" in markdown
    page = format_html(capabilities, "today")
    assert "conic-gradient" in page and page.count('class="cell"') == 100
