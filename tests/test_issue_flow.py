try:
    import issueflow as F
except ImportError:
    from voxa.agent import issueflow as F


def test_parse_matches():
    assert F.parse_post_issue("post an issue to github for spaced linux") == "spaced linux"
    assert F.parse_post_issue("post an issue on github") == ""
    assert F.parse_post_issue("post a github issue for voxa") == "voxa"
    assert F.parse_post_issue("open an issue for voxa on github") == "voxa"
    assert F.parse_post_issue("file an issue for spaced linux") == "spaced linux"
    assert F.parse_post_issue("create a github issue") == ""
    assert F.parse_post_issue("report a bug in voxa") == "voxa"
    assert F.parse_post_issue("report a bug for voxa") == "voxa"
    assert F.parse_post_issue("report a bug") == ""
    assert F.parse_post_issue("Make a github issue for the Voxa!") == "voxa"


def test_parse_nonmatches():
    assert F.parse_post_issue("what is a github issue") is None
    assert F.parse_post_issue("open github") is None
    assert F.parse_post_issue("post this to github") is None


def test_find_repo_known():
    assert F.find_repo("") == "crhy/voxa"
    assert F.find_repo("Voxa") == "crhy/voxa"
    assert F.find_repo("crhy/thing") == "crhy/thing"


def test_find_repo_fetch():
    seen = []

    def fake(url):
        seen.append(url)
        return '{"items": [{"full_name": "spacedlinux/spaced", "name": "spaced linux"}]}'

    assert F.find_repo("spaced linux", fetch=fake) == "spacedlinux/spaced"
    assert "spaced%20linux" in seen[0]


def test_find_repo_fetch_errors():
    def boom(url):
        raise RuntimeError("net")

    assert F.find_repo("nope", fetch=boom) is None
    assert F.find_repo("nope", fetch=lambda url: "{}") is None


def test_issue_url_quoting():
    u = F.issue_url("o/r", "a b&c#d", "x\ny")
    assert "title=a%20b%26c%23d" in u
    assert "body=x%0Ay" in u


def test_issue_url_cut():
    u = F.issue_url("o/r", "t", "a" * 7000)
    assert "a" * 6000 in u
    assert "a" * 6001 not in u


def test_full_flow():
    flow = F.IssueFlow("crhy/voxa")
    assert flow.caption == "Issue title"
    say, url = flow.feed("login broken", None)
    assert say == F.ASK_BODY and url is None
    assert flow.caption == "Issue description"
    assert flow.feed("The app crashes on start", None) == ("", None)
    assert flow.feed("It shows a blank screen", None) == ("", None)
    assert flow.feed("oops", "undo") == ("Removed.", None)
    assert flow.feed("Now it hangs", None) == ("", None)
    say, url = flow.feed("stop dictation", "stop")
    assert say == "Opening the issue for crhy/voxa. Check it and press Submit."
    assert url == F.issue_url("crhy/voxa", "Login broken", "The app crashes on start. Now it hangs.")
    assert not flow.active
    assert flow.caption == ""
    assert flow.feed("anything", None) == ("", None)


def test_cancel_at_title():
    flow = F.IssueFlow("crhy/voxa")
    say, url = flow.feed("cancel that", None)
    assert say == F.CANCELLED and url is None
    assert not flow.active


def test_stop_at_title_cancels():
    flow = F.IssueFlow("crhy/voxa")
    say, url = flow.feed("whatever", "stop")
    assert say == F.CANCELLED and url is None
    assert not flow.active


def test_undo_empty_parts():
    flow = F.IssueFlow("crhy/voxa")
    flow.feed("t", None)
    assert flow.feed("x", "undo") == ("Nothing to remove.", None)


def test_caption_per_step():
    flow = F.IssueFlow("crhy/voxa")
    assert flow.caption == "Issue title"
    flow.feed("t", None)
    assert flow.caption == "Issue description"
    flow.feed("done", "stop")
    assert flow.caption == ""
