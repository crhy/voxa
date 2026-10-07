import json

try:
    import issueflow as F
except ImportError:
    from voxa.agent import issueflow as F

OWNER_REPOS = [
    {"full_name": "crhy/spaced", "name": "spaced", "description": "Spaced Linux"},
    {"full_name": "crhy/spaced-apt", "name": "spaced-apt", "description": "Spaced Linux apt repository (GitHub Pages)"},
    {"full_name": "crhy/spacedupdate", "name": "spacedupdate", "description": "The Update App for SpacedLinux.com"},
    {"full_name": "crhy/rhYciv", "name": "rhYciv", "description": "An open-source re-implementation of Civilization 2"},
    {"full_name": "crhy/CardsWithCats", "name": "CardsWithCats", "description": None},
]


def test_pick_repo():
    assert F.pick_repo("spaced linux", OWNER_REPOS) == "crhy/spaced"
    assert F.pick_repo("spaced", OWNER_REPOS) == "crhy/spaced"
    assert F.pick_repo("spaced update", OWNER_REPOS) == "crhy/spacedupdate"
    assert F.pick_repo("cards with cats", OWNER_REPOS) == "crhy/CardsWithCats"
    assert F.pick_repo("rhyciv", OWNER_REPOS) == "crhy/rhYciv"
    assert F.pick_repo("civilization", OWNER_REPOS) == "crhy/rhYciv"
    assert F.pick_repo("photoshop", OWNER_REPOS) is None


def test_find_repo_owner_first():
    seen = []

    def fake(url):
        seen.append(url)
        if "/users/crhy/repos" in url:
            return json.dumps(OWNER_REPOS)
        return '{"items": [{"full_name": "jeromeollivon-star/spacedesk-linux", "name": "spacedesk-linux"}]}'

    assert F.find_repo("spaced linux", fetch=fake, owner="crhy") == "crhy/spaced"
    assert not any("search/repositories" in url for url in seen)


def test_find_repo_search_not_exact():
    def fake(url):
        return '{"items": [{"full_name": "x/spacedesk-linux", "name": "spacedesk-linux"}]}'

    assert F.find_repo("spaced linux", fetch=fake, owner="") is None


def test_find_repo_search_exact():
    def fake(url):
        return '{"items": [{"full_name": "torvalds/linux", "name": "linux"}]}'

    assert F.find_repo("linux", fetch=fake, owner="") == "torvalds/linux"


def test_find_repo_fetch_raises():
    def boom(url):
        raise RuntimeError("net")

    assert F.find_repo("nope", fetch=boom) is None


class Done:
    def __init__(self, returncode, stdout):
        self.returncode = returncode
        self.stdout = stdout


def test_detect_owner_gh():
    def runner(command, **kwargs):
        return Done(0, "crhy\n")

    assert F.detect_owner(runner=runner) == "crhy"


def test_detect_owner_git_fallback():
    def runner(command, **kwargs):
        if command[0] == "gh":
            return Done(1, "")
        return Done(0, "someone")

    assert F.detect_owner(runner=runner) == "someone"


def test_detect_owner_none():
    def runner(command, **kwargs):
        return Done(1, "")

    assert F.detect_owner(runner=runner) == ""


def test_detect_owner_bad_answer():
    def runner(command, **kwargs):
        return Done(0, "cr hy")

    assert F.detect_owner(runner=runner) == ""

    def runner2(command, **kwargs):
        return Done(0, "crhy/voxa")

    assert F.detect_owner(runner=runner2) == ""
