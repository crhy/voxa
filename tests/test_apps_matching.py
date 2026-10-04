from voxa.apps import build_aliases, match_app, parse_desktop_dump, phonetic_key

DUMP = """@@@/usr/share/applications/org.gimp.GIMP.desktop
[Desktop Entry]
Type=Application
Name=GNU Image Manipulation Program
Exec=gimp-3.0 %U
@@@/usr/share/applications/org.libreoffice.LibreOffice.writer.desktop
[Desktop Entry]
Type=Application
Name=LibreOffice Writer
@@@/usr/share/applications/org.libreoffice.LibreOffice.calc.desktop
[Desktop Entry]
Type=Application
Name=LibreOffice Calc
@@@/usr/share/applications/io.github.crhy.SpacedBazaar.desktop
[Desktop Entry]
Type=Application
Name=Spaced Bazaar
@@@/usr/share/applications/com.anthropic.Claude.desktop
[Desktop Entry]
Type=Application
Name=Claude
@@@/usr/share/applications/com.obsproject.Studio.desktop
[Desktop Entry]
Type=Application
Name=OBS Studio
@@@/usr/share/applications/io.github.crhy.BrutalChess.desktop
[Desktop Entry]
Type=Application
Name=Brutal Chess
@@@/usr/share/applications/com.brave.Browser.desktop
[Desktop Entry]
Type=Application
Name=Brave
"""

APPS = parse_desktop_dump(DUMP)


def _name(query: str) -> str | None:
    app = match_app(query, APPS)
    return app.name if app else None


def test_fuzzy_names():
    assert _name("gimp") == "GNU Image Manipulation Program"
    assert _name("Gimp") == "GNU Image Manipulation Program"
    assert _name("the GAMP") == "GNU Image Manipulation Program"
    assert _name("GNU image program") == "GNU Image Manipulation Program"
    assert _name("Libra Office right") == "LibreOffice Writer"
    assert _name("libre office writer") == "LibreOffice Writer"
    assert _name("writer") == "LibreOffice Writer"
    assert _name("libre office calc") == "LibreOffice Calc"
    assert _name("Space Bizarre") == "Spaced Bazaar"
    assert _name("spaced bazar") == "Spaced Bazaar"
    assert _name("cloud") == "Claude"
    assert _name("claude") == "Claude"
    assert _name("OBS") == "OBS Studio"
    assert _name("obs studio") == "OBS Studio"
    assert _name("brutal chess") == "Brutal Chess"
    assert _name("brave") == "Brave"


def test_no_match():
    for query in ("", "a", "the weather today", "images of a pony", "a new document", "xyzzy"):
        assert match_app(query, APPS) is None, query


def test_phonetic_key():
    assert phonetic_key("gimp") == phonetic_key("gamp")
    assert phonetic_key("Claude") == phonetic_key("cloud")
    assert phonetic_key("bizarre") == phonetic_key("bazaar")


def test_build_aliases_gimp():
    gimp = next(app for app in APPS if app.name == "GNU Image Manipulation Program")
    assert gimp.aliases == ("gimp",)
    assert build_aliases("GNU Image Manipulation Program", "org.gimp.GIMP.desktop", {"Exec": "gimp-3.0 %U"}) == ("gimp",)
