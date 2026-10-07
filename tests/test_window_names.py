import types

from voxa.agent.tools import windows as windows_mod


def mate_terminal(name="Terminal", path="/usr/share/applications/mate-terminal.desktop"):
    return types.SimpleNamespace(name=name, path=path, keywords="", aliases=())


FAKE_APPS = [
    mate_terminal(),
    types.SimpleNamespace(name="System Monitor", path="/usr/share/applications/gnome-system-monitor.desktop", keywords="", aliases=()),
    types.SimpleNamespace(name="NVIDIA Monitor", path="/usr/share/applications/nvidia-settings.desktop", keywords="", aliases=()),
    types.SimpleNamespace(name="Brave Web Browser", path="/usr/share/applications/brave-browser.desktop", keywords="", aliases=()),
]

INDEX = windows_mod.app_name_index(FAKE_APPS)


def test_friendly_label_menu_names():
    assert windows_mod.friendly_label("bash", "mate-terminal.Mate-terminal", INDEX) == "Terminal"
    assert windows_mod.friendly_label("x", "mate-volume-control.Mate-volume-control", {}) == "Volume Control"
    assert windows_mod.friendly_label("Amazon.com: tents - Brave", "brave.Browser", INDEX) == "Brave"
    assert windows_mod.friendly_label("64K", "mate-terminal.Mate-terminal", INDEX) == "Terminal"
    assert windows_mod.friendly_label("64K", "mate-terminal.Mate-terminal", {}) == "Terminal"
    assert windows_mod.friendly_label("", "foo_bar.Foo_bar", {}) == "Foo Bar"
    assert windows_mod.friendly_label("foo - 64K", "mate-terminal.Mate-terminal", {}) == "Terminal"


def test_app_name_index_keys():
    assert INDEX["mate-terminal"] == "Terminal"
    assert INDEX["terminal"] == "Terminal"
    assert INDEX["gnome-system-monitor"] == "System Monitor"
    assert INDEX["nvidia-settings"] == "NVIDIA Monitor"
    assert INDEX["brave-browser"] == "Brave"
    assert INDEX["brave web browser"] == "Brave"
    assert INDEX["bravebrowser"] == "Brave"


def test_app_name_index_trims():
    trimmed = windows_mod.app_name_index([mate_terminal(name="MATE Terminal")])
    assert trimmed["mate-terminal"] == "Terminal"
    assert trimmed["mate terminal"] == "Terminal"


def test_owner_eight_windows():
    windows = [
        ("bash", "mate-terminal.Mate-terminal"),
        ("top", "mate-terminal.Mate-terminal"),
        ("vim", "mate-terminal.Mate-terminal"),
        ("64K", "mate-terminal.Mate-terminal"),
        ("System Monitor", "gnome-system-monitor.Gnome-system-monitor"),
        ("NVIDIA Monitor", "nvidia-settings.Nvidia-settings"),
        ("Amazon.com: tents - Brave", "brave.Browser"),
        ("Volume Control", "mate-volume-control.Mate-volume-control"),
    ]
    labels = [windows_mod.friendly_label(title, wm, INDEX) for title, wm in windows]
    assert windows_mod.describe_windows(labels) == (
        "You have 8 windows open: four Terminal windows, System Monitor, NVIDIA Monitor, "
        "Brave and Volume Control."
    )
