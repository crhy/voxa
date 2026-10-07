"""Find and press buttons in other programs through the desktop's accessibility bus (AT-SPI over D-Bus).

Runs with the HOST's python3 and nothing but PyGObject's Gio: `python3 uihost.py <command> [args]`, printing one JSON
object. Commands:
  apps                              -> {"ok": true, "apps": [{"name": "spaced-update.py", "windows": ["Spaced Update"]}, ...]}
  list <app> [role ...]             -> {"ok": true, "items": [{"role": "push button", "name": "Install", "enabled": true, "showing": true}, ...]}
  press <app> <label> [role]        -> {"ok": true, "pressed": {"role": ..., "name": ...}} or {"ok": false, "error": "..."}
  wait <app> <label> <seconds>      -> waits until an item with that label is showing: {"ok": true, "found": {...}} / {"ok": false, ...}
  text <app>                        -> {"ok": true, "text": ["visible label 1", ...]}   (labels, headings, status lines)
Matching of <app> and <label> ignores case, punctuation, mnemonic underscores and trailing "…"; an exact match wins,
then "starts with", then "contains".
"""

import json
import re
import sys
import time

from gi.repository import Gio, GLib

ROOT = "/org/a11y/atspi/accessible/root"
ACC = "org.a11y.atspi.Accessible"
PRESSABLE = ("push button", "toggle button", "check box", "radio button", "menu item", "page tab", "link", "list item",
             "check menu item", "radio menu item", "combo box", "button")
TEXTUAL = ("label", "heading", "static", "status bar", "text", "paragraph", "info bar", "alert", "notification",
           "progress bar", "page tab")
STATE_ENABLED, STATE_SENSITIVE, STATE_SHOWING, STATE_VISIBLE = 8, 24, 25, 30
MAX_NODES = 6000


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").replace("_", "").casefold())).strip()


def connect():
    session = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    address = session.call_sync("org.a11y.Bus", "/org/a11y/bus", "org.a11y.Bus", "GetAddress", None,
                                GLib.VariantType("(s)"), 0, 3000, None).unpack()[0]
    flags = Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION
    return Gio.DBusConnection.new_for_address_sync(address, flags, None, None)


class Node:
    def __init__(self, bus, name, path):
        self.bus, self.bus_name, self.path = bus, name, path

    def _call(self, interface, method, args=None, reply=None, timeout=1500):
        return self.bus.call_sync(self.bus_name, self.path, interface, method, args,
                                  GLib.VariantType(reply) if reply else None, 0, timeout, None)

    def prop(self, name, interface=ACC):
        try:
            return self._call("org.freedesktop.DBus.Properties", "Get", GLib.Variant("(ss)", (interface, name)),
                              "(v)").unpack()[0]
        except GLib.Error:
            return None

    @property
    def name(self):
        return self.prop("Name") or ""

    @property
    def role(self):
        try:
            return self._call(ACC, "GetRoleName", None, "(s)").unpack()[0]
        except GLib.Error:
            return ""

    def states(self):
        try:
            low, high = self._call(ACC, "GetState", None, "(au)").unpack()[0]
        except (GLib.Error, ValueError):
            return set()
        bits = low | (high << 32)
        return {i for i in range(64) if bits >> i & 1}

    def children(self):
        try:
            return [Node(self.bus, n, p) for n, p in self._call(ACC, "GetChildren", None, "(a(so))").unpack()[0]]
        except GLib.Error:
            return []

    def press(self):
        """Do the default action (a button's "click"); True when the program accepted it.

        A notebook tab has no action of its own: it is chosen through its parent's selection.
        """
        if self.role != "page tab":
            try:
                if self._call("org.a11y.atspi.Action", "DoAction", GLib.Variant("(i)", (0,)), "(b)", 4000).unpack()[0]:
                    return True
            except GLib.Error:
                pass
        try:
            index = self._call(ACC, "GetIndexInParent", None, "(i)").unpack()[0]
            parent_name, parent_path = self.prop("Parent")
            parent = Node(self.bus, parent_name, parent_path)
            return bool(parent._call("org.a11y.atspi.Selection", "SelectChild", GLib.Variant("(i)", (index,)), "(b)",
                                     4000).unpack()[0])
        except (GLib.Error, TypeError, ValueError):
            return False

    def describe(self):
        states = self.states()
        return {"role": self.role, "name": self.name,
                # GTK 3 sets both bits on a usable widget, GTK 4 only "sensitive"; a greyed-out one has neither.
                "enabled": STATE_ENABLED in states or STATE_SENSITIVE in states,
                "showing": STATE_SHOWING in states and STATE_VISIBLE in states}


def applications(bus):
    return [node for node in Node(bus, "org.a11y.atspi.Registry", ROOT).children() if node.name]


def score(wanted, have):
    wanted, have = norm(wanted), norm(have)
    if not wanted or not have:
        return 0
    if wanted == have:
        return 3
    if have.startswith(wanted):
        return 2
    return 1 if wanted in have else 0


def find_app(bus, wanted):
    """The program whose own name, or one of whose window titles, matches best ("Spaced Update", "Probe Four")."""
    best_score, best = 0, None
    for app in applications(bus):
        value = max([score(wanted, app.name)] + [score(wanted, window.name) for window in app.children()])
        if value > best_score:
            best_score, best = value, app
    return best


def walk(node):
    seen, queue = 0, [node]
    while queue and seen < MAX_NODES:
        current = queue.pop(0)
        seen += 1
        yield current
        queue.extend(current.children())


def find_item(app, label, roles=PRESSABLE):
    best_score, best = 0, None
    for node in walk(app):
        name = node.name
        value = score(label, name)
        if value <= best_score:
            continue
        info = node.describe()
        if info["role"] in roles and info["showing"]:
            best_score, best = value, (node, info)
            if value == 3 and info["enabled"]:
                break
    return best


def main(argv):
    if not argv:
        return {"ok": False, "error": "no command"}
    command, args = argv[0], argv[1:]
    bus = connect()
    if command == "apps":
        return {"ok": True, "apps": [{"name": app.name, "windows": [w.name for w in app.children() if w.name]}
                                     for app in applications(bus)]}
    if not args:
        return {"ok": False, "error": "which program?"}
    deadline = time.monotonic() + (float(args[2]) if command == "wait" and len(args) > 2 else 0.0)
    app = find_app(bus, args[0])
    while app is None and time.monotonic() < deadline:
        time.sleep(0.4)
        app = find_app(bus, args[0])
    if app is None:
        return {"ok": False, "error": "program not found"}
    if command == "list":
        roles = tuple(args[1:]) or PRESSABLE
        items = [info for info in (node.describe() for node in walk(app)) if info["role"] in roles and info["name"]]
        return {"ok": True, "app": app.name, "items": items}
    if command == "text":
        lines = []
        for node in walk(app):
            info = node.describe()
            if info["role"] in TEXTUAL and info["name"] and info["showing"] and info["name"] not in lines:
                lines.append(info["name"])
        return {"ok": True, "app": app.name, "text": lines}
    if command in ("press", "wait"):
        if len(args) < 2:
            return {"ok": False, "error": "which button?"}
        roles = (args[2],) if command == "press" and len(args) > 2 else (PRESSABLE if command == "press" else PRESSABLE + TEXTUAL)
        found = find_item(app, args[1], roles)
        while found is None and time.monotonic() < deadline:
            time.sleep(0.5)
            found = find_item(app, args[1], roles)
        if found is None:
            return {"ok": False, "app": app.name, "error": "not found"}
        node, info = found
        if command == "wait":
            return {"ok": True, "app": app.name, "found": info}
        if not info["enabled"]:
            return {"ok": False, "app": app.name, "error": "disabled", "item": info}
        return {"ok": node.press(), "app": app.name, "pressed": info}
    return {"ok": False, "error": "unknown command"}


if __name__ == "__main__":
    try:
        print(json.dumps(main(sys.argv[1:])))
    except Exception as error:  # the bus is missing, or a program vanished mid-walk
        print(json.dumps({"ok": False, "error": str(error)}))
