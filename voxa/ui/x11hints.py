"""Xlib calls GTK 4 no longer offers: a small window that floats above the others at a screen position and never takes the keyboard focus."""

from __future__ import annotations

import ctypes
import ctypes.util

_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


class _Attributes(ctypes.Structure):
    _fields_ = [
        ("background_pixmap", ctypes.c_ulong),
        ("background_pixel", ctypes.c_ulong),
        ("border_pixmap", ctypes.c_ulong),
        ("border_pixel", ctypes.c_ulong),
        ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("colormap", ctypes.c_ulong),
        ("cursor", ctypes.c_ulong),
    ]


def corner_position(monitor: tuple[int, int, int, int], size: tuple[int, int], margin: int = 24) -> tuple[int, int]:
    """Top-left pixel that puts a window of *size* in the top-right corner of *monitor* (x, y, width, height)."""
    x, y, width, _height = monitor
    return (max(x, x + width - size[0] - margin), y + margin)


def _load():
    name = ctypes.util.find_library("X11")
    if not name:
        return None
    lib = ctypes.CDLL(name)
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XChangeWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    lib.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    lib.XMoveWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
    lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    lib.XSetErrorHandler.restype = ctypes.c_void_p
    lib.XSetErrorHandler.argtypes = [ctypes.c_void_p]
    return lib


def _with_display(action) -> bool:
    """Run action(lib, display) on a private X connection; X errors are swallowed, never fatal."""
    try:
        lib = _load()
        if lib is None:
            return False
        display = lib.XOpenDisplay(None)
        if not display:
            return False
        quiet = _ERROR_HANDLER(lambda _display, _event: 0)
        previous = lib.XSetErrorHandler(ctypes.cast(quiet, ctypes.c_void_p))
        try:
            action(lib, display)
            lib.XSync(display, 0)
        finally:
            lib.XSetErrorHandler(previous)
            lib.XCloseDisplay(display)
        return True
    except Exception:
        return False


def detach_from_window_manager(xid: int, x: int, y: int) -> bool:
    """Call after realize, BEFORE the window is shown.

    Marks the window override-redirect (as menus and tooltips are): the window manager gives it no frame, no
    taskbar entry and, above all, never the keyboard focus. Also puts it at (x, y).
    """

    def action(lib, display):
        attributes = _Attributes()
        attributes.override_redirect = 1
        lib.XChangeWindowAttributes(display, xid, 1 << 9, ctypes.byref(attributes))  # CWOverrideRedirect
        lib.XMoveWindow(display, xid, int(x), int(y))

    return _with_display(action)


def raise_at(xid: int, x: int, y: int) -> bool:
    """Call while the window is shown: back on top of every window and back at (x, y)."""

    def action(lib, display):
        lib.XMoveWindow(display, xid, int(x), int(y))
        lib.XRaiseWindow(display, xid)

    return _with_display(action)
