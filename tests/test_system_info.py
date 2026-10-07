from __future__ import annotations

import subprocess

from voxa.agent.intents import route
from voxa.agent.sysinfo import (
    human_size,
    ordinal,
    parse_df,
    parse_meminfo,
    parse_upower,
    spoken_battery,
    spoken_date,
    spoken_disk,
    spoken_memory,
    spoken_time,
)
from voxa.agent.tools import system

_ORIG_RUN = system._run

_MEGA = 1024**2
_GIGA = 1024**3
_TERA = 1024**4


def test_spoken_time():
    assert spoken_time(0, 0) == "It's midnight."
    assert spoken_time(0, 5) == "It's 12:05 in the morning."
    assert spoken_time(9, 30) == "It's 9:30 in the morning."
    assert spoken_time(12, 0) == "It's noon."
    assert spoken_time(12, 1) == "It's 12:01 in the afternoon."
    assert spoken_time(15, 0) == "It's 3 o'clock in the afternoon."
    assert spoken_time(18, 45) == "It's 6:45 in the evening."
    assert spoken_time(23, 10) == "It's 11:10 at night."


def test_spoken_date():
    assert spoken_date("Wednesday", 7, "October", 2026) == "It's Wednesday, the 7th of October 2026."


def test_ordinal():
    expected = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 11: "11th", 12: "12th", 13: "13th", 21: "21st", 22: "22nd", 23: "23rd", 31: "31st"}
    for n, want in expected.items():
        assert ordinal(n) == want


def test_human_size():
    assert human_size(512 * _MEGA) == "512 megabytes"
    assert human_size(int(1.5 * _GIGA)) == "1.5 gigabytes"
    assert human_size(24 * _GIGA) == "24 gigabytes"
    assert human_size(int(2.1 * _TERA)) == "2.1 terabytes"


def test_spoken_disk():
    assert spoken_disk(120 * _GIGA, 500 * _GIGA) == "You have 120 gigabytes free of 500 gigabytes, 24 percent."


def test_spoken_memory():
    assert spoken_memory(24 * _GIGA, 32 * _GIGA) == "24 gigabytes of memory are free, out of 32 gigabytes."


def test_spoken_battery():
    assert spoken_battery(None, "") == "This computer has no battery."
    assert spoken_battery(80, "charging") == "The battery is at 80 percent and charging."
    assert spoken_battery(80, "fully-charged") == "The battery is full."
    assert spoken_battery(80, "discharging") == "The battery is at 80 percent."


def test_parse_df():
    out = "        Avail         1B-blocks\n128849018880 536870912000\n"
    assert parse_df(out) == (128849018880, 536870912000)
    assert parse_df("garbage\nno numbers here\n") is None


def test_parse_meminfo():
    text = "MemTotal:       32000000 kB\nMemFree: 1000 kB\nMemAvailable:   24000000 kB\n"
    assert parse_meminfo(text) == (24000000 * 1024, 32000000 * 1024)
    assert parse_meminfo("nothing useful") is None


def test_parse_upower():
    text = "    state:               discharging\n    percentage:          80%\n"
    assert parse_upower(text) == (80, "discharging")
    assert parse_upower("no lines here") == (None, "")


def _fake_run(cmd, stdout):
    def fake(command, *args, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return fake


def test_tool_time():
    system._run = _fake_run(["date", "+%H %M"], "14 30")
    try:
        assert system.tell_time({}).speech == "It's 2:30 in the afternoon."
    finally:
        system._run = _ORIG_RUN


def test_tool_date():
    system._run = _fake_run(["date", "+%A|%-d|%B|%Y"], "Wednesday|7|October|2026")
    try:
        assert system.tell_date({}).speech == "It's Wednesday, the 7th of October 2026."
    finally:
        system._run = _ORIG_RUN


def test_tool_disk():
    def fake(command, *args, **kwargs):
        if command[:2] == ["sh", "-c"]:
            return subprocess.CompletedProcess(command, 0, "/home/me\n", "")
        return subprocess.CompletedProcess(command, 0, "        Avail         1B-blocks\n128849018880 536870912000\n", "")

    system._run = fake
    try:
        assert system.disk_space({}).speech == "You have 120 gigabytes free of 500 gigabytes, 24 percent."
    finally:
        system._run = _ORIG_RUN


def test_tool_memory():
    system._run = _fake_run(["cat", "/proc/meminfo"], "MemTotal: 32000000 kB\nMemAvailable: 24000000 kB\n")
    try:
        assert system.memory_free({}).speech == "23 gigabytes of memory are free, out of 31 gigabytes."
    finally:
        system._run = _ORIG_RUN


def test_tool_battery():
    def fake(command, *args, **kwargs):
        if command[:2] == ["sh", "-c"]:
            return subprocess.CompletedProcess(command, 0, "/sys/power/battery\n", "")
        return subprocess.CompletedProcess(command, 0, "    state:               charging\n    percentage:          80%\n", "")

    system._run = fake
    try:
        assert system.battery_level({}).speech == "The battery is at 80 percent and charging."
    finally:
        system._run = _ORIG_RUN


def test_tool_no_battery():
    system._run = _fake_run(["sh", "-c", "upower -e | grep -m1 BAT"], "")
    try:
        assert system.battery_level({}).speech == "This computer has no battery."
    finally:
        system._run = _ORIG_RUN


def test_tool_garbage():
    system._run = _fake_run(["date", "+%H %M"], "garbage")
    try:
        result = system.tell_time({})
        assert not result.ok
        assert result.speech == "I could not read that from this computer."
    finally:
        system._run = _ORIG_RUN


def test_router_phrases():
    assert route("what time is it").tool == "tell_time"
    assert route("what's the date").tool == "tell_date"
    assert route("how much disk space do i have").tool == "disk_space"
    assert route("how much memory is free").tool == "memory_free"
    assert route("how much battery is left").tool == "battery_level"


def test_router_keeps_others():
    assert route("what time is it in Tokyo") is None
    assert route("what is the date of the next election") is None
    assert route("what time does the store close") is None
