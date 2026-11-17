"""Voxa's own screen toolset: find a control by the text on it, then click or type - with accessibility off.

The screenshot, the OCR and the input sink are all fakes here; nothing looks at a real screen or sends real input.
"""

from __future__ import annotations

from voxa.agent import ui, visual_control


class FakeSink:
    """Records the input Voxa would send, and can refuse it the way a missing XTest client does."""

    def __init__(self, works=True):
        self.works = works
        self.clicks: list[tuple[int, int]] = []
        self.typed: list[str] = []

    def click(self, x, y):
        if not self.works:
            return False
        self.clicks.append((x, y))
        return True

    def type_text(self, text):
        if not self.works:
            return False
        self.typed.append(text)
        return True


def _capture(path="screen.png"):
    """A capture that hands the OCR a name instead of grabbing the real screen."""

    def capture():
        return path

    return capture


def _ocr(words):
    def ocr(path):
        return words

    return ocr


def test_finds_save_in_the_fake_ocr_result_and_clicks_its_middle():
    sink = FakeSink()
    words = [visual_control.Word("Save", 100, 200, 40, 20)]
    result = visual_control.press_text("GIMP", "Save", capture=_capture(), ocr=_ocr(words), sink=sink)
    assert result["ok"]
    assert result["clicked"] == {"text": "Save", "x": 120, "y": 210}
    assert sink.clicks == [(120, 210)]


def test_a_label_written_over_several_words_is_found_as_one_line():
    sink = FakeSink()
    words = [
        visual_control.Word("Install", 200, 300, 60, 20),
        visual_control.Word("Selected", 265, 300, 80, 20),
    ]
    result = visual_control.press_text(
        "Spaced Bazaar", "Install Selected", capture=_capture(), ocr=_ocr(words), sink=sink
    )
    assert result["ok"]
    assert result["clicked"]["text"] == "Install Selected"
    assert sink.clicks == [(272, 310)]


def test_the_control_that_matches_best_wins():
    sink = FakeSink()
    words = [
        visual_control.Word("Save as", 100, 100, 60, 20),
        visual_control.Word("Save", 100, 300, 40, 20),
    ]
    result = visual_control.press_text("GIMP", "Save", capture=_capture(), ocr=_ocr(words), sink=sink)
    assert result["ok"]
    assert sink.clicks == [(120, 310)]


def test_only_the_named_window_is_searched_and_clicked():
    sink = FakeSink()
    words = [visual_control.Word("Save", 100, 200, 40, 20)]
    result = visual_control.press_text("GIMP", "Save", region=(0, 0, 80, 80), capture=_capture(), ocr=_ocr(words), sink=sink)
    assert not result["ok"]
    assert result["error"] == "not found on screen"
    assert sink.clicks == []


def test_a_spoken_notice_goes_out_when_voxa_takes_control(caplog):
    spoken = []
    words = [visual_control.Word("Save", 100, 200, 40, 20)]
    with caplog.at_level("INFO", logger="voxa.agent.visual_control"):
        result = visual_control.press_text(
            "GIMP", "Save", capture=_capture(), ocr=_ocr(words), sink=FakeSink(), notice=spoken.append
        )
    assert result["ok"]
    assert spoken == [f"{visual_control.TAKING_CONTROL} GIMP"]
    assert visual_control.TAKING_CONTROL in caplog.text


def test_the_screen_layer_never_touches_the_accessibility_bus(monkeypatch):
    calls: list[tuple[str, bool]] = []

    class RecordingBus:
        def is_enabled(self):
            calls.append(("get", False))
            return False

        def set_enabled(self, value):
            calls.append(("set", value))
            return True

    monkeypatch.setattr("voxa.agent.a11y_scope.GdbusBus", RecordingBus)
    words = [visual_control.Word("Save", 100, 200, 40, 20)]
    result = visual_control.press_text("GIMP", "Save", capture=_capture(), ocr=_ocr(words), sink=FakeSink())
    assert result["ok"]
    assert calls == []


def test_clicking_into_a_field_and_typing_go_through_the_same_sink():
    sink = FakeSink()
    words = [visual_control.Word("Name", 400, 500, 50, 20)]
    result = visual_control.type_at("Pluma", "Name", "meeting notes", capture=_capture(), ocr=_ocr(words), sink=sink)
    assert result["ok"]
    assert sink.clicks == [(425, 510)]
    assert sink.typed == ["meeting notes"]


def test_a_screen_that_cannot_be_grabbed_is_reported_and_nothing_is_clicked():
    sink = FakeSink()
    result = visual_control.press_text("GIMP", "Save", capture=lambda: "", ocr=_ocr([visual_control.Word("Save", 1, 1, 1, 1)]), sink=sink)
    assert not result["ok"]
    assert result["error"] == "not found on screen"
    assert sink.clicks == []


def test_input_that_the_xtest_client_cannot_send_is_reported():
    sink = FakeSink(works=False)
    words = [visual_control.Word("Save", 100, 200, 40, 20)]
    result = visual_control.press_text("GIMP", "Save", capture=_capture(), ocr=_ocr(words), sink=sink)
    assert not result["ok"]
    assert result["error"] == "input failed"
    assert sink.clicks == []


def test_a_screenshot_is_read_by_tesseract_tsv_boxes():
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t120\t340\t50\t22\t95\tSave\n"
        "5\t1\t1\t1\t1\t2\t175\t340\t60\t22\t91\tAs\n"
        "4\t1\t1\t1\t1\t0\t120\t340\t115\t22\t-1\t\n"
    )
    words = visual_control.parse_words(tsv)
    assert [(word.text, word.x, word.y) for word in words] == [("Save", 120, 340), ("As", 175, 340)]
    lines = visual_control.group_lines(words)
    assert lines[0][0] == "Save As"


def test_the_screen_toolset_is_used_when_the_bus_cannot_reach_the_program(monkeypatch):
    sink = FakeSink()
    words = [visual_control.Word("Update System", 300, 400, 120, 22)]
    monkeypatch.setattr(visual_control, "capture_screen", _capture())
    monkeypatch.setattr(visual_control, "ocr_words", _ocr(words))
    monkeypatch.setattr(visual_control, "XTestSink", lambda: sink)
    monkeypatch.setattr(
        ui,
        "call",
        lambda command, *args, **kw: {"ok": False, "error": "program not found"},
    )
    result = ui.press("Spaced Update", "Update System")
    assert result["ok"]
    assert sink.clicks == [(360, 411)]
    assert result["layer"] == "visual"


def test_a_greyed_out_button_is_not_clicked_by_the_screen_toolset(monkeypatch):
    sink = FakeSink()
    monkeypatch.setattr(visual_control, "capture_screen", _capture())
    monkeypatch.setattr(visual_control, "ocr_words", _ocr([visual_control.Word("Save", 100, 200, 40, 20)]))
    monkeypatch.setattr(visual_control, "XTestSink", lambda: sink)
    monkeypatch.setattr(ui, "call", lambda command, *args, **kw: {"ok": False, "error": "disabled"})
    result = ui.press("GIMP", "Save")
    assert result["error"] == "disabled"
    assert sink.clicks == []


def test_pressing_through_the_screen_toolset_still_refuses_the_forbidden_labels():
    from voxa.agent.tools.uicontrol import press_button

    assert not press_button({"label": "delete"}).ok
    assert not press_button({"label": "pay now"}).ok


def test_nothing_at_all_happens_while_actions_are_simulated(monkeypatch):
    monkeypatch.setattr("voxa.simulation.actions_simulated", lambda: True)
    captured: list[str] = []
    monkeypatch.setattr(visual_control, "capture_screen", lambda: captured.append("screen") or "screen.png")
    result = visual_control.press_text("GIMP", "Save")
    assert result == {"ok": True, "simulated": True}
    assert captured == []
