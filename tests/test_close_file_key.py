import voxa.agent.tools.files as files


def test_close_file_uses_ctrl_w(monkeypatch):
    pressed = []

    def recorder(keys):
        pressed.append(keys)

    monkeypatch.setattr(files, "press_keys", recorder)
    monkeypatch.setattr(files, "xdotool_available", lambda: True)

    files.file_dialog({"action": "close"})
    assert pressed == ["ctrl+w"]
    assert "ctrl+q" not in pressed

    pressed.clear()
    files.file_dialog({"action": "load"})
    assert pressed == ["ctrl+o"]
