"""Tests for working_caption: the live caption shown under the face while a tool runs."""

from voxa.agent.progress import working_caption


def test_find_file():
    assert working_caption("find_file", {"name": "report"}) == "Looking for report…"


def test_find_large_files():
    assert working_caption("find_large_files", {}) == "Looking for large files…"


def test_copy_file():
    assert working_caption("copy_file", {"name": "report", "destination": "the documents"}) == "Copying report to Documents…"


def test_move_file():
    assert working_caption("move_file", {"name": "report", "destination": "my downloads folder"}) == "Moving report to Downloads folder…"


def test_trash_file():
    assert working_caption("trash_file", {"name": "report"}) == "Deleting report…"


def test_empty_trash():
    assert working_caption("empty_trash", {}) == "Emptying the trash…"


def test_rename_file():
    assert working_caption("rename_file", {"name": "notes"}) == "Renaming notes…"


def test_make_folder():
    assert working_caption("make_folder", {"name": "Archive"}) == "Creating the folder Archive…"


def test_open_folder():
    assert working_caption("open_folder", {"folder": "my downloads folder"}) == "Opening Downloads folder…"


def test_list_folder():
    assert working_caption("list_folder", {"folder": "the pictures"}) == "Looking in Pictures…"


def test_open_app():
    assert working_caption("open_app", {"name": "GIMP"}) == "Opening GIMP…"


def test_close_app():
    assert working_caption("close_app", {"name": "GIMP"}) == "Closing GIMP…"


def test_switch_to():
    assert working_caption("switch_to", {"name": "GIMP"}) == "Switching to GIMP…"


def test_open_site():
    assert working_caption("open_site", {"name": "Amazon"}) == "Opening Amazon…"


def test_browse():
    assert working_caption("browse", {"url": "https://example.com"}) == "Opening https://example.com…"


def test_web_search():
    assert working_caption("web_search", {"query": "budget tents"}) == "Searching for budget tents…"


def test_play_music():
    assert working_caption("play_music", {"query": "jazz"}) == "Finding jazz…"


def test_play_video():
    assert working_caption("play_video", {"query": "cooking"}) == "Finding cooking…"


def test_play_youtube():
    assert working_caption("play_youtube", {"query": "cooking"}) == "Finding cooking…"


def test_play_latest():
    assert working_caption("play_latest", {}) == "Finding the latest video…"


def test_search_youtube():
    assert working_caption("search_youtube", {"query": "tutorials"}) == "Searching YouTube for tutorials…"


def test_cleanup_text():
    assert working_caption("cleanup_text", {}) == "Cleaning up the text…"


def test_read_page():
    assert working_caption("read_page", {}) == "Reading the page…"


def test_click_on():
    assert working_caption("click_on", {"text": "Submit"}) == "Clicking Submit…"


def test_search_site():
    assert working_caption("search_site", {"query": "deals"}) == "Searching for deals…"


def test_deal_search_no_value():
    assert working_caption("search_flights", {}) == "Searching…"


def test_deal_search_with_value():
    assert working_caption("search_flights", {"destination": "Paris"}) == "Searching for Paris…"


def test_deal_search_with_query():
    assert working_caption("find_products", {"query": "tents"}) == "Searching for tents…"


def test_unknown_tool():
    assert working_caption("lock_screen", {}) == "Lock screen…"


def test_long_caption_cut():
    caption = working_caption("web_search", {"query": "the history of the roman empire in the third century"})
    assert caption.endswith("…")
    assert len(caption) <= 40
    assert not caption[:-1].endswith("roman e")


def test_missing_argument_generic():
    assert working_caption("find_file", {}) == "Looking…"


def test_missing_argument_copy():
    assert working_caption("copy_file", {}) == "Copying…"
