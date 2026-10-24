import pytest

from voxa.agent.intents import route


@pytest.mark.parametrize("prompt,tool", [
    ("Can you open the calculator for me", "open_app"),
    ("Hey please open the calculator", "open_app"),
    ("Can you switch to the calculator please", "switch_to"),
    ("Hey can you switch to the calculator", "switch_to"),
    ("Can you minimize the calculator", "minimize_app"),
    ("Hey please minimize the calculator window", "minimize_app"),
    ("Can you restore the calculator please", "restore_app"),
    ("Hey can you restore the calculator", "restore_app"),
    ("Can you maximize the calculator", "maximize_app"),
    ("Hey please maximize the calculator window", "maximize_app"),
    ("Can you close the calculator", "close_app"),
    ("Hey please close the calculator window", "close_app"),
    ("Hey find the file called notes please", "find_file"),
    ("Can you find the file named notes", "find_file"),
    ("Find all files bigger than 10 megabytes", "find_large_files"),
    ("Locate files larger than 10 megabytes", "find_large_files"),
    ("Can you create a folder called Voxa test folder in Documents", "make_folder"),
    ("Hey please make a folder called Voxa test folder in Documents", "make_folder"),
    ("Say hello world please", "say_text"),
    ("Say the words hello world", "say_text"),
    ("Hey can you repeat that please", "repeat_last"),
    ("Please repeat that", "repeat_last"),
    ("What window is this", "active_window_name"),
    ("Which window is in front", "active_window_name"),
    ("Can you read my clipboard", "read_clipboard_aloud"),
    ("Hey what is on my clipboard", "read_clipboard_aloud"),
    ("Can you forget the electricity bill", "date_forget"),
    ("Hey please forget the electricity bill", "date_forget"),
])
def test_paraphrase_routes(prompt, tool):
    result = route(prompt)
    assert result is not None, f"{prompt!r} routed to None"
    assert result.tool == tool, f"{prompt!r} routed to {result.tool!r}, expected {tool!r}"
