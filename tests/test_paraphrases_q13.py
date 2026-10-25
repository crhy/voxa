from voxa.agent.intents import route


def test_cap54_track_package_paraphrases():
    r = route("hey can you track package 1Z999AA10123456784")
    assert r and r.tool == "track_package" and r.args["tracking"] == "1Z999AA10123456784"
    r = route("please track the parcel 1Z999AA10123456784")
    assert r and r.tool == "track_package" and r.args["tracking"] == "1Z999AA10123456784"


def test_cap59_budget_set_paraphrases():
    r = route("hey can you set my food budget to 300 dollars")
    assert r and r.tool == "budget_set" and r.args["category"] == "food"
    r = route("please set the food budget to 300 dollars")
    assert r and r.tool == "budget_set" and r.args["category"] == "food"


def test_cap59_budget_status_paraphrases():
    r = route("hey how am i doing on my budget")
    assert r and r.tool == "budget_status"
    r = route("hey how much is left in my food budget")
    assert r and r.tool == "budget_status" and r.args["category"] == "food"


def test_cap73_make_folder_paraphrases():
    r = route("hey can you create a folder called Test Box in Documents")
    assert r and r.tool == "make_folder" and r.args["name"] == "Test Box"
    r = route("please make a new folder called Test Box in Documents")
    assert r and r.tool == "make_folder" and r.args["name"] == "Test Box"


def test_cap73_list_folder_paraphrases():
    r = route("hey what is in my Documents folder")
    assert r and r.tool == "list_folder" and r.args["folder"] == "Documents"
    r = route("list the files in my Documents folder")
    assert r and r.tool == "list_folder" and r.args["folder"] == "Documents"


def test_cap73_trash_file_paraphrases():
    r = route("hey can you delete the file Test Box in Documents")
    assert r and r.tool == "trash_file" and r.args["name"] == "Test Box"
    r = route("please remove Test Box from Documents")
    assert r and r.tool == "trash_file" and r.args["name"] == "Test Box"


def test_cap76_expense_spreadsheet_paraphrases():
    r = route("hey can you create a spreadsheet of my expenses")
    assert r and r.tool == "expense_spreadsheet"
    r = route("please export my expenses to a spreadsheet")
    assert r and r.tool == "expense_spreadsheet"


def test_cap46_travel_doc_add_paraphrases():
    r = route("hey save my passport expires on June 5 2028")
    assert r and r.tool == "travel_doc_add" and r.args["document"] == "passport"
    r = route("can you save my passport expires June 5 2028")
    assert r and r.tool == "travel_doc_add" and r.args["document"] == "passport"


def test_cap46_travel_doc_when_paraphrases():
    r = route("hey when does the passport expire")
    assert r and r.tool == "travel_doc_when" and r.args["document"] == "passport"
    r = route("hey when does my passport expires")
    assert r and r.tool == "travel_doc_when" and r.args["document"] == "passport"


def test_cap46_travel_doc_list_paraphrases():
    r = route("hey what travel documents do i have")
    assert r and r.tool == "travel_doc_list"
    r = route("can you show me my travel documents")
    assert r and r.tool == "travel_doc_list"


def test_cap46_travel_doc_forget_paraphrases():
    r = route("hey can you forget my passport")
    assert r and r.tool == "travel_doc_forget" and r.args["document"] == "passport"
    r = route("please remove my passport")
    assert r and r.tool == "travel_doc_forget" and r.args["document"] == "passport"


def test_guards_still_route():
    r = route("what's on my to-do list")
    assert r and r.tool == "list_read"
    r = route("what time is it")
    assert r and r.tool == "tell_time"
    assert route("what is the capital of France") is None
    r = route("open the calculator")
    assert r and r.tool == "open_app"
