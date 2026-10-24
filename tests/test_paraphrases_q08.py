from voxa.agent.intents import route


def test_cap65_week_p1():
    assert route("hey what's happening this week").tool == "calendar_week"
    assert route("hey what does my week look like please").tool == "calendar_week"


def test_cap65_week_person_p1():
    assert route("hey what's Emma doing this week").args["person"] == "Emma"
    assert route("hey what's on Emma's calendar this week please").args["person"] == "Emma"


def test_cap65_busy_p1():
    assert route("hey how busy is my week").tool == "calendar_busy"
    assert route("hey how many appointments do I have this week please").tool == "calendar_busy"


def test_cap85_slow_p1():
    assert route("hey why is my computer so slow").tool == "top_cpu"
    assert route("hey what's slowing down my computer").tool == "top_cpu"


def test_cap85_memory_p1():
    assert route("hey what's using up my memory").tool == "top_memory"
    assert route("hey what's eating my memory").tool == "top_memory"


def test_cap64_meal_idea_p1():
    assert route("hey what should I make for dinner tonight").tool == "meal_idea"
    assert route("hey suggest a dinner idea").tool == "meal_idea"


def test_cap64_meal_shop_p1():
    assert route("hey add the ingredients to my shopping list please").tool == "meal_shop"
    assert route("hey add those ingredients to my shopping list").tool == "meal_shop"


def test_cap64_clear_p1():
    assert route("hey empty the shopping list").tool == "list_clear"
    assert route("can you empty my shopping list please").tool == "list_clear"


def test_cap87_backup_now_p1():
    assert route("hey back up my data please").tool == "backup_now"
    assert route("can you make a backup please").tool == "backup_now"


def test_cap87_backup_last_p1():
    assert route("hey when was my last backup").tool == "backup_last"
    assert route("hey when did my last backup run").tool == "backup_last"


def test_cap87_sort_preview_p1():
    assert route("hey how does my downloads folder sort").args["folder"] == "downloads"
    assert route("hey what would my downloads folder sort").args["folder"] == "downloads"


def test_cap29_list_add_p1():
    assert route("hey add paint the fence to my to do list").tool == "list_add"
    assert route("can you add paint the fence to my to-do list please").tool == "list_add"


def test_cap29_list_add_kitchen_p1():
    assert route("hey add coffee filters to my kitchen list").tool == "list_add"
    assert route("can you add coffee filters to the kitchen list please").tool == "list_add"


def test_cap29_pending_p1():
    assert route("hey what is still open").tool == "list_pending"
    assert route("hey what's still pending please").tool == "list_pending"


def test_cap29_clear_todo_p1():
    assert route("hey empty my to do list").tool == "list_clear"
    assert route("can you clear my to-do list please").tool == "list_clear"


def test_cap29_clear_kitchen_p1():
    assert route("hey empty my kitchen list").tool == "list_clear"
    assert route("can you clear my kitchen list please").tool == "list_clear"


def test_guards_still_route():
    assert route("what's on my to-do list").tool == "list_read"
    assert route("what time is it").tool == "tell_time"
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"
