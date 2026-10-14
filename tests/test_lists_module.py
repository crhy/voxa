from voxa.agent.lists import (
    add,
    clean_item,
    clear,
    complete,
    find,
    list_name,
    parse,
    remove,
    render,
    spoken,
)

SAMPLE = """## To-do
- [ ] Call the vet
- [x] Pay the rent
## Shopping
- [ ] Milk
"""


def test_list_name():
    assert list_name("my to do list") == "To-do"
    assert list_name("todo") == "To-do"
    assert list_name("the to-do list") == "To-do"
    assert list_name("tasks") == "To-do"
    assert list_name("task list") == "To-do"
    assert list_name("shopping") == "Shopping"
    assert list_name("the shopping list") == "Shopping"
    assert list_name("grocery list") == "Shopping"
    assert list_name("groceries") == "Shopping"
    assert list_name("packing list") == "Packing"
    assert list_name("") == "To-do"


def test_parse_render_round_trip():
    lists = parse(SAMPLE)
    assert lists == {
        "To-do": [(False, "Call the vet"), (True, "Pay the rent")],
        "Shopping": [(False, "Milk")],
    }
    assert render(lists) == SAMPLE
    assert render({}) == ""
    assert render({"Packing": []}) == ""


def test_clean_item():
    assert clean_item("  milk. ") == "Milk"
    assert clean_item("buy eggs!") == "Buy eggs"
    assert clean_item("what is up?") == "What is up"


def test_add():
    assert add(SAMPLE, "To-do", "walk the dog.") == (
        "## To-do\n- [ ] Call the vet\n- [x] Pay the rent\n- [ ] Walk the dog\n"
        "## Shopping\n- [ ] Milk\n"
    )
    assert add(SAMPLE, "Packing", "socks") == (
        "## To-do\n- [ ] Call the vet\n- [x] Pay the rent\n## Shopping\n- [ ] Milk\n"
        "## Packing\n- [ ] Socks\n"
    )
    assert add(SAMPLE, "To-do", "call the VET") == SAMPLE


def test_find():
    items = parse(SAMPLE)["To-do"]
    assert find(items, "call the vet") == 0
    assert find(items, "vet") == 0
    assert find(items, "cal the vet") == 0
    assert find(items, "fly a kite") is None


def test_complete():
    new, text = complete(SAMPLE, "To-do", "vet")
    assert text == "Call the vet"
    assert new == (
        "## To-do\n- [x] Call the vet\n- [x] Pay the rent\n## Shopping\n- [ ] Milk\n"
    )
    assert complete(SAMPLE, "To-do", "fly a kite") == (SAMPLE, None)


def test_remove():
    new, text = remove(SAMPLE, "Shopping", "milk")
    assert text == "Milk"
    assert new == "## To-do\n- [ ] Call the vet\n- [x] Pay the rent\n"
    assert remove(SAMPLE, "Shopping", "bread") == (SAMPLE, None)


def test_clear():
    new, n = clear(SAMPLE, "To-do")
    assert n == 2
    assert new == "## Shopping\n- [ ] Milk\n"
    new, n = clear(SAMPLE, "To-do", done_only=True)
    assert n == 1
    assert new == "## To-do\n- [ ] Call the vet\n## Shopping\n- [ ] Milk\n"
    assert clear(SAMPLE, "Packing") == (SAMPLE, 0)


def test_spoken():
    lists = parse(SAMPLE)
    assert spoken(lists, "Shopping") == "You have one thing on your shopping list: Milk."
    assert spoken(lists, "To-do") == "You have one thing on your to-do list: Call the vet."
    assert spoken(lists, "Packing") == "Your packing list is empty."
    three = {"Shopping": [(False, "Milk"), (False, "Eggs"), (False, "Bread")]}
    assert spoken(three, "Shopping") == (
        "You have 3 things on your shopping list: Milk, Eggs and Bread."
    )
    assert "Pay the rent" not in spoken(lists, "To-do")
