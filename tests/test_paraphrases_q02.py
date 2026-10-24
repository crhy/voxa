from voxa.agent.intents import route


def test_cap19_paraphrases():
    cases = [
        ("hey the report is due on friday", "date_remember"),
        ("please remember the report deadline is friday", "date_remember"),
        ("hey when is the report due", "date_when"),
        ("hey when is the report due please", "date_when"),
        ("hey what's coming up this week", "date_upcoming"),
        ("what's due soon", "date_upcoming"),
        ("hey forget the report deadline", "date_forget"),
        ("please forget the report deadline", "date_forget"),
    ]
    for phrase, expected in cases:
        r = route(phrase)
        assert r is not None and r.tool == expected, f"{phrase!r} -> {r}"


def test_cap21_paraphrases():
    cases = [
        ("hey add buy stamps to my to do list", "list_add"),
        ("can you add buy stamps to my to-do list please", "list_add"),
        ("hey add pay the rent to my to-do list", "list_add"),
        ("please add pay the rent to my to do list", "list_add"),
        ("hey what's on my to do list", "list_read"),
        ("can you show me my to-do list", "list_read"),
        ("hey mark buy stamps as done", "list_done"),
        ("can you check off buy stamps from my to-do list", "list_done"),
        ("hey clear my to-do list", "list_clear"),
        ("please empty my to-do list", "list_clear"),
    ]
    for phrase, expected in cases:
        r = route(phrase)
        assert r is not None and r.tool == expected, f"{phrase!r} -> {r}"


def test_cap24_paraphrases():
    cases = [
        ("hey remind me to stretch in two hours", "set_reminder"),
        ("can you remind me to stretch in two hours", "set_reminder"),
        ("hey what are my reminders", "list_reminders"),
        ("show me my reminders", "list_reminders"),
        ("hey cancel my reminders", "cancel_reminders"),
        ("cancel all reminders", "cancel_reminders"),
        ("hey set a timer for one minute", "set_timer"),
        ("can you set a one minute timer", "set_timer"),
        ("hey remind me to check the oven in one minute", "set_reminder"),
        ("can you remind me to check the oven in one minute", "set_reminder"),
        ("hey what timers do I have", "list_reminders"),
        ("show me my timers", "list_reminders"),
    ]
    for phrase, expected in cases:
        r = route(phrase)
        assert r is not None and r.tool == expected, f"{phrase!r} -> {r}"


def test_cap26_paraphrases():
    cases = [
        ("hey add finish the report to my to-do list", "list_add"),
        ("can you add finish the report to my to do list", "list_add"),
        ("hey add water the plants to my to-do list", "list_add"),
        ("please add water the plants to my to do list", "list_add"),
        ("hey mark finish the report as done", "list_done"),
        ("can you check off finish the report from my to-do list", "list_done"),
        ("hey how am I doing", "list_progress"),
        ("can you give me a progress report", "list_progress"),
        ("hey what's still open", "list_pending"),
        ("what do I still have to do", "list_pending"),
        ("hey what should I do next", "list_next"),
        ("what do I do next", "list_next"),
    ]
    for phrase, expected in cases:
        r = route(phrase)
        assert r is not None and r.tool == expected, f"{phrase!r} -> {r}"


def test_cap30_paraphrases():
    cases = [
        ("hey remember that the bikes are in the shed", "remember_fact"),
        ("please keep in mind that the bikes are in the shed", "remember_fact"),
        ("hey remember that the wifi password is on the fridge", "remember_fact"),
        ("don't forget that the wifi password is on the fridge", "remember_fact"),
        ("hey what do you remember about the bikes", "recall_fact"),
        ("do you remember anything about the bikes", "recall_fact"),
        ("hey what do you know about the wifi password", "recall_fact"),
        ("do you remember anything about the wifi password", "recall_fact"),
        ("hey what do you remember", "list_facts"),
        ("what have I told you", "list_facts"),
        ("hey forget about the bikes", "forget_fact"),
        ("please forget what I said about the bikes", "forget_fact"),
        ("hey forget about the wifi password", "forget_fact"),
        ("please forget what I told you about the wifi password", "forget_fact"),
        ("hey take a note to call the vet", "take_note"),
        ("can you make a note to call the vet", "take_note"),
        ("hey read my notes", "read_notes"),
        ("can you show me my notes", "read_notes"),
        ("hey delete my last note", "delete_last_note"),
        ("please remove my last note", "delete_last_note"),
        ("hey make a note to water the plants", "take_note"),
        ("can you take a note to water the plants", "take_note"),
        ("hey read my notes", "read_notes"),
        ("what's in my notes", "read_notes"),
        ("hey delete my last note", "delete_last_note"),
        ("can you erase my last note", "delete_last_note"),
        ("hey make a note to buy milk", "take_note"),
        ("can you take a note to buy milk", "take_note"),
    ]
    for phrase, expected in cases:
        r = route(phrase)
        assert r is not None and r.tool == expected, f"{phrase!r} -> {r}"
