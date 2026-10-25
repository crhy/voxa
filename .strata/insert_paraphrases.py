#!/usr/bin/env python3
"""Insert paraphrase cases into JSON test files."""
import copy
import json


def load(path):
    with open(path) as f:
        return json.load(f)

def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

def insert_after(cases, original_id, paraphrases):
    """Insert paraphrase cases right after the case with original_id."""
    idx = next(i for i, c in enumerate(cases) if c["id"] == original_id)
    for offset, (_pid, prompt) in enumerate(paraphrases, 1):
        new_case = copy.deepcopy(cases[idx])
        new_case["id"] = f"{original_id}-p{offset}"
        new_case["prompt"] = prompt
        cases.insert(idx + offset, new_case)

# --- test_cases_cap.json ---
cap = load("voxatest/data/test_cases_cap.json")
cap19 = [
    ("cap19-01", [("hey the report is due on friday",), ("please remember the report deadline is friday",)]),
    ("cap19-02", [("hey when is the report due",), ("hey when is the report due please",)]),
    ("cap19-03", [("hey what's coming up this week",), ("what's due soon",)]),
    ("cap19-04", [("hey forget the report deadline",), ("please forget the report deadline",)]),
]
cap21 = [
    ("cap21-01", [("hey add buy stamps to my to do list",), ("can you add buy stamps to my to-do list please",)]),
    ("cap21-02", [("hey add pay the rent to my to-do list",), ("please add pay the rent to my to do list",)]),
    ("cap21-03", [("hey what's on my to do list",), ("can you show me my to-do list",)]),
    ("cap21-04", [("hey mark buy stamps as done",), ("can you check off buy stamps from my to-do list",)]),
    ("cap21-05", [("hey clear my to-do list",), ("please empty my to-do list",)]),
]
cap26 = [
    ("cap26-01", [("hey add finish the report to my to-do list",), ("can you add finish the report to my to do list",)]),
    ("cap26-02", [("hey add water the plants to my to-do list",), ("please add water the plants to my to do list",)]),
    ("cap26-03", [("hey mark finish the report as done",), ("can you check off finish the report from my to-do list",)]),
    ("cap26-04", [("hey how am I doing",), ("can you give me a progress report",)]),
    ("cap26-05", [("hey what's still open",), ("what do I still have to do",)]),
    ("cap26-06", [("hey what should I do next",), ("what do I do next",)]),
]
cap30 = [
    ("cap30-01", [("hey remember that the bikes are in the shed",), ("please keep in mind that the bikes are in the shed",)]),
    ("cap30-02", [("hey remember that the wifi password is on the fridge",), ("don't forget that the wifi password is on the fridge",)]),
    ("cap30-03", [("hey what do you remember about the bikes",), ("do you remember anything about the bikes",)]),
    ("cap30-04", [("hey what do you know about the wifi password",), ("do you remember anything about the wifi password",)]),
    ("cap30-05", [("hey what do you remember",), ("what have I told you",)]),
    ("cap30-06", [("hey forget about the bikes",), ("please forget what I said about the bikes",)]),
    ("cap30-07", [("hey forget about the wifi password",), ("please forget what I told you about the wifi password",)]),
]
for orig, paras in cap19 + cap21 + cap26 + cap30:
    insert_after(cap["cases"], orig, [(None, p[0]) for p in paras])
save("voxatest/data/test_cases_cap.json", cap)

# --- test_cases_2.json ---
f2 = load("voxatest/data/test_cases_2.json")
f2_paras = [
    ("files2-01", [("hey remind me to stretch in two hours",), ("can you remind me to stretch in two hours",)]),
    ("files2-02", [("hey what are my reminders",), ("show me my reminders",)]),
    ("files2-03", [("hey cancel my reminders",), ("cancel all reminders",)]),
    ("files2-04", [("hey take a note to call the vet",), ("can you make a note to call the vet",)]),
    ("files2-05", [("hey read my notes",), ("can you show me my notes",)]),
    ("files2-06", [("hey delete my last note",), ("please remove my last note",)]),
]
for orig, paras in f2_paras:
    insert_after(f2["cases"], orig, [(None, p[0]) for p in paras])
save("voxatest/data/test_cases_2.json", f2)

# --- test_cases_3.json ---
f3 = load("voxatest/data/test_cases_3.json")
f3_paras = [
    ("new-files-01", [("hey make a note to water the plants",), ("can you take a note to water the plants",)]),
    ("new-files-02", [("hey read my notes",), ("what's in my notes",)]),
    ("new-files-03", [("hey delete my last note",), ("can you erase my last note",)]),
    ("new-files-04", [("hey set a timer for one minute",), ("can you set a one minute timer",)]),
    ("new-files-05", [("hey remind me to check the oven in one minute",), ("can you remind me to check the oven in one minute",)]),
    ("new-files-06", [("hey what timers do I have",), ("show me my timers",)]),
]
for orig, paras in f3_paras:
    insert_after(f3["cases"], orig, [(None, p[0]) for p in paras])
save("voxatest/data/test_cases_3.json", f3)

# --- test_cases.json ---
f1 = load("voxatest/data/test_cases.json")
f1_paras = [
    ("files-04", [("hey make a note to buy milk",), ("can you take a note to buy milk",)]),
]
for orig, paras in f1_paras:
    insert_after(f1["cases"], orig, [(None, p[0]) for p in paras])
save("voxatest/data/test_cases.json", f1)

print("Done. Inserted paraphrases into all 4 JSON files.")
