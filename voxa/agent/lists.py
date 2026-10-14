from difflib import SequenceMatcher

FILE_NAME = "Voxa Lists.md"

Lists = dict[str, list[tuple[bool, str]]]

TODO_ALIASES = {"to do", "todo", "to", "task", "tasks"}
SHOPPING_ALIASES = {"shopping", "grocery", "groceries"}


def list_name(spoken: str) -> str:
    norm = " ".join(spoken.lower().replace("-", " ").split())
    for prefix in ("my ", "the "):
        if norm.startswith(prefix):
            norm = norm[len(prefix):]
    if norm.endswith(" list"):
        norm = norm[:-len(" list")]
    norm = norm.strip()
    if not norm:
        return "To-do"
    if norm in TODO_ALIASES:
        return "To-do"
    if norm in SHOPPING_ALIASES:
        return "Shopping"
    return norm.title()


def parse(content: str) -> Lists:
    lists: Lists = {}
    current: str | None = None
    for line in content.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            lists.setdefault(current, [])
        elif line.startswith("- [ ] "):
            if current is not None:
                lists[current].append((False, line[6:].strip()))
        elif line.startswith("- [x] ") or line.startswith("- [X] "):
            if current is not None:
                lists[current].append((True, line[6:].strip()))
    return lists


def render(lists: Lists) -> str:
    out = []
    for name, items in lists.items():
        if not items:
            continue
        out.append(f"## {name}\n")
        for done, text in items:
            mark = "x" if done else " "
            out.append(f"- [{mark}] {text}\n")
    return "".join(out)


def clean_item(text: str) -> str:
    t = text.strip()
    if t and t[-1] in ".!?":
        t = t[:-1]
    if t:
        t = t[0].upper() + t[1:]
    return t


def add(content: str, name: str, item: str) -> str:
    lists = parse(content)
    text = clean_item(item)
    items = lists.get(name)
    if items is None:
        lists[name] = [(False, text)]
    else:
        for done, existing in items:
            if not done and existing.lower() == text.lower():
                return content
        items.append((False, text))
    return render(lists)


def _drop_filler(text: str) -> str:
    return " ".join(w for w in text.lower().split() if w not in ("a", "an", "the"))


def find(items: list[tuple[bool, str]], spoken: str) -> int | None:
    words = spoken.lower().split()
    stages = []
    for i, (done, text) in enumerate(items):
        if text.lower() == spoken.lower():
            stages.append((not done, i))
    best = _pick(stages)
    if best is not None:
        return best
    stages = []
    for i, (done, text) in enumerate(items):
        low = text.lower()
        if words and all(w in low for w in words):
            stages.append((not done, i))
    best = _pick(stages)
    if best is not None:
        return best
    stages = []
    for i, (done, text) in enumerate(items):
        if SequenceMatcher(None, _drop_filler(spoken), _drop_filler(text)).ratio() >= 0.6:
            stages.append((not done, i))
    return _pick(stages)


def _pick(stages: list[tuple[bool, int]]) -> int | None:
    if not stages:
        return None
    stages.sort(key=lambda s: (not s[0],))
    return stages[0][1]


def complete(content: str, name: str, spoken: str) -> tuple[str, str | None]:
    lists = parse(content)
    items = lists.get(name)
    if items is None:
        return (content, None)
    idx = find(items, spoken)
    if idx is None:
        return (content, None)
    done, text = items[idx]
    items[idx] = (True, text)
    return (render(lists), text)


def remove(content: str, name: str, spoken: str) -> tuple[str, str | None]:
    lists = parse(content)
    items = lists.get(name)
    if items is None:
        return (content, None)
    idx = find(items, spoken)
    if idx is None:
        return (content, None)
    text = items.pop(idx)[1]
    return (render(lists), text)


def clear(content: str, name: str, done_only: bool = False) -> tuple[str, int]:
    lists = parse(content)
    items = lists.get(name)
    if items is None:
        return (content, 0)
    kept = [it for it in items if done_only and not it[0]] if done_only else []
    removed = len(items) - len(kept)
    if kept:
        lists[name] = kept
    else:
        lists.pop(name, None)
    return (render(lists), removed)


def spoken(lists: Lists, name: str) -> str:
    items = [text for done, text in lists.get(name, []) if not done]
    label = name.lower()
    if not items:
        return f"Your {label} list is empty."
    if len(items) == 1:
        return f"You have one thing on your {label} list: {items[0]}."
    joined = ", ".join(items[:-1]) + " and " + items[-1]
    return f"You have {len(items)} things on your {label} list: {joined}."


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def spoken_progress(lists: Lists, name: str = "To-do") -> str:
    items = lists.get(name, [])
    label = name.lower()
    if not items:
        return f"Your {label} list is empty."
    done = sum(1 for is_done, _ in items if is_done)
    total = len(items)
    if done == total:
        return f"Everything on your {label} list is done: {total} of {total}."
    open_items = [text for is_done, text in items if not is_done]
    report = f"You have finished {done} of {total} things on your {label} list."
    if open_items:
        report += f" Still open: {_join(open_items)}."
    return report


def spoken_pending(lists: Lists) -> str:
    counts = [(name, sum(1 for is_done, _ in items if not is_done)) for name, items in lists.items()]
    counts = [(name, count) for name, count in counts if count]
    total = sum(count for _, count in counts)
    if total == 0:
        return "Nothing is open. Well done."
    thing = "thing" if total == 1 else "things"
    parts = [f"{count} on your {name.lower()} list" for name, count in counts]
    return f"You have {total} {thing} open: {_join(parts)}."


def spoken_next(lists: Lists, due: list[str]) -> str:
    items = [text for is_done, text in lists.get("To-do", []) if not is_done]
    if not items:
        return "Nothing is waiting. Your to-do list is clear."
    due_low = [name.lower() for name in due if name]
    idx = next((i for i, text in enumerate(items) if any(name in text.lower() for name in due_low)), None)
    if idx is None:
        return f"Next on your list: {items[0]}."
    first = items[idx]
    rest = items[:idx] + items[idx + 1:]
    head = first[0].lower() + first[1:]
    out = f"Start with {head}: it is due soon."
    if rest:
        out += f" After that: {_join(rest)}."
    return out
