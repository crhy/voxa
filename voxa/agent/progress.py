"""One short line saying what a tool is doing right now, for the caption under the face."""


def working_caption(tool: str, args: dict[str, str]) -> str:
    """Return a caption (at most 40 characters) describing the running tool call."""
    template = _TEMPLATES.get(tool)
    if template is None and (tool.startswith("find_") or tool.startswith("search_")):
        value = args.get("query") or args.get("destination") or args.get("place")
        if value and value.strip():
            template = "Searching for {value}…"
            args = {"value": value}
        else:
            template = "Searching…"
    if template is None:
        return _fit(tool.replace("_", " ").capitalize() + "…")

    placeholders = [name for _, name, _, _ in _parse(template) if name is not None]
    for name in placeholders:
        if not (args.get(name) or "").strip():
            return _fit(_generic(template))

    filled = _fill(template, args)
    return _fit(filled)


def _generic(template: str) -> str:
    """Bracket-free generic form: the first word of the template plus an ellipsis."""
    first = template.split(" ", 1)[0]
    return first + "…"


def _fill(template: str, args: dict[str, str]) -> str:
    out = []
    for literal, name, _, _ in _parse(template):
        out.append(literal)
        if name is not None:
            value = (args.get(name) or "").strip()
            if name in _NORMALISED:
                value = _normalise_folder(value)
            out.append(value)
    return "".join(out)


def _parse(template: str) -> list[tuple[str, str | None, str, str]]:
    import string

    return list(string.Formatter().parse(template))


def _normalise_folder(value: str) -> str:
    lowered = value.lower()
    for prefix in ("the ", "my "):
        if lowered.startswith(prefix):
            value = value[len(prefix):]
            break
    if value:
        value = value[0].upper() + value[1:]
    return value


def _fit(caption: str) -> str:
    if len(caption) <= 40:
        return caption
    core = caption[:-1] if caption.endswith("…") else caption
    limit = 39
    cut = core[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    else:
        cut = core[:limit]
    return cut + "…"


_TEMPLATES: dict[str, str] = {
    "find_file": "Looking for {name}…",
    "find_large_files": "Looking for large files…",
    "copy_file": "Copying {name} to {destination}…",
    "move_file": "Moving {name} to {destination}…",
    "trash_file": "Deleting {name}…",
    "empty_trash": "Emptying the trash…",
    "rename_file": "Renaming {name}…",
    "make_folder": "Creating the folder {name}…",
    "open_folder": "Opening {folder}…",
    "list_folder": "Looking in {folder}…",
    "open_app": "Opening {name}…",
    "close_app": "Closing {name}…",
    "switch_to": "Switching to {name}…",
    "open_site": "Opening {name}…",
    "browse": "Opening {url}…",
    "web_search": "Searching for {query}…",
    "play_music": "Finding {query}…",
    "play_video": "Finding {query}…",
    "play_youtube": "Finding {query}…",
    "play_latest": "Finding the latest video…",
    "search_youtube": "Searching YouTube for {query}…",
    "cleanup_text": "Cleaning up the text…",
    "read_page": "Reading the page…",
    "click_on": "Clicking {text}…",
    "press_button": "Pressing {label}…",
    "read_window": "Reading the window…",
    "search_site": "Searching for {query}…",
    "update_system": "Updating Spaced Linux…",
    "check_system_version": "Checking the version…",
}

_NORMALISED = {"folder", "destination"}
