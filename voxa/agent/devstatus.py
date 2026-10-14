from __future__ import annotations

import json
import re


def _norm_name(text: str) -> str:
    norm = " ".join(text.lower().replace("-", " ").split())
    if norm.startswith("the "):
        norm = norm[len("the "):]
    if norm.endswith(" project"):
        norm = norm[: -len(" project")]
    return norm.strip()


def _basename(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def project_dir(name: str, home: str, candidates: list[str]) -> str | None:
    """Pick the candidate folder whose last part matches the spoken project name."""
    target = _norm_name(name)
    if not target:
        return None
    exact: list[str] = []
    prefix: list[str] = []
    for path in candidates:
        base = _norm_name(_basename(path))
        if base == target:
            exact.append(path)
        elif base.startswith(target):
            prefix.append(path)
    chosen = exact or prefix
    if not chosen:
        return None
    if home:
        for path in chosen:
            if path == home or path.startswith(home + "/"):
                return path
    return chosen[0]


def parse_git_status(porcelain_branch_text: str) -> tuple[str, int, int, int]:
    """(branch, changed files, ahead, behind) from `git status --porcelain=v1 --branch`."""
    lines = [line for line in porcelain_branch_text.splitlines() if line]
    if not lines:
        return ("", 0, 0, 0)
    header = lines[0][3:] if lines[0].startswith("##") else lines[0]
    branch = header.split("...")[0].split(" ")[0].strip()
    ahead = behind = 0
    match = re.search(r"\[ahead (\d+)(?:, behind (\d+))?\]", header)
    if match:
        ahead = int(match.group(1))
        behind = int(match.group(2)) if match.group(2) else 0
    return (branch, len(lines) - 1, ahead, behind)


def spoken_git(name: str, branch: str, changed: int, ahead: int, behind: int, last_subject: str) -> str:
    head = f"{name} is on {branch}"
    if changed == 0 and ahead == 0 and behind == 0:
        head += " and clean"
    else:
        seg = f"{changed} changed file" + ("s" if changed != 1 else "")
        if ahead:
            seg += f", {ahead} commit" + ("s" if ahead != 1 else "") + " not pushed"
        if behind:
            seg += f" and {behind} behind"
        head += f" with {seg}"
    out = head + "."
    if last_subject:
        out += f" The last commit was: {last_subject}."
    return out


def parse_task_line(line: str) -> tuple[str, str] | None:
    """Classify one status.txt line: ("working", name) | ("passed", name) | ("failed", name)."""
    line = line.strip()
    if not line:
        return None
    start = re.match(r"START\s+(.+?)(?:\s+->.*)?$", line)
    if start:
        return ("working", start.group(1).split()[-1])
    ok = re.match(r"\S+\s+(\S+)\s+OK\b", line)
    if ok:
        return ("passed", ok.group(1))
    fail = re.match(r"\S+\s+(\S+)\b(?=.*did not pass)", line)
    if fail:
        return ("failed", fail.group(1))
    return None


def _task_sentence(last_task_line: str | None) -> str:
    parsed = parse_task_line(last_task_line) if last_task_line else None
    if not parsed:
        return ""
    kind, name = parsed
    if kind == "working":
        return f" It is working on {name}."
    if kind == "failed":
        return f" Its last task, {name}, did not pass."
    return f" Its last task, {name}, passed."


def spoken_model(health_json_text: str | None, last_task_line: str | None) -> str:
    model = None
    if health_json_text:
        try:
            model = json.loads(health_json_text).get("model")
        except (ValueError, TypeError):
            model = None
    if model:
        out = f"The local model is running: {model}."
    else:
        out = "The local model is not running."
    return out + _task_sentence(last_task_line)
