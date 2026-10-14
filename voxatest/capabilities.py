"""The 100 assistant capabilities (GitHub issue 102): what has spoken tests, and how often they pass.

Every spoken case names the capability it is evidence for. This module reads the run reports,
works out a pass rate per capability from each case's most recent runs, and writes the result as
Markdown (with charts GitHub draws itself) and as a self-contained HTML page.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent / "data" / "capabilities.json"
TARGET = 0.9
RECENT_RUNS = 5  # a case's pass rate is taken over its last few runs, so a fix shows up quickly


@dataclass(slots=True)
class Capability:
    n: int
    category: str
    name: str
    desc: str
    needs_user: str = ""
    cases: dict[str, list[bool]] = field(default_factory=dict)  # case id -> outcomes, oldest first

    @property
    def runs(self) -> int:
        return sum(len(outcomes[-RECENT_RUNS:]) for outcomes in self.cases.values())

    @property
    def passed(self) -> int:
        return sum(sum(outcomes[-RECENT_RUNS:]) for outcomes in self.cases.values())

    @property
    def rate(self) -> float | None:
        return self.passed / self.runs if self.runs else None

    @property
    def status(self) -> str:
        """working (at or above target), partial (tested, below target), needs you, or planned."""
        rate = self.rate
        if rate is not None:
            return "working" if rate >= TARGET else "partial"
        return "needs you" if self.needs_user else "planned"


def load_capabilities(path: Path = DATA_PATH) -> list[Capability]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Capability(int(item["n"]), str(item["category"]), str(item["name"]), str(item["desc"]), str(item.get("needs_user", "")))
        for item in raw["capabilities"]
    ]


def read_reports(reports_dir: Path, since: str = "") -> list[dict[str, Any]]:
    """Every complete run report in the folder, oldest first (file names sort by time)."""
    reports = []
    for path in sorted(Path(reports_dir).glob("run_*.json")):
        if since and path.name < f"run_{since}":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("results"):
            reports.append(payload)
    return reports


def shipped_case_capabilities(data_dir: Path = DATA_PATH.parent) -> dict[str, int]:
    """Case id -> capability number for every shipped case file (reports written before the
    results carried the number themselves are matched through this)."""
    known: dict[str, int] = {}
    for path in sorted(Path(data_dir).glob("test_cases*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for case in payload.get("cases", []) if isinstance(payload, dict) else []:
            if isinstance(case, dict) and case.get("capability"):
                known[str(case.get("id", ""))] = int(case["capability"])
    return known


def tally(capabilities: list[Capability], reports: list[dict[str, Any]]) -> list[Capability]:
    by_number = {capability.n: capability for capability in capabilities}
    known = shipped_case_capabilities()
    for report in reports:
        for result in report.get("results", []):
            number = int(result.get("capability") or 0) or known.get(str(result.get("case_id", "")), 0)
            capability = by_number.get(number)
            if capability is not None:
                capability.cases.setdefault(str(result.get("case_id", "")), []).append(bool(result.get("passed")))
    return capabilities


def summary(capabilities: list[Capability]) -> dict[str, Any]:
    counts = {"working": 0, "partial": 0, "planned": 0, "needs you": 0}
    for capability in capabilities:
        counts[capability.status] += 1
    tested = [capability for capability in capabilities if capability.runs]
    runs = sum(capability.runs for capability in tested)
    passed = sum(capability.passed for capability in tested)
    categories: dict[str, dict[str, int]] = {}
    for capability in capabilities:
        entry = categories.setdefault(capability.category, {"working": 0, "partial": 0, "planned": 0, "needs you": 0})
        entry[capability.status] += 1
    return {
        "counts": counts,
        "tested": len(tested),
        "runs": runs,
        "passed": passed,
        "rate": passed / runs if runs else 0.0,
        "categories": categories,
    }


def _percent(rate: float | None) -> str:
    return "—" if rate is None else f"{round(rate * 100)}%"


def format_markdown(capabilities: list[Capability], stamp: str = "") -> str:
    info = summary(capabilities)
    counts = info["counts"]
    lines = [
        "## Voxa capability status",
        "",
        f"Measured by VoxaTest speaking each request aloud to Voxa and checking what she really did{f' — updated {stamp}' if stamp else ''}.",
        "",
        f"**{counts['working']} working** (at least {round(TARGET * 100)}% of spoken tests pass) · "
        f"**{counts['partial']} partly working** · **{counts['planned']} not built yet** · "
        f"**{counts['needs you']} need your accounts or approval**",
        "",
        f"Spoken checks counted: {info['passed']} passed of {info['runs']} ({_percent(info['rate'])}), "
        f"over each case's last {RECENT_RUNS} runs.",
        "",
        "```mermaid",
        "pie showData title The 100 capabilities",
        f'    "Working" : {counts["working"]}',
        f'    "Partly working" : {counts["partial"]}',
        f'    "Not built yet" : {counts["planned"]}',
        f'    "Needs you" : {counts["needs you"]}',
        "```",
        "",
    ]
    tested = [capability for capability in capabilities if capability.runs]
    if tested:
        labels = ", ".join(f'"{capability.n}"' for capability in tested)
        values = ", ".join(str(round((capability.rate or 0) * 100)) for capability in tested)
        lines += [
            "```mermaid",
            "xychart-beta",
            '    title "Pass rate of each tested capability (%)"',
            f"    x-axis [{labels}]",
            '    y-axis "Passed" 0 --> 100',
            f"    bar [{values}]",
            "```",
            "",
        ]
    lines += ["| Category | Working | Partly | Not built | Needs you |", "| --- | ---: | ---: | ---: | ---: |"]
    for category, entry in info["categories"].items():
        lines.append(f"| {category} | {entry['working']} | {entry['partial']} | {entry['planned']} | {entry['needs you']} |")
    lines += ["", "| # | Capability | Status | Passed | Spoken tests |", "| ---: | --- | --- | ---: | ---: |"]
    marks = {"working": "✅ working", "partial": "🟡 partly", "planned": "⬜ not built", "needs you": "🔒 needs you"}
    for capability in capabilities:
        note = f" ({capability.needs_user})" if capability.status == "needs you" else ""
        lines.append(
            f"| {capability.n} | {capability.name} | {marks[capability.status]}{note} | "
            f"{_percent(capability.rate)} | {len(capability.cases) or ''} |"
        )
    return "\n".join(lines) + "\n"


_COLOURS = {"working": "#40cc85", "partial": "#f9c24d", "planned": "#4a4f5e", "needs you": "#6f8cf0"}


def format_html(capabilities: list[Capability], stamp: str = "") -> str:
    info = summary(capabilities)
    counts = info["counts"]
    total = max(1, len(capabilities))
    segments, offset = [], 0.0
    for status, colour in _COLOURS.items():
        share = counts[status] / total * 100
        segments.append(f"{colour} {offset:.2f}% {offset + share:.2f}%")
        offset += share
    cells = "".join(
        f'<div class="cell" style="background:{_COLOURS[c.status]}" title="{c.n}. {html.escape(c.name)} — {c.status}'
        f'{"" if c.rate is None else f", {_percent(c.rate)} passed"}">{c.n}</div>'
        for c in capabilities
    )
    bars = "".join(
        f'<div class="row"><span class="label">{c.n}. {html.escape(c.name)}</span>'
        f'<span class="track"><span class="fill" style="width:{(c.rate or 0) * 100:.0f}%;background:{_COLOURS[c.status]}"></span>'
        f'<span class="mark"></span></span><span class="value">{_percent(c.rate)} · {c.passed}/{c.runs}</span></div>'
        for c in capabilities
        if c.runs
    )
    legend = "".join(
        f'<span class="key"><i style="background:{colour}"></i>{counts[status]} {status}</span>' for status, colour in _COLOURS.items()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Voxa capabilities</title><style>
body{{background:#14161c;color:#e9ecf3;font:15px/1.45 system-ui,sans-serif;margin:0;padding:32px}}
h1{{margin:0 0 4px;font-size:26px}} .sub{{opacity:.6;margin-bottom:24px}}
.top{{display:flex;gap:36px;align-items:center;flex-wrap:wrap;background:#1d2029;border-radius:18px;padding:26px}}
.donut{{width:190px;height:190px;border-radius:50%;background:conic-gradient({", ".join(segments)});display:grid;place-items:center}}
.donut div{{width:130px;height:130px;border-radius:50%;background:#1d2029;display:grid;place-items:center;text-align:center}}
.donut b{{font-size:34px;display:block}} .key{{margin-right:18px;white-space:nowrap}} .key i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px}}
.grid{{display:grid;grid-template-columns:repeat(20,1fr);gap:5px;margin:24px 0}}
.cell{{aspect-ratio:1;border-radius:6px;display:grid;place-items:center;font-size:11px;font-weight:700;color:#14161c}}
.row{{display:grid;grid-template-columns:330px 1fr 110px;gap:14px;align-items:center;margin:7px 0}}
.label{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}} .track{{position:relative;height:12px;background:#2a2e3a;border-radius:6px}}
.fill{{position:absolute;inset:0 auto 0 0;border-radius:6px}} .mark{{position:absolute;left:{TARGET * 100:.0f}%;top:-4px;bottom:-4px;width:2px;background:#e9ecf3;opacity:.6}}
.value{{opacity:.75;font-variant-numeric:tabular-nums}} .card{{background:#1d2029;border-radius:18px;padding:22px 26px;margin-top:22px}}
</style></head><body><h1>Voxa capabilities</h1><div class="sub">The 100 assistant capabilities of issue 102, measured by spoken VoxaTest runs{f" · {html.escape(stamp)}" if stamp else ""}</div>
<div class="top"><div class="donut"><div><span><b>{counts['working']}</b>of 100 working</span></div></div>
<div><div style="font-size:19px;margin-bottom:10px">{info['passed']} of {info['runs']} spoken checks passed ({_percent(info['rate'])})</div>{legend}
<div style="opacity:.6;margin-top:10px">Working means at least {round(TARGET * 100)}% of a capability's spoken tests pass over each test's last {RECENT_RUNS} runs.</div></div></div>
<div class="grid">{cells}</div>
<div class="card"><div style="font-weight:700;margin-bottom:10px">Pass rate of each tested capability (the line is the {round(TARGET * 100)}% target)</div>{bars or "No spoken results yet."}</div>
</body></html>"""


def write_status(reports_dir: Path, markdown_path: Path | None, html_path: Path | None, since: str = "", stamp: str = "") -> dict[str, Any]:
    capabilities = tally(load_capabilities(), read_reports(reports_dir, since))
    if markdown_path is not None:
        Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_path).write_text(format_markdown(capabilities, stamp), encoding="utf-8")
    if html_path is not None:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        Path(html_path).write_text(format_html(capabilities, stamp), encoding="utf-8")
    return summary(capabilities)
