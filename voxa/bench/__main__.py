"""``python -m voxa.bench <subcommand>`` - the hearing bench command line.

B1 implements ``render``; B2 implements ``run`` (the file-driven pipeline runner) and ``agree``
(runner transcripts vs direct ASR). ``check`` is named here so the bench interface is fixed while
roadmap B3 fills it in; calling it before it exists exits 2 and says which task adds it.
"""

from __future__ import annotations

import sys

from . import agree, render, runner

# Subcommand -> the roadmap task that implements it.
NOT_IMPLEMENTED = {
    "check": "B3 (voxa/bench/report.py)",
}

USAGE = """usage: python -m voxa.bench <subcommand> [options]

  render   render a seeded synthetic clip set (roadmap B1)
  run      run the file-driven pipeline over a clip set (roadmap B2)
  agree    compare runner output with direct ASR (roadmap B2)
  check    assert metric conditions on a report ({task})
""".format(task="added by a later roadmap task")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "render":
        return render.main(rest)
    if command == "run":
        return runner.run(rest)
    if command == "agree":
        return agree.main(rest)
    task = NOT_IMPLEMENTED.get(command)
    if task is not None:
        print(f"voxa.bench {command} is not implemented yet; roadmap task {task} adds it.", file=sys.stderr)
        return 2
    print(USAGE, file=sys.stderr)
    print(f"voxa.bench: unknown subcommand {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
