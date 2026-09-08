"""Synchronize linked source snippets embedded in Markdown notes.

Markers have the form ``<!-- code:relative/path.py:10-24 -->`` immediately
before a fenced code block. The block is replaced with the current source
lines, while the prose around it remains untouched.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MARKER = re.compile(r"^(?P<indent>\s*)<!--\s*code:(?P<path>[^:#]+):(?P<start>\d+)-(?P<end>\d+)\s*-->\s*$")
FENCE = re.compile(r"^(?P<indent>\s*)```(?P<lang>[^\n]*)\n(?P<body>.*?)(?P<close>^\s*```\s*$)", re.MULTILINE | re.DOTALL)


def sync_note(note: Path, root: Path, *, write: bool) -> tuple[str, int]:
    original = note.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    output: list[str] = []
    changed = 0
    i = 0
    while i < len(lines):
        match = MARKER.match(lines[i].rstrip("\n\r"))
        if not match:
            output.append(lines[i]); i += 1; continue
        output.append(lines[i])
        if i + 1 >= len(lines) or not lines[i + 1].lstrip().startswith("```"):
            raise ValueError(f"{note}: marker on line {i + 1} must be followed by a fenced code block")
        fence = lines[i + 1]
        lang = fence.strip()[3:].strip()
        end = i + 2
        while end < len(lines) and not lines[end].lstrip().startswith("```"):
            end += 1
        if end == len(lines):
            raise ValueError(f"{note}: unterminated code fence after line {i + 1}")
        source = root / match.group("path")
        if not source.is_file():
            raise FileNotFoundError(f"{note}: linked source does not exist: {source}")
        source_lines = source.read_text(encoding="utf-8").splitlines()
        start, finish = int(match.group("start")), int(match.group("end"))
        if start < 1 or finish < start or finish > len(source_lines):
            raise ValueError(f"{note}: invalid range {start}-{finish} for {source} ({len(source_lines)} lines)")
        block = "".join(f"{line}\n" for line in source_lines[start - 1:finish])
        replacement = [f"{match.group('indent')}```{lang}\n", block, f"{match.group('indent')}```\n"]
        old = lines[i + 1:end + 1]
        new = "".join(replacement)
        if "".join(old) != new:
            changed += 1
        output.extend(replacement)
        i = end + 1
    rendered = "".join(output)
    if write and rendered != original:
        note.write_text(rendered, encoding="utf-8")
    return rendered, changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Update linked code blocks in Markdown notes")
    parser.add_argument("notes", nargs="+", type=Path, help="Markdown note files")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root for relative source paths")
    parser.add_argument("--check", action="store_true", help="fail if any note is stale; do not write")
    args = parser.parse_args()
    stale = 0
    for note in args.notes:
        rendered, changed = sync_note(note, args.root, write=not args.check)
        stale += changed
        print(f"{note}: {'stale' if changed else 'up to date'} ({changed} linked block(s))")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
