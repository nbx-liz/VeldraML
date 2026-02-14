from __future__ import annotations

import re
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parents[1] / "HISTORY.md"

DATE_HEADER_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2})（作業/PR: .+）$")
DECISION_RE = re.compile(r"^\s*-\s*Decision:\s*(provisional|confirmed)(?:（[^）]+）)?\s*$")
ENGLISH_SECTION_LABELS = (
    "**Context**",
    "**Plan**",
    "**Changes**",
    "**Decisions**",
    "**Results**",
    "**Risks / Notes**",
    "**Open Questions**",
)


def _read_history_text() -> str:
    return HISTORY_PATH.read_text(encoding="utf-8")


def test_history_headers_are_sorted_ascending() -> None:
    dates: list[str] = []
    for line in _read_history_text().splitlines():
        match = DATE_HEADER_RE.match(line)
        if match:
            dates.append(match.group(1))

    assert dates, "No history entry headers found in HISTORY.md"
    assert dates == sorted(dates), "History entries must be sorted in ascending date order"


def test_history_has_no_english_section_labels() -> None:
    text = _read_history_text()
    for label in ENGLISH_SECTION_LABELS:
        assert label not in text, f"English section label must not remain: {label}"


def test_history_decision_status_is_limited() -> None:
    invalid_lines: list[str] = []
    for line in _read_history_text().splitlines():
        if "Decision:" not in line:
            continue
        if DECISION_RE.match(line):
            continue
        invalid_lines.append(line)

    assert not invalid_lines, f"Invalid Decision status lines found: {invalid_lines}"
