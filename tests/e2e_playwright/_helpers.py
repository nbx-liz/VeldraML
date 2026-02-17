from __future__ import annotations

from pathlib import Path


def goto(page, base_url: str, path: str) -> None:
    page.goto(f"{base_url}{path}", wait_until="domcontentloaded")
    page.wait_for_selector("#page-content", state="attached")
    page.wait_for_timeout(250)


def assert_ids(page, ids: list[str], *, state: str = "attached") -> None:
    for id_ in ids:
        page.wait_for_selector(f"#{id_}", state=state)


def set_input_value(page, selector: str, value: str) -> None:
    page.fill(selector, "")
    page.fill(selector, value)


def first_existing_path(*candidates: str) -> str:
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""
