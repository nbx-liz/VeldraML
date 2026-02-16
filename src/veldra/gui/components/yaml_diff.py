"""Simple YAML diff rendering."""

from __future__ import annotations

import difflib

from dash import html


def render_yaml_diff(left_yaml: str, right_yaml: str) -> html.Pre:
    left = (left_yaml or "").splitlines()
    right = (right_yaml or "").splitlines()
    diff = "\n".join(
        difflib.unified_diff(left, right, fromfile="Run A", tofile="Run B", lineterm="")
    )
    text = diff or "No differences."
    return html.Pre(text, className="small", style={"maxHeight": "360px", "overflowY": "auto"})
