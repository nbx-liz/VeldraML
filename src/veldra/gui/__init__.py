"""Dash GUI adapter package."""

from __future__ import annotations

from typing import Any


def create_app() -> Any:
    from veldra.gui.app import create_app as _create_app

    return _create_app()


__all__ = ["create_app"]
