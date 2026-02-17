"""Feature importance helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _resolve_booster(booster: Any) -> Any:
    if hasattr(booster, "booster_"):
        return booster.booster_
    return booster


def compute_importance(
    booster: Any,
    importance_type: str = "gain",
    top_n: int = 20,
) -> pd.DataFrame:
    """Return sorted feature importance from LightGBM booster-like object."""
    resolved = _resolve_booster(booster)
    if not hasattr(resolved, "feature_name") or not hasattr(resolved, "feature_importance"):
        return pd.DataFrame(columns=["feature", "importance"])

    names = list(resolved.feature_name())
    values = np.asarray(resolved.feature_importance(importance_type=importance_type), dtype=float)
    if values.size != len(names):
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame({"feature": names, "importance": values})
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    return frame.head(max(1, int(top_n)))
