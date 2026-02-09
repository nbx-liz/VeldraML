"""Tabular data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from veldra.api.exceptions import VeldraValidationError


def load_tabular_data(path: str) -> pd.DataFrame:
    """Load CSV/Parquet into a DataFrame based on file extension."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)

    raise VeldraValidationError(
        f"Unsupported data format: '{suffix}'. Supported formats are .csv and .parquet."
    )
