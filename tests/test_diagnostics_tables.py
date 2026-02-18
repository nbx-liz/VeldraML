from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.diagnostics.tables import (
    build_binary_table,
    build_dr_table,
    build_drdid_table,
    build_frontier_table,
    build_multiclass_table,
    build_regression_table,
)


def test_table_builders_include_expected_columns() -> None:
    x = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    reg = build_regression_table(x, [1.0, 2.0], [1, 2], [0.8, 2.2], ["in", "out"])
    assert {"prediction", "residual", "fold_id"} <= set(reg.columns)

    bin_tbl = build_binary_table(x, [0, 1], [1, 2], [0.2, 0.8], ["in", "out"])
    assert {"score", "fold_id"} <= set(bin_tbl.columns)

    mc_tbl = build_multiclass_table(
        x,
        [0, 1],
        [1, 2],
        np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        ["in", "out"],
    )
    assert "proba_class_0" in mc_tbl.columns

    fr_tbl = build_frontier_table(x, [1.0, 2.0], [1, 2], [1.1, 2.2], [0.9, 0.95])
    assert {"prediction", "efficiency"} <= set(fr_tbl.columns)

    dr = build_dr_table(pd.DataFrame({"weight": [1.0, 2.0]}))
    drdid = build_drdid_table(pd.DataFrame({"weight": [1.0, 2.0]}))
    assert not dr.empty
    assert not drdid.empty
