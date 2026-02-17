"""CSV-oriented table builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_regression_table(X, y, fold_ids, predictions, in_out_labels) -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    frame["y_true"] = np.asarray(y, dtype=float)
    frame["fold_id"] = np.asarray(fold_ids)
    frame["in_out_label"] = np.asarray(in_out_labels)
    frame["prediction"] = np.asarray(predictions, dtype=float)
    frame["residual"] = frame["y_true"] - frame["prediction"]
    return frame


def build_binary_table(X, y, fold_ids, scores, in_out_labels) -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    frame["y_true"] = np.asarray(y, dtype=int)
    frame["fold_id"] = np.asarray(fold_ids)
    frame["in_out_label"] = np.asarray(in_out_labels)
    frame["score"] = np.asarray(scores, dtype=float)
    return frame


def build_multiclass_table(X, y, fold_ids, class_probas, in_out_labels) -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    frame["y_true"] = np.asarray(y, dtype=int)
    frame["fold_id"] = np.asarray(fold_ids)
    frame["in_out_label"] = np.asarray(in_out_labels)
    proba = np.asarray(class_probas, dtype=float)
    for idx in range(proba.shape[1]):
        frame[f"proba_class_{idx}"] = proba[:, idx]
    return frame


def build_frontier_table(X, y, fold_ids, predictions, efficiency) -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    frame["y_true"] = np.asarray(y, dtype=float)
    frame["fold_id"] = np.asarray(fold_ids)
    frame["prediction"] = np.asarray(predictions, dtype=float)
    frame["efficiency"] = np.asarray(efficiency, dtype=float)
    return frame


def build_dr_table(observation_table) -> pd.DataFrame:
    return pd.DataFrame(observation_table).copy()


def build_drdid_table(observation_table) -> pd.DataFrame:
    return pd.DataFrame(observation_table).copy()
