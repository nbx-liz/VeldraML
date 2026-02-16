"""Binary classification training routines with OOF calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling.utils import (
    resolve_auto_num_leaves,
    resolve_feature_weights,
    resolve_ratio_params,
    split_for_early_stopping,
)
from veldra.split import iter_cv_splits


@dataclass(slots=True)
class BinaryTrainingOutput:
    model_text: str
    metrics: dict[str, Any]
    cv_results: pd.DataFrame
    feature_schema: dict[str, Any]
    calibrator: LogisticRegression
    calibration_curve: pd.DataFrame
    threshold: dict[str, Any]
    threshold_curve: pd.DataFrame | None = None
    training_history: dict[str, Any] | None = None


def _booster_iteration_stats(booster: Any, fallback_rounds: int) -> tuple[int, int]:
    current_iteration_fn = getattr(booster, "current_iteration", None)
    current_iteration = (
        int(current_iteration_fn()) if callable(current_iteration_fn) else int(fallback_rounds)
    )
    best_iteration = int(getattr(booster, "best_iteration", 0) or current_iteration)
    if best_iteration <= 0:
        best_iteration = current_iteration
    return best_iteration, current_iteration


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_feature_frame(
    config: RunConfig,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[Any]]:
    target = config.data.target
    if target not in data.columns:
        raise VeldraValidationError(f"Target column '{target}' was not found in input data.")
    if data.empty:
        raise VeldraValidationError("Input data is empty.")

    y = data[target].copy()
    if y.isna().any():
        raise VeldraValidationError("Target column contains null values.")

    unique = pd.unique(y)
    if len(unique) != 2:
        raise VeldraValidationError(
            f"Binary task requires exactly two target classes, got {len(unique)}."
        )
    classes = sorted((_to_python_scalar(v) for v in unique), key=lambda v: str(v))
    class_mapping = {classes[0]: 0, classes[1]: 1}
    y_encoded = y.map(class_mapping)
    if y_encoded.isna().any():
        raise VeldraValidationError("Failed to encode target classes into binary labels.")

    drop_cols = set(config.data.drop_cols + config.data.id_cols + [target])
    feature_cols = [col for col in data.columns if col not in drop_cols]
    if not feature_cols:
        raise VeldraValidationError(
            "No training features remain after applying drop/id/target columns."
        )

    x = data.loc[:, feature_cols].copy()
    return x, y_encoded.astype(int), classes


def _train_single_booster(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: RunConfig,
    evaluation_history: dict[str, Any] | None = None,
) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": config.train.seed,
        **config.train.lgb_params,
    }
    if config.train.top_k is not None:
        # Use custom precision@k metric as the monitored metric when configured.
        params["metric"] = "None"
    resolved_leaves = resolve_auto_num_leaves(config)
    if resolved_leaves is not None:
        params["num_leaves"] = resolved_leaves
    params.update(resolve_ratio_params(config, len(x_train)))
    feature_weights = resolve_feature_weights(config, list(x_train.columns))
    if feature_weights is not None:
        params["feature_weights"] = feature_weights
        params["feature_pre_filter"] = False
    if config.train.class_weight is not None:
        neg_weight = float(config.train.class_weight.get("0", 1.0))
        pos_weight = float(config.train.class_weight.get("1", 1.0))
        pos_count = int((y_train == 1).sum())
        neg_count = int((y_train == 0).sum())
        if pos_count > 0 and neg_count > 0 and "scale_pos_weight" not in params:
            params["scale_pos_weight"] = (neg_count * pos_weight) / (pos_count * neg_weight)
    elif (
        config.train.auto_class_weight
        and "is_unbalance" not in params
        and "scale_pos_weight" not in params
    ):
        params["is_unbalance"] = True

    categorical = [col for col in config.data.categorical if col in x_train.columns]
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        categorical_feature=[col for col in categorical if col in x_valid.columns],
        free_raw_data=False,
    )
    callbacks = []
    if config.train.early_stopping_rounds:
        callbacks.append(lgb.early_stopping(config.train.early_stopping_rounds, verbose=False))
    if evaluation_history is not None:
        callbacks.append(lgb.record_evaluation(evaluation_history))
    feval = None
    if config.train.top_k is not None:
        feval = _make_precision_at_k_feval(int(config.train.top_k))

    return lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[valid_set],
        num_boost_round=config.train.num_boost_round,
        callbacks=callbacks,
        feval=feval,
    )


def _binary_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p_pred, 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(y_true, clipped)),
        "logloss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, clipped)),
    }


def _binary_label_metrics(
    y_true: np.ndarray, p_pred: np.ndarray, threshold: float
) -> dict[str, float]:
    label_pred = (p_pred >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, label_pred)),
        "f1": float(f1_score(y_true, label_pred, zero_division=0)),
        "precision": float(precision_score(y_true, label_pred, zero_division=0)),
        "recall": float(recall_score(y_true, label_pred, zero_division=0)),
    }


def _precision_at_k(y_true: np.ndarray, p_pred: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    n_top = min(int(k), len(y_true))
    if n_top <= 0:
        return 0.0
    order = np.argsort(-p_pred)
    top_idx = order[:n_top]
    return float(np.mean(y_true[top_idx]))


def _make_precision_at_k_feval(k: int):
    def _precision_at_k_feval(y_pred: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
        y_true = np.asarray(dataset.get_label(), dtype=int)
        y_score = np.asarray(y_pred, dtype=float)
        value = _precision_at_k(y_true, y_score, k)
        return f"precision_at_{k}", value, True

    return _precision_at_k_feval


def _find_best_threshold_f1(y_true: np.ndarray, p_pred: np.ndarray) -> tuple[float, pd.DataFrame]:
    candidates = np.arange(0.01, 1.0, 0.01)
    records: list[dict[str, float]] = []
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidates:
        label_pred = (p_pred >= threshold).astype(int)
        tp = float(np.sum((y_true == 1) & (label_pred == 1)))
        fp = float(np.sum((y_true == 0) & (label_pred == 1)))
        fn = float(np.sum((y_true == 1) & (label_pred == 0)))

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

        records.append(
            {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, pd.DataFrame.from_records(records)


def train_binary_with_cv(config: RunConfig, data: pd.DataFrame) -> BinaryTrainingOutput:
    """Train binary model with CV and OOF probability calibration.

    Notes
    -----
    - Base LightGBM probabilities are generated out-of-fold first, then a
      calibration model (Platt) is fitted only on OOF scores.
    - This flow prevents calibration leakage from in-fold predictions.
    - Optional threshold optimization evaluates candidate cutoffs on calibrated
      OOF probabilities and records the selected threshold contract.
    """
    if config.task.type != "binary":
        raise VeldraValidationError("train_binary_with_cv only supports task.type='binary'.")
    if not config.data.path:
        raise VeldraValidationError("data.path is required for fit.")

    if config.split.type == "timeseries":
        data = data.sort_values(config.split.time_col).reset_index(drop=True)

    x, y, target_classes = _build_feature_frame(config, data)
    splits = iter_cv_splits(config, data, x, y)

    oof_raw = np.full(len(x), np.nan, dtype=float)
    fold_records: list[dict[str, float | int]] = []
    history_folds: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise VeldraValidationError("Encountered an empty train/valid split.")
        x_fold_train = x.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        x_es_train, x_es_valid, y_es_train, y_es_valid = split_for_early_stopping(
            x_fold_train, y_fold_train, config
        )
        eval_history: dict[str, Any] = {}
        booster = _train_single_booster(
            x_train=x_es_train,
            y_train=y_es_train,
            x_valid=x_es_valid,
            y_valid=y_es_valid,
            config=config,
            evaluation_history=eval_history,
        )
        pred_raw = np.asarray(
            booster.predict(x.iloc[valid_idx], num_iteration=booster.best_iteration),
            dtype=float,
        )
        pred_raw = np.clip(pred_raw, 1e-7, 1 - 1e-7)
        oof_raw[valid_idx] = pred_raw

        fold_metrics = _binary_metrics(y.iloc[valid_idx].to_numpy(), pred_raw)
        best_iteration, num_iterations = _booster_iteration_stats(
            booster, config.train.num_boost_round
        )
        history_folds.append(
            {
                "fold": fold_idx,
                "best_iteration": best_iteration if best_iteration > 0 else num_iterations,
                "num_iterations": num_iterations,
                "eval_history": eval_history.get("valid_0", eval_history),
            }
        )
        fold_records.append(
            {
                "fold": fold_idx,
                "auc": fold_metrics["auc"],
                "logloss": fold_metrics["logloss"],
                "brier": fold_metrics["brier"],
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
            }
        )

    if np.isnan(oof_raw).any():
        raise VeldraValidationError(
            "OOF predictions contain missing values. Check split configuration."
        )

    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=config.train.seed,
        max_iter=1000,
    )
    calibrator.fit(oof_raw.reshape(-1, 1), y.to_numpy())
    oof_cal = calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]

    y_true = y.to_numpy()
    mean_raw = _binary_metrics(y_true, oof_raw)
    mean_cal = _binary_metrics(y_true, oof_cal)
    cv_results = pd.DataFrame.from_records(fold_records)

    prob_true, prob_pred = calibration_curve(
        y.to_numpy(),
        np.clip(oof_cal, 1e-7, 1 - 1e-7),
        n_bins=10,
        strategy="uniform",
    )
    calibration_df = pd.DataFrame(
        {
            "prob_pred": prob_pred.astype(float),
            "prob_true": prob_true.astype(float),
        }
    )

    x_final_train, x_final_valid, y_final_train, y_final_valid = split_for_early_stopping(
        x, y, config
    )
    final_eval_history: dict[str, Any] = {}
    final_model = _train_single_booster(
        x_train=x_final_train,
        y_train=y_final_train,
        x_valid=x_final_valid,
        y_valid=y_final_valid,
        config=config,
        evaluation_history=final_eval_history,
    )
    final_best_iteration, final_num_iterations = _booster_iteration_stats(
        final_model, config.train.num_boost_round
    )
    training_history = {
        "folds": history_folds,
        "final_model": {
            "best_iteration": (
                final_best_iteration if final_best_iteration > 0 else final_num_iterations
            ),
            "num_iterations": final_num_iterations,
            "eval_history": final_eval_history.get("valid_0", final_eval_history),
        },
    }

    feature_schema = {
        "feature_names": x.columns.tolist(),
        "target": config.data.target,
        "task_type": config.task.type,
        "target_classes": target_classes,
    }
    threshold_optimization = config.postprocess.threshold_optimization
    use_optimization = bool(threshold_optimization and threshold_optimization.enabled)

    threshold_curve: pd.DataFrame | None = None
    if use_optimization:
        best_threshold, threshold_curve = _find_best_threshold_f1(y.to_numpy(), oof_cal)
        threshold = {
            "policy": "optimized_f1",
            "value": float(best_threshold),
            "source": "oof_p_cal",
        }
    else:
        threshold_value = (
            float(config.postprocess.threshold) if config.postprocess.threshold is not None else 0.5
        )
        threshold = {"policy": "fixed", "value": threshold_value}

    mean_label = _binary_label_metrics(y_true, oof_cal, float(threshold["value"]))
    mean_all = {**mean_cal, **mean_label}
    if config.train.top_k is not None:
        top_k = int(config.train.top_k)
        top_key = f"precision_at_{top_k}"
        mean_all[top_key] = _precision_at_k(y_true, oof_cal, top_k)
        mean_raw[top_key] = _precision_at_k(y_true, oof_raw, top_k)
    metrics = {
        "folds": fold_records,
        "mean": mean_all,
        "mean_raw": mean_raw,
    }

    return BinaryTrainingOutput(
        model_text=final_model.model_to_string(),
        metrics=metrics,
        cv_results=cv_results,
        feature_schema=feature_schema,
        calibrator=calibrator,
        calibration_curve=calibration_df,
        threshold=threshold,
        threshold_curve=threshold_curve,
        training_history=training_history,
    )
