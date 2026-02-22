"""Notebook section/spec generator for quick-reference UC-01..UC-13."""

from __future__ import annotations

# ruff: noqa: E501
import argparse
import textwrap
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
DEFAULT_NOTEBOOK_SUBDIR = "quick_reference"

NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

SUMMARY_CELL = textwrap.dedent(
    """
    SUMMARY = {
        "uc": UC_ID,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "artifact_path": str(artifact_path_for_summary),
        "outputs": [str(p) for p in summary_outputs],
        "metrics": metrics_df.round(6).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(SUMMARY, indent=2), encoding="utf-8")
    SUMMARY
    """
).strip()


def _setup_block(uc_id: str, out_dir: str) -> str:
    return f"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

from veldra.api import estimate_dr, evaluate, fit, tune
from veldra.api.artifact import Artifact
from veldra.diagnostics import (
    binary_metrics,
    build_binary_table,
    build_dr_table,
    build_drdid_table,
    build_frontier_table,
    build_multiclass_table,
    build_regression_table,
    compute_balance_smd,
    compute_importance,
    compute_overlap_stats,
    compute_shap,
    compute_shap_multiclass,
    frontier_metrics,
    multiclass_metrics,
    plot_confusion_matrix,
    plot_error_histogram,
    plot_feature_importance,
    plot_frontier_scatter,
    plot_if_distribution,
    plot_learning_curve,
    plot_love_plot,
    plot_parallel_trends,
    plot_pinball_histogram,
    plot_propensity_distribution,
    plot_roc_comparison,
    plot_roc_multiclass,
    plot_shap_summary,
    plot_timeseries_prediction,
    plot_timeseries_residual,
    plot_weight_distribution,
    regression_metrics,
)

ROOT = Path('.').resolve()
OUT_DIR = ROOT / 'examples' / 'out' / '{out_dir}'
OUT_DIR.mkdir(parents=True, exist_ok=True)
diag_dir = OUT_DIR / 'diagnostics'
diag_dir.mkdir(parents=True, exist_ok=True)
UC_ID = '{uc_id}'
"""


def _build_notebook(
    title: str,
    setup: str,
    sections: list[tuple[str, str]],
) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = NOTEBOOK_METADATA
    nb.cells = [
        nbformat.v4.new_markdown_cell(f"# {title}"),
        nbformat.v4.new_markdown_cell("## Setup"),
        nbformat.v4.new_code_cell(textwrap.dedent(setup).strip() + "\n"),
    ]
    for section_title, code in sections:
        nb.cells.append(nbformat.v4.new_markdown_cell(section_title))
        nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(code).strip() + "\n"))
    nb.cells.append(nbformat.v4.new_markdown_cell("## Result Summary"))
    nb.cells.append(nbformat.v4.new_code_cell(SUMMARY_CELL + "\n"))
    return nb


def _sections_uc01() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
from sklearn.model_selection import train_test_split

source_df = pd.read_csv(ROOT / 'data' / 'ames_housing.csv')
train_df, test_df = train_test_split(source_df, test_size=0.25, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = OUT_DIR / 'train.parquet'
test_path = OUT_DIR / 'test.parquet'
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 2. Feature Selection and Category Casting",
            """
target_col = 'SalePrice'
feature_cols = [col for col in train_df.columns if col != target_col]
cat_cols = train_df.loc[:, feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    categories = sorted(source_df[col].astype(str).dropna().unique().tolist())
    train_df[col] = pd.Categorical(train_df[col], categories=categories)
    test_df[col] = pd.Categorical(test_df[col], categories=categories)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 3. Config",
            """
config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {
        'path': str(train_path),
        'target': target_col,
        'categorical': cat_cols,
    },
    'split': {'type': 'kfold', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Fit Model",
            """
run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
""",
        ),
        (
            "## 5. Prediction",
            """
x_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].to_numpy(dtype=float)
x_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].to_numpy(dtype=float)
pred_train = np.asarray(artifact.predict(x_train), dtype=float)
pred_test = np.asarray(artifact.predict(x_test), dtype=float)

score_table = build_regression_table(
    pd.concat([x_train, x_test], ignore_index=True),
    np.concatenate([y_train, y_test]),
    np.concatenate([np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]),
    np.concatenate([pred_train, pred_test]),
    ['in_sample'] * len(x_train) + ['out_of_sample'] * len(x_test),
)
score_path = OUT_DIR / 'regression_scores.csv'
score_table.to_csv(score_path, index=False)
""",
        ),
        (
            "## 6. Metrics",
            """
metrics_df = pd.DataFrame(
    [
        regression_metrics(y_train, pred_train, label='in_sample'),
        regression_metrics(y_test, pred_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 7. Residual Distribution",
            """
residual_path = diag_dir / 'residual_hist.png'
residual_fig = plot_error_histogram(
    y_train - pred_train,
    y_test - pred_test,
    metrics_df.iloc[0].to_dict(),
    metrics_df.iloc[1].to_dict(),
    residual_path,
)
display(residual_fig)
""",
        ),
        (
            "## 8. Learning Curve",
            """
learning_curve_path = diag_dir / 'learning_curve.png'
learning_curve_fig = plot_learning_curve(artifact.training_history, learning_curve_path)
display(learning_curve_fig)
""",
        ),
        (
            "## 9. Feature Importance (Split) Plot",
            """
booster = artifact._get_booster()
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 10. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 11. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 12. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 13. SHAP Plot",
            """
shap_input = x_train.head(min(len(x_train), 64))
shap_frame = compute_shap(booster, shap_input)
shap_path = diag_dir / 'shap_summary.png'
shap_fig = plot_shap_summary(shap_frame, shap_input, shap_path)
display(shap_fig)
""",
        ),
        (
            "## 14. SHAP Table",
            """
shap_mean_abs_df = (
    shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'mean_abs_shap'}
    )
)
shap_mean_abs_path = OUT_DIR / 'shap_mean_abs.csv'
shap_mean_abs_df.to_csv(shap_mean_abs_path, index=False)
display(shap_mean_abs_df)

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    residual_path,
    learning_curve_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
    shap_path,
    shap_mean_abs_path,
    score_path,
]
artifact_path_for_summary = run_result.artifact_path
""",
        ),
    ]


def _sections_uc02() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
from sklearn.model_selection import train_test_split

source_df = pd.read_csv(ROOT / 'data' / 'titanic.csv')
train_df, test_df = train_test_split(
    source_df,
    test_size=0.25,
    random_state=42,
    stratify=source_df['Survived'],
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = OUT_DIR / 'train.parquet'
test_path = OUT_DIR / 'test.parquet'
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 2. Feature Selection and Category Casting",
            """
target_col = 'Survived'
feature_cols = [col for col in train_df.columns if col != target_col]
cat_cols = train_df.loc[:, feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    categories = sorted(source_df[col].astype(str).dropna().unique().tolist())
    train_df[col] = pd.Categorical(train_df[col], categories=categories)
    test_df[col] = pd.Categorical(test_df[col], categories=categories)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 3. Config",
            """
config = {
    'config_version': 1,
    'task': {'type': 'binary'},
    'data': {
        'path': str(train_path),
        'target': target_col,
        'categorical': cat_cols,
    },
    'split': {'type': 'stratified', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['logloss', 'auc'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'postprocess': {'calibration': 'platt'},
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Fit Model",
            """
run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
""",
        ),
        (
            "## 5. Prediction",
            """
x_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].to_numpy(dtype=int)
x_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].to_numpy(dtype=int)

pred_train = artifact.predict(x_train)
pred_test = artifact.predict(x_test)
score_train = pred_train['p_cal'].to_numpy(dtype=float)
score_test = pred_test['p_cal'].to_numpy(dtype=float)
pred_label_test = pred_test['label_pred'].to_numpy(dtype=int)

score_table = build_binary_table(
    pd.concat([x_train, x_test], ignore_index=True),
    np.concatenate([y_train, y_test]),
    np.concatenate([np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]),
    np.concatenate([score_train, score_test]),
    ['in_sample'] * len(x_train) + ['out_of_sample'] * len(x_test),
)
score_path = OUT_DIR / 'binary_scores.csv'
score_table.to_csv(score_path, index=False)
""",
        ),
        (
            "## 6. Metrics",
            """
metrics_df = pd.DataFrame(
    [
        binary_metrics(y_train, score_train, label='in_sample'),
        binary_metrics(y_test, score_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 7. ROC",
            """
roc_path = diag_dir / 'roc_in_out.png'
roc_fig = plot_roc_comparison(y_train, score_train, y_test, score_test, roc_path)
display(roc_fig)
""",
        ),
        (
            "## 8. Confusion Matrix",
            """
cm_path = diag_dir / 'confusion_matrix.png'
cm_fig = plot_confusion_matrix(y_test, pred_label_test, ['0', '1'], cm_path)
display(cm_fig)
""",
        ),
        (
            "## 9. Learning Curve",
            """
learning_curve_path = diag_dir / 'learning_curve.png'
learning_curve_fig = plot_learning_curve(artifact.training_history, learning_curve_path)
display(learning_curve_fig)
""",
        ),
        (
            "## 10. Feature Importance (Split) Plot",
            """
booster = artifact._get_booster()
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 11. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 12. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 13. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 14. SHAP Plot",
            """
shap_input = x_train.head(min(len(x_train), 64))
shap_frame = compute_shap(booster, shap_input)
shap_path = diag_dir / 'shap_summary.png'
shap_fig = plot_shap_summary(shap_frame, shap_input, shap_path)
display(shap_fig)
""",
        ),
        (
            "## 15. SHAP Table",
            """
shap_mean_abs_df = (
    shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'mean_abs_shap'}
    )
)
shap_mean_abs_path = OUT_DIR / 'shap_mean_abs.csv'
shap_mean_abs_df.to_csv(shap_mean_abs_path, index=False)
display(shap_mean_abs_df)

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    score_path,
    roc_path,
    cm_path,
    learning_curve_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
    shap_path,
    shap_mean_abs_path,
]
artifact_path_for_summary = run_result.artifact_path
""",
        ),
    ]


def _sections_uc03() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
from sklearn.model_selection import train_test_split

source_df = pd.read_csv(ROOT / 'data' / 'penguins.csv')
train_df, test_df = train_test_split(
    source_df,
    test_size=0.25,
    random_state=42,
    stratify=source_df['species'],
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = OUT_DIR / 'train.parquet'
test_path = OUT_DIR / 'test.parquet'
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 2. Feature Selection and Category Casting",
            """
target_col = 'species'
feature_cols = [col for col in train_df.columns if col != target_col]
cat_cols = train_df.loc[:, feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    categories = sorted(source_df[col].astype(str).dropna().unique().tolist())
    train_df[col] = pd.Categorical(train_df[col], categories=categories)
    test_df[col] = pd.Categorical(test_df[col], categories=categories)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 3. Config",
            """
config = {
    'config_version': 1,
    'task': {'type': 'multiclass'},
    'data': {
        'path': str(train_path),
        'target': target_col,
        'categorical': cat_cols,
    },
    'split': {'type': 'stratified', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['multi_logloss', 'multi_error'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Fit Model",
            """
run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
""",
        ),
        (
            "## 5. Prediction",
            """
x_train = train_df.drop(columns=[target_col])
x_test = test_df.drop(columns=[target_col])

target_classes = [str(v) for v in artifact.feature_schema['target_classes']]
class_to_idx = {label: idx for idx, label in enumerate(target_classes)}
y_train_idx = train_df[target_col].map(class_to_idx).to_numpy(dtype=int)
y_test_idx = test_df[target_col].map(class_to_idx).to_numpy(dtype=int)

pred_train = artifact.predict(x_train)
pred_test = artifact.predict(x_test)
prob_cols = [f'proba_{label}' for label in target_classes]
proba_train = pred_train[prob_cols].to_numpy(dtype=float)
proba_test = pred_test[prob_cols].to_numpy(dtype=float)
pred_test_idx = pred_test['label_pred'].map(class_to_idx).to_numpy(dtype=int)

score_table = build_multiclass_table(
    pd.concat([x_train, x_test], ignore_index=True),
    np.concatenate([y_train_idx, y_test_idx]),
    np.concatenate([np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]),
    np.vstack([proba_train, proba_test]),
    ['in_sample'] * len(x_train) + ['out_of_sample'] * len(x_test),
)
score_path = OUT_DIR / 'multiclass_scores.csv'
score_table.to_csv(score_path, index=False)
""",
        ),
        (
            "## 6. Metrics",
            """
metrics_df = pd.DataFrame(
    [
        multiclass_metrics(y_train_idx, proba_train, label='in_sample'),
        multiclass_metrics(y_test_idx, proba_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 7. OvR ROC",
            """
roc_path = diag_dir / 'roc_multiclass.png'
roc_fig = plot_roc_multiclass(y_test_idx, proba_test, target_classes, roc_path)
display(roc_fig)
""",
        ),
        (
            "## 8. Confusion Matrix",
            """
cm_path = diag_dir / 'confusion_matrix.png'
cm_fig = plot_confusion_matrix(y_test_idx, pred_test_idx, target_classes, cm_path)
display(cm_fig)
""",
        ),
        (
            "## 9. Learning Curve",
            """
learning_curve_path = diag_dir / 'learning_curve.png'
learning_curve_fig = plot_learning_curve(artifact.training_history, learning_curve_path)
display(learning_curve_fig)
""",
        ),
        (
            "## 10. Feature Importance (Split) Plot",
            """
booster = artifact._get_booster()
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 11. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 12. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 13. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 14. SHAP Plot",
            """
shap_input = x_train.head(min(len(x_train), 64))
pred_train_idx = np.argmax(proba_train, axis=1)
shap_frame = compute_shap_multiclass(
    booster,
    shap_input,
    pred_train_idx[: len(shap_input)],
    n_classes=len(target_classes),
)
shap_path = diag_dir / 'shap_summary.png'
shap_fig = plot_shap_summary(shap_frame, shap_input, shap_path)
display(shap_fig)
""",
        ),
        (
            "## 15. SHAP Table",
            """
shap_mean_abs_df = (
    shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'mean_abs_shap'}
    )
)
shap_mean_abs_path = OUT_DIR / 'shap_mean_abs.csv'
shap_mean_abs_df.to_csv(shap_mean_abs_path, index=False)
display(shap_mean_abs_df)

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    score_path,
    roc_path,
    cm_path,
    learning_curve_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
    shap_path,
    shap_mean_abs_path,
]
artifact_path_for_summary = run_result.artifact_path
""",
        ),
    ]


def _sections_uc04() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
source_df = pd.read_csv(ROOT / 'data' / 'bike_sharing.csv')
source_df = source_df.sort_values('dteday').reset_index(drop=True)
cut = int(len(source_df) * 0.8)
train_df = source_df.iloc[:cut].reset_index(drop=True)
test_df = source_df.iloc[cut:].reset_index(drop=True)

full_path = OUT_DIR / 'timeseries_full.parquet'
train_path = OUT_DIR / 'train.parquet'
test_path = OUT_DIR / 'test.parquet'
source_df.to_parquet(full_path, index=False)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 2. Feature Selection and Category Casting",
            """
target_col = 'cnt'
time_col = 'dteday'
feature_cols = [col for col in source_df.columns if col != target_col]
model_feature_cols = [col for col in feature_cols if col != time_col]
cat_cols = source_df.loc[:, model_feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    categories = sorted(source_df[col].astype(str).dropna().unique().tolist())
    source_df[col] = pd.Categorical(source_df[col], categories=categories)
    train_df[col] = pd.Categorical(train_df[col], categories=categories)
    test_df[col] = pd.Categorical(test_df[col], categories=categories)
source_df.to_parquet(full_path, index=False)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 3. Config",
            """
config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {
        'path': str(full_path),
        'target': target_col,
        'drop_cols': [time_col],
        'categorical': cat_cols,
    },
    'split': {
        'type': 'timeseries',
        'time_col': time_col,
        'n_splits': 4,
        'timeseries_mode': 'expanding',
    },
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Fit Model",
            """
run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
""",
        ),
        (
            "## 5. Prediction",
            """
x_all = source_df.loc[:, model_feature_cols]
y_all = source_df[target_col].to_numpy(dtype=float)
x_train = train_df.loc[:, model_feature_cols]
y_train = train_df[target_col].to_numpy(dtype=float)
x_test = test_df.loc[:, model_feature_cols]
y_test = test_df[target_col].to_numpy(dtype=float)

pred_all = np.asarray(artifact.predict(x_all), dtype=float)
pred_train = np.asarray(artifact.predict(x_train), dtype=float)
pred_test = np.asarray(artifact.predict(x_test), dtype=float)

score_table = build_regression_table(
    x_all,
    y_all,
    np.concatenate([np.zeros(len(train_df), dtype=int), np.ones(len(test_df), dtype=int)]),
    pred_all,
    ['in_sample'] * len(train_df) + ['out_of_sample'] * len(test_df),
)
score_path = OUT_DIR / 'timeseries_detail.csv'
score_table.to_csv(score_path, index=False)
""",
        ),
        (
            "## 6. Metrics",
            """
metrics_df = pd.DataFrame(
    [
        regression_metrics(y_train, pred_train, label='in_sample'),
        regression_metrics(y_test, pred_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 7. Time Series Prediction Plot",
            """
time_index = pd.to_datetime(source_df[time_col])
split_point = pd.to_datetime(test_df[time_col].iloc[0])
pred_path = diag_dir / 'timeseries_prediction.png'
pred_fig = plot_timeseries_prediction(time_index, y_all, pred_all, split_point, pred_path)
display(pred_fig)
""",
        ),
        (
            "## 8. Time Series Residual Plot",
            """
residual_path = diag_dir / 'timeseries_residual.png'
residual_fig = plot_timeseries_residual(time_index, y_all - pred_all, split_point, residual_path)
display(residual_fig)
""",
        ),
        (
            "## 9. Learning Curve",
            """
learning_curve_path = diag_dir / 'learning_curve.png'
learning_curve_fig = plot_learning_curve(artifact.training_history, learning_curve_path)
display(learning_curve_fig)
""",
        ),
        (
            "## 10. Feature Importance (Split) Plot",
            """
booster = artifact._get_booster()
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 11. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 12. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 13. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 14. SHAP Plot",
            """
shap_input = x_train.head(min(len(x_train), 64))
shap_frame = compute_shap(booster, shap_input)
shap_path = diag_dir / 'shap_summary.png'
shap_fig = plot_shap_summary(shap_frame, shap_input, shap_path)
display(shap_fig)
""",
        ),
        (
            "## 15. SHAP Table",
            """
shap_mean_abs_df = (
    shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'mean_abs_shap'}
    )
)
shap_mean_abs_path = OUT_DIR / 'shap_mean_abs.csv'
shap_mean_abs_df.to_csv(shap_mean_abs_path, index=False)
display(shap_mean_abs_df)

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    score_path,
    pred_path,
    residual_path,
    learning_curve_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
    shap_path,
    shap_mean_abs_path,
]
artifact_path_for_summary = run_result.artifact_path
""",
        ),
    ]


def _sections_uc05() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
source_df = pd.read_csv(ROOT / 'data' / 'demo' / 'frontier' / 'train_eval.csv')
source_df = source_df.sort_values('month').reset_index(drop=True)
unique_months = sorted(source_df['month'].unique().tolist())
cut_idx = max(1, int(len(unique_months) * 0.75))
train_months = set(unique_months[:cut_idx])
train_df = source_df[source_df['month'].isin(train_months)].reset_index(drop=True)
test_df = source_df[~source_df['month'].isin(train_months)].reset_index(drop=True)

train_path = OUT_DIR / 'train.parquet'
test_path = OUT_DIR / 'test.parquet'
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 2. Feature Selection and Category Casting",
            """
target_col = 'net_sales'
feature_cols = [col for col in train_df.columns if col != target_col]
cat_cols = train_df.loc[:, feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    categories = sorted(source_df[col].astype(str).dropna().unique().tolist())
    train_df[col] = pd.Categorical(train_df[col], categories=categories)
    test_df[col] = pd.Categorical(test_df[col], categories=categories)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 3. Config",
            """
alpha = 0.9
config = {
    'config_version': 1,
    'task': {'type': 'frontier'},
    'data': {
        'path': str(train_path),
        'target': target_col,
        'categorical': cat_cols,
    },
    'split': {'type': 'kfold', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['quantile'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'frontier': {'alpha': alpha},
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Fit Model",
            """
run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
latest_artifact_path = OUT_DIR / 'latest_artifact_path.txt'
latest_artifact_path.write_text(run_result.artifact_path, encoding='utf-8')
""",
        ),
        (
            "## 5. Prediction",
            """
x_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].to_numpy(dtype=float)
x_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].to_numpy(dtype=float)

pred_train = artifact.predict(x_train)['frontier_pred'].to_numpy(dtype=float)
pred_test = artifact.predict(x_test)['frontier_pred'].to_numpy(dtype=float)

joined_pred = np.concatenate([pred_train, pred_test])
joined_true = np.concatenate([y_train, y_test])
safe_pred = np.where(np.abs(joined_pred) < 1e-12, np.nan, joined_pred)
score_table = build_frontier_table(
    pd.concat([x_train, x_test], ignore_index=True),
    joined_true,
    np.concatenate([np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]),
    joined_pred,
    joined_true / safe_pred,
)
score_path = OUT_DIR / 'frontier_scores.csv'
score_table.to_csv(score_path, index=False)
""",
        ),
        (
            "## 6. Metrics (Frontier + Regression)",
            """
frontier_df = pd.DataFrame(
    [
        frontier_metrics(y_train, pred_train, alpha=alpha, label='in_sample'),
        frontier_metrics(y_test, pred_test, alpha=alpha, label='out_of_sample'),
    ]
)
regression_df = pd.DataFrame(
    [
        regression_metrics(y_train, pred_train, label='in_sample'),
        regression_metrics(y_test, pred_test, label='out_of_sample'),
    ]
).rename(
    columns={
        'rmse': 'reg_rmse',
        'mae': 'reg_mae',
        'mape': 'reg_mape',
        'r2': 'reg_r2',
        'huber': 'reg_huber',
    }
)
metrics_df = frontier_df.merge(regression_df, on='label', how='left')
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 7. Pinball Distribution",
            """
pinball_train = np.maximum(alpha * (y_train - pred_train), (alpha - 1.0) * (y_train - pred_train))
pinball_test = np.maximum(alpha * (y_test - pred_test), (alpha - 1.0) * (y_test - pred_test))
pinball_path = diag_dir / 'pinball_hist.png'
plot_pinball_histogram(pinball_train, pinball_test, pinball_path)
display(Image(filename=str(pinball_path)))
""",
        ),
        (
            "## 8. Frontier Scatter Plot",
            """
scatter_path = diag_dir / 'frontier_scatter.png'
plot_frontier_scatter(y_test, pred_test, scatter_path)
display(Image(filename=str(scatter_path)))
""",
        ),
        (
            "## 9. Learning Curve",
            """
learning_curve_path = diag_dir / 'learning_curve.png'
learning_curve_fig = plot_learning_curve(artifact.training_history, learning_curve_path)
display(learning_curve_fig)
""",
        ),
        (
            "## 10. Feature Importance (Split) Plot",
            """
booster = artifact._get_booster()
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 11. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 12. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 13. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 14. SHAP Plot",
            """
shap_input = x_train.head(min(len(x_train), 64))
shap_frame = compute_shap(booster, shap_input)
shap_path = diag_dir / 'shap_summary.png'
shap_fig = plot_shap_summary(shap_frame, shap_input, shap_path)
display(shap_fig)
""",
        ),
        (
            "## 15. SHAP Table",
            """
shap_mean_abs_df = (
    shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'mean_abs_shap'}
    )
)
shap_mean_abs_path = OUT_DIR / 'shap_mean_abs.csv'
shap_mean_abs_df.to_csv(shap_mean_abs_path, index=False)
display(shap_mean_abs_df)

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    score_path,
    pinball_path,
    scatter_path,
    learning_curve_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
    shap_path,
    shap_mean_abs_path,
    latest_artifact_path,
]
artifact_path_for_summary = run_result.artifact_path
""",
        ),
    ]


def _sections_uc06() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
source_df = pd.read_csv(ROOT / 'data' / 'lalonde.csv')
source_path = OUT_DIR / 'lalonde.csv'
source_df.to_csv(source_path, index=False)
target_col = 'outcome'
treatment_col = 'treatment'
""",
        ),
        (
            "## 2. Feature Selection",
            """
covariate_cols = [col for col in source_df.columns if col not in {target_col, treatment_col}]
if not covariate_cols:
    raise ValueError('No covariate columns found for DR estimation.')
covariates = source_df.loc[:, covariate_cols].reset_index(drop=True)
""",
        ),
        (
            "## 3. Config",
            """
config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {
        'path': str(source_path),
        'target': target_col,
    },
    'split': {'type': 'kfold', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'causal': {
        'method': 'dr',
        'treatment_col': treatment_col,
        'estimand': 'att',
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Run Estimation",
            """
result = estimate_dr(config)
obs_df = pd.read_parquet(result.metadata['observation_path'])
""",
        ),
        (
            "## 5. Metrics Summary",
            """
overlap_stats = compute_overlap_stats(
    obs_df['e_hat'].to_numpy(dtype=float),
    obs_df['treatment'].to_numpy(dtype=int),
)
metrics_df = pd.DataFrame(
    [
        {
            'label': 'dr_att',
            'estimate': result.estimate,
            'std_error': result.std_error,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            'overlap_metric': result.metrics.get('overlap_metric'),
            'smd_max_unweighted': result.metrics.get('smd_max_unweighted'),
            'smd_max_weighted': result.metrics.get('smd_max_weighted'),
            **overlap_stats,
        }
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 6. DR Table",
            """
dr_table = build_dr_table(obs_df)
dr_table_path = OUT_DIR / 'dr_table.csv'
dr_table.to_csv(dr_table_path, index=False)
display(dr_table.head(10))
""",
        ),
        (
            "## 7. Balance Table",
            """
smd_unweighted = compute_balance_smd(covariates, source_df[treatment_col])
smd_weighted = compute_balance_smd(covariates, source_df[treatment_col], obs_df['weight'])
balance_df = smd_unweighted.merge(
    smd_weighted,
    on='feature',
    suffixes=('_unweighted', '_weighted'),
    how='outer',
)
balance_path = OUT_DIR / 'balance_smd.csv'
balance_df.to_csv(balance_path, index=False)
display(balance_df)
""",
        ),
        (
            "## 8. Influence Function Distribution",
            """
if_path = diag_dir / 'if_distribution.png'
plot_if_distribution(obs_df['psi'].to_numpy(dtype=float), if_path)
display(Image(filename=str(if_path)))
""",
        ),
        (
            "## 9. Propensity Distribution",
            """
prop_path = diag_dir / 'propensity_distribution.png'
plot_propensity_distribution(
    obs_df['e_hat'].to_numpy(dtype=float),
    obs_df['treatment'].to_numpy(dtype=int),
    prop_path,
)
display(Image(filename=str(prop_path)))
""",
        ),
        (
            "## 10. Weight Distribution",
            """
weight_path = diag_dir / 'weight_distribution.png'
plot_weight_distribution(obs_df['weight'].to_numpy(dtype=float), weight_path)
display(Image(filename=str(weight_path)))
""",
        ),
        (
            "## 11. Love Plot",
            """
love_path = diag_dir / 'love_plot.png'
plot_love_plot(smd_unweighted, smd_weighted, love_path)
display(Image(filename=str(love_path)))
""",
        ),
        (
            "## 12. Prepare Summary",
            """
summary_outputs = [
    source_path,
    metrics_path,
    dr_table_path,
    balance_path,
    if_path,
    prop_path,
    weight_path,
    love_path,
]
artifact_path_for_summary = Path(result.metadata.get('summary_path', OUT_DIR)).parent
""",
        ),
    ]


def _sections_uc07() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
source_df = pd.read_csv(ROOT / 'data' / 'cps_panel.csv')
source_path = OUT_DIR / 'cps_panel.csv'
source_df.to_csv(source_path, index=False)
target_col = 'target'
treatment_col = 'treatment'
""",
        ),
        (
            "## 2. Feature Selection",
            """
covariate_cols = [col for col in ['age', 'skill'] if col in source_df.columns]
if not covariate_cols:
    raise ValueError('No covariate columns found for DR-DiD estimation.')
""",
        ),
        (
            "## 3. Config",
            """
config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {
        'path': str(source_path),
        'target': target_col,
    },
    'split': {'type': 'group', 'n_splits': 3, 'group_col': 'unit_id', 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'causal': {
        'method': 'dr_did',
        'treatment_col': treatment_col,
        'estimand': 'att',
        'design': 'panel',
        'time_col': 'time',
        'post_col': 'post',
        'unit_id_col': 'unit_id',
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Run Estimation",
            """
result = estimate_dr(config)
obs_df = pd.read_parquet(result.metadata['observation_path'])
""",
        ),
        (
            "## 5. Metrics Summary",
            """
metrics_df = pd.DataFrame(
    [
        {
            'label': 'drdid_att',
            'estimate': result.estimate,
            'std_error': result.std_error,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            'overlap_metric': result.metrics.get('overlap_metric'),
            'smd_max_unweighted': result.metrics.get('smd_max_unweighted'),
            'smd_max_weighted': result.metrics.get('smd_max_weighted'),
        }
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 6. DR-DiD Table",
            """
drdid_table = build_drdid_table(obs_df)
drdid_table_path = OUT_DIR / 'drdid_table.csv'
drdid_table.to_csv(drdid_table_path, index=False)
display(drdid_table.head(10))
""",
        ),
        (
            "## 7. Balance Table",
            """
base_cov = (
    source_df.loc[source_df['post'] == 0, ['unit_id', *covariate_cols, treatment_col]]
    .sort_values('unit_id')
    .reset_index(drop=True)
)
smd_unweighted = compute_balance_smd(base_cov[covariate_cols], base_cov[treatment_col])
smd_weighted = compute_balance_smd(base_cov[covariate_cols], base_cov[treatment_col], obs_df['weight'])
balance_df = smd_unweighted.merge(
    smd_weighted,
    on='feature',
    suffixes=('_unweighted', '_weighted'),
    how='outer',
)
balance_path = OUT_DIR / 'balance_smd.csv'
balance_df.to_csv(balance_path, index=False)
display(balance_df)
""",
        ),
        (
            "## 8. Influence Function Distribution",
            """
if_path = diag_dir / 'if_distribution.png'
plot_if_distribution(obs_df['psi'].to_numpy(dtype=float), if_path)
display(Image(filename=str(if_path)))
""",
        ),
        (
            "## 9. Parallel Trends",
            """
means = (
    source_df.groupby(['post', treatment_col])[target_col]
    .mean()
    .unstack(treatment_col)
    .reindex(index=[0, 1], columns=[0, 1])
)
parallel_path = diag_dir / 'parallel_trends.png'
plot_parallel_trends(
    means[1].to_numpy(dtype=float),
    means[0].to_numpy(dtype=float),
    ['pre', 'post'],
    parallel_path,
)
display(Image(filename=str(parallel_path)))
""",
        ),
        (
            "## 10. Propensity / Weight Distribution",
            """
prop_path = diag_dir / 'propensity_distribution.png'
plot_propensity_distribution(
    obs_df['e_hat'].to_numpy(dtype=float),
    obs_df['treatment'].to_numpy(dtype=int),
    prop_path,
)
weight_path = diag_dir / 'weight_distribution.png'
plot_weight_distribution(obs_df['weight'].to_numpy(dtype=float), weight_path)
display(Image(filename=str(prop_path)))
display(Image(filename=str(weight_path)))
""",
        ),
        (
            "## 11. Love Plot",
            """
love_path = diag_dir / 'love_plot.png'
plot_love_plot(smd_unweighted, smd_weighted, love_path)
display(Image(filename=str(love_path)))
""",
        ),
        (
            "## 12. Prepare Summary",
            """
summary_outputs = [
    source_path,
    metrics_path,
    drdid_table_path,
    balance_path,
    if_path,
    parallel_path,
    prop_path,
    weight_path,
    love_path,
]
artifact_path_for_summary = Path(result.metadata.get('summary_path', OUT_DIR)).parent
""",
        ),
    ]


def _sections_uc08() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Resolve Artifact",
            """
def _resolve_artifact_path() -> Path:
    uc11_base = ROOT / 'examples' / 'out' / 'phase35_uc11_frontier_tune_evaluate'
    uc05_base = ROOT / 'examples' / 'out' / 'phase35_uc05_frontier_fit_evaluate'
    marker_candidates = [
        uc11_base / 'latest_artifact_path.txt',
        uc05_base / 'latest_artifact_path.txt',
    ]
    for marker in marker_candidates:
        if not marker.exists():
            continue
        candidate = Path(marker.read_text(encoding='utf-8').strip())
        if candidate.exists():
            return candidate

    for base_dir in [uc11_base, uc05_base]:
        candidates = sorted((base_dir / 'artifacts').glob('*'))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError('No UC-11/UC-5 artifact found.')

artifact_path = _resolve_artifact_path()
artifact = Artifact.load(artifact_path)
target_col = str(artifact.run_config.data.target)
booster = artifact._get_booster()
""",
        ),
        (
            "## 2. Data Preparation and Category Alignment",
            """
latest_df = pd.read_csv(ROOT / 'data' / 'demo' / 'frontier' / 'latest.csv')
cat_cols = [str(col) for col in artifact.run_config.data.categorical if col in latest_df.columns]
for col, categories in zip(cat_cols, booster.pandas_categorical, strict=False):
    latest_df[col] = pd.Categorical(latest_df[col], categories=categories)
latest_path = OUT_DIR / 'latest.csv'
latest_df.to_csv(latest_path, index=False)
""",
        ),
        (
            "## 3. Run evaluate()",
            """
eval_result = evaluate(artifact, latest_df)
""",
        ),
        (
            "## 4. Prediction and Evaluation Table",
            """
x_eval = latest_df.drop(columns=[target_col])
y_eval = latest_df[target_col].to_numpy(dtype=float)
pred_frame = artifact.predict(x_eval)
if isinstance(pred_frame, pd.DataFrame):
    pred_eval = pred_frame['frontier_pred'].to_numpy(dtype=float)
else:
    pred_eval = np.asarray(pred_frame, dtype=float)

eval_table = build_regression_table(
    x_eval,
    y_eval,
    np.ones(len(x_eval), dtype=int),
    pred_eval,
    ['out_of_time'] * len(x_eval),
)
eval_table_path = OUT_DIR / 'eval_table.csv'
eval_table.to_csv(eval_table_path, index=False)
display(eval_table.head(10))
""",
        ),
        (
            "## 5. Metrics Summary",
            """
recomputed = regression_metrics(y_eval, pred_eval, label='artifact_recomputed')
metrics_df = pd.DataFrame(
    [
        {'label': 'artifact_evaluate', **eval_result.metrics},
        recomputed,
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 6. Compare Table",
            """
compare_rows = []
for metric in ['rmse', 'mae', 'mape', 'r2', 'huber']:
    artifact_value = eval_result.metrics.get(metric)
    recomputed_value = recomputed.get(metric)
    if artifact_value is None and recomputed_value is None:
        continue
    compare_rows.append(
        {
            'metric': metric,
            'evaluate_value': None if artifact_value is None else float(artifact_value),
            'recomputed_value': None if recomputed_value is None else float(recomputed_value),
            'delta': (
                None
                if artifact_value is None or recomputed_value is None
                else float(recomputed_value) - float(artifact_value)
            ),
        }
    )
compare_df = pd.DataFrame(compare_rows)
if compare_df.empty:
    compare_df = pd.DataFrame(
        columns=['metric', 'evaluate_value', 'recomputed_value', 'delta']
    )
compare_path = OUT_DIR / 'compare.csv'
compare_df.to_csv(compare_path, index=False)
display(compare_df)
""",
        ),
        (
            "## 7. Residual Visualization",
            """
residual = y_eval - pred_eval
residual_path = diag_dir / 'residual_hist.png'
residual_fig = plot_error_histogram(
    residual,
    residual,
    metrics_df.iloc[-1].to_dict(),
    metrics_df.iloc[-1].to_dict(),
    residual_path,
)
display(residual_fig)
""",
        ),
        (
            "## 8. Feature Importance (Split) Plot",
            """
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 9. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 10. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 11. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 12. Prepare Summary",
            """
summary_outputs = [
    latest_path,
    metrics_path,
    eval_table_path,
    compare_path,
    residual_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
]
artifact_path_for_summary = artifact_path
""",
        ),
    ]


def _sections_uc09() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
from sklearn.model_selection import train_test_split

source_df = pd.read_csv(ROOT / 'data' / 'titanic.csv')
train_df, test_df = train_test_split(
    source_df,
    test_size=0.25,
    random_state=42,
    stratify=source_df['Survived'],
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = OUT_DIR / 'train.parquet'
test_path = OUT_DIR / 'test.parquet'
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 2. Feature Selection and Category Casting",
            """
target_col = 'Survived'
feature_cols = [col for col in train_df.columns if col != target_col]
cat_cols = train_df.loc[:, feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    categories = sorted(source_df[col].astype(str).dropna().unique().tolist())
    train_df[col] = pd.Categorical(train_df[col], categories=categories)
    test_df[col] = pd.Categorical(test_df[col], categories=categories)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 3. Base Config",
            """
base_config = {
    'config_version': 1,
    'task': {'type': 'binary'},
    'data': {
        'path': str(train_path),
        'target': target_col,
        'categorical': cat_cols,
    },
    'split': {'type': 'stratified', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['logloss', 'auc'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'postprocess': {'calibration': 'platt'},
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Tuning Config",
            """
tune_config = json.loads(json.dumps(base_config))
tune_config['tuning'] = {
    'enabled': True,
    'n_trials': 3,
    'resume': True,
    'preset': 'standard',
    'objective': 'brier',
}
""",
        ),
        (
            "## 5. Run tune()",
            """
tune_result = tune(tune_config)
trials_df = pd.read_parquet(tune_result.metadata['trials_path'])
tuning_trials_path = OUT_DIR / 'tuning_trials.csv'
trials_df.to_csv(tuning_trials_path, index=False)
best_params_path = OUT_DIR / 'best_params.json'
best_params_path.write_text(json.dumps(tune_result.best_params, indent=2), encoding='utf-8')
display(trials_df.head(10))
""",
        ),
        (
            "## 6. Apply best_params",
            """
fit_config = json.loads(json.dumps(base_config))
for key, value in tune_result.best_params.items():
    if key.startswith('train.'):
        fit_config['train'][key.split('.', 1)[1]] = value
    else:
        fit_config['train'].setdefault('lgb_params', {})[key] = value
""",
        ),
        (
            "## 7. Fit Model",
            """
run_result = fit(fit_config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
""",
        ),
        (
            "## 8. Prediction",
            """
x_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].to_numpy(dtype=int)
x_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].to_numpy(dtype=int)

pred_train = artifact.predict(x_train)
pred_test = artifact.predict(x_test)
score_train = pred_train['p_cal'].to_numpy(dtype=float)
score_test = pred_test['p_cal'].to_numpy(dtype=float)
pred_label_test = pred_test['label_pred'].to_numpy(dtype=int)

score_table = build_binary_table(
    pd.concat([x_train, x_test], ignore_index=True),
    np.concatenate([y_train, y_test]),
    np.concatenate([np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]),
    np.concatenate([score_train, score_test]),
    ['in_sample'] * len(x_train) + ['out_of_sample'] * len(x_test),
)
score_path = OUT_DIR / 'binary_scores.csv'
score_table.to_csv(score_path, index=False)
""",
        ),
        (
            "## 9. Metrics",
            """
metrics_df = pd.DataFrame(
    [
        binary_metrics(y_train, score_train, label='in_sample'),
        binary_metrics(y_test, score_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 10. ROC",
            """
roc_path = diag_dir / 'roc_in_out.png'
roc_fig = plot_roc_comparison(y_train, score_train, y_test, score_test, roc_path)
display(roc_fig)
""",
        ),
        (
            "## 11. Confusion Matrix",
            """
cm_path = diag_dir / 'confusion_matrix.png'
cm_fig = plot_confusion_matrix(y_test, pred_label_test, ['0', '1'], cm_path)
display(cm_fig)
""",
        ),
        (
            "## 12. Learning Curve",
            """
learning_curve_path = diag_dir / 'learning_curve.png'
learning_curve_fig = plot_learning_curve(artifact.training_history, learning_curve_path)
display(learning_curve_fig)
""",
        ),
        (
            "## 13. Feature Importance (Split) Plot",
            """
booster = artifact._get_booster()
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 14. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 15. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 16. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 17. SHAP Plot",
            """
shap_input = x_train.head(min(len(x_train), 64))
shap_frame = compute_shap(booster, shap_input)
shap_path = diag_dir / 'shap_summary.png'
shap_fig = plot_shap_summary(shap_frame, shap_input, shap_path)
display(shap_fig)
""",
        ),
        (
            "## 18. SHAP Table",
            """
shap_mean_abs_df = (
    shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'mean_abs_shap'}
    )
)
shap_mean_abs_path = OUT_DIR / 'shap_mean_abs.csv'
shap_mean_abs_df.to_csv(shap_mean_abs_path, index=False)
display(shap_mean_abs_df)

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    score_path,
    tuning_trials_path,
    best_params_path,
    roc_path,
    cm_path,
    learning_curve_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
    shap_path,
    shap_mean_abs_path,
]
artifact_path_for_summary = run_result.artifact_path
""",
        ),
    ]


def _sections_uc10() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
source_df = pd.read_csv(ROOT / 'data' / 'bike_sharing.csv')
source_df = source_df.sort_values('dteday').reset_index(drop=True)
cut = int(len(source_df) * 0.8)
train_df = source_df.iloc[:cut].reset_index(drop=True)
test_df = source_df.iloc[cut:].reset_index(drop=True)

full_path = OUT_DIR / 'timeseries_full.parquet'
train_path = OUT_DIR / 'train.parquet'
test_path = OUT_DIR / 'test.parquet'
source_df.to_parquet(full_path, index=False)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 2. Feature Selection and Category Casting",
            """
target_col = 'cnt'
time_col = 'dteday'
feature_cols = [col for col in source_df.columns if col != target_col]
model_feature_cols = [col for col in feature_cols if col != time_col]
cat_cols = source_df.loc[:, model_feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    categories = sorted(source_df[col].astype(str).dropna().unique().tolist())
    source_df[col] = pd.Categorical(source_df[col], categories=categories)
    train_df[col] = pd.Categorical(train_df[col], categories=categories)
    test_df[col] = pd.Categorical(test_df[col], categories=categories)
source_df.to_parquet(full_path, index=False)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 3. Base Config",
            """
base_config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {
        'path': str(full_path),
        'target': target_col,
        'drop_cols': [time_col],
        'categorical': cat_cols,
    },
    'split': {
        'type': 'timeseries',
        'time_col': time_col,
        'n_splits': 4,
        'timeseries_mode': 'expanding',
    },
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Tuning Config",
            """
tune_config = json.loads(json.dumps(base_config))
tune_config['tuning'] = {
    'enabled': True,
    'n_trials': 3,
    'resume': True,
    'preset': 'standard',
    'objective': 'rmse',
}
""",
        ),
        (
            "## 5. Run tune()",
            """
tune_result = tune(tune_config)
trials_df = pd.read_parquet(tune_result.metadata['trials_path'])
tuning_trials_path = OUT_DIR / 'tuning_trials.csv'
trials_df.to_csv(tuning_trials_path, index=False)
best_params_path = OUT_DIR / 'best_params.json'
best_params_path.write_text(json.dumps(tune_result.best_params, indent=2), encoding='utf-8')
display(trials_df.head(10))
""",
        ),
        (
            "## 6. Apply best_params",
            """
fit_config = json.loads(json.dumps(base_config))
for key, value in tune_result.best_params.items():
    if key.startswith('train.'):
        fit_config['train'][key.split('.', 1)[1]] = value
    else:
        fit_config['train'].setdefault('lgb_params', {})[key] = value
""",
        ),
        (
            "## 7. Fit Model",
            """
run_result = fit(fit_config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
""",
        ),
        (
            "## 8. Prediction",
            """
x_all = source_df.loc[:, model_feature_cols]
y_all = source_df[target_col].to_numpy(dtype=float)
x_train = train_df.loc[:, model_feature_cols]
y_train = train_df[target_col].to_numpy(dtype=float)
x_test = test_df.loc[:, model_feature_cols]
y_test = test_df[target_col].to_numpy(dtype=float)

pred_all = np.asarray(artifact.predict(x_all), dtype=float)
pred_train = np.asarray(artifact.predict(x_train), dtype=float)
pred_test = np.asarray(artifact.predict(x_test), dtype=float)

score_table = build_regression_table(
    x_all,
    y_all,
    np.concatenate([np.zeros(len(train_df), dtype=int), np.ones(len(test_df), dtype=int)]),
    pred_all,
    ['in_sample'] * len(train_df) + ['out_of_sample'] * len(test_df),
)
score_path = OUT_DIR / 'timeseries_detail.csv'
score_table.to_csv(score_path, index=False)
""",
        ),
        (
            "## 9. Metrics",
            """
metrics_df = pd.DataFrame(
    [
        regression_metrics(y_train, pred_train, label='in_sample'),
        regression_metrics(y_test, pred_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 10. Time Series Prediction Plot",
            """
time_index = pd.to_datetime(source_df[time_col])
split_point = pd.to_datetime(test_df[time_col].iloc[0])
pred_path = diag_dir / 'timeseries_prediction.png'
pred_fig = plot_timeseries_prediction(time_index, y_all, pred_all, split_point, pred_path)
display(pred_fig)
""",
        ),
        (
            "## 11. Time Series Residual Plot",
            """
residual_path = diag_dir / 'timeseries_residual.png'
residual_fig = plot_timeseries_residual(time_index, y_all - pred_all, split_point, residual_path)
display(residual_fig)
""",
        ),
        (
            "## 12. Learning Curve",
            """
learning_curve_path = diag_dir / 'learning_curve.png'
learning_curve_fig = plot_learning_curve(artifact.training_history, learning_curve_path)
display(learning_curve_fig)
""",
        ),
        (
            "## 13. Feature Importance (Split) Plot",
            """
booster = artifact._get_booster()
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 14. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 15. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 16. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 17. SHAP Plot",
            """
shap_input = x_train.head(min(len(x_train), 64))
shap_frame = compute_shap(booster, shap_input)
shap_path = diag_dir / 'shap_summary.png'
shap_fig = plot_shap_summary(shap_frame, shap_input, shap_path)
display(shap_fig)
""",
        ),
        (
            "## 18. SHAP Table",
            """
shap_mean_abs_df = (
    shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'mean_abs_shap'}
    )
)
shap_mean_abs_path = OUT_DIR / 'shap_mean_abs.csv'
shap_mean_abs_df.to_csv(shap_mean_abs_path, index=False)
display(shap_mean_abs_df)

summary_outputs = [
    full_path,
    train_path,
    test_path,
    metrics_path,
    score_path,
    tuning_trials_path,
    best_params_path,
    pred_path,
    residual_path,
    learning_curve_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
    shap_path,
    shap_mean_abs_path,
]
artifact_path_for_summary = run_result.artifact_path
""",
        ),
    ]


def _sections_uc11() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
source_df = pd.read_csv(ROOT / 'data' / 'demo' / 'frontier' / 'train_eval.csv')
source_df = source_df.sort_values('month').reset_index(drop=True)
unique_months = sorted(source_df['month'].unique().tolist())
cut_idx = max(1, int(len(unique_months) * 0.75))
train_months = set(unique_months[:cut_idx])
train_df = source_df[source_df['month'].isin(train_months)].reset_index(drop=True)
test_df = source_df[~source_df['month'].isin(train_months)].reset_index(drop=True)

train_path = OUT_DIR / 'train.parquet'
test_path = OUT_DIR / 'test.parquet'
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 2. Feature Selection and Category Casting",
            """
target_col = 'net_sales'
feature_cols = [col for col in train_df.columns if col != target_col]
cat_cols = train_df.loc[:, feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    categories = sorted(source_df[col].astype(str).dropna().unique().tolist())
    train_df[col] = pd.Categorical(train_df[col], categories=categories)
    test_df[col] = pd.Categorical(test_df[col], categories=categories)
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
""",
        ),
        (
            "## 3. Base Config",
            """
alpha = 0.9
base_config = {
    'config_version': 1,
    'task': {'type': 'frontier'},
    'data': {
        'path': str(train_path),
        'target': target_col,
        'categorical': cat_cols,
    },
    'split': {'type': 'kfold', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['quantile'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'frontier': {'alpha': alpha},
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Tuning Config",
            """
tune_config = json.loads(json.dumps(base_config))
tune_config['tuning'] = {
    'enabled': True,
    'n_trials': 3,
    'resume': True,
    'preset': 'standard',
    'objective': 'pinball_coverage_penalty',
}
""",
        ),
        (
            "## 5. Run tune()",
            """
tune_result = tune(tune_config)
trials_df = pd.read_parquet(tune_result.metadata['trials_path'])
tuning_trials_path = OUT_DIR / 'tuning_trials.csv'
trials_df.to_csv(tuning_trials_path, index=False)
best_params_path = OUT_DIR / 'best_params.json'
best_params_path.write_text(json.dumps(tune_result.best_params, indent=2), encoding='utf-8')
display(trials_df.head(10))
""",
        ),
        (
            "## 6. Apply best_params",
            """
fit_config = json.loads(json.dumps(base_config))
for key, value in tune_result.best_params.items():
    if key.startswith('train.'):
        fit_config['train'][key.split('.', 1)[1]] = value
    else:
        fit_config['train'].setdefault('lgb_params', {})[key] = value
""",
        ),
        (
            "## 7. Fit Model",
            """
run_result = fit(fit_config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
latest_artifact_path = OUT_DIR / 'latest_artifact_path.txt'
latest_artifact_path.write_text(run_result.artifact_path, encoding='utf-8')
""",
        ),
        (
            "## 8. Prediction",
            """
x_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].to_numpy(dtype=float)
x_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].to_numpy(dtype=float)

pred_train = artifact.predict(x_train)['frontier_pred'].to_numpy(dtype=float)
pred_test = artifact.predict(x_test)['frontier_pred'].to_numpy(dtype=float)

joined_pred = np.concatenate([pred_train, pred_test])
joined_true = np.concatenate([y_train, y_test])
safe_pred = np.where(np.abs(joined_pred) < 1e-12, np.nan, joined_pred)
score_table = build_frontier_table(
    pd.concat([x_train, x_test], ignore_index=True),
    joined_true,
    np.concatenate([np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]),
    joined_pred,
    joined_true / safe_pred,
)
score_path = OUT_DIR / 'frontier_scores.csv'
score_table.to_csv(score_path, index=False)
""",
        ),
        (
            "## 9. Metrics（Frontier + Regression）",
            """
frontier_df = pd.DataFrame(
    [
        frontier_metrics(y_train, pred_train, alpha=alpha, label='in_sample'),
        frontier_metrics(y_test, pred_test, alpha=alpha, label='out_of_sample'),
    ]
)
regression_df = pd.DataFrame(
    [
        regression_metrics(y_train, pred_train, label='in_sample'),
        regression_metrics(y_test, pred_test, label='out_of_sample'),
    ]
).rename(
    columns={
        'rmse': 'reg_rmse',
        'mae': 'reg_mae',
        'mape': 'reg_mape',
        'r2': 'reg_r2',
        'huber': 'reg_huber',
    }
)
metrics_df = frontier_df.merge(regression_df, on='label', how='left')
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 10. Pinball Distribution",
            """
pinball_train = np.maximum(alpha * (y_train - pred_train), (alpha - 1.0) * (y_train - pred_train))
pinball_test = np.maximum(alpha * (y_test - pred_test), (alpha - 1.0) * (y_test - pred_test))
pinball_path = diag_dir / 'pinball_hist.png'
plot_pinball_histogram(pinball_train, pinball_test, pinball_path)
display(Image(filename=str(pinball_path)))
""",
        ),
        (
            "## 11. Frontier Scatter Plot",
            """
scatter_path = diag_dir / 'frontier_scatter.png'
plot_frontier_scatter(y_test, pred_test, scatter_path)
display(Image(filename=str(scatter_path)))
""",
        ),
        (
            "## 12. Learning Curve",
            """
learning_curve_path = diag_dir / 'learning_curve.png'
learning_curve_fig = plot_learning_curve(artifact.training_history, learning_curve_path)
display(learning_curve_fig)
""",
        ),
        (
            "## 13. Feature Importance (Split) Plot",
            """
booster = artifact._get_booster()
importance_split_df = compute_importance(booster, importance_type='split', top_n=20)
importance_split_path = diag_dir / 'importance_split.png'
importance_split_fig = plot_feature_importance(importance_split_df, 'split', importance_split_path)
display(importance_split_fig)
""",
        ),
        (
            "## 14. Feature Importance (Split) Table",
            """
importance_split_csv_path = OUT_DIR / 'importance_split.csv'
importance_split_df.to_csv(importance_split_csv_path, index=False)
display(importance_split_df)
""",
        ),
        (
            "## 15. Feature Importance (Gain) Plot",
            """
importance_gain_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_gain_path = diag_dir / 'importance_gain.png'
importance_gain_fig = plot_feature_importance(importance_gain_df, 'gain', importance_gain_path)
display(importance_gain_fig)
""",
        ),
        (
            "## 16. Feature Importance (Gain) Table",
            """
importance_gain_csv_path = OUT_DIR / 'importance_gain.csv'
importance_gain_df.to_csv(importance_gain_csv_path, index=False)
display(importance_gain_df)
""",
        ),
        (
            "## 17. SHAP Plot",
            """
shap_input = x_train.head(min(len(x_train), 64))
shap_frame = compute_shap(booster, shap_input)
shap_path = diag_dir / 'shap_summary.png'
shap_fig = plot_shap_summary(shap_frame, shap_input, shap_path)
display(shap_fig)
""",
        ),
        (
            "## 18. SHAP Table",
            """
shap_mean_abs_df = (
    shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'mean_abs_shap'}
    )
)
shap_mean_abs_path = OUT_DIR / 'shap_mean_abs.csv'
shap_mean_abs_df.to_csv(shap_mean_abs_path, index=False)
display(shap_mean_abs_df)

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    score_path,
    tuning_trials_path,
    best_params_path,
    pinball_path,
    scatter_path,
    learning_curve_path,
    importance_split_path,
    importance_split_csv_path,
    importance_gain_path,
    importance_gain_csv_path,
    shap_path,
    shap_mean_abs_path,
    latest_artifact_path,
]
artifact_path_for_summary = run_result.artifact_path
""",
        ),
    ]


def _sections_uc12() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
source_df = pd.read_csv(ROOT / 'data' / 'lalonde.csv')
source_path = OUT_DIR / 'lalonde.csv'
source_df.to_csv(source_path, index=False)
target_col = 'outcome'
treatment_col = 'treatment'
""",
        ),
        (
            "## 2. Feature Selection",
            """
covariate_cols = [col for col in source_df.columns if col not in {target_col, treatment_col}]
if not covariate_cols:
    raise ValueError('No covariate columns found for DR tuning/estimation.')
covariates = source_df.loc[:, covariate_cols].reset_index(drop=True)
""",
        ),
        (
            "## 3. Base Config",
            """
base_config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {
        'path': str(source_path),
        'target': target_col,
    },
    'split': {'type': 'kfold', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'causal': {
        'method': 'dr',
        'treatment_col': treatment_col,
        'estimand': 'att',
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Tuning Config",
            """
tune_config = json.loads(json.dumps(base_config))
tune_config['tuning'] = {
    'enabled': True,
    'n_trials': 3,
    'resume': True,
    'preset': 'standard',
    'objective': 'dr_balance_priority',
}
""",
        ),
        (
            "## 5. Run tune()",
            """
tune_result = tune(tune_config)
trials_df = pd.read_parquet(tune_result.metadata['trials_path'])
tuning_trials_path = OUT_DIR / 'tuning_trials.csv'
trials_df.to_csv(tuning_trials_path, index=False)
best_params_path = OUT_DIR / 'best_params.json'
best_params_path.write_text(json.dumps(tune_result.best_params, indent=2), encoding='utf-8')
display(trials_df.head(10))
""",
        ),
        (
            "## 6. Apply best_params",
            """
estimate_config = json.loads(json.dumps(base_config))
for key, value in tune_result.best_params.items():
    if key.startswith('train.'):
        estimate_config['train'][key.split('.', 1)[1]] = value
    else:
        estimate_config['train'].setdefault('lgb_params', {})[key] = value
""",
        ),
        (
            "## 7. Run Estimation",
            """
result = estimate_dr(estimate_config)
obs_df = pd.read_parquet(result.metadata['observation_path'])
""",
        ),
        (
            "## 8. Metrics Summary",
            """
overlap_stats = compute_overlap_stats(
    obs_df['e_hat'].to_numpy(dtype=float),
    obs_df['treatment'].to_numpy(dtype=int),
)
metrics_df = pd.DataFrame(
    [
        {
            'label': 'dr_att',
            'estimate': result.estimate,
            'std_error': result.std_error,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            'overlap_metric': result.metrics.get('overlap_metric'),
            'smd_max_unweighted': result.metrics.get('smd_max_unweighted'),
            'smd_max_weighted': result.metrics.get('smd_max_weighted'),
            **overlap_stats,
        }
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 9. DR Table",
            """
dr_table = build_dr_table(obs_df)
dr_table_path = OUT_DIR / 'dr_table.csv'
dr_table.to_csv(dr_table_path, index=False)
display(dr_table.head(10))
""",
        ),
        (
            "## 10. Balance Table",
            """
smd_unweighted = compute_balance_smd(covariates, source_df[treatment_col])
smd_weighted = compute_balance_smd(covariates, source_df[treatment_col], obs_df['weight'])
balance_df = smd_unweighted.merge(
    smd_weighted,
    on='feature',
    suffixes=('_unweighted', '_weighted'),
    how='outer',
)
balance_path = OUT_DIR / 'balance_smd.csv'
balance_df.to_csv(balance_path, index=False)
display(balance_df)
""",
        ),
        (
            "## 11. Influence Function Distribution",
            """
if_path = diag_dir / 'if_distribution.png'
plot_if_distribution(obs_df['psi'].to_numpy(dtype=float), if_path)
display(Image(filename=str(if_path)))
""",
        ),
        (
            "## 12. Propensity Distribution",
            """
prop_path = diag_dir / 'propensity_distribution.png'
plot_propensity_distribution(
    obs_df['e_hat'].to_numpy(dtype=float),
    obs_df['treatment'].to_numpy(dtype=int),
    prop_path,
)
display(Image(filename=str(prop_path)))
""",
        ),
        (
            "## 13. Weight Distribution",
            """
weight_path = diag_dir / 'weight_distribution.png'
plot_weight_distribution(obs_df['weight'].to_numpy(dtype=float), weight_path)
display(Image(filename=str(weight_path)))
""",
        ),
        (
            "## 14. Love Plot",
            """
love_path = diag_dir / 'love_plot.png'
plot_love_plot(smd_unweighted, smd_weighted, love_path)
display(Image(filename=str(love_path)))
""",
        ),
        (
            "## 15. Prepare Summary",
            """
summary_outputs = [
    source_path,
    metrics_path,
    dr_table_path,
    balance_path,
    tuning_trials_path,
    best_params_path,
    if_path,
    prop_path,
    weight_path,
    love_path,
]
artifact_path_for_summary = Path(result.metadata.get('summary_path', OUT_DIR)).parent
""",
        ),
    ]


def _sections_uc13() -> list[tuple[str, str]]:
    return [
        (
            "## 1. Data Preparation",
            """
source_df = pd.read_csv(ROOT / 'data' / 'cps_panel.csv')
source_path = OUT_DIR / 'cps_panel.csv'
source_df.to_csv(source_path, index=False)
target_col = 'target'
treatment_col = 'treatment'
""",
        ),
        (
            "## 2. Feature Selection",
            """
covariate_cols = [col for col in ['age', 'skill'] if col in source_df.columns]
if not covariate_cols:
    raise ValueError('No covariate columns found for DR-DiD tuning/estimation.')
""",
        ),
        (
            "## 3. Base Config",
            """
base_config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {
        'path': str(source_path),
        'target': target_col,
    },
    'split': {'type': 'group', 'n_splits': 3, 'group_col': 'unit_id', 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 1200,
        'early_stopping_rounds': 120,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.02,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'first_metric_only': True,
        },
    },
    'causal': {
        'method': 'dr_did',
        'treatment_col': treatment_col,
        'estimand': 'att',
        'design': 'panel',
        'time_col': 'time',
        'post_col': 'post',
        'unit_id_col': 'unit_id',
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}
""",
        ),
        (
            "## 4. Tuning Config",
            """
tune_config = json.loads(json.dumps(base_config))
tune_config['tuning'] = {
    'enabled': True,
    'n_trials': 3,
    'resume': True,
    'preset': 'standard',
    'objective': 'drdid_balance_priority',
}
""",
        ),
        (
            "## 5. Run tune()",
            """
tune_result = tune(tune_config)
trials_df = pd.read_parquet(tune_result.metadata['trials_path'])
tuning_trials_path = OUT_DIR / 'tuning_trials.csv'
trials_df.to_csv(tuning_trials_path, index=False)
best_params_path = OUT_DIR / 'best_params.json'
best_params_path.write_text(json.dumps(tune_result.best_params, indent=2), encoding='utf-8')
display(trials_df.head(10))
""",
        ),
        (
            "## 6. Apply best_params",
            """
estimate_config = json.loads(json.dumps(base_config))
for key, value in tune_result.best_params.items():
    if key.startswith('train.'):
        estimate_config['train'][key.split('.', 1)[1]] = value
    else:
        estimate_config['train'].setdefault('lgb_params', {})[key] = value
""",
        ),
        (
            "## 7. Run Estimation",
            """
result = estimate_dr(estimate_config)
obs_df = pd.read_parquet(result.metadata['observation_path'])
""",
        ),
        (
            "## 8. Metrics Summary",
            """
metrics_df = pd.DataFrame(
    [
        {
            'label': 'drdid_att',
            'estimate': result.estimate,
            'std_error': result.std_error,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            'overlap_metric': result.metrics.get('overlap_metric'),
            'smd_max_unweighted': result.metrics.get('smd_max_unweighted'),
            'smd_max_weighted': result.metrics.get('smd_max_weighted'),
        }
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
display(metrics_df)
""",
        ),
        (
            "## 9. DR-DiD Table",
            """
drdid_table = build_drdid_table(obs_df)
drdid_table_path = OUT_DIR / 'drdid_table.csv'
drdid_table.to_csv(drdid_table_path, index=False)
display(drdid_table.head(10))
""",
        ),
        (
            "## 10. Balance Table",
            """
base_cov = (
    source_df.loc[source_df['post'] == 0, ['unit_id', *covariate_cols, treatment_col]]
    .sort_values('unit_id')
    .reset_index(drop=True)
)
smd_unweighted = compute_balance_smd(base_cov[covariate_cols], base_cov[treatment_col])
smd_weighted = compute_balance_smd(base_cov[covariate_cols], base_cov[treatment_col], obs_df['weight'])
balance_df = smd_unweighted.merge(
    smd_weighted,
    on='feature',
    suffixes=('_unweighted', '_weighted'),
    how='outer',
)
balance_path = OUT_DIR / 'balance_smd.csv'
balance_df.to_csv(balance_path, index=False)
display(balance_df)
""",
        ),
        (
            "## 11. Influence Function Distribution",
            """
if_path = diag_dir / 'if_distribution.png'
plot_if_distribution(obs_df['psi'].to_numpy(dtype=float), if_path)
display(Image(filename=str(if_path)))
""",
        ),
        (
            "## 12. Parallel Trends",
            """
means = (
    source_df.groupby(['post', treatment_col])[target_col]
    .mean()
    .unstack(treatment_col)
    .reindex(index=[0, 1], columns=[0, 1])
)
parallel_path = diag_dir / 'parallel_trends.png'
plot_parallel_trends(
    means[1].to_numpy(dtype=float),
    means[0].to_numpy(dtype=float),
    ['pre', 'post'],
    parallel_path,
)
display(Image(filename=str(parallel_path)))
""",
        ),
        (
            "## 13. Propensity / Weight Distribution",
            """
prop_path = diag_dir / 'propensity_distribution.png'
plot_propensity_distribution(
    obs_df['e_hat'].to_numpy(dtype=float),
    obs_df['treatment'].to_numpy(dtype=int),
    prop_path,
)
weight_path = diag_dir / 'weight_distribution.png'
plot_weight_distribution(obs_df['weight'].to_numpy(dtype=float), weight_path)
display(Image(filename=str(prop_path)))
display(Image(filename=str(weight_path)))
""",
        ),
        (
            "## 14. Love Plot",
            """
love_path = diag_dir / 'love_plot.png'
plot_love_plot(smd_unweighted, smd_weighted, love_path)
display(Image(filename=str(love_path)))
""",
        ),
        (
            "## 15. Prepare Summary",
            """
summary_outputs = [
    source_path,
    metrics_path,
    drdid_table_path,
    balance_path,
    tuning_trials_path,
    best_params_path,
    if_path,
    parallel_path,
    prop_path,
    weight_path,
    love_path,
]
artifact_path_for_summary = Path(result.metadata.get('summary_path', OUT_DIR)).parent
""",
        ),
    ]


def _execute_notebook(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def generate_quick_reference_notebooks(
    *,
    execute: bool = True,
    notebook_subdir: str = DEFAULT_NOTEBOOK_SUBDIR,
) -> list[Path]:
    if notebook_subdir not in {"quick_reference", "quick_reference_preview"}:
        raise ValueError(
            "notebook_subdir must be one of {'quick_reference', 'quick_reference_preview'}."
        )
    notebook_dir = NB_DIR / notebook_subdir
    notebook_dir.mkdir(parents=True, exist_ok=True)
    spec: list[tuple[str, str, str, str, list[tuple[str, str]]]] = [
        (
            "reference_01_regression_fit_evaluate.ipynb",
            "Phase35 UC-1 Regression Fit/Evaluate",
            "UC-1",
            "phase35_uc01_regression_fit_evaluate",
            _sections_uc01(),
        ),
        (
            "reference_02_binary_fit_evaluate.ipynb",
            "Phase35 UC-2 Binary Fit/Evaluate",
            "UC-2",
            "phase35_uc02_binary_fit_evaluate",
            _sections_uc02(),
        ),
        (
            "reference_03_multiclass_fit_evaluate.ipynb",
            "Phase35 UC-3 Multiclass Fit/Evaluate",
            "UC-3",
            "phase35_uc03_multiclass_fit_evaluate",
            _sections_uc03(),
        ),
        (
            "reference_04_timeseries_fit_evaluate.ipynb",
            "Phase35 UC-4 Timeseries Fit/Evaluate",
            "UC-4",
            "phase35_uc04_timeseries_fit_evaluate",
            _sections_uc04(),
        ),
        (
            "reference_05_frontier_fit_evaluate.ipynb",
            "Phase35 UC-5 Frontier Fit/Evaluate",
            "UC-5",
            "phase35_uc05_frontier_fit_evaluate",
            _sections_uc05(),
        ),
        (
            "reference_06_dr_estimate.ipynb",
            "Phase35 UC-6 DR Estimate",
            "UC-6",
            "phase35_uc06_dr_estimate",
            _sections_uc06(),
        ),
        (
            "reference_07_drdid_estimate.ipynb",
            "Phase35 UC-7 DR-DiD Estimate",
            "UC-7",
            "phase35_uc07_drdid_estimate",
            _sections_uc07(),
        ),
        (
            "reference_08_artifact_evaluate.ipynb",
            "Phase35 UC-8 Artifact Evaluate",
            "UC-8",
            "phase35_uc08_artifact_evaluate",
            _sections_uc08(),
        ),
        (
            "reference_09_binary_tune_evaluate.ipynb",
            "Phase35 UC-9 Binary Tune/Evaluate",
            "UC-9",
            "phase35_uc09_binary_tune_evaluate",
            _sections_uc09(),
        ),
        (
            "reference_10_timeseries_tune_evaluate.ipynb",
            "Phase35 UC-10 Timeseries Tune/Evaluate",
            "UC-10",
            "phase35_uc10_timeseries_tune_evaluate",
            _sections_uc10(),
        ),
        (
            "reference_11_frontier_tune_evaluate.ipynb",
            "Phase35 UC-11 Frontier Tune/Evaluate",
            "UC-11",
            "phase35_uc11_frontier_tune_evaluate",
            _sections_uc11(),
        ),
        (
            "reference_12_dr_tune_estimate.ipynb",
            "Phase35 UC-12 DR Tune/Estimate",
            "UC-12",
            "phase35_uc12_dr_tune_estimate",
            _sections_uc12(),
        ),
        (
            "reference_13_drdid_tune_estimate.ipynb",
            "Phase35 UC-13 DR-DiD Tune/Estimate",
            "UC-13",
            "phase35_uc13_drdid_tune_estimate",
            _sections_uc13(),
        ),
    ]

    generated: list[Path] = []
    for name, title, uc_id, out_name, sections in spec:
        setup = _setup_block(uc_id, out_name)
        nb = _build_notebook(title, setup, sections)
        path = notebook_dir / name
        path.write_text(nbformat.writes(nb), encoding="utf-8")
        generated.append(path)

    if execute:
        for path in generated:
            _execute_notebook(path)

    return generated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quick-reference notebooks.")
    parser.add_argument(
        "--output-dir",
        choices=["quick_reference", "quick_reference_preview"],
        default=DEFAULT_NOTEBOOK_SUBDIR,
        help="Notebook output subdirectory under notebooks/.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Generate notebooks without execution.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generate_quick_reference_notebooks(
        execute=not args.no_execute,
        notebook_subdir=args.output_dir,
    )


if __name__ == "__main__":
    main()
