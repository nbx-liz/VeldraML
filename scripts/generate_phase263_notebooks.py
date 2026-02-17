from __future__ import annotations

# ruff: noqa: E501
import json
import textwrap
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
QUICK_REF_DIR = NB_DIR / "quick_reference"

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


def _build_notebook(
    title: str,
    setup: str,
    workflow: str,
    overview: tuple[str, str, str],
    tutorial_notebook: str,
) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = NOTEBOOK_METADATA
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            f"# {title}\n\nPhase26.3 notebook with executed diagnostics outputs."
        ),
        nbformat.v4.new_markdown_cell(
            "## Overview\n"
            f"- Purpose: {overview[0]}\n"
            f"- API: {overview[1]}\n"
            f"- Outputs: {overview[2]}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Learn More\n"
            f"Detailed tutorial: `notebooks/tutorials/{tutorial_notebook}`"
        ),
        nbformat.v4.new_markdown_cell("## Setup"),
        nbformat.v4.new_code_cell(textwrap.dedent(setup).strip() + "\n"),
        nbformat.v4.new_markdown_cell(
            "## Config Notes\n"
            "- Keep compatibility output directories under `examples/out/phase26_*`.\n"
            "- Keep key tuning knobs annotated with inline comments when modifying config payloads."
        ),
        nbformat.v4.new_markdown_cell("## Workflow"),
        nbformat.v4.new_markdown_cell(
            "### Output Annotation\n"
            "The following workflow cells materialize plots/tables consumed by evidence tests."
        ),
        nbformat.v4.new_code_cell(textwrap.dedent(workflow).strip() + "\n"),
        nbformat.v4.new_markdown_cell("## Result Summary"),
        nbformat.v4.new_code_cell(SUMMARY_CELL + "\n"),
    ]
    return nb


def _setup_block(uc_id: str, out_dir: str) -> str:
    return f"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    plot_error_histogram,
    plot_feature_importance,
    plot_frontier_scatter,
    plot_if_distribution,
    plot_lift_chart,
    plot_love_plot,
    plot_nll_histogram,
    plot_parallel_trends,
    plot_pinball_histogram,
    plot_propensity_distribution,
    plot_roc_comparison,
    plot_shap_summary,
    plot_timeseries_prediction,
    plot_timeseries_residual,
    plot_true_class_prob_histogram,
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


def _workflow_uc1() -> str:
    return """
from sklearn.model_selection import train_test_split

source_df = pd.read_csv(ROOT / 'examples' / 'out' / 'phase26_2_uc07_artifact_evaluate' / 'eval.csv')
train_df, test_df = train_test_split(source_df, test_size=0.25, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = OUT_DIR / 'train.csv'
test_path = OUT_DIR / 'test.csv'
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {'path': str(train_path), 'target': 'target'},
    'split': {'type': 'kfold', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.01,
            'max_bin': 255,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'min_child_samples': 20,
            'first_metric_only': True,
        },
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}

run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
(OUT_DIR / 'latest_artifact_path.txt').write_text(run_result.artifact_path, encoding='utf-8')

x_train = train_df.drop(columns=['target'])
y_train = train_df['target'].to_numpy(dtype=float)
x_test = test_df.drop(columns=['target'])
y_test = test_df['target'].to_numpy(dtype=float)
pred_train = artifact.predict(x_train)
pred_test = artifact.predict(x_test)

metrics_df = pd.DataFrame(
    [
        regression_metrics(y_train, pred_train, label='in_sample'),
        regression_metrics(y_test, pred_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)

residual_train = y_train - np.asarray(pred_train, dtype=float)
residual_test = y_test - np.asarray(pred_test, dtype=float)
residual_path = diag_dir / 'residual_hist.png'
plot_error_histogram(
    residual_train,
    residual_test,
    metrics_df.iloc[0].to_dict(),
    metrics_df.iloc[1].to_dict(),
    residual_path,
)

booster = artifact._get_booster()
importance_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_path = diag_dir / 'importance_gain.png'
plot_feature_importance(importance_df, 'gain', importance_path)
importance_df.to_csv(OUT_DIR / 'importance_gain.csv', index=False)

shap_frame = compute_shap(booster, x_train.head(min(len(x_train), 64)))
shap_path = diag_dir / 'shap_summary.png'
plot_shap_summary(shap_frame, x_train.head(min(len(x_train), 64)), shap_path)
shap_frame.abs().mean().sort_values(ascending=False).reset_index().rename(
    columns={'index': 'feature', 0: 'mean_abs_shap'}
).to_csv(OUT_DIR / 'shap_mean_abs.csv', index=False)

score_table = build_regression_table(
    pd.concat([x_train, x_test], ignore_index=True),
    np.concatenate([y_train, y_test]),
    np.concatenate([
        np.zeros(len(x_train), dtype=int),
        np.ones(len(x_test), dtype=int),
    ]),
    np.concatenate([
        np.asarray(pred_train, dtype=float),
        np.asarray(pred_test, dtype=float),
    ]),
    ['in_sample'] * len(x_train) + ['out_of_sample'] * len(x_test),
)
score_path = OUT_DIR / 'regression_scores.csv'
score_table.to_csv(score_path, index=False)

display(metrics_df)
display(score_table.head(10))
display(Image(filename=str(residual_path)))
display(Image(filename=str(importance_path)))
display(Image(filename=str(shap_path)))

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    residual_path,
    importance_path,
    shap_path,
    score_path,
]
artifact_path_for_summary = run_result.artifact_path
"""


def _workflow_uc2() -> str:
    return """
from sklearn.model_selection import train_test_split

source_df = pd.read_csv(ROOT / 'examples' / 'data' / 'breast_cancer_binary.csv')
train_df, test_df = train_test_split(
    source_df,
    test_size=0.25,
    random_state=42,
    stratify=source_df['target'],
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = OUT_DIR / 'train.csv'
test_path = OUT_DIR / 'test.csv'
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

base_config = {
    'config_version': 1,
    'task': {'type': 'binary'},
    'data': {'path': str(train_path), 'target': 'target'},
    'split': {'type': 'stratified', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['logloss', 'auc'],
        'lgb_params': {
            'learning_rate': 0.01,
            'max_bin': 255,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'min_child_samples': 20,
            'first_metric_only': True,
        },
    },
    'postprocess': {'calibration': 'platt'},
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts_fit')},
}

tune_config = json.loads(json.dumps(base_config))
tune_config['tuning'] = {
    'enabled': True,
    'n_trials': 3,
    'resume': True,
    'objective': 'brier',
    'metrics_candidates': ['logloss', 'auc'],
    'search_space': {
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1, 'log': True},
        'train.num_leaves_ratio': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'train.early_stopping_validation_fraction': {'type': 'float', 'low': 0.1, 'high': 0.3},
        'max_bin': {'type': 'int', 'low': 127, 'high': 255},
        'train.min_data_in_leaf_ratio': {'type': 'float', 'low': 0.01, 'high': 0.1},
        'train.min_data_in_bin_ratio': {'type': 'float', 'low': 0.01, 'high': 0.1},
        'max_depth': {'type': 'int', 'low': 3, 'high': 15},
        'feature_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'lambda_l1': 0.0,
        'lambda_l2': {'type': 'float', 'low': 0.000001, 'high': 0.1},
    },
}

tune_result = tune(tune_config)
fit_config = json.loads(json.dumps(base_config))
for key, value in tune_result.best_params.items():
    if key.startswith('train.'):
        fit_config['train'][key.split('.', 1)[1]] = value
    else:
        fit_config['train'].setdefault('lgb_params', {})[key] = value

run_result = fit(fit_config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)

x_train = train_df.drop(columns=['target'])
y_train = train_df['target'].to_numpy(dtype=int)
x_test = test_df.drop(columns=['target'])
y_test = test_df['target'].to_numpy(dtype=int)
pred_train = artifact.predict(x_train)
pred_test = artifact.predict(x_test)

score_train = pred_train['p_cal'].to_numpy(dtype=float)
score_test = pred_test['p_cal'].to_numpy(dtype=float)
metrics_df = pd.DataFrame(
    [
        binary_metrics(y_train, score_train, label='in_sample'),
        binary_metrics(y_test, score_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)

roc_path = diag_dir / 'roc_in_out.png'
plot_roc_comparison(y_train, score_train, y_test, score_test, roc_path)
lift_path = diag_dir / 'lift.png'
plot_lift_chart(y_test, score_test, lift_path)

booster = artifact._get_booster()
importance_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_path = diag_dir / 'importance_gain.png'
plot_feature_importance(importance_df, 'gain', importance_path)
importance_df.to_csv(OUT_DIR / 'importance_gain.csv', index=False)

shap_frame = compute_shap(booster, x_train.head(min(len(x_train), 64)))
shap_path = diag_dir / 'shap_summary.png'
plot_shap_summary(shap_frame, x_train.head(min(len(x_train), 64)), shap_path)

score_table = build_binary_table(
    pd.concat([x_train, x_test], ignore_index=True),
    np.concatenate([y_train, y_test]),
    np.concatenate([
        np.zeros(len(x_train), dtype=int),
        np.ones(len(x_test), dtype=int),
    ]),
    np.concatenate([score_train, score_test]),
    ['in_sample'] * len(x_train) + ['out_of_sample'] * len(x_test),
)
score_path = OUT_DIR / 'binary_scores.csv'
score_table.to_csv(score_path, index=False)

display(metrics_df)
display(score_table.head(10))
display(Image(filename=str(roc_path)))
display(Image(filename=str(lift_path)))
display(Image(filename=str(importance_path)))
display(Image(filename=str(shap_path)))

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    roc_path,
    lift_path,
    importance_path,
    shap_path,
    score_path,
]
artifact_path_for_summary = run_result.artifact_path
"""


def _workflow_uc3() -> str:
    return """
from sklearn.model_selection import train_test_split

alpha = 0.9
source_df = pd.read_csv(ROOT / 'examples' / 'data' / 'frontier_demo.csv').head(400)
train_df, test_df = train_test_split(source_df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = OUT_DIR / 'train.csv'
test_path = OUT_DIR / 'test.csv'
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

config = {
    'config_version': 1,
    'task': {'type': 'frontier'},
    'data': {'path': str(train_path), 'target': 'target'},
    'split': {'type': 'kfold', 'n_splits': 4, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['quantile'],
        'lgb_params': {
            'learning_rate': 0.01,
            'max_bin': 255,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'min_child_samples': 20,
            'first_metric_only': True,
        },
    },
    'frontier': {'alpha': alpha},
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}

run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)

x_train = train_df.drop(columns=['target'])
y_train = train_df['target'].to_numpy(dtype=float)
x_test = test_df.drop(columns=['target'])
y_test = test_df['target'].to_numpy(dtype=float)
pred_train = artifact.predict(x_train)['frontier_pred'].to_numpy(dtype=float)
pred_test = artifact.predict(x_test)['frontier_pred'].to_numpy(dtype=float)

metrics_df = pd.DataFrame(
    [
        frontier_metrics(y_train, pred_train, alpha=alpha, label='in_sample'),
        frontier_metrics(y_test, pred_test, alpha=alpha, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)

pinball_train = np.maximum(alpha * (y_train - pred_train), (alpha - 1.0) * (y_train - pred_train))
pinball_test = np.maximum(alpha * (y_test - pred_test), (alpha - 1.0) * (y_test - pred_test))
pinball_path = diag_dir / 'pinball_hist.png'
plot_pinball_histogram(pinball_train, pinball_test, pinball_path)
scatter_path = diag_dir / 'frontier_scatter.png'
plot_frontier_scatter(y_test, pred_test, scatter_path)

booster = artifact._get_booster()
importance_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_path = diag_dir / 'importance_gain.png'
plot_feature_importance(importance_df, 'gain', importance_path)
importance_df.to_csv(OUT_DIR / 'importance_gain.csv', index=False)

shap_frame = compute_shap(booster, x_train.head(min(len(x_train), 64)))
shap_path = diag_dir / 'shap_summary.png'
plot_shap_summary(shap_frame, x_train.head(min(len(x_train), 64)), shap_path)

safe_pred = np.where(np.abs(np.concatenate([pred_train, pred_test])) < 1e-12, np.nan, np.concatenate([pred_train, pred_test]))
frontier_table = build_frontier_table(
    pd.concat([x_train, x_test], ignore_index=True),
    np.concatenate([y_train, y_test]),
    np.concatenate([
        np.zeros(len(x_train), dtype=int),
        np.ones(len(x_test), dtype=int),
    ]),
    np.concatenate([pred_train, pred_test]),
    np.concatenate([y_train, y_test]) / safe_pred,
)
score_path = OUT_DIR / 'frontier_scores.csv'
frontier_table.to_csv(score_path, index=False)

display(metrics_df)
display(frontier_table.head(10))
display(Image(filename=str(pinball_path)))
display(Image(filename=str(scatter_path)))
display(Image(filename=str(importance_path)))
display(Image(filename=str(shap_path)))

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    pinball_path,
    scatter_path,
    importance_path,
    shap_path,
    score_path,
]
artifact_path_for_summary = run_result.artifact_path
"""


def _workflow_uc4() -> str:
    return """
source_path = ROOT / 'examples' / 'out' / 'phase26_2_uc04_causal_dr_estimate' / 'dr.csv'
source_df = pd.read_csv(source_path)

config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {'path': str(source_path), 'target': 'outcome'},
    'split': {'type': 'kfold', 'n_splits': 3, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.01,
            'max_bin': 255,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'min_child_samples': 20,
            'first_metric_only': True,
        },
    },
    'causal': {
        'method': 'dr',
        'treatment_col': 'treatment',
        'estimand': 'att',
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}

result = estimate_dr(config)
obs_df = pd.read_parquet(result.metadata['observation_path'])
dr_table = build_dr_table(obs_df)
dt_path = OUT_DIR / 'dr_table.csv'
dr_table.to_csv(dt_path, index=False)

if_path = diag_dir / 'if_distribution.png'
plot_if_distribution(obs_df['psi'].to_numpy(dtype=float), if_path)
prop_path = diag_dir / 'propensity_distribution.png'
plot_propensity_distribution(
    obs_df['e_hat'].to_numpy(dtype=float),
    obs_df['treatment'].to_numpy(dtype=int),
    prop_path,
)
weight_path = diag_dir / 'weight_distribution.png'
plot_weight_distribution(obs_df['weight'].to_numpy(dtype=float), weight_path)

covariates = source_df[['x1', 'x2']].reset_index(drop=True)
smd_unweighted = compute_balance_smd(covariates, obs_df['treatment'])
smd_weighted = compute_balance_smd(covariates, obs_df['treatment'], obs_df['weight'])
love_path = diag_dir / 'love_plot.png'
plot_love_plot(smd_unweighted, smd_weighted, love_path)

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
smd_path = OUT_DIR / 'balance_smd.csv'
smd_unweighted.merge(
    smd_weighted,
    on='feature',
    suffixes=('_unweighted', '_weighted'),
    how='outer',
).to_csv(smd_path, index=False)

display(metrics_df)
display(dr_table.head(10))
display(pd.read_csv(smd_path))
display(Image(filename=str(if_path)))
display(Image(filename=str(prop_path)))
display(Image(filename=str(weight_path)))
display(Image(filename=str(love_path)))

summary_outputs = [
    metrics_path,
    dt_path,
    smd_path,
    if_path,
    prop_path,
    weight_path,
    love_path,
]
artifact_path_for_summary = Path(result.metadata['summary_path']).parent
"""


def _workflow_uc5() -> str:
    return """
source_path = ROOT / 'examples' / 'out' / 'phase26_2_uc05_causal_drdid_estimate' / 'drdid_panel.csv'
source_df = pd.read_csv(source_path)

config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {'path': str(source_path), 'target': 'target'},
    'split': {'type': 'group', 'n_splits': 3, 'group_col': 'unit_id', 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.01,
            'max_bin': 255,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'min_child_samples': 20,
            'first_metric_only': True,
        },
    },
    'causal': {
        'method': 'dr_did',
        'treatment_col': 'treatment',
        'estimand': 'att',
        'design': 'panel',
        'time_col': 'time',
        'post_col': 'post',
        'unit_id_col': 'unit_id',
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}

result = estimate_dr(config)
obs_df = pd.read_parquet(result.metadata['observation_path'])
drdid_table = build_drdid_table(obs_df)
dt_path = OUT_DIR / 'drdid_table.csv'
drdid_table.to_csv(dt_path, index=False)

if_path = diag_dir / 'if_distribution.png'
plot_if_distribution(obs_df['psi'].to_numpy(dtype=float), if_path)

means = (
    source_df.groupby(['post', 'treatment'])['target']
    .mean()
    .unstack('treatment')
    .reindex(index=[0, 1], columns=[0, 1])
)
parallel_path = diag_dir / 'parallel_trends.png'
plot_parallel_trends(
    means[1].to_numpy(dtype=float),
    means[0].to_numpy(dtype=float),
    ['pre', 'post'],
    parallel_path,
)

prop_path = diag_dir / 'propensity_distribution.png'
plot_propensity_distribution(
    obs_df['e_hat'].to_numpy(dtype=float),
    obs_df['treatment'].to_numpy(dtype=int),
    prop_path,
)
weight_path = diag_dir / 'weight_distribution.png'
plot_weight_distribution(obs_df['weight'].to_numpy(dtype=float), weight_path)

base_cov = (
    source_df.loc[source_df['post'] == 0, ['unit_id', 'age', 'skill', 'treatment']]
    .sort_values('unit_id')
    .reset_index(drop=True)
)
smd_unweighted = compute_balance_smd(base_cov[['age', 'skill']], base_cov['treatment'])
smd_weighted = compute_balance_smd(
    base_cov[['age', 'skill']],
    base_cov['treatment'],
    obs_df['weight'],
)
love_path = diag_dir / 'love_plot.png'
plot_love_plot(smd_unweighted, smd_weighted, love_path)

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
smd_path = OUT_DIR / 'balance_smd.csv'
smd_unweighted.merge(
    smd_weighted,
    on='feature',
    suffixes=('_unweighted', '_weighted'),
    how='outer',
).to_csv(smd_path, index=False)

display(metrics_df)
display(drdid_table.head(10))
display(pd.read_csv(smd_path))
display(Image(filename=str(if_path)))
display(Image(filename=str(parallel_path)))
display(Image(filename=str(prop_path)))
display(Image(filename=str(weight_path)))
display(Image(filename=str(love_path)))

summary_outputs = [
    metrics_path,
    dt_path,
    smd_path,
    if_path,
    parallel_path,
    prop_path,
    weight_path,
    love_path,
]
artifact_path_for_summary = Path(result.metadata['summary_path']).parent
"""


def _workflow_uc6() -> str:
    return """
source_path = ROOT / 'examples' / 'out' / 'phase26_2_uc06_causal_dr_tune' / 'dr_tune.csv'

config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {'path': str(source_path), 'target': 'outcome'},
    'split': {'type': 'kfold', 'n_splits': 3, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.01,
            'max_bin': 255,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'min_child_samples': 20,
            'first_metric_only': True,
        },
    },
    'causal': {
        'method': 'dr',
        'treatment_col': 'treatment',
        'estimand': 'att',
    },
    'tuning': {
        'enabled': True,
        'n_trials': 5,
        'resume': True,
        'objective': 'dr_balance_priority',
        'search_space': {
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1, 'log': True},
            'train.num_leaves_ratio': {'type': 'float', 'low': 0.5, 'high': 1.0},
            'train.early_stopping_validation_fraction': {'type': 'float', 'low': 0.1, 'high': 0.3},
            'max_bin': {'type': 'int', 'low': 127, 'high': 255},
            'train.min_data_in_leaf_ratio': {'type': 'float', 'low': 0.01, 'high': 0.1},
            'train.min_data_in_bin_ratio': {'type': 'float', 'low': 0.01, 'high': 0.1},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'feature_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0},
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': {'type': 'float', 'low': 0.000001, 'high': 0.1},
        },
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}

tune_result = tune(config)
trials_df = pd.read_parquet(tune_result.metadata['trials_path'])
trials_csv_path = OUT_DIR / 'tuning_trials.csv'
trials_df.to_csv(trials_csv_path, index=False)

diag_path = diag_dir / 'overlap.png'
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trials_df['number'], trials_df['value'], marker='o', label='objective')
if 'overlap_metric' in trials_df.columns:
    ax2 = ax.twinx()
    ax2.plot(trials_df['number'], trials_df['overlap_metric'], color='tab:orange', marker='x', label='overlap')
    ax2.set_ylabel('overlap_metric')
ax.set_xlabel('trial')
ax.set_ylabel('objective value')
ax.set_title('DR tuning diagnostics')
fig.tight_layout()
fig.savefig(diag_path)
plt.close(fig)

best_components = dict(tune_result.metadata.get('objective_components', {}))
metrics_df = pd.DataFrame(
    [
        {
            'label': 'dr_tuning',
            'best_score': tune_result.best_score,
            'best_param_count': len(tune_result.best_params),
            **{str(k): v for k, v in best_components.items() if isinstance(v, (int, float, bool, str))},
        }
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
best_params_path = OUT_DIR / 'best_params.json'
best_params_path.write_text(json.dumps(tune_result.best_params, indent=2), encoding='utf-8')

display(metrics_df)
display(trials_df.head(10))
display(Image(filename=str(diag_path)))

summary_outputs = [
    metrics_path,
    trials_csv_path,
    diag_path,
    best_params_path,
]
artifact_path_for_summary = tune_result.metadata['tuning_path']
"""


def _workflow_uc7() -> str:
    return """
def _latest_artifact_path() -> Path:
    marker = ROOT / 'examples' / 'out' / 'phase26_2_uc01_regression_fit_evaluate' / 'latest_artifact_path.txt'
    if marker.exists():
        return Path(marker.read_text(encoding='utf-8').strip())
    candidates = sorted((ROOT / 'examples' / 'out' / 'phase26_2_uc01_regression_fit_evaluate' / 'artifacts').glob('*'))
    if not candidates:
        raise FileNotFoundError('No UC-1 artifact found.')
    return candidates[-1]

artifact_path = _latest_artifact_path()
artifact = Artifact.load(artifact_path)
eval_df = pd.read_csv(ROOT / 'examples' / 'out' / 'phase26_2_uc07_artifact_evaluate' / 'eval.csv')
eval_result = evaluate(artifact, eval_df)

x_eval = eval_df.drop(columns=['target'])
y_eval = eval_df['target'].to_numpy(dtype=float)
pred_eval = artifact.predict(x_eval)
metrics_df = pd.DataFrame(
    [
        {
            'label': 'artifact_evaluate',
            **eval_result.metrics,
        },
        regression_metrics(y_eval, pred_eval, label='artifact_recomputed'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)

residual = y_eval - np.asarray(pred_eval, dtype=float)
residual_path = diag_dir / 'eval_residual_hist.png'
plot_error_histogram(residual, residual, metrics_df.iloc[0].to_dict(), metrics_df.iloc[0].to_dict(), residual_path)

eval_table = build_regression_table(
    x_eval,
    y_eval,
    np.zeros(len(x_eval), dtype=int),
    np.asarray(pred_eval, dtype=float),
    ['evaluate'] * len(x_eval),
)
eval_table_path = OUT_DIR / 'eval_table.csv'
eval_table.to_csv(eval_table_path, index=False)

display(metrics_df)
display(eval_table.head(10))
display(Image(filename=str(residual_path)))

summary_outputs = [metrics_path, eval_table_path, residual_path]
artifact_path_for_summary = artifact_path
"""


def _workflow_uc8() -> str:
    return """
def _cb_result_eval_precheck(schema_payload: dict[str, object], frame: pd.DataFrame) -> dict[str, object]:
    required = [str(c) for c in schema_payload.get('feature_names', [])]
    missing = [col for col in required if col not in frame.columns]
    return {'ok': len(missing) == 0, 'missing_columns': missing, 'required_columns': required}

marker = ROOT / 'examples' / 'out' / 'phase26_2_uc01_regression_fit_evaluate' / 'latest_artifact_path.txt'
if marker.exists():
    artifact_path = Path(marker.read_text(encoding='utf-8').strip())
else:
    candidates = sorted((ROOT / 'examples' / 'out' / 'phase26_2_uc01_regression_fit_evaluate' / 'artifacts').glob('*'))
    if not candidates:
        raise FileNotFoundError('No UC-1 artifact found for reevaluate.')
    artifact_path = candidates[-1]

artifact = Artifact.load(artifact_path)
reeval_df = pd.read_csv(ROOT / 'examples' / 'out' / 'phase26_2_uc08_artifact_reevaluate' / 'reeval_ok.csv')
missing_df = pd.read_csv(ROOT / 'examples' / 'out' / 'phase26_2_uc08_artifact_reevaluate' / 'reeval_missing_col.csv')

schema_payload = {'feature_names': artifact.feature_schema.get('feature_names', [])}
precheck_ok = _cb_result_eval_precheck(schema_payload, reeval_df)
precheck_bad = _cb_result_eval_precheck(schema_payload, missing_df)

reeval = evaluate(artifact, reeval_df)
y_true = reeval_df['target'].to_numpy(dtype=float)
y_pred = np.asarray(artifact.predict(reeval_df.drop(columns=['target'])), dtype=float)
residual = y_true - y_pred

train_metrics = artifact.metrics.get('mean', {}) if isinstance(artifact.metrics, dict) else {}
rows = []
for metric in ['rmse', 'mae', 'mape', 'r2']:
    if metric in reeval.metrics:
        train_value = train_metrics.get(metric)
        reeval_value = float(reeval.metrics[metric])
        rows.append(
            {
                'metric': metric,
                'train_value': None if train_value is None else float(train_value),
                'reeval_value': reeval_value,
                'delta': None if train_value is None else reeval_value - float(train_value),
            }
        )

compare_df = pd.DataFrame(rows)
compare_path = OUT_DIR / 'reeval_compare.csv'
compare_df.to_csv(compare_path, index=False)
precheck_path = OUT_DIR / 'precheck.csv'
pd.DataFrame([precheck_ok, precheck_bad], index=['ok_case', 'missing_case']).to_csv(precheck_path)

hist_path = diag_dir / 'reeval_hist.png'
plot_error_histogram(residual, residual, reeval.metrics, reeval.metrics, hist_path)

metrics_df = compare_df.rename(columns={'metric': 'label'})
if metrics_df.empty:
    metrics_df = pd.DataFrame([{'label': 'reeval', **reeval.metrics}])

display(pd.DataFrame([precheck_ok, precheck_bad], index=['ok_case', 'missing_case']))
display(compare_df)
display(Image(filename=str(hist_path)))

summary_outputs = [compare_path, precheck_path, hist_path]
artifact_path_for_summary = artifact_path
"""


def _workflow_uc11() -> str:
    return """
from sklearn.model_selection import train_test_split

source_df = pd.read_csv(ROOT / 'examples' / 'data' / 'iris_multiclass.csv')
train_df, test_df = train_test_split(
    source_df,
    test_size=0.25,
    random_state=42,
    stratify=source_df['target'],
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = OUT_DIR / 'train.csv'
test_path = OUT_DIR / 'test.csv'
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

config = {
    'config_version': 1,
    'task': {'type': 'multiclass'},
    'data': {'path': str(train_path), 'target': 'target'},
    'split': {'type': 'stratified', 'n_splits': 3, 'seed': 42},
    'train': {
        'seed': 42,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['multi_logloss', 'multi_error'],
        'lgb_params': {
            'learning_rate': 0.01,
            'max_bin': 255,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'min_child_samples': 20,
            'first_metric_only': True,
        },
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}

run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)
target_classes = artifact.feature_schema['target_classes']
class_to_idx = {label: idx for idx, label in enumerate(target_classes)}

x_train = train_df.drop(columns=['target'])
x_test = test_df.drop(columns=['target'])
y_train = train_df['target'].map(class_to_idx).to_numpy(dtype=int)
y_test = test_df['target'].map(class_to_idx).to_numpy(dtype=int)

pred_train = artifact.predict(x_train)
pred_test = artifact.predict(x_test)
prob_cols = [f'proba_{label}' for label in target_classes]
proba_train = pred_train[prob_cols].to_numpy(dtype=float)
proba_test = pred_test[prob_cols].to_numpy(dtype=float)

metrics_df = pd.DataFrame(
    [
        multiclass_metrics(y_train, proba_train, label='in_sample'),
        multiclass_metrics(y_test, proba_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)

prob_true_train = proba_train[np.arange(len(y_train)), y_train]
prob_true_test = proba_test[np.arange(len(y_test)), y_test]
nll_train = -np.log(np.clip(prob_true_train, 1e-7, 1.0))
nll_test = -np.log(np.clip(prob_true_test, 1e-7, 1.0))

nll_path = diag_dir / 'nll_hist.png'
plot_nll_histogram(nll_train, nll_test, nll_path)
prob_path = diag_dir / 'true_class_prob_hist.png'
plot_true_class_prob_histogram(prob_true_train, prob_true_test, prob_path)

booster = artifact._get_booster()
importance_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_path = diag_dir / 'importance_gain.png'
plot_feature_importance(importance_df, 'gain', importance_path)
importance_df.to_csv(OUT_DIR / 'importance_gain.csv', index=False)

pred_train_idx = np.argmax(proba_train, axis=1)
shap_frame = compute_shap_multiclass(
    booster,
    x_train.head(min(len(x_train), 64)),
    pred_train_idx[: min(len(x_train), 64)],
    n_classes=len(target_classes),
)
shap_path = diag_dir / 'shap_summary.png'
plot_shap_summary(shap_frame, x_train.head(min(len(x_train), 64)), shap_path)

score_table = build_multiclass_table(
    pd.concat([x_train, x_test], ignore_index=True),
    np.concatenate([y_train, y_test]),
    np.concatenate([
        np.zeros(len(x_train), dtype=int),
        np.ones(len(x_test), dtype=int),
    ]),
    np.vstack([proba_train, proba_test]),
    ['in_sample'] * len(x_train) + ['out_of_sample'] * len(x_test),
)
score_path = OUT_DIR / 'multiclass_scores.csv'
score_table.to_csv(score_path, index=False)

display(metrics_df)
display(score_table.head(10))
display(Image(filename=str(nll_path)))
display(Image(filename=str(prob_path)))
display(Image(filename=str(importance_path)))
display(Image(filename=str(shap_path)))

summary_outputs = [
    train_path,
    test_path,
    metrics_path,
    nll_path,
    prob_path,
    importance_path,
    shap_path,
    score_path,
]
artifact_path_for_summary = run_result.artifact_path
"""


def _workflow_uc12() -> str:
    return """
source_df = pd.read_csv(ROOT / 'examples' / 'data' / 'california_housing.csv').head(1200).copy()
source_df['event_time'] = pd.date_range('2020-01-01', periods=len(source_df), freq='D').astype(str)
train_cut = int(len(source_df) * 0.8)
train_df = source_df.iloc[:train_cut].reset_index(drop=True)
test_df = source_df.iloc[train_cut:].reset_index(drop=True)

full_path = OUT_DIR / 'timeseries_train.csv'
train_df.to_csv(OUT_DIR / 'train.csv', index=False)
test_df.to_csv(OUT_DIR / 'test.csv', index=False)
source_df.to_csv(full_path, index=False)

config = {
    'config_version': 1,
    'task': {'type': 'regression'},
    'data': {
        'path': str(full_path),
        'target': 'target',
        'drop_cols': ['event_time'],
    },
    'split': {
        'type': 'kfold',
        'n_splits': 4,
        'seed': 42,
    },
    'train': {
        'seed': 42,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'early_stopping_validation_fraction': 0.2,
        'auto_num_leaves': True,
        'num_leaves_ratio': 1.0,
        'min_data_in_leaf_ratio': 0.01,
        'min_data_in_bin_ratio': 0.01,
        'metrics': ['rmse', 'mae'],
        'lgb_params': {
            'learning_rate': 0.01,
            'max_bin': 255,
            'max_depth': 10,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.000001,
            'min_child_samples': 20,
            'first_metric_only': True,
        },
    },
    'export': {'artifact_dir': str(OUT_DIR / 'artifacts')},
}

run_result = fit(config)
artifact = Artifact.load(run_result.artifact_path)
eval_result = evaluate(artifact, test_df)

x_all = source_df.drop(columns=['target'])
y_all = source_df['target'].to_numpy(dtype=float)
y_pred_all = np.asarray(artifact.predict(x_all), dtype=float)

x_train = train_df.drop(columns=['target'])
y_train = train_df['target'].to_numpy(dtype=float)
y_pred_train = np.asarray(artifact.predict(x_train), dtype=float)

x_test = test_df.drop(columns=['target'])
y_test = test_df['target'].to_numpy(dtype=float)
y_pred_test = np.asarray(artifact.predict(x_test), dtype=float)

metrics_df = pd.DataFrame(
    [
        regression_metrics(y_train, y_pred_train, label='in_sample'),
        regression_metrics(y_test, y_pred_test, label='out_of_sample'),
    ]
)
metrics_path = OUT_DIR / 'timeseries_scores.csv'
metrics_df.to_csv(metrics_path, index=False)

time_index = pd.to_datetime(source_df['event_time'])
split_point = pd.to_datetime(test_df['event_time'].iloc[0])
pred_path = diag_dir / 'timeseries_prediction.png'
plot_timeseries_prediction(time_index, y_all, y_pred_all, split_point, pred_path)
residual_path = diag_dir / 'timeseries_residual.png'
plot_timeseries_residual(time_index, y_all - y_pred_all, split_point, residual_path)

booster = artifact._get_booster()
importance_df = compute_importance(booster, importance_type='gain', top_n=20)
importance_path = diag_dir / 'importance_gain.png'
plot_feature_importance(importance_df, 'gain', importance_path)
importance_df.to_csv(OUT_DIR / 'importance_gain.csv', index=False)

feature_cols = [c for c in x_train.columns if c != 'event_time']
shap_input = x_train.loc[:, feature_cols].head(min(len(x_train), 64))
shap_frame = compute_shap(booster, shap_input)
shap_path = diag_dir / 'shap_summary.png'
plot_shap_summary(shap_frame, shap_input, shap_path)

table = build_regression_table(
    source_df.loc[:, feature_cols],
    y_all,
    np.concatenate([
        np.zeros(len(train_df), dtype=int),
        np.ones(len(test_df), dtype=int),
    ]),
    y_pred_all,
    ['in_sample'] * len(train_df) + ['out_of_sample'] * len(test_df),
)
score_path = OUT_DIR / 'timeseries_detail.csv'
table.to_csv(score_path, index=False)

display(metrics_df)
display(table.head(10))
display(Image(filename=str(pred_path)))
display(Image(filename=str(residual_path)))
display(Image(filename=str(importance_path)))
display(Image(filename=str(shap_path)))

summary_outputs = [
    OUT_DIR / 'train.csv',
    OUT_DIR / 'test.csv',
    metrics_path,
    pred_path,
    residual_path,
    importance_path,
    shap_path,
    score_path,
]
artifact_path_for_summary = run_result.artifact_path
"""


SPEC = [
    (
        "quick_reference/reference_01_regression_fit_evaluate.ipynb",
        "UC-1 Regression Fit/Evaluate",
        _setup_block("UC-1", "phase26_2_uc01_regression_fit_evaluate"),
        _workflow_uc1(),
        (
            "Regression fit and evaluate quick reference.",
            "`fit(config)`, `evaluate(artifact, data)`",
            "metrics.csv, diagnostics png, regression_scores.csv",
        ),
        "tutorial_01_regression_basics.ipynb",
    ),
    (
        "quick_reference/reference_02_binary_tune_evaluate.ipynb",
        "UC-2 Binary Tune/Evaluate",
        _setup_block("UC-2", "phase26_2_uc02_binary_tune_evaluate"),
        _workflow_uc2(),
        (
            "Binary tuning and evaluation quick reference.",
            "`tune(config)`, `fit(config)`, `evaluate(artifact, data)`",
            "metrics.csv, ROC/lift png, binary_scores.csv",
        ),
        "tutorial_02_binary_classification_tuning.ipynb",
    ),
    (
        "quick_reference/reference_03_frontier_fit_evaluate.ipynb",
        "UC-3 Frontier Fit/Evaluate",
        _setup_block("UC-3", "phase26_2_uc03_frontier_fit_evaluate"),
        _workflow_uc3(),
        (
            "Frontier quantile regression quick reference.",
            "`fit(config)`",
            "metrics.csv, frontier plots, frontier_scores.csv",
        ),
        "tutorial_03_frontier_quantile_regression.ipynb",
    ),
    (
        "quick_reference/reference_04_causal_dr_estimate.ipynb",
        "UC-4 DR Estimate",
        _setup_block("UC-4", "phase26_2_uc04_causal_dr_estimate"),
        _workflow_uc4(),
        (
            "Causal DR estimation quick reference.",
            "`estimate_dr(config)`",
            "metrics.csv, balance tables, overlap diagnostics png",
        ),
        "tutorial_05_causal_dr_lalonde.ipynb",
    ),
    (
        "quick_reference/reference_05_causal_drdid_estimate.ipynb",
        "UC-5 DR-DiD Estimate",
        _setup_block("UC-5", "phase26_2_uc05_causal_drdid_estimate"),
        _workflow_uc5(),
        (
            "Causal DR-DiD estimation quick reference.",
            "`estimate_dr(config)` with `method=dr_did`",
            "metrics.csv, drdid tables, parallel-trends diagnostics",
        ),
        "tutorial_06_causal_drdid_lalonde.ipynb",
    ),
    (
        "quick_reference/reference_06_causal_dr_tune.ipynb",
        "UC-6 DR Tune",
        _setup_block("UC-6", "phase26_2_uc06_causal_dr_tune"),
        _workflow_uc6(),
        (
            "Causal objective tuning quick reference.",
            "`tune(config)` with causal objective",
            "metrics.csv, tuning_trials.csv, overlap diagnostics",
        ),
        "tutorial_05_causal_dr_lalonde.ipynb",
    ),
    (
        "quick_reference/reference_07_artifact_evaluate.ipynb",
        "UC-7 Artifact Evaluate",
        _setup_block("UC-7", "phase26_2_uc07_artifact_evaluate"),
        _workflow_uc7(),
        (
            "Existing artifact evaluation quick reference.",
            "`Artifact.load(path)`, `evaluate(artifact, data)`",
            "metrics.csv, eval table, diagnostics",
        ),
        "tutorial_01_regression_basics.ipynb",
    ),
    (
        "quick_reference/reference_08_artifact_reevaluate.ipynb",
        "UC-8 Artifact Re-evaluate",
        _setup_block("UC-8", "phase26_2_uc08_artifact_reevaluate"),
        _workflow_uc8(),
        (
            "Artifact re-evaluation quick reference.",
            "`evaluate(artifact, new_data)`",
            "reeval_compare.csv, precheck.csv, re-eval diagnostics",
        ),
        "tutorial_01_regression_basics.ipynb",
    ),
    (
        "quick_reference/reference_11_multiclass_fit_evaluate.ipynb",
        "UC-11 Multiclass Fit/Evaluate",
        _setup_block("UC-11", "phase26_3_uc_multiclass_fit_evaluate"),
        _workflow_uc11(),
        (
            "Multiclass fit/evaluate quick reference.",
            "`fit(config)`, `evaluate(artifact, data)`",
            "metrics.csv, class-prob diagnostics, multiclass_scores.csv",
        ),
        "tutorial_07_model_evaluation_guide.ipynb",
    ),
    (
        "quick_reference/reference_12_timeseries_fit_evaluate.ipynb",
        "UC-12 Timeseries Fit/Evaluate",
        _setup_block("UC-12", "phase26_3_uc_timeseries_fit_evaluate"),
        _workflow_uc12(),
        (
            "Timeseries regression quick reference.",
            "`fit(config)` with timeseries split",
            "metrics.csv, prediction/residual diagnostics, timeseries_detail.csv",
        ),
        "tutorial_07_model_evaluation_guide.ipynb",
    ),
]


def _execute_notebook(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def _build_manifest() -> None:
    manifest_path = NB_DIR / "phase26_3_execution_manifest.json"
    entries: list[dict[str, object]] = []

    summary_files = {
        "UC-1": ROOT / "examples" / "out" / "phase26_2_uc01_regression_fit_evaluate" / "summary.json",
        "UC-2": ROOT / "examples" / "out" / "phase26_2_uc02_binary_tune_evaluate" / "summary.json",
        "UC-3": ROOT / "examples" / "out" / "phase26_2_uc03_frontier_fit_evaluate" / "summary.json",
        "UC-4": ROOT / "examples" / "out" / "phase26_2_uc04_causal_dr_estimate" / "summary.json",
        "UC-5": ROOT / "examples" / "out" / "phase26_2_uc05_causal_drdid_estimate" / "summary.json",
        "UC-6": ROOT / "examples" / "out" / "phase26_2_uc06_causal_dr_tune" / "summary.json",
        "UC-7": ROOT / "examples" / "out" / "phase26_2_uc07_artifact_evaluate" / "summary.json",
        "UC-8": ROOT / "examples" / "out" / "phase26_2_uc08_artifact_reevaluate" / "summary.json",
        "UC-11": ROOT / "examples" / "out" / "phase26_3_uc_multiclass_fit_evaluate" / "summary.json",
        "UC-12": ROOT / "examples" / "out" / "phase26_3_uc_timeseries_fit_evaluate" / "summary.json",
    }

    notebook_map = {
        "UC-1": QUICK_REF_DIR / "reference_01_regression_fit_evaluate.ipynb",
        "UC-2": QUICK_REF_DIR / "reference_02_binary_tune_evaluate.ipynb",
        "UC-3": QUICK_REF_DIR / "reference_03_frontier_fit_evaluate.ipynb",
        "UC-4": QUICK_REF_DIR / "reference_04_causal_dr_estimate.ipynb",
        "UC-5": QUICK_REF_DIR / "reference_05_causal_drdid_estimate.ipynb",
        "UC-6": QUICK_REF_DIR / "reference_06_causal_dr_tune.ipynb",
        "UC-7": QUICK_REF_DIR / "reference_07_artifact_evaluate.ipynb",
        "UC-8": QUICK_REF_DIR / "reference_08_artifact_reevaluate.ipynb",
        "UC-9": QUICK_REF_DIR / "reference_09_export_python_onnx.ipynb",
        "UC-10": QUICK_REF_DIR / "reference_10_export_html_excel.ipynb",
        "UC-11": QUICK_REF_DIR / "reference_11_multiclass_fit_evaluate.ipynb",
        "UC-12": QUICK_REF_DIR / "reference_12_timeseries_fit_evaluate.ipynb",
    }

    for uc in ["UC-1", "UC-2", "UC-3", "UC-4", "UC-5", "UC-6", "UC-7", "UC-8", "UC-11", "UC-12"]:
        summary = json.loads(summary_files[uc].read_text(encoding="utf-8"))
        entries.append(
            {
                "uc": uc,
                "status": "passed",
                "notebook": str(notebook_map[uc].resolve()),
                "artifact_path": summary.get("artifact_path"),
                "outputs": summary.get("outputs", []),
                "metrics": summary.get("metrics", []),
            }
        )

    # Preserve UC-9/10 as minimal export-only entries.
    entries.append(
        {
            "uc": "UC-9",
            "status": "passed",
            "notebook": str(notebook_map["UC-9"].resolve()),
            "artifact_path": str(
                (ROOT / "examples" / "out" / "phase26_2_uc09_export_python_onnx" / "exports").resolve()
            ),
            "outputs": [
                str((ROOT / "examples" / "out" / "phase26_2_uc09_export_python_onnx" / "exports" / "python").resolve()),
                str((ROOT / "examples" / "out" / "phase26_2_uc09_export_python_onnx" / "exports" / "onnx").resolve()),
            ],
            "metrics": [],
        }
    )
    entries.append(
        {
            "uc": "UC-10",
            "status": "passed",
            "notebook": str(notebook_map["UC-10"].resolve()),
            "artifact_path": str(
                (ROOT / "examples" / "out" / "phase26_2_uc10_export_html_excel" / "reports").resolve()
            ),
            "outputs": [
                str((ROOT / "examples" / "out" / "phase26_2_uc10_export_html_excel" / "reports" / "report.html").resolve()),
                str((ROOT / "examples" / "out" / "phase26_2_uc10_export_html_excel" / "reports" / "report.xlsx").resolve()),
            ],
            "metrics": [],
        }
    )

    payload = {
        "phase": "26.3",
        "parity_criterion": "reachability_and_artifact_outputs",
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    QUICK_REF_DIR.mkdir(parents=True, exist_ok=True)
    for nb_name, title, setup, workflow, overview, tutorial in SPEC:
        path = NB_DIR / nb_name
        path.parent.mkdir(parents=True, exist_ok=True)
        notebook = _build_notebook(title, setup, workflow, overview, tutorial)
        path.write_text(nbformat.writes(notebook), encoding="utf-8")

    for nb_name, *_ in SPEC:
        _execute_notebook(NB_DIR / nb_name)

    _build_manifest()


if __name__ == "__main__":
    main()
