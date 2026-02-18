"""Help text dictionary for Phase26.2 guide components."""

from __future__ import annotations

HELP_TEXTS: dict[str, dict[str, str]] = {
    "task_type_regression": {
        "short": "Predict continuous numeric outcomes.",
        "detail": "Use this when the target is continuous (sales, price, temperature).",
    },
    "task_type_binary": {
        "short": "Predict one of two classes.",
        "detail": "Use this when labels are binary (0/1, yes/no).",
    },
    "task_type_multiclass": {
        "short": "Predict one label among multiple classes.",
        "detail": "Use this when the target has 3 or more discrete classes.",
    },
    "task_type_frontier": {
        "short": "Estimate frontier / quantile behavior.",
        "detail": "Use this for frontier analysis and quantile-like objectives.",
    },
    "frontier_alpha": {
        "short": "Alpha controls frontier confidence level.",
        "detail": "Typical values are 0.90 for strict frontier and 0.50 for median-like behavior.",
    },
    "causal_dr": {
        "short": "DR for cross-sectional treatment effect estimation.",
        "detail": "Use DR when you do not model pre/post treatment timing explicitly.",
    },
    "causal_drdid": {
        "short": "DR-DiD for before/after treatment effect estimation.",
        "detail": "Use DR-DiD for panel or repeated cross-section with time and post flags.",
    },
    "treatment_col": {
        "short": "Treatment must be binary.",
        "detail": "Select a 0/1 treatment indicator column.",
    },
    "unit_id_col": {
        "short": "Unit identifier for panel-style diagnostics.",
        "detail": "Recommended for DR-DiD and grouped validation flows.",
    },
    "exclude_cols": {
        "short": "Exclude leakage-prone features from training.",
        "detail": "Drop columns that may directly leak target information.",
    },
    "split_kfold": {
        "short": "Standard cross-validation split.",
        "detail": "Use when data order has no temporal meaning.",
    },
    "split_stratified": {
        "short": "Preserve class balance in each fold.",
        "detail": "Recommended for binary/multiclass tasks.",
    },
    "split_group": {
        "short": "Keep groups separated across folds.",
        "detail": "Use when samples in the same group should not leak across train/validation.",
    },
    "split_timeseries": {
        "short": "Prevent future-to-past leakage.",
        "detail": "Use for ordered time data where chronology must be respected.",
    },
    "timeseries_gap": {
        "short": "Gap creates a buffer between train and validation.",
        "detail": "Use a positive gap to reduce temporal leakage risk.",
    },
    "timeseries_embargo": {
        "short": "Embargo blocks nearby points after validation windows.",
        "detail": "Useful when short-term autocorrelation is strong.",
    },
    "timeseries_mode": {
        "short": "Expanding accumulates data; blocked uses fixed windows.",
        "detail": "Choose expanding for long-term learning, blocked for local stationarity.",
    },
    "train_learning_rate": {
        "short": "Smaller values train slower but can improve stability.",
        "detail": "Common range: 0.01 to 0.1.",
    },
    "train_num_leaves": {
        "short": "Controls tree complexity.",
        "detail": "Higher leaves increase capacity and overfitting risk.",
    },
    "train_max_depth": {
        "short": "Limits tree depth.",
        "detail": "Use -1 for unlimited depth and control with other regularization settings.",
    },
    "train_min_child_samples": {
        "short": "Minimum samples per leaf.",
        "detail": "Higher values reduce overfitting.",
    },
    "train_early_stopping": {
        "short": "Stop when validation no longer improves.",
        "detail": "Set enough rounds to avoid stopping too early.",
    },
    "objective_rmse": {
        "short": "Square-error objective.",
        "detail": "Sensitive to larger errors and outliers.",
    },
    "objective_mae": {
        "short": "Absolute-error objective.",
        "detail": "More robust to outliers than RMSE.",
    },
    "objective_auc": {
        "short": "Ranking/selection quality for binary classification.",
        "detail": "Useful when class ranking matters more than thresholded labels.",
    },
    "objective_f1": {
        "short": "Balance of precision and recall.",
        "detail": "Useful for imbalanced classes with thresholded decisions.",
    },
    "objective_logloss": {
        "short": "Probability calibration quality.",
        "detail": "Penalizes over-confident wrong predictions.",
    },
    "objective_pinball": {
        "short": "Quantile/frontier loss.",
        "detail": "Standard objective for frontier-style modeling.",
    },
    "objective_dr_balance_priority": {
        "short": "Prioritize covariate balance first.",
        "detail": "Recommended default for causal optimization.",
    },
    "objective_dr_std_error": {
        "short": "Minimize standard error.",
        "detail": "Can trade off some covariate balance quality.",
    },
    "objective_dr_overlap_penalty": {
        "short": "Penalize poor overlap.",
        "detail": "Helps avoid unstable causal estimates.",
    },
}
