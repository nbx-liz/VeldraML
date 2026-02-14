"""Doubly robust DiD estimation (2-period MVP)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from veldra.api.exceptions import VeldraValidationError
from veldra.causal.diagnostics import max_standardized_mean_difference, overlap_metric
from veldra.causal.dr import DREstimationOutput, run_dr_estimation
from veldra.config.models import RunConfig


def _to_binary(series: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise VeldraValidationError(f"{name} must be binary (0/1).")
    uniques = sorted(pd.unique(numeric).tolist())
    if len(uniques) != 2 or not set(uniques).issubset({0, 1}):
        raise VeldraValidationError(
            f"{name} must contain exactly two binary values (0/1), got {uniques}."
        )
    return numeric.astype(int)


def _base_validation(config: RunConfig, frame: pd.DataFrame) -> tuple[str, str, str]:
    if config.causal is None:
        raise VeldraValidationError("causal config is required for DR-DiD estimation.")
    if config.causal.method != "dr_did":
        raise VeldraValidationError(f"Unsupported causal method '{config.causal.method}'.")
    if config.causal.design is None:
        raise VeldraValidationError("causal.design is required for DR-DiD estimation.")
    if config.causal.time_col is None:
        raise VeldraValidationError("causal.time_col is required for DR-DiD estimation.")
    if config.causal.post_col is None:
        raise VeldraValidationError("causal.post_col is required for DR-DiD estimation.")
    if frame.empty:
        raise VeldraValidationError("Input data is empty.")
    if config.task.type == "binary" and config.causal.estimand != "att":
        raise VeldraValidationError(
            "DR-DiD binary supports only causal.estimand='att' in current phase."
        )

    treatment_col = config.causal.treatment_col
    target_col = config.data.target
    time_col = config.causal.time_col
    post_col = config.causal.post_col
    required = [treatment_col, target_col, time_col, post_col]
    if config.causal.design == "panel":
        if config.causal.unit_id_col is None:
            raise VeldraValidationError("causal.unit_id_col is required for panel DR-DiD.")
        required.append(config.causal.unit_id_col)

    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise VeldraValidationError(f"DR-DiD input is missing required columns: {missing}")

    _to_binary(frame[treatment_col], name=treatment_col)
    _to_binary(frame[post_col], name=post_col)
    if config.task.type == "binary":
        _to_binary(frame[target_col], name=target_col)
    else:
        try:
            pd.to_numeric(frame[target_col], errors="raise")
        except Exception as exc:
            raise VeldraValidationError(
                "Outcome values must be numeric for DR-DiD regression."
            ) from exc

    return treatment_col, target_col, post_col


def _dr_config_from_drdid(config: RunConfig) -> RunConfig:
    dr_cfg = config.model_copy(deep=True)
    if dr_cfg.causal is None:
        raise VeldraValidationError("causal config is required for DR-DiD estimation.")
    dr_cfg.causal.method = "dr"
    # DR-DiD pseudo outcomes are continuous by construction even for binary endpoints.
    dr_cfg.task.type = "regression"
    return dr_cfg


def _panel_to_pseudo_frame(
    config: RunConfig,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert config.causal is not None
    assert config.causal.unit_id_col is not None
    treatment_col = config.causal.treatment_col
    target_col = config.data.target
    post_col = config.causal.post_col
    unit_col = config.causal.unit_id_col

    post = pd.to_numeric(frame[post_col], errors="coerce").astype(int)
    frame = frame.copy()
    frame[post_col] = post

    counts = frame.groupby([unit_col, post_col]).size().unstack(fill_value=0)
    if not (0 in counts.columns and 1 in counts.columns):
        raise VeldraValidationError("Panel DR-DiD requires both pre and post observations.")
    if ((counts[0] != 1) | (counts[1] != 1)).any():
        raise VeldraValidationError(
            "Panel DR-DiD requires exactly one pre and one post row per unit."
        )

    pre = frame.loc[frame[post_col] == 0].sort_values(unit_col).reset_index(drop=True)
    post_df = frame.loc[frame[post_col] == 1].sort_values(unit_col).reset_index(drop=True)
    if not pre[unit_col].equals(post_df[unit_col]):
        raise VeldraValidationError("Panel DR-DiD unit alignment failed between pre and post rows.")

    t_pre = pd.to_numeric(pre[treatment_col], errors="coerce").astype(int)
    t_post = pd.to_numeric(post_df[treatment_col], errors="coerce").astype(int)
    if not (t_pre == t_post).all():
        raise VeldraValidationError(
            "Panel DR-DiD requires treatment to be stable per unit across periods."
        )

    drop_cols = set(
        config.data.drop_cols
        + config.data.id_cols
        + [
            target_col,
            treatment_col,
            config.causal.time_col or "",
            config.causal.post_col or "",
            unit_col,
        ]
    )
    feature_cols = [c for c in pre.columns if c not in drop_cols]
    if not feature_cols:
        raise VeldraValidationError("No feature columns remain for panel DR-DiD estimation.")

    pseudo = pd.get_dummies(pre.loc[:, feature_cols], drop_first=False).astype(float)
    if pseudo.empty:
        raise VeldraValidationError("No usable feature columns remain after encoding.")

    y_pre = pd.to_numeric(pre[target_col], errors="raise").to_numpy(dtype=float)
    y_post = pd.to_numeric(post_df[target_col], errors="raise").to_numpy(dtype=float)
    pseudo[target_col] = y_post - y_pre
    pseudo[treatment_col] = t_pre.to_numpy(dtype=int)
    pseudo[unit_col] = pre[unit_col].to_numpy()

    observation_meta = pd.DataFrame(
        {
            "unit_id": pre[unit_col].to_numpy(),
            "y_pre": y_pre,
            "y_post": y_post,
            "y_diff": y_post - y_pre,
            "treatment": t_pre.to_numpy(dtype=int),
        }
    )
    return pseudo, observation_meta


def _repeated_cs_to_pseudo_frame(
    config: RunConfig,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert config.causal is not None
    treatment_col = config.causal.treatment_col
    target_col = config.data.target
    post_col = config.causal.post_col

    pseudo = frame.copy()
    post = pd.to_numeric(pseudo[post_col], errors="coerce").astype(int).to_numpy(dtype=float)
    y = pd.to_numeric(pseudo[target_col], errors="coerce").to_numpy(dtype=float)
    p_post = float(np.mean(post))
    if not (0.0 < p_post < 1.0):
        raise VeldraValidationError(
            "DR-DiD repeated cross-section requires both pre and post rows."
        )
    y_tilde = y * post / p_post - y * (1.0 - post) / (1.0 - p_post)
    pseudo[target_col] = y_tilde

    observation_meta = pd.DataFrame(
        {
            "post": post.astype(int),
            "outcome": y,
            "outcome_tilde": y_tilde,
            "treatment": pd.to_numeric(pseudo[treatment_col], errors="coerce").astype(int),
        }
    )
    return pseudo, observation_meta


def run_dr_did_estimation(config: RunConfig, frame: pd.DataFrame) -> DREstimationOutput:
    """Run two-period DR-DiD estimation for panel or repeated cross-section data.

    Notes
    -----
    - Input is transformed into pseudo-outcome form (panel difference or
      repeated-cross-section tilting), then fed to DR estimation.
    - Binary outcome DR-DiD is interpreted as Risk Difference ATT in current
      scope.
    - Additional diagnostics (overlap, weighted/unweighted SMD) are computed to
      support balance-priority tuning objectives.
    """
    treatment_col, _target_col, post_col = _base_validation(config, frame)
    assert config.causal is not None

    n_pre = int((pd.to_numeric(frame[post_col], errors="coerce") == 0).sum())
    n_post = int((pd.to_numeric(frame[post_col], errors="coerce") == 1).sum())
    treated = pd.to_numeric(frame[treatment_col], errors="coerce").astype(int)
    post = pd.to_numeric(frame[post_col], errors="coerce").astype(int)
    n_treated_pre = int(((treated == 1) & (post == 0)).sum())
    n_treated_post = int(((treated == 1) & (post == 1)).sum())

    if config.causal.design == "panel":
        pseudo_frame, obs_meta = _panel_to_pseudo_frame(config, frame)
    else:
        pseudo_frame, obs_meta = _repeated_cs_to_pseudo_frame(config, frame)

    smd_drop_cols = {
        config.data.target,
        treatment_col,
        config.causal.time_col or "",
        config.causal.post_col or "",
        config.causal.unit_id_col or "",
    }
    smd_covariates = pseudo_frame.drop(
        columns=[c for c in smd_drop_cols if c in pseudo_frame.columns]
    )

    dr_cfg = _dr_config_from_drdid(config)
    dr_out = run_dr_estimation(dr_cfg, pseudo_frame)

    obs = dr_out.observation_table.copy()
    for col in obs_meta.columns:
        obs[col] = obs_meta[col].to_numpy()

    overlap = float(
        dr_out.metrics.get(
            "overlap_metric",
            overlap_metric(
                obs["e_hat"].to_numpy(dtype=float),
                obs["treatment"].to_numpy(dtype=int),
            ),
        )
    )
    t_np = obs["treatment"].to_numpy(dtype=int)
    smd_max_unweighted = float(
        dr_out.metrics.get(
            "smd_max_unweighted",
            max_standardized_mean_difference(smd_covariates, t_np),
        )
    )
    if "smd_max_weighted" in dr_out.metrics:
        smd_max_weighted = float(dr_out.metrics["smd_max_weighted"])
    else:
        e_np = np.clip(obs["e_hat"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        if config.causal.estimand == "ate":
            balance_weights = np.where(t_np == 1, 1.0 / e_np, 1.0 / (1.0 - e_np))
        else:
            balance_weights = np.where(t_np == 1, 1.0, e_np / (1.0 - e_np))
        smd_max_weighted = max_standardized_mean_difference(
            smd_covariates, t_np, weights=balance_weights
        )

    metrics = dict(dr_out.metrics)
    metrics["overlap_metric"] = overlap
    metrics["drdid"] = metrics.get("dr", dr_out.estimate)
    metrics["smd_max_unweighted"] = smd_max_unweighted
    metrics["smd_max_weighted"] = smd_max_weighted

    summary = dict(dr_out.summary)
    summary.update(
        {
            "method": "dr_did",
            "design": config.causal.design,
            "time_col": config.causal.time_col,
            "post_col": config.causal.post_col,
            "unit_id_col": config.causal.unit_id_col,
            "n_pre": n_pre,
            "n_post": n_post,
            "n_treated_pre": n_treated_pre,
            "n_treated_post": n_treated_post,
            "overlap_metric": overlap,
            "smd_max_unweighted": smd_max_unweighted,
            "smd_max_weighted": smd_max_weighted,
            "binary_outcome": bool(config.task.type == "binary"),
            "outcome_scale": (
                "risk_difference_att" if config.task.type == "binary" else "continuous_att"
            ),
            "metrics": metrics,
        }
    )

    return replace(
        dr_out,
        method="dr_did",
        metrics=metrics,
        observation_table=obs,
        summary=summary,
    )
