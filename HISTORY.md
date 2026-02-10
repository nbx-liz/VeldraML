# HISTORY.md
（AIエージェント作業履歴・意思決定ログ）

## ルール
- 1エントリ = 1作業単位（会話/セッション/PRなど）
- 「何をしたか」だけでなく「なぜそうしたか」を残す
- 仕様の決定は **Decision** に書く（provisional/confirmed を明記）
- 未確定は **Open Questions** に積む（放置しない）

---

## Template（このままコピペして追記）
### YYYY-MM-DD (Session/PR: XXXXX)
**Context**
- 背景 / 依頼 / 目的：

**Plan**
- 今日やること（箇条書き）

**Changes**
- 実装変更：
  - 例）`veldra.modeling.cv` にOOF出力を追加
- ドキュメント変更：
  - 例）DESIGN_BLUEPRINTのArtifact項を更新
- テスト変更：
  - 例）校正のリーク防止テスト追加

**Decisions**
- Decision: provisional | confirmed
  - 内容：
  - 理由：
  - 影響範囲：Config / Artifact / API / GUI / 性能 など

**Results**
- 動作確認結果：
  - 例）1M行相当のスモーク（サンプル）で完走
  - 例）binary校正前後でBrier改善

**Risks / Notes**
- リスクや留意点：

**Open Questions**
- [ ] 未決事項1
- [ ] 未決事項2

---

## Log
（ここに上のTemplateで追記していく）

### 2026-02-09 (Session/PR: bootstrap-mvp-scaffold)
**Context**
- 背景 / 依頼 / 目的：
  - VeldraMLのMVP骨格を計画に沿って実装し、`uv` で再現可能な開発環境を構築する。

**Plan**
- `uv` でPython 3.11 + 依存固定環境を構築する
- `src/veldra` にAPI/Config/Artifact/Splitterの最小骨格を実装する
- MVP向けの基本テストを追加する
- 設計図と履歴に決定事項を反映する

**Changes**
- 実装変更：
  - `pyproject.toml` を厳密固定版へ更新（runtime/dev を `==` 固定）
  - `src/veldra/api/*` に公開API入口（runner/artifact/types/exceptions/logging）を追加
  - `src/veldra/config/*` に RunConfigモデルとYAML I/Oを追加
  - `src/veldra/artifact/*` に manifest生成とsave/load基盤を追加
  - `src/veldra/split/time_series.py` に leakage-safe な時系列splitterを追加
  - `src/veldra/data|modeling|postprocess|simulate` にMVPプレースホルダを追加
  - `.gitignore` を追加（`.venv/.uv_cache/.uv_python` 等を除外）
- ドキュメント変更：
  - `DESIGN_BLUEPRINT.md` に「2026-02-09 MVP固定インターフェース」を追記
  - Open Questions に優先度（P1/P2）を付与
- テスト変更：
  - `tests/test_runconfig_validation.py`
  - `tests/test_splitter_contract.py`
  - `tests/test_artifact_roundtrip.py`
  - `tests/test_api_surface.py`
  - `tests/test_logging_contract.py`

**Decisions**
- Decision: confirmed
  - 内容：
    - 初期スコープはMVP骨格（公開API面の固定 + 未実装領域は統一例外で明示）に限定する。
  - 理由：
    - 安全性と再現性を確保しつつ、後続実装の変更コストを最小化するため。
  - 影響範囲：
    - Config / Artifact / API
- Decision: confirmed
  - 内容：
    - 依存管理は `pyproject.toml` + `uv.lock` の厳密固定運用（バージョン固定）で進める。
  - 理由：
    - 再現性最優先の原則に一致し、環境差異による不具合を抑制できるため。
  - 影響範囲：
    - Config / Artifact / API / 性能
- Decision: provisional
  - 内容：
    - Python 3.11 取得不能時のみ一時的に 3.12 を許容する。
  - 理由：
    - ネットワーク制約下でも作業停止を防ぐ運用上の保険として必要なため。
  - 影響範囲：
    - API / 性能 / 運用

**Results**
- 動作確認結果：
  - `uv python install 3.11` で 3.11.14 を導入し `.python-version` を3.11で固定
  - `uv add` / `uv lock` / `uv sync --dev` 実行済み
  - `uv run ruff check .` : All checks passed
  - `uv run pytest -q` : 10 passed

**Risks / Notes**
- PowerShellプロファイル由来の `[Console]::OutputEncoding` 警告が毎コマンドで混入するが、
  実行自体は継続可能。
- `uv` が一時ロックファイルを `%LOCALAPPDATA%\\Temp` に作成しようとした際、環境によって
  アクセス警告が出る場合がある。

**Open Questions**
- [ ] `fit` のMVP実装を「Artifact生成のみ」から「最小学習（1 fold smoke）」へいつ拡張するか
- [ ] `manifest_version` の次回更新ポリシー（破壊変更判定ルール）をどこで固定するか

### 2026-02-09 (Session/PR: phase4-examples-california-demo)
**Context**
- Add runnable examples for the current regression workflow (fit/predict/evaluate).
- Provide demo scripts that generate concrete output files for onboarding and verification.

**Plan**
- Add `examples/prepare_demo_data.py`, `examples/run_demo_regression.py`, and `examples/evaluate_demo_artifact.py`.
- Add shared helpers in `examples/common.py`.
- Add tests for prepare/run/evaluate example scripts.
- Update design and history documents with the examples contract.

**Changes**
- Added `examples/` scripts and `examples/README.md`.
- Added tests:
  - `tests/test_examples_prepare_demo_data.py`
  - `tests/test_examples_run_demo_regression.py`
  - `tests/test_examples_evaluate_demo_artifact.py`
- Updated `.gitignore` for `.codex/`, `examples/out/`, and generated CSV in `examples/data/`.

**Decisions**
- Decision: confirmed
  - Policy:
    - Example scripts are maintained as adapter-level executable references and do not alter `veldra.api.*` signatures.
  - Reason:
    - Reduces onboarding friction while preserving stable API compatibility.
  - Impact area:
    - API / Docs / Operability

- Decision: confirmed
  - Policy:
    - California Housing is the default demo source; local CSV must be created by `prepare_demo_data.py`.
  - Reason:
    - Makes demo flow explicit and reproducible with a consistent local input contract.
  - Impact area:
    - Data / Docs / Operability

**Results**
- Script-level outputs are produced under `examples/out/<timestamp>/`.
- `uv run ruff check .` passed.
- `uv run pytest -q` passed in this workspace before demo execution (15 passed).
- `uv run python examples/prepare_demo_data.py` failed in this environment due blocked network
  (`WinError 10061` while downloading California Housing), and exited with the expected hint.
- `uv run python examples/run_demo_regression.py` succeeded with local labeled CSV at the default path:
  - `run_id`: `ffff7218fba24318a1eaf4db85342f78`
  - `rmse`: `0.391954`
  - `mae`: `0.314148`
  - `r2`: `0.872247`
- `uv run python examples/evaluate_demo_artifact.py --artifact-path ...` succeeded:
  - `rmse`: `0.186839`
  - `mae`: `0.106562`
  - `r2`: `0.971317`
  - `n_rows`: `500`

**Risks / Notes**
- `prepare_demo_data.py` requires network access at fetch time.
- If download is unavailable, the script exits with a retry/network hint.

**Open Questions**
- [ ] Should the next example phase prioritize binary calibration flow or multiclass baseline first?
- [ ] Should artifact re-evaluation examples accept file-path input directly in addition to DataFrame-only API usage?

### 2026-02-09 (Session/PR: phase5-binary-fit-predict-evaluate-oof-calibration)
**Context**
- Extend MVP from regression-only runtime to binary runtime with OOF-safe calibration.
- Preserve stable API signatures and enforce artifact-based reproducibility.

**Plan**
- Add binary training core with CV and OOF Platt calibration.
- Extend artifact persistence for calibrator/calibration diagnostics/fixed threshold.
- Enable binary predict/evaluate through existing stable API entrypoints.
- Add binary-focused unit tests and update design/history docs.

**Changes**
- Code changes:
  - Added `src/veldra/modeling/binary.py`.
  - Updated `src/veldra/modeling/__init__.py` exports.
  - Updated `src/veldra/api/runner.py` for binary `fit/predict/evaluate`.
  - Updated `src/veldra/api/artifact.py` for binary prediction path.
  - Updated `src/veldra/artifact/store.py` to persist/load binary extras.
  - Updated `src/veldra/config/models.py` for binary calibration/threshold validation.
- Tests:
  - Added `tests/test_binary_fit_smoke.py`
  - Added `tests/test_binary_oof_calibration.py`
  - Added `tests/test_binary_predict_contract.py`
  - Added `tests/test_binary_evaluate_metrics.py`
  - Added `tests/test_binary_artifact_roundtrip.py`
  - Updated `tests/test_api_surface.py`
  - Updated `tests/test_runconfig_validation.py`

**Decisions**
- Decision: confirmed
  - Policy:
    - OOF-only fit is mandatory for probability calibration in binary flow.
  - Reason:
    - Prevent calibration leakage and keep evaluation defensible.
  - Impact area:
    - Modeling / Validation / Reproducibility

- Decision: confirmed
  - Policy:
    - Binary prediction contract is `p_cal` (default), plus `p_raw` and `label_pred`.
  - Reason:
    - Balance operational simplicity with auditability/debuggability.
  - Impact area:
    - API / Artifact / Consumer contract

- Decision: confirmed
  - Policy:
    - Binary evaluation metrics are `auc`, `logloss`, `brier`.
  - Reason:
    - Covers discrimination and probability quality.
  - Impact area:
    - Evaluation / Reporting

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : 35 passed.
- `uv run coverage run -m pytest -q && uv run coverage report -m` : TOTAL 94%.
- Binary sample run (`fit` + `evaluate`) metrics snapshot:
  - fit mean: `auc=0.9534`, `logloss=0.3247`, `brier=0.0955`
  - evaluate: `auc=1.0000`, `logloss=0.1065`, `brier=0.0103`

**Risks / Notes**
- Threshold optimization remains out of scope in this phase (fixed 0.5 only).
- `tune/simulate/export` remain unimplemented.

**Open Questions**
- [ ] Should Phase 6 prioritize binary threshold optimization or multiclass baseline first?
- [ ] Should binary `label_pred` support original class label restoration in API output?

### 2026-02-09 (Session/PR follow-up: phase5-binary-examples)
**Context**
- Extend examples after Phase 5 merge so binary workflow is reproducible end-to-end like regression.

**Changes**
- Added scripts:
  - `examples/prepare_demo_data_binary.py`
  - `examples/run_demo_binary.py`
  - `examples/evaluate_demo_binary_artifact.py`
- Updated docs:
  - `examples/README.md` with binary commands and outputs.
  - `DESIGN_BLUEPRINT.md` with binary examples note.
- Added tests:
  - `tests/test_examples_prepare_demo_data_binary.py`
  - `tests/test_examples_run_demo_binary.py`
  - `tests/test_examples_evaluate_demo_binary_artifact.py`

**Decisions**
- Decision: confirmed
  - Policy:
    - Binary examples are included as first-class adapter scripts in the repository.
  - Reason:
    - Keep onboarding and regression/binary parity for reproducible demo flows.
  - Impact area:
    - Examples / Documentation / QA

### 2026-02-10 (Session planning: phase6-multiclass-fit-predict-evaluate-examples)
**Context**
- Start Phase 6 to extend runtime from regression/binary to multiclass MVP.
- Keep stable API signatures unchanged and preserve artifact-first reproducibility.

**Decisions**
- Decision: provisional
  - Policy:
    - Multiclass prediction output contract is `label_pred` + `proba_<class>`.
  - Reason:
    - Supports both operational classification and probability-level analysis.
  - Impact area:
    - API / Artifact / Consumer contract

- Decision: provisional
  - Policy:
    - Multiclass evaluation metrics are `accuracy`, `macro_f1`, `logloss`.
  - Reason:
    - Balances label-quality and probability-quality checks.
  - Impact area:
    - Evaluation / Reporting

- Decision: provisional
  - Policy:
    - Multiclass examples are implemented in the same phase as core/API.
  - Reason:
    - Keeps onboarding parity across regression, binary, multiclass workflows.
  - Impact area:
    - Examples / Documentation / QA

### 2026-02-10 (Session/PR: phase6-multiclass-fit-predict-evaluate-examples)
**Context**
- Implement multiclass runtime (`fit/predict/evaluate`) and keep stable API signatures unchanged.
- Add runnable multiclass examples and matching tests in the same phase.

**Changes**
- Code changes:
  - Added `src/veldra/modeling/multiclass.py`.
  - Updated `src/veldra/modeling/__init__.py` exports.
  - Updated `src/veldra/api/runner.py` for multiclass `fit/predict/evaluate`.
  - Updated `src/veldra/api/artifact.py` for multiclass prediction contract.
  - Updated `examples/common.py` with multiclass default dataset path.
  - Added multiclass examples:
    - `examples/prepare_demo_data_multiclass.py`
    - `examples/run_demo_multiclass.py`
    - `examples/evaluate_demo_multiclass_artifact.py`
  - Updated `README.md` with multiclass status and commands.
- Tests:
  - Added `tests/test_multiclass_fit_smoke.py`
  - Added `tests/test_multiclass_predict_contract.py`
  - Added `tests/test_multiclass_evaluate_metrics.py`
  - Added `tests/test_multiclass_artifact_roundtrip.py`
  - Added `tests/test_examples_prepare_demo_data_multiclass.py`
  - Added `tests/test_examples_run_demo_multiclass.py`
  - Added `tests/test_examples_evaluate_demo_multiclass_artifact.py`
  - Updated `tests/test_api_surface.py`
  - Updated `tests/test_runconfig_validation.py`

**Decisions**
- Decision: confirmed
  - Policy:
    - Multiclass prediction contract is `label_pred` + `proba_<class>`.
  - Reason:
    - Provides both decision output and probability diagnostics from one API contract.
  - Impact area:
    - API / Artifact / Consumer contract

- Decision: confirmed
  - Policy:
    - Multiclass evaluation metrics are `accuracy`, `macro_f1`, `logloss`.
  - Reason:
    - Covers class-level quality and probability calibration quality.
  - Impact area:
    - Evaluation / Reporting

- Decision: confirmed
  - Policy:
    - Multiclass examples are shipped in the same phase as core/API support.
  - Reason:
    - Keeps runnable onboarding parity across task types.
  - Impact area:
    - Examples / Documentation / QA

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : 53 passed.
- Multiclass demo run (`examples/run_demo_multiclass.py`) snapshot:
  - `accuracy=0.900000`
  - `macro_f1=0.899749`
  - `logloss=0.818215`
- Multiclass artifact re-evaluation (`examples/evaluate_demo_multiclass_artifact.py`) snapshot:
  - `accuracy=0.980000`
  - `macro_f1=0.979998`
  - `logloss=0.163656`
  - `n_rows=150`

### 2026-02-10 (Session/PR: phase7-binary-threshold-optimization-optin)
**Context**
- Add binary threshold optimization while keeping default runtime behavior unchanged.
- Ensure optimization is explicit opt-in and does not interfere with typical usage.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - added `postprocess.threshold_optimization` config
    - added validation for binary-only usage and conflict with fixed threshold
  - Updated `src/veldra/modeling/binary.py`:
    - added optional OOF `p_cal` threshold optimization (F1)
    - added optional threshold curve generation
  - Updated `src/veldra/artifact/store.py` and `src/veldra/api/artifact.py`:
    - added optional `threshold_curve.csv` save/load
  - Updated `src/veldra/api/runner.py`:
    - binary evaluate now includes threshold-dependent metrics
    - binary metadata includes threshold policy/value
  - Updated examples/docs:
    - `examples/run_demo_binary.py` adds `--optimize-threshold` flag
    - `examples/evaluate_demo_binary_artifact.py` prints threshold-dependent metrics
    - `README.md` documents opt-in threshold optimization usage

**Decisions**
- Decision: confirmed
  - Policy:
    - Threshold optimization is opt-in only; default remains fixed threshold.
  - Reason:
    - Avoid changing established behavior in common production usage.
  - Impact area:
    - Compatibility / Operability / API behavior

- Decision: confirmed
  - Policy:
    - Threshold optimization uses OOF calibrated probabilities only.
  - Reason:
    - Prevent leakage and preserve defensible evaluation.
  - Impact area:
    - Modeling / Validation

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (phase7 implementation set).
- Binary default path remained fixed-threshold compatible.
- `--optimize-threshold` example path produced optimized threshold policy and optional curve file.

**Risks / Notes**
- Threshold optimization objective is fixed to F1 in this phase.
- The feature is intentionally binary-only and opt-in to avoid runtime surprise for standard users.

**Open Questions**
- [ ] Should additional threshold objectives (e.g., precision/recall constrained) be added in a future
      phase, or kept out of stable API for now?
- [ ] Should threshold policy be exposed in CLI/GUI presets after API stabilization?

### 2026-02-10 (Session/PR: phase7.1-doc-closure-and-phase8-tune-mvp)
**Context**
- Close remaining doc consistency tasks and implement `runner.tune` MVP.
- Keep stable API signatures unchanged and avoid behavioral regressions in existing paths.

**Plan**
- Phase 7.1:
  - complete unfinished Phase 7 history section
  - add capability matrix and historical clarification in design docs
- Phase 8:
  - implement `tune` runtime for regression/binary/multiclass
  - persist tuning artifacts under `artifacts/tuning/<run_id>/`
  - add full test coverage for tune smoke/validation/artifact outputs

**Changes**
- Code changes:
  - Added `src/veldra/modeling/tuning.py` (Optuna-backed tuning engine).
  - Updated `src/veldra/modeling/__init__.py` exports.
  - Updated `src/veldra/api/runner.py`:
    - `tune` now validates config and executes tuning
    - writes `study_summary.json` and `trials.parquet`
    - returns populated `TuneResult`
- Dependency updates:
  - Added runtime dependency: `optuna==4.0.0`
  - Updated lockfile (`uv.lock`)
- Tests:
  - Added `tests/test_tune_smoke_regression.py`
  - Added `tests/test_tune_smoke_binary.py`
  - Added `tests/test_tune_smoke_multiclass.py`
  - Added `tests/test_tune_validation.py`
  - Added `tests/test_tune_artifacts.py`
  - Updated `tests/test_api_surface.py`
  - Updated `tests/test_runner_additional.py`
- Docs:
  - Updated `DESIGN_BLUEPRINT.md` with capability matrix, historical note, and Phase 8 section.
  - Updated `README.md` to reflect `tune` support and usage.

**Decisions**
- Decision: confirmed
  - Policy:
    - `tune` requires `tuning.enabled=true`; otherwise validation error.
  - Reason:
    - Prevent accidental tuning execution from standard training configs.
  - Impact area:
    - API behavior / Operability

- Decision: confirmed
  - Policy:
    - Tuning objectives are fixed by task in MVP:
      regression=`rmse`(min), binary=`auc`(max), multiclass=`macro_f1`(max).
  - Reason:
    - Keep MVP deterministic, simple, and auditable.
  - Impact area:
    - Modeling / Evaluation

- Decision: confirmed
  - Policy:
    - Binary threshold optimization stays fully opt-in for prediction/evaluation flow and is disabled
      in tuning objective evaluation.
  - Reason:
    - Keep tuning comparisons focused on probability quality and preserve non-intrusive default policy.
  - Impact area:
    - Compatibility / Modeling

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed with newly added tune tests.
- `uv run coverage run -m pytest -q` + `uv run coverage report -m` : passed.
- `tune` now returns non-empty `best_params` / `best_score` for regression, binary, and multiclass.
- Tuning artifacts are generated under `artifacts/tuning/<run_id>/`.

**Risks / Notes**
- Search-space DSL is intentionally minimal in MVP; advanced conditional spaces are out of scope.
- `simulate/export/frontier runtime` remain unimplemented by design.

**Open Questions**
- [ ] Should next phase add user-selectable optimization metric per task, or keep fixed objective
      contracts for API stability?
- [ ] Should tuning results be loadable through Artifact-like API, or remain file-based outputs?

**Traceability note**
- The `Decision: provisional` entries in the 2026-02-10 Phase 6 planning section are now resolved by
  the confirmed implementation entry: `2026-02-10 (Session/PR: phase6-multiclass-fit-predict-evaluate-examples)`.

### 2026-02-10 (Session planning: phase8.1-tune-expansion)
**Context**
- Expand tune runtime for practical optimization workflows (objective selection, resume, progress
  logging, and tune examples).

**Decisions**
- Decision: provisional
  - Policy:
    - Objective is selectable with task-specific allowed choices.
  - Reason:
    - Preserve safety and clarity while avoiding free-form invalid objective names.
  - Impact area:
    - Config / Tuning / API behavior

- Decision: provisional
  - Policy:
    - Resume is implemented using Optuna SQLite (`study.db`) in artifact tuning directory.
  - Reason:
    - Ensure trial-level durability and restartability for interrupted runs.
  - Impact area:
    - Operability / Reproducibility

- Decision: provisional
  - Policy:
    - Tune progress logs are emitted during optimization with selectable log level.
  - Reason:
    - Improve observability during long optimization runs.
  - Impact area:
    - Logging / Operations

### 2026-02-10 (Session/PR: phase8.1-tune-expansion-implementation)
**Context**
- Extend tune runtime to support objective selection, resume, progress logging, and runnable examples.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - tuning fields added: `objective`, `resume`, `study_name`, `log_level`
    - task-constrained objective validation added
  - Updated `src/veldra/modeling/tuning.py`:
    - objective-aware direction mapping
    - deterministic/default `study_name` generation
    - SQLite storage support with resume handling
    - trial callback persistence (`study_summary.json`, `trials.parquet`)
  - Updated `src/veldra/api/runner.py`:
    - structured progress logs per trial (`tune trial completed`)
    - log level mapped from config
    - resume/non-resume study behavior and storage path management
  - Updated `src/veldra/modeling/binary.py`:
    - added binary threshold-dependent mean metrics for tune objective compatibility
  - Added `examples/run_demo_tune.py`:
    - task switch, objective override, resume, study-name, log-level, search-space-file
- Tests added:
  - `tests/test_tune_objective_selection.py`
  - `tests/test_tune_resume.py`
  - `tests/test_tune_logging.py`
  - `tests/test_examples_run_demo_tune.py`
- Tests updated:
  - `tests/test_tune_validation.py`
  - `tests/test_tuning_internal.py`

**Decisions**
- Decision: confirmed
  - Policy:
    - Tuning objective is selectable, but constrained by task-specific allowed metrics.
  - Reason:
    - Enable flexibility while preventing invalid objective configurations.
  - Impact area:
    - Config / Tuning / API behavior

- Decision: confirmed
  - Policy:
    - Resume uses Optuna SQLite in `artifacts/tuning/<study_name>/study.db`.
  - Reason:
    - Preserve progress across interruptions and allow controlled continuation.
  - Impact area:
    - Reproducibility / Operability

- Decision: confirmed
  - Policy:
    - Progress logs are emitted per trial with configurable level from `tuning.log_level`.
  - Reason:
    - Improve observability for long-running optimization.
  - Impact area:
    - Logging / Operations

- Decision: confirmed
  - Policy:
    - `tuning.search_space` remains the primary contract to choose target parameters and ranges.
  - Reason:
    - Keep optimization boundary explicit and controllable by users.
  - Impact area:
    - Tuning UX / Reproducibility

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed.
- Tune now supports:
  - objective selection
  - resume continuation
  - per-trial persisted progress
  - configurable logging level
  - unified tune demo script with CLI overrides

**Risks / Notes**
- Progress persistence writes per trial; very high trial counts can increase artifact I/O.
- For non-resume runs with existing study name, explicit conflict error is returned by design.

**Open Questions**
- [ ] Should pruning strategy (e.g., median pruner) be introduced in a later phase?
- [ ] Should tune results have a dedicated load API (Artifact-like) beyond file outputs?

### 2026-02-10 (Session planning: phase9-frontier-fit-predict-evaluate-mvp)
**Context**
- Implement the next runtime capability for `task.type="frontier"` with minimal MVP scope.
- Preserve stable API signatures and avoid regressions for existing tasks.

**Decisions**
- Decision: provisional
  - Policy:
    - Frontier runtime MVP scope is `fit/predict/evaluate` only.
  - Reason:
    - Delivers practical runtime coverage without coupling to larger simulate/export work.
  - Impact area:
    - API behavior / Modeling / Artifact contract

- Decision: provisional
  - Policy:
    - Frontier default quantile alpha is `0.90`.
  - Reason:
    - Provides a deterministic default while keeping explicit override support.
  - Impact area:
    - Config / Modeling / Evaluation

- Decision: provisional
  - Policy:
    - Frontier prediction output contract is DataFrame with `frontier_pred`, and optional `u_hat`
      when labeled input is provided.
  - Reason:
    - Keeps unlabeled inference simple while supporting immediate inefficiency inspection on labeled
      data.
  - Impact area:
    - API contract / Operability

### 2026-02-10 (Session/PR: phase9-frontier-fit-predict-evaluate-mvp)
**Context**
- Implement frontier runtime as the next minimal production path after tune expansion.
- Keep stable API signatures unchanged and preserve existing task behavior.

**Changes**
- Code changes:
  - Added `src/veldra/modeling/frontier.py`:
    - `train_frontier_with_cv` with quantile objective and CV evaluation
    - frontier metrics: `pinball`, `mae`, `mean_u_hat`, `coverage`
  - Updated `src/veldra/config/models.py`:
    - added `FrontierConfig` (`alpha` default `0.90`)
    - added frontier validation rules (`alpha` bounds, split restrictions, non-frontier guard)
  - Updated `src/veldra/modeling/__init__.py` exports for frontier trainer.
  - Updated `src/veldra/api/runner.py`:
    - `fit/predict/evaluate` now support `task.type="frontier"`
    - evaluate returns frontier metrics and metadata (`frontier_alpha`)
  - Updated `src/veldra/api/artifact.py`:
    - frontier prediction path with `frontier_pred` and optional `u_hat`
  - Updated `examples/common.py` with `DEFAULT_FRONTIER_DATA_PATH`.
  - Added examples:
    - `examples/prepare_demo_data_frontier.py`
    - `examples/run_demo_frontier.py`
    - `examples/evaluate_demo_frontier_artifact.py`
- Tests added:
  - `tests/test_frontier_fit_smoke.py`
  - `tests/test_frontier_predict_contract.py`
  - `tests/test_frontier_evaluate_metrics.py`
  - `tests/test_frontier_artifact_roundtrip.py`
  - `tests/test_frontier_config_validation.py`
  - `tests/test_examples_prepare_demo_data_frontier.py`
  - `tests/test_examples_run_demo_frontier.py`
  - `tests/test_examples_evaluate_demo_frontier_artifact.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_runner_additional.py`
  - `tests/test_artifact_additional.py`
  - `tests/test_runconfig_validation.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Frontier quantile default alpha is `0.90`.
  - Reason:
    - Ensures deterministic behavior while preserving explicit override capability.
  - Impact area:
    - Config / Modeling / Evaluation

- Decision: confirmed
  - Policy:
    - Frontier MVP scope is `fit/predict/evaluate` only.
  - Reason:
    - Delivers runtime value now without coupling to simulate/export backlog.
  - Impact area:
    - API behavior / Delivery risk

- Decision: confirmed
  - Policy:
    - Frontier prediction contract is `frontier_pred` (+ optional `u_hat` when target present).
  - Reason:
    - Supports both unlabeled inference and labeled inefficiency inspection without API branching.
  - Impact area:
    - API contract / Operability

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed with frontier tests included.
- `uv run coverage run -m pytest -q` + `uv run coverage report -m` : passed.

**Risks / Notes**
- `u_hat` output on frontier prediction is available only when target column is present in input.
- `tune(frontier)`, `simulate`, and `export` remain intentionally unimplemented.

**Open Questions**
- [ ] Should frontier `tune` objective support only pinball initially, or include coverage-constrained
      variants from the first release?
- [ ] Should frontier prediction always return `u_hat` when target is absent by allowing explicit
      target argument, or keep current implicit contract?

### 2026-02-10 (Session/PR: phase10-simulate-mvp-scenario-dsl)
**Context**
- Implement `simulate` MVP as the next runtime feature while keeping stable API signatures unchanged.
- Keep delivery non-intrusive: no behavior change in existing `fit/predict/evaluate/tune`.

**Changes**
- Code changes:
  - Added `src/veldra/simulate/engine.py`:
    - scenario normalization (`dict` / `list[dict]`)
    - action application (`set/add/mul/clip`)
    - task-specific simulation output builder
  - Updated `src/veldra/simulate/__init__.py` exports.
  - Updated `src/veldra/api/runner.py`:
    - implemented `simulate(artifact, data, scenarios)` for regression/binary/multiclass/frontier
    - added structured completion log (`simulate completed`)
  - Updated `src/veldra/api/artifact.py`:
    - implemented `Artifact.simulate(df, scenario)` single-scenario shortcut
  - Added `examples/run_demo_simulate.py`.
- Tests added:
  - `tests/test_simulate_engine_actions.py`
  - `tests/test_simulate_runner_regression.py`
  - `tests/test_simulate_runner_binary.py`
  - `tests/test_simulate_runner_multiclass.py`
  - `tests/test_simulate_runner_frontier.py`
  - `tests/test_examples_run_demo_simulate.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_runner_additional.py`
  - `tests/test_artifact_additional.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - `simulate` MVP is delivered for all implemented task runtimes
      (regression/binary/multiclass/frontier).
  - Reason:
    - Closes major runtime gap with minimal API surface change.
  - Impact area:
    - API behavior / Scenario runtime

- Decision: confirmed
  - Policy:
    - Scenario DSL action set is fixed to `set/add/mul/clip` for MVP.
  - Reason:
    - Covers common scenario operations while minimizing complexity and risk.
  - Impact area:
    - Validation / Operability

- Decision: confirmed
  - Policy:
    - Simulation output contract is long-form with shared keys
      (`row_id`, `scenario`, `task_type`) plus task-specific comparison columns.
  - Reason:
    - Keeps downstream processing consistent across tasks.
  - Impact area:
    - API contract / Analytics usability

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (`150 passed`).

**Risks / Notes**
- Simulation operations currently target numeric columns only.
- `export` and `tune(frontier)` remain intentionally unimplemented.

**Open Questions**
- [ ] Should Phase 11 `export` MVP start with python-only inference package, or include ONNX in first release?
- [ ] Should `tune(frontier)` start with pinball-only objective, or include coverage-constrained variants?

### 2026-02-10 (Session planning: phase10-simulate-mvp)
**Context**
- Implement `simulate` MVP as the next runtime capability after frontier and tuning expansions.
- Preserve stable API signatures and keep existing task behavior unchanged.
- Status:
  - Superseded by `Session/PR: phase10-simulate-mvp-scenario-dsl` with `Decision: confirmed`.

**Decisions**
- Decision: provisional
  - Policy:
    - Phase 10 scope is `simulate` only, using one PR with non-intrusive changes.
  - Reason:
    - Close the largest remaining runtime gap while isolating risk from `export` and `tune(frontier)`.
  - Impact area:
    - API behavior / Scenario runtime / Examples

- Decision: provisional
  - Policy:
    - Scenario DSL minimal operations are `set/add/mul/clip`.
  - Reason:
    - Provides practical simulation controls without introducing search/optimization complexity.
  - Impact area:
    - Config/runtime contract / Validation

- Decision: provisional
  - Policy:
    - `SimulationResult.data` is long-form with shared keys (`row_id`, `scenario`, `task_type`) and
      task-specific comparison columns.
  - Reason:
    - Keeps downstream analysis format stable while supporting all implemented task types.
  - Impact area:
    - API contract / Operability

### 2026-02-10 (Session planning: phase11-export-mvp-python-onnx-optional)
**Context**
- Implement `export` MVP as the next runtime capability after `simulate`.
- Keep existing runtime behavior stable and add optional ONNX support without making it mandatory.

**Decisions**
- Decision: provisional
  - Policy:
    - `runner.export` supports `python` and `onnx` formats with unchanged signature.
  - Reason:
    - Closes remaining stable API gap while preserving compatibility.
  - Impact area:
    - API behavior / Artifact distribution

- Decision: provisional
  - Policy:
    - ONNX export is optional-dependency based; missing dependency raises explicit validation error.
  - Reason:
    - Avoids forcing heavy dependencies while keeping ONNX path available.
  - Impact area:
    - Packaging / Operability

- Decision: provisional
  - Policy:
    - Python export is always available and task-agnostic (regression/binary/multiclass/frontier).
  - Reason:
    - Guarantees baseline export usability in constrained environments.
  - Impact area:
    - User experience / Runtime reliability

### 2026-02-10 (Session/PR: phase11-export-mvp-python-onnx-optional)
**Context**
- Implement `export` MVP to close the remaining stable API runtime gap.
- Keep `fit/predict/evaluate/tune/simulate` behavior unchanged.

**Changes**
- Code changes:
  - Added `src/veldra/artifact/exporter.py`:
    - `export_python_package(artifact, out_dir)`
    - `export_onnx_model(artifact, out_dir)`
    - optional ONNX dependency loading with explicit error guidance
  - Updated `src/veldra/api/runner.py`:
    - implemented `export(artifact, format)`
    - supported formats: `python`, `onnx`
    - structured export completion log
  - Added `examples/run_demo_export.py`
  - Updated `pyproject.toml`:
    - added optional dependency group `export-onnx`
- Tests added:
  - `tests/test_export_python_mvp.py`
  - `tests/test_export_onnx_optional.py`
  - `tests/test_export_runner_contract.py`
  - `tests/test_examples_run_demo_export.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_runner_additional.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - `export` supports `python` and `onnx` with unchanged API signature.
  - Reason:
    - Completes the stable API runtime surface with minimal compatibility risk.
  - Impact area:
    - API behavior / Distribution workflow

- Decision: confirmed
  - Policy:
    - ONNX export remains optional-dependency based and non-blocking for python export.
  - Reason:
    - Keeps default setup lightweight while enabling ONNX where required.
  - Impact area:
    - Packaging / Operability

- Decision: confirmed
  - Policy:
    - ONNX export for `frontier` is explicitly unsupported in MVP (`VeldraNotImplementedError`).
  - Reason:
    - Avoids silent behavior ambiguity and keeps conversion contract explicit.
  - Impact area:
    - API contract / Reliability

**Results**
- `uv run ruff check .` : pending
- `uv run pytest -q` : pending

**Risks / Notes**
- ONNX conversion success depends on optional toolchain availability and converter compatibility.
- `tune(frontier)` remains intentionally unimplemented.

**Open Questions**
- [ ] Should Phase 12 prioritize `tune(frontier)` or frontier ONNX export support first?
