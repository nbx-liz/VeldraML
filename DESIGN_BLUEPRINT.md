# DESIGN_BLUEPRINT.md
（AIエージェント向け設計図）

## 0. 概要
プロジェクト名（仮）：VeldraML（Pythonパッケージ：`veldra`）
LightGBMベースで、回帰・二値分類・多値分類・フロンティア（SFA近似）を
Config駆動で学習・評価・最適化（HPO）・配布（Export/Artifact）・シミュレーションまで扱う。

前提：
- 主ユーザーはデータサイエンティスト寄り
- 100万行程度に対応
- 特徴量生成はスコープ外（入力特徴量は外部で作る）

---

## 1. 主要ユースケース
### 1.1 学習・評価
- 回帰/二値/多値/フロンティアを学習し、CVで評価し、結果を保存する
- 時系列CVを選べる（time_col指定）

### 1.2 HPO（自動最適化）
- Optuna等で探索
- デフォルトは **CV内包Objective**（安定性優先）
- tuning時は “fast preset” で軽量化できる

### 1.3 推論・配布
- Artifactを読み込み、同一スキーマで推論できる
- Exportで推論テンプレ（任意）を出力できる

### 1.4 シミュレーション
- Scenario（入力変更ルール）を適用し、予測差分を比較
- リソース配賦（総量固定・上限/下限・枠）をサポート
- タスク別に結果の見せ方を切り替える

### 1.5 Frontier（SFA目的）
- 生産フロンティア：quantile回帰で上側境界を推定（alpha高分位）
- 非効率：`u_hat = max(0, frontier_pred - y_obs)` を標準出力

### 1.6 二値分類の確率校正
- Platt / Isotonic（初期はPlatt推奨）
- **OOF予測で校正器fit**（リーク回避）
- 閾値最適化（metric/cost/constraint）

---

## 2. 非目標（スコープ外）
- 特徴量生成（FE）パイプライン
- 厳密なSFAの最尤推定（分布仮定でvとuを分離する推定）
- 巨大分散処理（Spark等）は当面対象外

---

## 3. アーキテクチャ（レイヤ）
### 3.1 Core（純ライブラリ）
- GUI/CLI/APIを知らない
- RunConfigを入力として、学習・評価・保存を完結

### 3.2 Adapters（外側）
- CLI：Coreを呼ぶだけ
- GUI：Config編集・可視化・Run管理
- API：将来用（ジョブ化するなら）

---

## 4. パッケージ構成（推奨）
- `veldra.api`：Stable API（Runner / Artifactの公開窓口）
- `veldra.config`：RunConfig（pydantic）とI/O
- `veldra.data`：loader/schema/validate
- `veldra.split`：splitter群（kfold/stratified/group/timeseries）
- `veldra.modeling`：task/trainer/cv/tuning/metrics
- `veldra.postprocess`：calibration/threshold/reliability（二値中心）
- `veldra.simulate`：Scenario DSL / engine / optimize（任意）
- `veldra.artifact`：save/load/export/manifest

---

## 5. Stable API（壊さない）
### 5.1 Runner
- `fit(config) -> RunResult`
- `tune(config) -> TuneResult`
- `evaluate(artifact_or_config, data) -> EvalResult`
- `predict(artifact, data) -> Prediction`
- `simulate(artifact, data, scenarios) -> SimulationResult`
- `export(artifact, format=...) -> ExportResult`

### 5.2 Artifact
- `Artifact.load(path)`
- `artifact.save(path)`
- `artifact.predict(df)`
- `artifact.simulate(df, scenario)`

---

## 6. RunConfig（骨格）
- `task.type`: regression | binary | multiclass | frontier
- `data`: path/target/id_cols/categorical/drop_cols
- `split`: type/n_splits/time_col/group_col/seed
- `train`: lgb_params/early_stopping/seed
- `tuning`: enabled/n_trials/search_space/preset
- `postprocess`: calibration/threshold（binaryのみ）
- `simulation`: scenarios/actions/constraints
- `export`: artifact_dir/inference_package

### Configバージョニング
- `config_version` を必須化
- 変更時は `veldra.config.migrate` に移行ロジックを追加（後方互換）

---

## 7. Task仕様（差分点）
### 7.1 Regression
- objective/metricの標準セット
- 出力：y_pred

### 7.2 Binary
- 出力：p_raw, p_cal（校正有効時）
- threshold policy：metric/cost/constraint

### 7.3 Multiclass
- 出力：class_proba, label_pred
- 指標：macro/micro等はconfigで選択可能

### 7.4 Frontier（production）
- objective：quantile（alpha高分位）
- 出力：frontier_pred, u_hat
- オプション：monotone constraints（必要なら）

---

## 8. CV / Splitter
- KFold / StratifiedKFold / GroupKFold
- TimeSeriesSplit（基本）
- 将来：blocked/purged/embargoを追加できる抽象（base interfaceを先に固定）

---

## 9. HPO（Optuna等）
- デフォルト：CV内包Objective
- 早期停止必須、探索空間はtask/presetで初期値提供
- 1M行対応のため tuning preset（fast/standard）を用意

---

## 10. 二値：確率校正と閾値
### 10.1 校正
- fitはOOFで行う（データリーク防止）
- 手法：Platt（デフォ）/ Isotonic（オプション）

### 10.2 閾値
- strategy：metric / cost / constraint
- thresholdと根拠（params）をArtifactへ保存

---

## 11. シミュレーション（Scenario DSL）
### 11.1 actions（最小セット）
- `set` / `add` / `mul` / `clip`
- `allocate_total`（総量固定配賦）
  - weight_by（例：inefficiency_rank等）を許容

### 11.2 constraints
- total固定、unit上限/下限、増減率上限、group枠

### 11.3 outputs
- 予測差分（Δ）、集計（unit/period/group）
- FrontierはΔu_hat を標準で出す
- 分布外警告（学習分布からの逸脱）を出す

---

## 12. Artifactフォーマット（固定）
最低限：
- manifest.json（version/hash/依存）
- run_config.yaml
- feature_schema.json
- model.lgb.txt
- metrics.json
- cv_results.parquet

binary追加：
- calibrator.pkl / threshold.json / calibration_curve.csv

frontier追加：
- frontier_pred / inefficiency(u_hat)

export（任意）：
- predict.py / requirements.txt / Dockerfile

---

## 13. GUI（MVP方針）
- GUIはConfig編集＋実行＋可視化＋Artifact管理に徹する
- 入口は用途テンプレ：
  - 学習、スコアリング、シミュレーション、エクスポート
- Coreロジックは持たない（Runnerを呼ぶだけ）

---

## 14. テスト戦略
- スモーク：fit→save→load→predict の一連
- 重要：OOF校正が守られていること（リーク検知）
- splitter：時系列の順序性、groupリークの防止
- artifact：同一入力で同一出力（固定seed）を確認

---

## 15. Open Questions（都度更新）
- [P1] [ ] Frontierのalphaデフォルトと評価指標（pinball loss等）
- [P1] [ ] 時系列splitの強化（blocked/gap）をいつ入れるか
- [P2] [ ] 自動最適化（配賦）をMVPに入れるか、後回しにするか
- [P2] [ ] Export（ONNX等）をサポートするか

---

## 16. 2026-02-09 MVP固定インターフェース（Phase 1）
### 16.1 Scope（固定）
- このフェーズでは `src/veldra` の骨格、RunConfig、Artifact I/O、Stable APIの入口を固定する。
- HPO/校正/シミュレーション本体は未実装とし、公開APIでは統一例外で明示する。

### 16.2 Stable API挙動（MVP）
- `fit(config)` は RunConfigを検証し、Artifactの最小構成（manifest/run_config/feature_schema）を保存する。
- `tune/evaluate/predict/simulate/export` は `VeldraNotImplementedError` を返し、黙って失敗しない。
- `Artifact.load/save` は動作保証対象とする。
- `Artifact.predict/simulate` は `VeldraNotImplementedError` を返す。

### 16.3 RunConfigバリデーション（MVP）
- `config_version` は必須かつ `>=1`。
- `split.type=timeseries` では `split.time_col` 必須。
- `split.type=group` では `split.group_col` 必須。
- `postprocess.calibration/threshold` は `task.type=binary` のときのみ許容。

### 16.4 Artifact最小契約（MVP）
- 必須ファイル: `manifest.json`, `run_config.yaml`, `feature_schema.json`。
- manifestには `manifest_version`, `project_version`, `run_id`, `task_type`,
  `config_version`, `config_hash`, `python_version`, `dependencies`, `created_at_utc` を保存する。

---

## 17. 2026-02-09 Examples and Demo Flow (Phase 4)
### 17.1 Goal
- Add runnable examples that execute the regression workflow on real demo data.
- Keep the stable API unchanged while making onboarding and reproducibility easier.

### 17.2 Included scripts
- `examples/prepare_demo_data.py`
  - Downloads California Housing with `fetch_california_housing(as_frame=True)`.
  - Writes `examples/data/california_housing.csv` with a normalized target column name: `target`.
  - Uses `examples/data/sklearn_data` as local dataset cache to avoid permission issues.
- `examples/run_demo_regression.py`
  - Reads local CSV, performs train/test split, runs `fit`, `evaluate`, and `predict`.
  - Stores outputs under `examples/out/<timestamp>/`.
- `examples/evaluate_demo_artifact.py`
  - Loads an existing Artifact and re-runs `evaluate` on labeled tabular data.

### 17.3 Output contract for demo run
- `run_result.json`
- `eval_result.json`
- `predictions_sample.csv`
- `used_config.yaml`
- Artifact files under `artifacts/<run_id>/`

### 17.4 Design constraints
- Examples are adapter-side helpers only and do not introduce business logic into core modules.
- Demo data file is generated locally and is not committed.
- Failure cases must give actionable hints (missing CSV, target column mismatch, invalid artifact path).

### 17.5 Open Questions updates
- [Closed] Should examples with real data be included in MVP follow-up phases?
  - Decision: yes (Phase 4).
- [Next] Binary calibration demo order:
  - Candidate A: implement binary training first, then examples.
  - Candidate B: ship a synthetic-only binary example before full binary training.

---

## 18. 2026-02-09 Binary Workflow (Phase 5)
### 18.1 Goal
- Implement binary `fit/predict/evaluate` with OOF-based probability calibration.
- Keep stable API signatures unchanged and add behavior only.

### 18.2 Implemented scope
- `fit` supports `task.type='binary'`.
- `predict` supports binary artifacts and returns `p_cal`, `p_raw`, `label_pred`.
- `evaluate` supports binary artifacts and returns `auc`, `logloss`, `brier`.
- OOF raw predictions are used to fit Platt calibrator.

### 18.3 Binary artifact additions
- `calibrator.pkl`
- `calibration_curve.csv`
- `threshold.json` (fixed threshold 0.5)

### 18.4 Validation and constraints
- Binary target must contain exactly two classes.
- For this phase, `postprocess.calibration` supports `platt` only.
- Threshold value validation: `0.0 <= threshold <= 1.0`.

### 18.5 API behavior notes
- Stable API signatures remain unchanged.
- `Artifact.predict` now supports regression and binary.
- `runner.evaluate` now supports regression and binary for Artifact input.

### 18.6 Examples extension for binary workflow
- Added binary demo scripts under `examples/`:
  - `prepare_demo_data_binary.py` for local dataset preparation (Breast Cancer CSV).
  - `run_demo_binary.py` for end-to-end `fit/predict/evaluate`.
  - `evaluate_demo_binary_artifact.py` for artifact re-evaluation.
- Binary example output contract follows regression examples:
  - `run_result.json`, `eval_result.json`, `predictions_sample.csv`, `used_config.yaml`.
- This extension is adapter-side only and does not change `veldra.api.*` signatures.

---

## 19. 2026-02-10 Multiclass Workflow Proposal (Phase 6)
### 19.1 Goal
- Add `task.type='multiclass'` runtime support for `fit/predict/evaluate`.
- Keep stable API signatures unchanged while extending behavior.

### 19.2 Proposed runtime contract
- `fit` supports multiclass with CV metrics and artifact generation.
- `predict` returns tabular output with:
  - `label_pred`
  - `proba_<class>` columns aligned to `feature_schema.target_classes`.
- `evaluate` (Artifact input) returns:
  - `accuracy`, `macro_f1`, `logloss`.

### 19.3 Artifact and schema
- No new required artifact files for multiclass.
- Existing files remain mandatory:
  - `manifest.json`, `run_config.yaml`, `feature_schema.json`.
- Existing optional training payload remains used:
  - `model.lgb.txt`, `metrics.json`, `cv_results.parquet`.
- `feature_schema.json` includes multiclass `target_classes` order contract.

### 19.4 Scope notes
- In scope:
  - multiclass core training path + API path + examples + tests.
- Out of scope:
  - calibration/threshold optimization for multiclass.
  - `tune/simulate/export` implementations.

### 19.5 Implemented status (Phase 6)
- Added multiclass runtime for:
  - `fit(config)`
  - `predict(artifact, data)`
  - `evaluate(artifact, data)`
- Prediction contract:
  - `label_pred`
  - `proba_<class>` columns ordered by `feature_schema.target_classes`
- Evaluation metrics:
  - `accuracy`, `macro_f1`, `logloss`
- Added multiclass examples:
  - `prepare_demo_data_multiclass.py`
  - `run_demo_multiclass.py`
  - `evaluate_demo_multiclass_artifact.py`
- Stable API signatures remain unchanged.

---

## 20. 2026-02-10 Binary Threshold Optimization (Phase 7)
### 20.1 Goal
- Add optional binary threshold optimization without changing default behavior.
- Keep existing jobs fully compatible unless optimization is explicitly enabled.

### 20.2 Non-intrusive default policy
- `postprocess.threshold_optimization` is disabled by default.
- When disabled:
  - threshold remains fixed (`postprocess.threshold` or fallback `0.5`)
  - no additional optimization artifact files are produced
  - predict/evaluate runtime path stays lightweight

### 20.3 Opt-in optimization policy
- Enabled only via explicit config:
  - `postprocess.threshold_optimization.enabled: true`
  - `objective: "f1"` (current phase)
- Optimization data source:
  - OOF calibrated probabilities (`p_cal`) only.
- Saved outputs:
  - `threshold.json` with `policy: optimized_f1`
  - `threshold_curve.csv` (optional, optimization enabled only)

### 20.4 Binary evaluate extension
- Keep existing probability metrics:
  - `auc`, `logloss`, `brier`
- Add threshold-dependent metrics:
  - `accuracy`, `f1`, `precision`, `recall`, `threshold`

---

## 21. Current Capability Matrix (2026-02-10)
| Area | Status | Notes |
| --- | --- | --- |
| `fit` | Implemented | regression, binary, multiclass |
| `predict` | Implemented | regression, binary, multiclass |
| `evaluate` | Implemented | Artifact input path for regression, binary, multiclass, frontier |
| `tune` | Implemented (Phase 8 MVP) | regression, binary, multiclass (Optuna TPE) |
| `simulate` | Not implemented | Kept as stable API stub |
| `export` | Not implemented | Kept as stable API stub |
| `frontier` task runtime | Implemented (Phase 9 MVP) | fit/predict/evaluate with quantile baseline |

## 22. Historical Note for Section 16
- Section 16 records the **Phase 1 point-in-time state**.
- Statements in Section 16 that mention unimplemented `evaluate/predict` are historical and do not
  represent the current runtime status after Phases 2-8.

## 23. Open Questions (Updated)
- [Closed] Threshold optimization MVP inclusion
  - Status: implemented as binary-only **opt-in** feature in Phase 7.
- [Closed] Frontier alpha default and objective finalization
  - Status: Phase 9 sets default `alpha=0.90`, objective `quantile`, and metric contract
    (`pinball`, `mae`, `mean_u_hat`, `coverage`).
- [P1] Time-series split advanced options (`blocked/gap/embargo`) timeline.
- [P2] Export ONNX support prioritization.

## 24. 2026-02-10 Hyperparameter Tuning MVP (Phase 8)
### 24.1 Goal
- Activate `runner.tune(config)` while preserving stable API signatures.
- Keep behavior non-intrusive for existing `fit/predict/evaluate` paths.

### 24.2 Scope
- Supported tasks: regression, binary, multiclass.
- Optimization backend: Optuna TPE sampler with fixed seed (`train.seed`).
- Outputs:
  - `artifacts/tuning/<run_id>/study_summary.json`
  - `artifacts/tuning/<run_id>/trials.parquet`

### 24.3 Objective contract
- regression: minimize `rmse`
- binary: maximize `auc`
- multiclass: maximize `macro_f1`

### 24.4 Design constraints
- `tuning.enabled=true` is required for `tune`.
- `data.path` is required for `tune`.
- Binary threshold optimization remains opt-in for prediction/evaluation flow, but is disabled inside
  tuning objective evaluation to avoid adding threshold-policy effects to probability metric search.

## 25. 2026-02-10 Hyperparameter Tuning Expansion Proposal (Phase 8.1)
### 25.1 Proposal
- Extend tuning to support:
  - selectable objective per task (task-constrained choices)
  - resumable study execution via Optuna SQLite storage
  - per-trial progress persistence (`study_summary.json`, `trials.parquet`)
  - progress logs with configurable log level
  - single runnable tune example script with CLI overrides

### 25.2 Rationale
- Improves practical usability for long-running optimization jobs.
- Reduces risk of losing work on interruptions.
- Keeps stable API signatures unchanged while expanding behavior.

### 25.3 Compatibility notes
- Existing `fit/predict/evaluate` behavior remains unchanged.
- Binary threshold optimization remains fully opt-in and separated from tuning objectives.

## 26. 2026-02-10 Hyperparameter Tuning Expansion (Phase 8.1, implemented)
### 26.1 Added capabilities
- Objective selection with task-constrained allowed values:
  - regression: `rmse|mae|r2`
  - binary: `auc|logloss|brier|accuracy|f1|precision|recall`
  - multiclass: `accuracy|macro_f1|logloss`
- Resumable tuning via Optuna SQLite:
  - `artifacts/tuning/<study_name>/study.db`
- Trial-by-trial persistence:
  - `study_summary.json`
  - `trials.parquet`
- Progress logging with configurable level (`DEBUG|INFO|WARNING|ERROR`).
- Single tune demo script:
  - `examples/run_demo_tune.py`

### 26.2 Runtime contract
- `tune` remains `veldra.api.runner.tune(config) -> TuneResult` (signature unchanged).
- `tuning.enabled=true` is required for execution.
- `tuning.resume=true` continues existing study; otherwise existing study name is rejected.
- `tuning.search_space` is the formal user contract for optimization target parameters and ranges.

### 26.3 Non-intrusive behavior
- `fit/predict/evaluate` are unaffected.
- Binary threshold optimization remains opt-in and is disabled during tuning objective scoring.

## 27. 2026-02-10 Frontier Runtime MVP Proposal (Phase 9)
### 27.1 Goal
- Activate `task.type="frontier"` runtime for `fit/predict/evaluate` while preserving stable API
  signatures.
- Keep implementation minimal and non-intrusive for existing regression/binary/multiclass paths.

### 27.2 Confirmed proposal defaults
- Frontier quantile objective uses `objective="quantile"` and `metric="quantile"`.
- Frontier default alpha is `0.90` when not explicitly configured.
- MVP scope is limited to:
  - `fit(config)`
  - `predict(artifact, data)`
  - `evaluate(artifact, data)` (Artifact input path only)

### 27.3 Proposed config contract
- Add `frontier.alpha: float = 0.90` under `RunConfig`.
- Validation rules:
  - `0 < frontier.alpha < 1` for `task.type="frontier"`.
  - `split.type="stratified"` is forbidden for frontier.
  - Non-frontier tasks must not set non-default `frontier` values.

### 27.4 Proposed metric contract
- CV/evaluation metrics for frontier:
  - `pinball`
  - `mae`
  - `mean_u_hat` where `u_hat = max(0, frontier_pred - y_true)`
  - `coverage` where `coverage = mean(y_true <= frontier_pred)`

### 27.5 Proposed prediction contract
- `Artifact.predict(df)` for frontier returns DataFrame with:
  - `frontier_pred` (always)
  - `u_hat` (only when target column exists in `df`)

### 27.6 Compatibility notes
- `tune(frontier)`, `simulate`, and `export` remain intentionally unimplemented.
- Existing task contracts and outputs remain unchanged.

## 28. 2026-02-10 Frontier Runtime MVP (Phase 9, implemented)
### 28.1 Added capabilities
- `fit` now supports `task.type="frontier"` with LightGBM quantile objective.
- `predict` now supports frontier and returns DataFrame with:
  - `frontier_pred` always
  - `u_hat` when labeled input contains target column
- `evaluate(artifact, data)` now supports frontier and returns:
  - `pinball`, `mae`, `mean_u_hat`, `coverage`

### 28.2 Config/runtime contract
- Added `RunConfig.frontier.alpha` with default `0.90`.
- Frontier validation:
  - requires `0 < alpha < 1`
  - rejects `split.type='stratified'`
  - non-frontier tasks cannot override default frontier config

### 28.3 Artifact and compatibility notes
- Existing artifact files remain unchanged; frontier extras are stored in existing:
  - `feature_schema.json` (`frontier_alpha`)
  - `metrics.json`
- `tune(frontier)`, `simulate`, and `export` remain intentionally unimplemented.
