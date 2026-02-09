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
