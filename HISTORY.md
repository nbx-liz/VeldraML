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

### 2026-02-09 (Session/PR: phase2-pr1-regression-fit-predict-cv)
**Context**
- 背景 / 依頼 / 目的：
  - Phase 2として回帰 `fit/predict` を実装し、MVP骨格を実学習可能な状態へ進める。

**Plan**
- CSV/Parquetローダー追加
- 回帰CV学習とArtifact拡張（model/metrics/cv_results）を実装
- runner/artifactのpredict経路を回帰で実装

**Changes**
- 実装変更：
  - `src/veldra/data/loader.py` を追加し、CSV/Parquet読み込みを実装
  - `src/veldra/modeling/regression.py` を追加し、CV学習・OOF評価・最終モデル学習を実装
  - `src/veldra/api/runner.py` の `fit/predict` を回帰対応へ更新
  - `src/veldra/api/artifact.py` にモデル保持・回帰推論を実装
  - `src/veldra/artifact/store.py` に `model.lgb.txt/metrics.json/cv_results.parquet` の保存/読込を追加
- テスト変更：
  - `tests/test_api_surface.py` を新仕様（data.path必須、predict実装）に整合

**Decisions**
- Decision: confirmed
  - 内容：
    - Phase 2の第一段として、`task.type='regression'` の `fit/predict` を先行実装する。
  - 理由：
    - Stable APIを壊さず、学習系の最小実行価値を最短で提供するため。
  - 影響範囲：
    - API / Artifact / Config
- Decision: confirmed
  - 内容：
    - 予測時の余剰列は無視し、不足列のみエラーにする。
  - 理由：
    - 推論時の運用利便性を確保しつつ、必須特徴量欠落は安全側で失敗させるため。
  - 影響範囲：
    - API / 推論運用

**Results**
- 動作確認結果：
  - `uv run ruff check .` : All checks passed
  - `uv run pytest -q` : 10 passed（PR1時点）

**Risks / Notes**
- `evaluate/tune/simulate/export` は未実装のまま（`VeldraNotImplementedError`）。

**Open Questions**
- [ ] `evaluate(...)` 実装を次PRで入れるか（Artifact入力優先か）
- [ ] binary校正（OOF）の着手順をどこに置くか

### 2026-02-09 (Session/PR: phase2-pr2-tests-docs)
**Context**
- 背景 / 依頼 / 目的：
  - 回帰 `fit/predict` 実装に対するテスト補強と設計文書更新を別PRで行う。

**Plan**
- 回帰学習スモーク・roundtrip・再現性・ローダー形式のテストを追加
- DESIGN_BLUEPRINTのPhase 2実装状態とOpen Questionsを更新
- HISTORYへPR1/PR2の意思決定ログを記録

**Changes**
- テスト変更：
  - `tests/test_regression_fit_smoke.py` を追加
  - `tests/test_regression_predict_roundtrip.py` を追加
  - `tests/test_data_loader_formats.py` を追加
  - `tests/test_fit_reproducibility.py` を追加
- ドキュメント変更：
  - `DESIGN_BLUEPRINT.md` に Phase 2（回帰fit/predict）の実装済み仕様を追記
  - Open Questions を次優先課題に更新
  - `HISTORY.md` に PR1/PR2 のログを追記

**Decisions**
- Decision: confirmed
  - 内容：
    - PR2では機能拡張ではなく、テストとドキュメントの整備に限定する。
  - 理由：
    - PR1との差分責務を分離し、レビュー容易性とリスク分離を高めるため。
  - 影響範囲：
    - テスト / ドキュメント

**Results**
- 動作確認結果：
  - `uv run ruff check .` : All checks passed
  - `uv run pytest -q` : 15 passed

**Risks / Notes**
- PR2はPR1の機能実装に依存するため、PRベースは一時的にPR1ブランチとする。

**Open Questions**
- [ ] PR1マージ後にPR2のbaseを `main` へ切り替えるタイミング
