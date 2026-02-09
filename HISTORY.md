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

### 2026-02-09 (Session/PR: phase3-pr3-evaluate-regression-artifact)
**Context**
- 背景 / 依頼 / 目的：
  - Phase 3として `evaluate(...)` を回帰タスクで実装し、再評価APIを有効化する。

**Plan**
- `evaluate(artifact, data)` の回帰実装
- config入力経路は未実装維持
- 入力バリデーションと構造化ログを追加

**Changes**
- 実装変更：
  - `src/veldra/api/runner.py` の `evaluate` を実装
  - 入力契約（Artifact限定、DataFrame+target必須、空データ禁止）を追加
  - 指標計算（rmse/mae/r2）と `EvalResult.metadata` を追加
  - `evaluate completed` の構造化ログを追加

**Decisions**
- Decision: confirmed
  - 内容：
    - `evaluate` はPhase 3で Artifact入力限定にする。
  - 理由：
    - 既存の回帰fit/predict資産を活かし、仕様の分岐を最小化するため。
  - 影響範囲：
    - API / テスト
- Decision: confirmed
  - 内容：
    - `data` は target列を含む DataFrame 契約で固定する。
  - 理由：
    - 利用側I/Oを単純化し、誤入力時の検知を明確にするため。
  - 影響範囲：
    - API / 利用者入力契約

**Results**
- 動作確認結果：
  - `uv run ruff check .` : All checks passed
  - `uv run pytest -q` : 10 passed（PR3時点）

**Risks / Notes**
- 非回帰タスクの `evaluate` は未実装（`VeldraNotImplementedError`）。

**Open Questions**
- [ ] `evaluate(config, data)` を次フェーズで入れるか
- [ ] binary評価経路を `evaluate` にいつ統合するか

### 2026-02-09 (Session/PR: phase3-pr4-tests-docs)
**Context**
- 背景 / 依頼 / 目的：
  - Phase 3の `evaluate` 実装に対するテスト補強と設計文書更新を分離PRで実施する。

**Plan**
- `tests/test_evaluate_regression.py` を追加
- `tests/test_api_surface.py` の期待値を更新
- DESIGN/HISTORYのPhase 3記述を更新

**Changes**
- テスト変更：
  - `tests/test_evaluate_regression.py` を追加
  - `tests/test_api_surface.py` を更新（evaluate正常系を追加）
- ドキュメント変更：
  - `DESIGN_BLUEPRINT.md` にPhase 2/Phase 3 実装済みセクションを追記
  - Open Questionsの優先順位を更新
  - `HISTORY.md` にPR3/PR4ログを追記

**Decisions**
- Decision: confirmed
  - 内容：
    - PR4はテスト/文書更新のみとし、機能差分を混在させない。
  - 理由：
    - レビュー容易性と差分の責務分離を維持するため。
  - 影響範囲：
    - テスト / ドキュメント

**Results**
- 動作確認結果：
  - `uv run ruff check .` : All checks passed
  - `uv run pytest -q` : 13 passed

**Risks / Notes**
- PR4はPR3依存のため、PR作成時のbaseは `feat/evaluate-regression-artifact` とする。

**Open Questions**
- [ ] PR3マージ後のPR4 base切替タイミング
