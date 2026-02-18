# HISTORY.md
（AIエージェント作業履歴・意思決定ログ / 再編版）

- 詳細原本: `HISTORY_ARCHIVE.md`（2026-02-14退避、参照専用）
- 本書の目的: 長期運用しやすい要約履歴を時系列で維持する

## ルール
- 1エントリ = 1作業単位（会話/セッション/PR）
- 日付・時刻の基準は日本時間（JST, UTC+09:00）とする
- 同一フェーズ（`phaseN` / `phaseN.M`）の `Session planning` と `Session/PR` は1エントリに統合する
- 記載順は日付の古い順（昇順）。同日内はフェーズ番号昇順、その後に非フェーズを旧出現順で記載する
- 仕様決定は `Decision` で明示し、`provisional`（暫定）または `confirmed`（確定）だけを使う
- 1エントリの上限は、背景2項目・変更内容6項目・決定事項2件・検証結果3項目とする
- 重複ログ列挙は避け、代表コマンドと結果要約へ圧縮する

## テンプレート（追記時に利用）
### YYYY-MM-DD（作業/PR: XXXXX）
**背景**
- 背景 / 目的

**実施計画**
- 必要時のみ記載

**変更内容**
- 実装・設定・文書・テストの要点

**決定事項**
- `Decision` の使用値は `provisional（暫定）` または `confirmed（確定）`
  - 内容:
  - 理由:
  - 影響範囲:

**検証結果**
- 代表的な確認結果

**リスク・補足**
- 必要時のみ記載

**未決事項**
- 必要時のみ記載

---

## Log

### 2026-02-09（作業/PR: phase4-examples-california-demo）
**背景**
- MVP足場の上で、回帰ユースケースの再現可能なデモ導線を整備する必要があった。

**変更内容**
- `examples/` に California Housing デモの実行スクリプト群を追加。
- `examples/README.md` を追加し、準備・学習・評価の手順を明文化。
- `tests/test_examples_*` を拡充し、データ準備から評価までのスモーク契約を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: examples は「実行可能な最小ワークフロー」を優先し、Core契約のサンプルとして維持する。
  - 理由: 新規利用者の導入コストと検証コストを同時に下げるため。
  - 影響範囲: examples / docs / テスト運用

**検証結果**
- 代表コマンド: `uv run pytest -q`（examples系テストを含む）を通過。

### 2026-02-09（作業/PR: phase5-binary-fit-predict-evaluate-oof-calibration）
**背景**
- binaryタスクの本実装と、OOF確率校正の契約を確立する必要があった。
- `phase5-binary-examples` の追補を本エントリに統合した。

**変更内容**
- `src/veldra/modeling/binary.py` を追加し、学習・予測・評価の実装を導入。
- `src/veldra/api/runner.py` / `src/veldra/api/artifact.py` / `src/veldra/artifact/store.py` を更新し、binaryの永続化と推論契約を接続。
- `src/veldra/config/models.py` を更新し、校正・閾値最適化関連のバリデーションを追加。
- `tests/test_binary_*` と `tests/test_api_surface.py` を更新し、OOF校正・roundtrip・評価指標を検証。
- examples の binary 手順を追加して再現導線を補完。

**決定事項**
- Decision: confirmed（確定）
  - 内容: binary確率校正はOOF経由で実施し、リークを避ける設計を標準化する。
  - 理由: 評価値の過大推定を防ぎ、再現性を担保するため。
  - 影響範囲: modeling / Artifact / metrics

**検証結果**
- `tests/test_binary_oof_calibration.py` を含む binary 契約テスト群を通過。

### 2026-02-09（作業/PR: bootstrap-mvp-scaffold）
**背景**
- 初期MVPとして、RunConfig駆動の全体骨格を早期に固定する必要があった。

**変更内容**
- `src/veldra/` の基本モジュール骨格とAPI入口を整備。
- RunConfig/Artifact を中心に、task実行の最小導線を実装。
- 基本スモークテストと初期ドキュメントを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Coreは外部UI層から独立させ、共通入口を RunConfig に統一する。
  - 理由: CLI/GUI/API で同一契約を維持するため。
  - 影響範囲: アーキテクチャ全体

**検証結果**
- 初期スモークテストで API 主要導線の動作を確認。

### 2026-02-10（作業/PR: phase6-multiclass-fit-predict-evaluate-examples）
**背景**
- multiclass タスクを regression/binary と同等の契約レベルへ引き上げる必要があった。
- `Session planning` と `Session/PR` を統合した。

**変更内容**
- `src/veldra/modeling/multiclass.py` を追加し、CV学習・推論・評価を実装。
- `src/veldra/api/runner.py` / `src/veldra/api/artifact.py` を更新し、multiclass導線を接続。
- multiclass 用 examples を追加し、README へ実行手順を追記。
- `tests/test_multiclass_*` と examples系テストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: multiclass の予測契約を `label_pred` + `proba_<class>` で統一する。
  - 理由: 下流分析の一貫性を保つため。
  - 影響範囲: API契約 / Artifact

**検証結果**
- multiclass の fit/predict/evaluate/roundtrip テストが通過。

### 2026-02-10（作業/PR: phase7-binary-threshold-optimization-optin）
**背景**
- binary運用で閾値最適化の要望があり、互換性を壊さず opt-in で導入する必要があった。

**変更内容**
- binary設定に閾値最適化オプションを追加。
- 学習・評価・Artifact保存に最適閾値の反映経路を追加。
- examples/README と関連テストを更新。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 閾値最適化は既定OFFの opt-in で提供する。
  - 理由: 既存運用の後方互換を維持するため。
  - 影響範囲: Config / binary評価

**検証結果**
- binary閾値最適化テスト（`tests/test_binary_threshold_optimization.py`）を通過。

### 2026-02-10（作業/PR: phase7.1-doc-closure-and-phase8-tune-mvp）
**背景**
- ドキュメント上の機能差分を解消しつつ、tune機能のMVP導入を進める必要があった。

**変更内容**
- tune機能の基本導線を整理し、公開文書との齟齬を解消。
- README / DESIGN_BLUEPRINT / テスト構成を実装状態へ同期。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 実装状態と文書を同一リリースで同期し、差分を残さない。
  - 理由: 誤用防止と運用信頼性確保のため。
  - 影響範囲: docs / 開発運用

**検証結果**
- 文書契約と主要スモークが通過。

### 2026-02-10（作業/PR: phase8.1-tune-expansion-implementation）
**背景**
- tuneを task横断で実用化するため、探索・保存・再開まわりを拡張する必要があった。
- `phase8.1-tune-expansion` の planning を統合した。

**変更内容**
- `src/veldra/modeling/tuning.py` を追加し、Optunaベースの探索エンジンを導入。
- `src/veldra/api/runner.py` を更新し、`tune` 実行導線と結果契約を拡張。
- `optuna` 依存と `uv.lock` を更新。
- `tests/test_tune_*` と `tests/test_runner_additional.py` を拡充。
- `README.md` / `DESIGN_BLUEPRINT.md` を更新。

**決定事項**
- Decision: confirmed（確定）
  - 内容: tune は resume 可能な永続化前提で提供する。
  - 理由: 長時間探索の運用性を担保するため。
  - 影響範囲: tuning / Artifact / 実行運用

**検証結果**
- `uv run ruff check .` と `tests/test_tune_*` を通過。

### 2026-02-10（作業/PR: phase9-frontier-fit-predict-evaluate-mvp）
**背景**
- frontier タスクを共通API体系へ統合する必要があった。
- planning/実装ログを統合した。

**変更内容**
- frontier 学習・推論・評価導線を実装。
- `fit/predict/evaluate` の task分岐に frontier を追加。
- frontier 契約テストと examples を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: frontier 予測契約は `frontier_pred` を基本とし、target存在時に `u_hat` を補助指標として返す。
  - 理由: 最小契約と分析拡張の両立のため。
  - 影響範囲: frontier API / 評価

**検証結果**
- frontier の fit/predict/evaluate スモークを通過。

### 2026-02-10（作業/PR: phase10-simulate-mvp-scenario-dsl）
**背景**
- 学習後の政策シミュレーションを、Artifact中心の再現可能フローで提供する必要があった。
- `phase10-simulate-mvp` planning を統合した。

**変更内容**
- `simulate` API を実装し、Scenario DSL（`set/add/mul/clip`）を導入。
- `src/veldra/simulate/engine.py` を追加し、シナリオ正規化と適用ロジックを実装。
- simulate examples と契約テストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: DSLはMVPで最小演算（`set/add/mul/clip`）に限定する。
  - 理由: 安全性と再現性を優先し、拡張余地を残すため。
  - 影響範囲: simulate API / Scenario契約

**検証結果**
- simulate スモークと DSL 妥当性テストを通過。

### 2026-02-10（作業/PR: phase11-export-mvp-python-onnx-optional）
**背景**
- 学習成果物の配布可能性を高めるため、export のMVPが必要だった。
- planning/実装を統合した。

**変更内容**
- `export(format="python")` を実装。
- `export(format="onnx")` を optional dependency 前提で導入。
- export関連テストと README を更新。

**決定事項**
- Decision: confirmed（確定）
  - 内容: ONNXは optional とし、依存未導入時は明示エラーで案内する。
  - 理由: 最小依存環境を壊さず拡張性を確保するため。
  - 影響範囲: export / 依存管理

**検証結果**
- python export と onnx optional 契約テストを通過。

### 2026-02-10（作業/PR: phase12-tune-frontier-mvp）
**背景**
- frontier向け tuning を運用可能にするため、objective設計を含む統合が必要だった。
- planning/実装を統合した。

**変更内容**
- frontier tuning objective を実装し、設定バリデーションを追加。
- tuning結果の保存・再利用経路を整備。
- frontier tuning テストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: frontier tuning の既定 objective は既存互換を維持する。
  - 理由: 既存ワークフローの安定性を優先するため。
  - 影響範囲: frontier tune / Config

**検証結果**
- frontier tuning スモークテストを通過。

### 2026-02-10（作業/PR: phase13-frontier-onnx-export-mvp）
**背景**
- frontier モデルの ONNX 配布ニーズに対応する必要があった。
- planning/実装を統合した。

**変更内容**
- frontier で ONNX export を可能にする変換経路を追加。
- export 検証テストとドキュメントを更新。

**決定事項**
- Decision: confirmed（確定）
  - 内容: frontier ONNX は既存 export 契約に追加し、公開シグネチャは変更しない。
  - 理由: Stable API 方針を維持するため。
  - 影響範囲: export / frontier

**検証結果**
- frontier ONNX export のスモークを通過。

### 2026-02-10（作業/PR: phase14-export-validation-tooling-mvp）
**背景**
- export後の利用時に妥当性検証を自動化する要望があった。
- planning/実装を統合した。

**変更内容**
- export validation レポート生成を追加。
- 主要契約に対する検証補助ユーティリティを整備。
- 関連テストとREADMEを更新。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 配布成果物には検証情報を同梱し、再現性監査を容易にする。
  - 理由: 配布後トラブルの切り分けを高速化するため。
  - 影響範囲: export / 運用性

**検証結果**
- export validation 契約テストを通過。

### 2026-02-10（作業/PR: phase15-onnx-quantization-mvp）
**背景**
- ONNX推論の軽量化要件に対応する必要があった。
- planning/実装を統合した。

**変更内容**
- ONNX dynamic quantization を opt-in で追加。
- quantization エラーハンドリングとテストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 量子化は既定OFFとし、明示指定時のみ適用する。
  - 理由: 予期しない精度変化リスクを避けるため。
  - 影響範囲: export / ONNX

**検証結果**
- `tests/test_export_onnx_optimization*.py` を通過。

### 2026-02-10（作業/PR: phase16-timeseries-split-advanced-mvp）
**背景**
- 時系列分割の運用要件（リーク防止、期間制約）を満たす拡張が必要だった。
- planning/実装を統合した。

**変更内容**
- timeseries splitter の高度設定（期間制約・fold制御）を実装。
- runconfig バリデーションと split テストを拡充。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 時系列分割は順序保全を最優先し、無効設定は明示エラーとする。
  - 理由: リーク防止と検証妥当性確保のため。
  - 影響範囲: splitter / Config

**検証結果**
- timeseries split の追加テストを通過。

**リスク・補足**
- `History hygiene` で記録されていた「planning tail 重複」は本再編で本エントリへ吸収した。

### 2026-02-10（作業/PR: phase17-frontier-tune-coverage-objective-mvp）
**背景**
- frontier tuning で coverage 目標を明示的に最適化したい要望があった。
- planning/実装を統合した。

**変更内容**
- `pinball_coverage_penalty` objective を追加（opt-in）。
- `coverage_target` / `coverage_tolerance` / `penalty_weight` の設定経路を追加。
- frontier tuning 追加テストを実装。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 既定 objective は `pinball` 維持、coverage重視は opt-in で提供する。
  - 理由: 後方互換と運用調整性を両立するため。
  - 影響範囲: frontier tune / Config

**検証結果**
- `uv run ruff check .` と frontier tuning 関連 pytest を通過。

**リスク・補足**
- `History hygiene` の superseded note は本エントリに吸収済み。

### 2026-02-10（作業/PR: phase18-evaluate-config-input-mvp）
**背景**
- 共通入口原則に沿って、`evaluate(config, data)` を実装する必要があった。
- planning/実装を統合した。

**変更内容**
- `src/veldra/api/runner.py` に config入力分岐を追加し、ephemeral学習評価を実装。
- 既存の `evaluate(artifact, data)` 振る舞いは維持。
- `tests/test_evaluate_config_path.py` と validation テストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: configモード評価は永続Artifactを作らない一時実行とする。
  - 理由: API互換を維持しつつ運用導線を拡張するため。
  - 影響範囲: evaluate API / 実行メタデータ

**検証結果**
- evaluate(config, data) 契約テストが通過。

### 2026-02-10（作業/PR: phase19-causal-dr-mvp-att-calibrated）
**背景**
- 因果推論（DR）のMVP導線を共通APIへ統合する必要があった。
- planning/実装を統合した。

**変更内容**
- `estimate_dr` 導線を実装し、ATT既定の因果推論フローを追加。
- propensity 校正（platt）と診断メトリクス返却を整備。
- DR系テスト群を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: DRの既定 estimand は ATT、propensity 校正は platt を採用する。
  - 理由: 実務利用の説明性と安定運用を優先するため。
  - 影響範囲: causal API / metrics

**検証結果**
- DRスモークと契約テスト（`tests/test_dr_*`）を通過。

### 2026-02-11（作業/PR: phase19.1-lalonde-dr-analysis-notebook）
**背景**
- DR runtime の利用例として、Lalonde 分析ノートブックを整備する必要があった。
- planning/実装を統合した。

**変更内容**
- `notebooks/lalonde_dr_analysis_workflow.ipynb` を追加。
- URL取り込み + ローカルキャッシュ再実行の導線を整備。
- notebook 契約テストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: notebook でも ATT/platt の既定を明示記載する。
  - 理由: 分析前提の透明性を確保するため。
  - 影響範囲: notebooks / 再現性

**検証結果**
- notebook 構造テスト群（15件）が通過。

**リスク・補足**
- `History hygiene` の phase19.1 superseded note は本エントリに吸収済み。

### 2026-02-11（作業/PR: phase20-drdid-and-causal-tune-mvp）
**背景**
- DR-DiD 本体実装と causal tuning objective を同時に整備する必要があった。
- planning/実装を統合した。

**変更内容**
- `causal.method="dr_did"` の panel / repeated cross section 経路を実装。
- causal tune objective を DR-DiD 含めて接続。
- `tests/test_drdid_*` を追加・更新。

**決定事項**
- Decision: confirmed（確定）
  - 内容: DR-DiD は2時点MVPを正式サポートし、診断メトリクスを共通返却する。
  - 理由: 因果比較の再現性と検証容易性を確保するため。
  - 影響範囲: causal runtime / tuning

**検証結果**
- DR-DiD スモークと validation テストを通過。

### 2026-02-11（作業/PR: phase20.1-lalonde-drdid-analysis-notebook）
**背景**
- DR-DiD runtime の実践例として、Lalonde向け notebook が必要だった。
- planning/実装を統合した。

**変更内容**
- `notebooks/lalonde_drdid_analysis_workflow.ipynb` を追加。
- panel変換キャッシュと比較診断（Naive/IPW/DR/DR-DiD）を実装。
- notebook 契約テストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: DR-DiD notebook では panel設計（`re75`→`re78`）を標準例として採用する。
  - 理由: 目的変数解釈と再現導線を揃えるため。
  - 影響範囲: notebooks / causal分析

**検証結果**
- notebook 構造・パス検証が通過。

### 2026-02-11（作業/PR: phase20.2-design-blueprint-reorg-and-gap-audit）
**背景**
- 実装進行に合わせて、設計文書の整合性を再監査する必要があった。

**変更内容**
- `DESIGN_BLUEPRINT.md` を再編し、Current State と未実装ギャップを明確化。
- HISTORY運用と設計文書の責務分離を明記。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 詳細時系列は `HISTORY.md`、設計の現況要約は `DESIGN_BLUEPRINT.md` を正本とする。
  - 理由: ドキュメント役割の混線を防ぐため。
  - 影響範囲: docs運用

**検証結果**
- 設計文書の章構成と用語整合を確認。

### 2026-02-11（作業/PR: phase21-dash-gui-mvp）
**背景**
- GUIアダプタをMVPとして導入し、RunConfig共通入口をGUIでも成立させる必要があった。
- planning/実装を統合した。

**変更内容**
- `src/veldra/gui/` に DashベースGUI（Config/Run/Artifacts）を実装。
- GUI経由で `veldra.api.runner` を呼ぶアダプタ構成を確立。
- GUIスモークテストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: GUI層はCoreロジックを持たず、API呼び出し専用アダプタとする。
  - 理由: 設計原則（Core独立）を維持するため。
  - 影響範囲: GUI / アーキテクチャ

**検証結果**
- GUI主要フローのスモークを通過。

### 2026-02-12（作業/PR: phase22-config-migrate-mvp）
**背景**
- RunConfigバージョン運用のため、migrationユーティリティが必要だった。
- planning/実装を統合した。

**変更内容**
- Python API: `migrate_run_config_payload` / `migrate_run_config_file` を追加。
- CLI: `veldra config migrate` を追加。
- migrationのバリデーション/エラーハンドリングを実装。
- `tests/test_config_migrate_*` を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: migration は上書き禁止・未知キー明示エラーを基本方針とする。
  - 理由: 設定破損リスクを最小化するため。
  - 影響範囲: Config運用 / CLI

**検証結果**
- config migrate のCLI/APIテストを通過。

### 2026-02-12（作業/PR: phase23-drdid-binary-riskdiff-mvp）
**背景**
- binary outcome をDR-DiDへ適用する要件に対応する必要があった。
- planning/実装を統合した。

**変更内容**
- `task.type="binary"` の DR-DiD を実装し、Risk Difference ATT 解釈を追加。
- `CausalResult.metadata` に binary関連メタデータを追加。
- `tests/test_drdid_binary_*` を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: DR-DiD binary は estimand を ATT に限定して提供する。
  - 理由: 因果解釈の一貫性を保つため。
  - 影響範囲: causal binary契約

**検証結果**
- DR-DiD binary スモークと validation を通過。

### 2026-02-12（作業/PR: phase24-causal-tune-balance-priority）
**背景**
- 因果チューニングをSE中心から balance優先へ拡張する必要があった。
- planning/実装を統合した。

**変更内容**
- `dr_balance_priority` / `drdid_balance_priority` objective を追加。
- `tuning.causal_balance_threshold` を追加。
- trial attributes と診断メトリクス返却を強化。

**決定事項**
- Decision: confirmed（確定）
  - 内容: causal tune の既定 objective は balance-priority 系へ更新する。
  - 理由: 実運用での共変量バランス重視方針に合わせるため。
  - 影響範囲: tuning / causal

**検証結果**
- causal tuning 系テストを通過。

### 2026-02-12（作業/PR: phase25-gui-async-jobs-migrate-workflow-mvp）
**背景**
- GUIの運用性を高めるため、非同期ジョブ基盤とmigrate統合を先行導入する必要があった。
- planning/実装を統合した。

**変更内容**
- GUIジョブキュー（SQLite永続 + single worker）を実装。
- queued/running の best-effort cancel を導入。
- `/config` に config migrate（preview/diff/apply）を統合。
- GUIジョブストア/ワーカーのテストを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: GUIジョブ実行は single worker 既定で後方互換を優先する。
  - 理由: 実装複雑度を抑えて運用安定性を確保するため。
  - 影響範囲: GUI運用 / job queue

**検証結果**
- GUIジョブ関連スモークを通過。

### 2026-02-14（作業/PR: phase25-gui-operability-completion）
**背景**
- phase25 の運用強化を完了させ、GUI主要導線を安定運用状態へ引き上げる必要があった。

**変更内容**
- Data→Config→Run→Results の callback 互換性を補強。
- uploadデータ例外をユーザー向けエラーへ変換。
- 旧テスト契約との後方互換フォールバックを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: GUI callback 契約は既存呼び出し互換を維持したまま堅牢化する。
  - 理由: UI修正時の回帰影響を最小化するため。
  - 影響範囲: GUI callback / 互換性

**検証結果**
- `pytest -q` で `405 passed` を確認（当時記録）。

### 2026-02-14（作業/PR: gui-scroll-layout-stabilization-batch）
**背景**
- Data画面でスクロールジャンプ、空白押し下げ、横スクロール消失が連続発生した。
- 統合元:
  - `data-layout-root-cause-fix-row-col-height-and-overflow-scope`
  - `remove-scrollto-side-effects-and-restore-data-preview-scroll`
  - `rollback-internal-scroll-shell-to-restore-usability`
  - `app-shell-internal-scroll-refactor`
  - `data-inspection-single-trigger-and-animation-isolation`
  - `gui-data-preview-datatable-removal-for-scroll-stability`
  - `gui-narrow-width-scroll-jump-hardening`
  - `gui-data-preview-scroll-jump-regression-fix`
  - `gui-config-quick-run-and-preview-stability`
  - `gui-inline-preview-readiness-and-jst-labels`
  - `gui-preview-modal-and-artifact-autoselect-fallback`
  - `gui-ux-autoselect-and-scroll-stabilization`

**変更内容**
- `src/veldra/gui/app.py` で Data系 clientside `window.scrollTo` を撤去し、不要なスクロール補正状態を削除。
- `src/veldra/gui/assets/style.css` のグローバル `overflow` 上書きを廃止し、局所スコープへ限定。
- Data preview の横/縦スクロールと最大高を局所設定へ統一。
- Data領域内アニメーションとアンカー挙動の適用範囲を明確化。
- 画面遷移時の auto-select / fallback を整理し、初期表示のブレを抑制。

**決定事項**
- Decision: confirmed（確定）
  - 内容: スクロール制御はブラウザ標準を基本とし、JS補正は原則禁止する。
  - 理由: レイアウト再計算との競合による副作用を避けるため。
  - 影響範囲: GUI / UX

**検証結果**
- 代表コマンド: `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
- 結果: `73 passed, 1 warning`（当時記録）。

### 2026-02-14（作業/PR: gui-run-results-compatibility-batch）
**背景**
- Run/Results/Data連携で互換崩れが連続し、GUI運用の再現性を損なっていた。
- 統合元:
  - `gui-run-default-config-bootstrap-fix`
  - `gui-results-artifact-compat-fix`
  - `gui-data-scroll-and-run-auto-navigation-fix`
  - `gui-test-regression-compat-restoration`
  - `gui-data-page-upload-state-and-scroll-fix`
  - `gui-config-to-run-state-sync-fix`
  - `gui-results-feature-importance-fallback`
  - `gui-results-metrics-mean-plot-fix`
  - `gui-results-runconfig-json-serialization-fix`

**変更内容**
- Config→Run 状態同期の崩れを修正し、デフォルト設定生成の安定性を改善。
- Artifact互換フォールバックを追加し、Results表示失敗を低減。
- Run完了時の自動遷移・Data状態保持を調整。
- RunConfig JSON 化の安全化（直列化エラー回避）を実装。
- feature importance / metrics 描画での欠損時フォールバックを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Results 画面は「失敗よりフォールバック優先」で可視化継続を選ぶ。
  - 理由: 操作中断を減らしデバッグ情報を残すため。
  - 影響範囲: GUI results / run flow

**検証結果**
- GUI回帰テスト群（services/app callbacks/results）を通過。

### 2026-02-14（作業/PR: gui-lazy-import-and-test-separation）
**背景**
- GUI import 時のメモリ消費が大きく、coverage 実行時にOOMリスクがあった。
- 統合元:
  - `gui-memory-investigation-and-lazy-import`
  - `gui-lazy-import-reapply-and-test-separation`

**変更内容**
- `src/veldra/gui/app.py` / `src/veldra/gui/services.py` の重量依存 import を遅延解決化。
- `tests/conftest.py` と `pyproject.toml` に `gui` marker を追加し、テスト分離運用を導入。
- `src/veldra/gui/app.py` の `_to_jsonable` に循環参照/Mock安全化を追加。
- GUI関連テストのモック実装を軽量化し、メモリ負荷を低減。

**決定事項**
- Decision: confirmed（確定）
  - 内容: GUIテストは `gui` / `not gui` に分離し、2段階実行を標準手順とする。
  - 理由: 低メモリ環境での実行可能性を確保するため。
  - 影響範囲: テスト運用 / GUI性能

**検証結果**
- marker分離収集: `gui=70`, `not gui=335`（当時記録）。
- GUI関連追加テスト（jsonable安全化含む）を通過。

**リスク・補足**
- 当時環境では `uv run` の外部解決制約があり、一部検証は `.venv/bin/python -m pytest` で実施した。

### 2026-02-14（作業/PR: ruff-repo-wide-cleanup）
**背景**
- `uv run ruff check .` が全体で失敗し、品質ゲートの信頼性が低下していた。

**変更内容**
- `ruff check --fix` / `ruff format` を適用し、自動修正可能な違反を解消。
- `src/veldra/gui/app.py` などで `E501/E722/E402/F821` を手動修正。
- GUI marker分離運用に合わせて pytest 実行手順を整理。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Ruffルールは緩和せず、コード修正のみで通過させる。
  - 理由: 品質ゲートの基準を下げないため。
  - 影響範囲: 開発品質 / GUI adapter / tests

**検証結果**
- `uv run ruff check .`: passed
- `uv run pytest -q -m "not gui"`: passed
- `uv run pytest -q -m "gui"`: passed

### 2026-02-15（作業/PR: api-doc-contract-and-algorithm-docs）
**背景**
- 公開APIのdocstring契約が不足し、利用者向け参照性が低かった。
- 注記: このエントリ日付 `2026-02-15` は再編実施日 `2026-02-14` 時点では未来日記録。日付は原記録を維持した。

**変更内容**
- `src/veldra/api/runner.py` / `types.py` / `artifact.py` / `logging.py` の公開docstringをNumPy形式で統一。
- `src/veldra/modeling/*`、`src/veldra/causal/*`、`src/veldra/simulate/engine.py` の主要関数へ `Notes` を追記。
- `tests/test_doc_contract.py` を追加し、公開API doc 契約の品質ゲートを導入。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 公開API docstring は NumPy形式を標準とする。
  - 理由: 契約の可読性・機械検証性を高めるため。
  - 影響範囲: APIドキュメント / テスト品質

**検証結果**
- `uv run ruff check .` と `uv run pytest -q tests/test_doc_contract.py` を通過（当時記録）。

### 2026-02-15（作業/PR: readme-why-runbook-runconfig-reference）
**背景**
- README に Why・運用導線・RunConfig完全参照が不足していた。
- 注記: このエントリ日付 `2026-02-15` は再編実施日 `2026-02-14` 時点では未来日記録。日付は原記録を維持した。

**変更内容**
- `README.md` に `Why VeldraML?` と `From Quick Start to Production` を追加。
- `RunConfig Reference (Complete)` をREADMEへ埋め込み、生成ブロックで管理。
- `scripts/generate_runconfig_reference.py` を追加し、`--write/--check` を提供。
- `tests/test_readme_runconfig_reference.py` を追加し、README参照整合を自動検証。

**決定事項**
- Decision: confirmed（確定）
  - 内容: RunConfig参照はREADME内埋め込みを正本とし、生成+検証で同期する。
  - 理由: 文書と実装の同期漏れを防ぐため。
  - 影響範囲: README運用 / docs品質

**検証結果**
- `uv run pytest -q tests/test_readme_runconfig_reference.py` を通過（当時記録）。

### 2026-02-15（作業/PR: phase25.5-test-improvement-planning-and-bootstrap）
**背景**
- テストスイートで、データ生成重複・task間非対称・private依存テストが同時に残っていた。
- Phase25.5の実装着手前に、設計と意思決定を文書で固定する必要があった。

**変更内容**
- `DESIGN_BLUEPRINT.md` の `Phase25.5` を、DRY/対称性/API化の3本柱で全置換。
- Wave方式（Wave1-3）で42ファイルのデータファクトリー移行方針を明文化。
- regression契約テスト4本の新設方針と、`split/cv.py`・`causal/diagnostics.py` の公開化方針を明記。
- 検証コマンドと完了条件（Stable API非破壊を含む）を明記。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase25.5では `iter_cv_splits` と因果診断メトリクスを公開ユーティリティへ抽出する。
  - 理由: private実装依存を減らし、重複ロジックの保守点を集約するため。
  - 影響範囲: split / causal / tests / docs
- Decision: provisional（暫定）
  - 内容: テストデータ生成ロジックは `tests/conftest.py` の共通ファクトリーへWave移行する。
  - 理由: 重複削減とseed再現性の統一を同時に満たすため。
  - 影響範囲: tests全体

**検証結果**
- 文書更新時点のため、コード検証は未実施（実装フェーズで実施予定）。

### 2026-02-15（作業/PR: gui-test-hardening-app-services-phase25.5）
**背景**
- GUIテストは本数は十分だが、callback_map依存と重複ケースにより保守性が低かった。
- `app.py` / `services.py` の未カバー分岐を埋め、品質ゲートを明確化する必要があった。

**変更内容**
- `tests/test_gui_app_coverage.py` / `tests/test_gui_app_coverage_2.py` を廃止し、
  `tests/test_gui_app_pure_callbacks.py` / `tests/test_gui_app_job_flow.py` /
  `tests/test_gui_app_results_flow.py` に再編。
- `tests/test_gui_app_callbacks_internal.py` を配線最小確認へ整理し、未実装 `pass` を解消。
- `tests/test_gui_services_core.py` を追加し、lazy import, default path, エラー正規化,
  `run_action` 必須入力分岐を拡充。
- `README.md` に GUI coverage gate（`app.py` / `services.py` 90%目標）を追記。

**決定事項**
- Decision: confirmed（確定）
  - 内容: callback_map依存テストは配線確認の最小範囲に限定し、ロジック検証は `_cb_*` 直接呼び出しを標準とする。
  - 理由: Dash内部仕様変化による脆さを減らすため。
  - 影響範囲: gui tests / 開発運用

**検証結果**
- GUIテスト群・カバレッジ確認を実施し、`app.py` / `services.py` のカバレッジ改善を確認。

### 2026-02-15（作業/PR: phase25.6-gui-ux-polish-css-layout-only）
**背景**
- ダークテーマ上の補助テキスト可読性、データプレビューの視認性、Config導線の一貫性に改善余地があった。
- 機能追加を避け、既存契約を維持したまま CSS/レイアウトのみでUXを改善する必要があった。

**変更内容**
- `src/veldra/gui/assets/style.css` に `text-muted-readable`、データプレビュー sticky header fallback、active step glow を追加。
- `src/veldra/gui/pages/data_page.py` で補助テキストclassを `text-muted-readable` へ置換し、プレビュー `thead th` に sticky style を追加。
- `src/veldra/gui/pages/config_page.py` で `cfg-container-id-cols` を Split Strategy 配下へ移設し、進行ボタン色を `primary` へ統一、export preset初期値を `artifacts` に固定。
- `src/veldra/gui/app.py` の `_stepper_bar` で完了済みコネクタの色を `var(--success)` に変更。
- `tests/test_gui_pages_logic.py` / `tests/test_gui_pages_and_init.py` / `tests/test_gui_app_callbacks_config.py` / `tests/test_gui_app_additional_branches.py` を更新し、新UX契約を検証。
- `DESIGN_BLUEPRINT.md` に `12.6 Phase25.6` セクションを追記。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Phase25.6は CSS/HTML レイアウト変更に限定し、GUI callback ID契約と公開API契約を維持する。
  - 理由: UX改善を最小リスクで実施し、回帰範囲を限定するため。
  - 影響範囲: GUI presentation / tests / docs

**検証結果**
- `uv run pytest tests/test_gui_app_callbacks_internal.py tests/test_gui_app_pure_callbacks.py tests/test_gui_app_job_flow.py -v` を通過。
- `uv run pytest tests/test_gui_pages_logic.py tests/test_gui_pages_and_init.py tests/test_gui_app_callbacks_config.py tests/test_gui_app_additional_branches.py -v` を通過。
- `uv run pytest tests -x --tb=short` を通過。

### 2026-02-15（作業/PR: phase25.6-timeseries-time-column-warning-ui）
**背景**
- Time Series split では `split.time_col` が必須だが、GUI上で必須条件が明示されず入力漏れ余地が残っていた。

**変更内容**
- `src/veldra/gui/pages/config_page.py` の Time Column 直下に `cfg-timeseries-time-warning` を追加。
- `src/veldra/gui/app.py` に `_cb_timeseries_time_warning` を追加し、
  `split=timeseries` かつ `time_col` 未選択時のみ警告を表示する callback を配線。
- `tests/test_gui_app_additional_branches.py` に warning callback の分岐テストを追加。
- `tests/test_gui_pages_and_init.py` / `tests/test_gui_app_callbacks_internal.py` を更新し、
  レイアウト存在と callback wiring を検証。
- `DESIGN_BLUEPRINT.md` の Phase25.6成果物へ Time Column 必須警告を追記。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Time Column セレクタの表示条件は `Time Series` 選択時のみを維持し、未選択時は警告表示で補助する。
  - 理由: 既存UI導線を崩さず、必須入力漏れを事前に減らすため。
  - 影響範囲: GUI UX / gui tests / docs

**検証結果**
- `uv run pytest tests/test_gui_app_additional_branches.py tests/test_gui_pages_and_init.py tests/test_gui_app_callbacks_internal.py -v` を通過。
- `uv run pytest tests -x --tb=short` を通過。

### 2026-02-15（作業/PR: phase25.6-remove-id-columns-visual-ui）
**背景**
- Split Strategy 上で `Group Column` と `ID Columns (Optional - for Group K-Fold)` が併存し、導線が重複していた。

**変更内容**
- `src/veldra/gui/pages/config_page.py` から `ID Columns (Optional - for Group K-Fold)` の表示UIを削除。
- `cfg-data-id-cols` / `cfg-container-id-cols` は callback互換のため Data Settings 内の非表示要素として維持。
- `src/veldra/gui/app.py` の `cfg-container-id-cols.style` 表示制御callbackを削除。
- `tests/test_gui_pages_and_init.py` / `tests/test_gui_app_callbacks_config.py` を更新し、新しいUI契約へ合わせた。
- `DESIGN_BLUEPRINT.md` の Phase25.6成果物記述を「移動」から「表示削除」へ更新。

**決定事項**
- Decision: confirmed（確定）
  - 内容: ID ColumnsはGUI表示から外し、必要最小限の内部互換IDのみ非表示で残す。
  - 理由: UXの重複を減らしつつ、既存callback契約を壊さないため。
  - 影響範囲: GUI layout / callback wiring / tests / docs

**検証結果**
- `uv run pytest tests/test_gui_app_additional_branches.py tests/test_gui_pages_and_init.py tests/test_gui_app_callbacks_internal.py -v` を通過。
- `uv run pytest tests -x --tb=short` を通過。

### 2026-02-15（作業/PR: phase25.6-split-timeseries-visibility-callback-fix）
**背景**
- Split Strategy の Time Series ブロックが表示されず、Time Column セレクタに到達できない不具合があった。

**変更内容**
- `src/veldra/gui/app.py` に `cfg-container-group/cfg-container-timeseries` の表示切替 callback 配線を復元。
- `tests/test_gui_app_callbacks_internal.py` の必須callback配線検証に
  `cfg-container-timeseries.style` を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Split種別に応じた Group/TimeSeries パネル表示は `_cb_update_split_options` で一元制御する。
  - 理由: レイアウト条件分岐の欠落再発を防ぐため。
  - 影響範囲: GUI callback wiring / tests

**検証結果**
- `uv run pytest tests/test_gui_app_additional_branches.py tests/test_gui_pages_and_init.py tests/test_gui_app_callbacks_internal.py -v` を通過。
- `uv run pytest tests -x --tb=short` を通過。

### 2026-02-15（作業/PR: phase25.6-data-settings-categorical-hide-and-split-refresh）
**背景**
- Data Settings に `Categorical Columns (Optional override)` が残存し、簡素化方針と不一致だった。
- Split Strategy の Time Series 必須入力案内を、よりUI上で明確にする必要があった。

**変更内容**
- `src/veldra/gui/pages/config_page.py` で `Categorical Columns (Optional override)` 表示を除去し、
  `cfg-data-cat-cols` は互換のため非表示コンテナで維持。
- Time Column の placeholder を `Select required time column...` へ変更し、
  `Required for Time Series validation.` の常時補助文言を追加。
- `tests/test_gui_pages_and_init.py` を更新し、Categorical表示の非存在とTime Column UI契約を検証。
- `DESIGN_BLUEPRINT.md` の Phase25.6成果物へ Data Settings簡素化を追記。

**決定事項**
- Decision: confirmed（確定）
  - 内容: `cfg-data-cat-cols` の内部ID契約は維持しつつ、Data Settings の操作UIからは非表示にする。
  - 理由: callback互換を維持しながら利用者導線を簡素化するため。
  - 影響範囲: GUI layout / gui tests / docs

**検証結果**
- `uv run pytest tests/test_gui_app_additional_branches.py tests/test_gui_pages_and_init.py tests/test_gui_app_callbacks_internal.py -v` を通過。
- `uv run pytest tests -x --tb=short` を通過。

### 2026-02-16（作業/PR: phase25.7-lightgbm-enhancements-step1-8）
**背景**
- Phase25.7 設計を実装へ反映し、LightGBM学習契約（split/ES/weight/history）を RunConfig 駆動で統合する必要があった。
- 既存 Stable API と Artifact 後方互換を維持したまま、GUI と config migrate まで含めて完了させる必要があった。

**変更内容**
- `TrainConfig` に `num_boost_round` / `early_stopping_validation_fraction` / `auto_class_weight` / `class_weight` を追加し、関連バリデーションを実装。
- 全 task 学習器で `num_boost_round` を反映し、CV/最終モデルともに train 部分から ES 用 validation を分割する実装へ変更（OOF valid を ES 監視から分離）。
- binary/multiclass の class weight（auto/manual）を実装し、`split.type=kfold` の binary/multiclass は内部的に stratified を自動適用。
- causal DR cross-fit 分割を拡張し、`split.group_col` または panel の `unit_id_col` が利用可能な場合は GroupKFold、不可時は KFold へフォールバック。
- Artifact に `training_history`（`training_history.json`）を追加し、save/load の後方互換（旧 artifact は欠損許容）を維持。
- GUI builder を更新して `Num Boost Round` / class weight 入力を追加し、`train.num_boost_round` 出力へ切替。`config migrate` に `train.lgb_params.n_estimators -> train.num_boost_round` 変換を追加。

**決定事項**
- Decision: provisional（暫定）
  - 内容: causal cross-fit の group 分割は「利用可能なら GroupKFold、利用不可なら KFold」のフォールバック方針で実装する。
  - 理由: 設定必須化による互換破壊を避けつつ、group情報があるケースでは漏洩リスクを下げるため。
  - 影響範囲: causal/dr cross-fit / split 方針
- Decision: confirmed（確定）
  - 内容: Phase25.7 は Step1-8 を完遂し、学習履歴保存・GUI・migration までを同一フェーズで閉じる。
  - 理由: 設定/学習/配布導線を同時に整合させることで運用上の齟齬を防ぐため。
  - 影響範囲: config/models, modeling, artifact, gui, migrate, tests, docs

**検証結果**
- `uv run ruff check`（今回変更ファイル対象）を通過。
- 追加/更新テスト群（config train fields, class weight, auto split, ES split, training history, migrate, GUI, DR internal）を通過。
- `uv run pytest tests -x --tb=short` で **468 passed, 0 failed** を確認。

### 2026-02-16（作業/PR: phase25.8-lightgbm-params-topk-feature-weights）
**背景**
- Phase25.8 の設計案を実装へ落とし込み、LightGBMパラメータ拡張を RunConfig 駆動で統合する必要があった。
- binary tuning objective の一部（accuracy/f1/precision/recall）が学習出力 `metrics.mean` と不整合で失敗する既知ギャップを同時に解消する必要があった。

**変更内容**
- `src/veldra/config/models.py` に `auto_num_leaves / num_leaves_ratio / min_data_in_leaf_ratio / min_data_in_bin_ratio / feature_weights / top_k` を追加し、競合・範囲・task制約・`precision_at_k` objective 連動バリデーションを実装。
- `src/veldra/modeling/utils.py` に `resolve_auto_num_leaves / resolve_ratio_params / resolve_feature_weights` を追加し、`regression/binary/multiclass/frontier` の全学習器へ適用。
- `src/veldra/modeling/binary.py` に `precision_at_k`（feval）を追加し、`top_k` 指定時の ES を custom metric 優先化。`metrics.mean` に threshold系指標と `precision_at_{k}` を追加。
- `src/veldra/modeling/tuning.py` で `precision_at_k` objective を `precision_at_{top_k}` へ解決するロジックを追加。standard search space を `lambda_l1/lambda_l2/path_smooth/min_gain_to_split` まで拡張。
- `src/veldra/api/runner.py` の binary evaluate に `precision_at_{k}` 返却を追加（`top_k` 設定時のみ）。
- `src/veldra/gui/pages/config_page.py` / `src/veldra/gui/app.py` を拡張し、Phase25.8 GUI項目・YAMLマッピング・Top K 表示制御・binary tune objective 候補更新を実装。
- `scripts/generate_runconfig_reference.py` と `README.md` の RunConfig Reference を更新。
- テスト追加/更新:
  - 新規: `tests/test_lgb_param_resolution.py`, `tests/test_top_k_precision.py`
  - 更新: `tests/test_config_train_fields.py`, `tests/test_tuning_internal.py`, `tests/test_tune_objective_selection.py`, `tests/test_tune_validation.py`, `tests/test_binary_evaluate_metrics.py`, GUI関連テスト

**決定事項**
- Decision: provisional（暫定）
  - 内容: `train.top_k` 指定時は ES 監視を `precision_at_{k}` 優先で実行する（`metric=None` + `feval`）。
  - 理由: top-k最適化の運用意図と学習停止基準を一致させるため。
  - 影響範囲: binary training / tuning / training history
- Decision: provisional（暫定）
  - 内容: `train.feature_weights` の未知特徴量キーは無視せずエラーにする。
  - 理由: 設定typoの静かな混入を防ぎ、再現性を維持するため。
  - 影響範囲: modeling utils / 全task学習器 / config運用

**検証結果**
- `uv run ruff check .` を通過。
- 追加/更新テスト群（config/lgb resolver/top_k/tuning/evaluate/gui）を通過。
- `uv run pytest -q -m "not gui"`: **385 passed**
- `uv run pytest -q -m "gui"`: **100 passed**

### 2026-02-16（作業/PR: phase25.9-test-gap-closure-plan-lock）
**背景**
- Phase25.7/25.8 の LightGBM 強化は主要機能が実装済みだが、学習ループ適用・search space・objective override・Artifact整合性の一部が未検証だった。
- 実装前に、テスト主導で不足ギャップを閉じる方針を文書として固定する必要があった。

**変更内容**
- `DESIGN_BLUEPRINT.md` の `12.9 Phase25.9` を更新し、実装方針を「不足テスト + 必要時の最小修正」に固定。
- 早期停止 `best_iteration` 検証を「実学習依存」から「monkeypatch 契約検証優先」へ更新。
- Causal GroupKFold 項目を `tests/test_auto_split_selection.py` 前提から `tests/test_dr_internal.py` の `unit_id_col` 経路検証へ更新。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase25.9 は不足テスト追加で差分が出た場合に、同フェーズ内で最小本体修正まで実施する。
  - 理由: テスト未整備のままギャップを先送りせず、回帰防止と安定運用を同時に満たすため。
  - 影響範囲: modeling / tests / docs
- Decision: provisional（暫定）
  - 内容: Causal GroupKFold は既存 `group_col` 分岐に加え、`unit_id_col` 経路を `tests/test_dr_internal.py` で追加検証する。
  - 理由: 既存カバレッジ重複を避けつつ、未検証分岐のみを効率よく補完するため。
  - 影響範囲: causal/dr tests / docs

**検証結果**
- 文書更新段階のため、コード検証は未実施（次ステップで実装・検証予定）。

### 2026-02-16（作業/PR: phase25.9-lightgbm-test-gap-closure）
**背景**
- Phase25.9 で特定した不足ギャップ（学習ループ適用、search space、objective override、Artifact整合、best_iteration契約、causal unit_id経路）をテストで閉じる必要があった。
- 追加テストで不足が見つかった場合は、同フェーズ内で最小修正まで行う方針（provisional）を確定させる必要があった。

**変更内容**
- 新規テストを追加:
  - `tests/test_auto_num_leaves.py`
  - `tests/test_ratio_params.py`
  - `tests/test_feature_weights.py`
  - `tests/test_tuning_search_space.py`
  - `tests/test_objective_override.py`
  - `tests/test_artifact_param_roundtrip.py`
- 既存テストを拡張:
  - `tests/test_num_boost_round.py`（`num_boost_round` 既定値300の後方互換）
  - `tests/test_early_stopping_validation.py`（monkeypatchで `best_iteration` 記録契約を検証）
  - `tests/test_dr_internal.py`（`unit_id_col` 経路の GroupKFold 選択と KFold フォールバック）
- 本体の最小修正を実施:
  - `src/veldra/modeling/regression.py`
  - `src/veldra/modeling/binary.py`
  - `src/veldra/modeling/multiclass.py`
  - `src/veldra/modeling/frontier.py`
  - `feature_weights` 指定時に `params["feature_pre_filter"] = False` を付与し、学習パラメータ適用契約を固定。
- `DESIGN_BLUEPRINT.md` の Phase25.9 を更新し、best_iteration/causal項目を実装方針に一致させた。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Phase25.9 は不足テスト追加で差分が出た場合に、同フェーズ内で最小本体修正まで実施する。
  - 理由: 回帰防止と品質ゲートを同一フェーズで閉じるため。
  - 影響範囲: modeling / tests / docs
- Decision: confirmed（確定）
  - 内容: Causal GroupKFold は `group_col` 既存検証を維持しつつ、`unit_id_col` 経路を `tests/test_dr_internal.py` で補完する。
  - 理由: 重複を増やさず未検証分岐のみを効率的に埋めるため。
  - 影響範囲: causal/dr tests / docs
- Decision: confirmed（確定）
  - 内容: 早期停止 `best_iteration` は実学習依存ではなく、monkeypatch による契約検証を主方式とする。
  - 理由: CI環境差による不安定性を回避し、履歴記録契約を安定的に担保するため。
  - 影響範囲: early stopping tests

**検証結果**
- `uv run pytest -q tests/test_auto_num_leaves.py tests/test_ratio_params.py tests/test_feature_weights.py` : **6 passed**
- `uv run pytest -q tests/test_tuning_search_space.py tests/test_objective_override.py tests/test_artifact_param_roundtrip.py` : **4 passed**
- `uv run pytest -q tests/test_num_boost_round.py tests/test_early_stopping_validation.py tests/test_dr_internal.py` : **26 passed**
- `uv run ruff check .` : passed
- `uv run pytest -q -m "not gui"` : **399 passed, 100 deselected**

### 2026-02-16（作業/PR: phase26-ux-ui-implementation）
**背景**
- Phase26（UX/UI 改善）を 4画面構成から 6+2画面構成へ再編し、初学者導線の完遂率を上げる必要があった。
- 実装着手前に、ロールアウト方式・Export 範囲・`/config` 互換方針を固定する必要があった。

**変更内容**
- 画面と導線を拡張: `Target / Validation / Train / Runs / Compare` を新設し、`/config` は互換メッセージ付きで `/target` 導線へ集約。
- `workflow-state` を拡張し、`_build_config_from_state` / `_state_from_config_payload` を導入して state 駆動で RunConfig YAML を生成。
- `services.py` に `GuardRailChecker`、`infer_task_type`、Artifact 比較ロジック、Excel/HTML レポート生成を追加。
- `job_store.py` に `get_jobs` / `delete_jobs` を追加し、Runs ページの複数選択操作（削除/比較/複製）を実装。
- Results を拡張し、Learning Curves タブ・Config タブ・Export ジョブ（`export_excel` / `export_html_report`）を追加。
- テスト追加/更新:
  - 新規: `tests/test_gui_target_page.py`, `tests/test_gui_validation_page.py`, `tests/test_gui_train_page.py`, `tests/test_gui_runs_page.py`, `tests/test_gui_compare_page.py`, `tests/test_gui_guardrail.py`, `tests/test_gui_workflow_state.py`, `tests/test_gui_config_from_state.py`, `tests/test_gui_stepper.py`, `tests/test_gui_export_excel.py`, `tests/test_gui_export_html.py`, `tests/test_gui_results_enhanced.py`, `tests/test_gui_data_enhanced.py`, `tests/test_gui_e2e_flow.py`
  - 更新: `tests/test_gui_pages_logic.py`, `tests/test_gui_app_additional_branches.py`

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase26 は Stage A/B/C の 3 段階分割で実装し、各 Stage ごとに GUI テストを通して段階収束する。
  - 理由: 変更範囲が `app.py`/ページ/非同期ジョブに跨るため、単発導入より回帰抑止がしやすい。
  - 影響範囲: gui app/pages/services/tests/docs
- Decision: provisional（暫定）
  - 内容: Phase26 の Export は Excel + HTML までを必須実装とし、SHAP は `export-report` extra 導入時のみ有効化する。
  - 理由: GUI 基盤改修と同時に必須依存を増やし過ぎず、機能を段階導入するため。
  - 影響範囲: gui services/job actions/dependencies
- Decision: provisional（暫定）
  - 内容: `/config` 導線は 1 フェーズ互換を維持し、画面導線は `/target` へ誘導する。
  - 理由: 既存テストと運用導線の急激な破壊を避けつつ、新導線へ移行するため。
  - 影響範囲: gui routing/tests/user workflow
- Decision: confirmed（確定）
  - 内容: Stage A/B/C の 3 段階分割方針で Phase26 実装を完了し、各 Stage の機能（画面分割/ガードレール/Results拡張/Runs比較/Export）を同一PRで収束させる。
  - 理由: 段階ごとの回帰確認を維持しつつ、利用者導線を中途半端な状態で残さないため。
  - 影響範囲: gui app/pages/services/components/tests/docs
- Decision: confirmed（確定）
  - 内容: Export は Excel + HTML を GUI 非同期ジョブとして実装し、SHAP は依存未導入時にシートへ「未生成」表示でフォールバックする。
  - 理由: 依存差異でジョブを失敗させず、レポート導線を常時利用可能にするため。
  - 影響範囲: gui services/results page/job queue/dependencies
- Decision: confirmed（確定）
  - 内容: `/config` は互換経路として維持し、新規操作導線は `Data -> Target -> Validation -> Train -> Run -> Results` を正規経路とする。
  - 理由: 既存テスト契約と新UX導線を両立するため。
  - 影響範囲: gui routing/stepper/sidebar/tests

**検証結果**
- `uv run ruff check src/veldra/gui tests` を通過。
- `uv run pytest -q -m "gui"`: **126 passed, 399 deselected**
- `uv run pytest tests -x --tb=short`: **525 passed, 0 failed**

### 2026-02-16（作業/PR: design-blueprint-phase25-summary-compression）
**背景**
- `DESIGN_BLUEPRINT.md` の Phase25〜25.9 が詳細化により長文化し、完了済みフェーズの参照コストが高かった。
- 実装・決定・検証結果を保持したまま、要点中心に圧縮する必要があった。

**変更内容**
- `DESIGN_BLUEPRINT.md` の `12`〜`12.9` を再構成し、各フェーズを「目的/実装要点/検証結果」中心の短縮版へ置換。
- 重複していた背景説明・擬似コード・詳細手順を削減し、以下を明示的に保持:
  - Phase25: 非同期ジョブ基盤、config migrate、callback堅牢化
  - Phase25.5: テストDRY化、regression契約補完、公開ユーティリティ化
  - Phase25.6: CSS/HTML限定のUX改善と非スコープ
  - Phase25.7: TrainConfig拡張、ES分割、class weight、自動split、training_history、GUI/migration
  - Phase25.8: 追加パラメータ、top_k連携、GUI拡張、search space拡張、provisional decision
  - Phase25.9: 不足テスト補完、最小本体修正、confirmed decision
- Phase26 以降の記述は未変更。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 完了済みフェーズ（25〜25.9）は、将来の実装手順書ではなく運用参照向けに圧縮表現へ統一する。
  - 理由: 主要契約・意思決定・検証実績を維持しつつ、読み取りコストを下げるため。
  - 影響範囲: docs（`DESIGN_BLUEPRINT.md`）

**検証結果**
- ドキュメント編集のみ（コード/テスト変更なし）。
- `DESIGN_BLUEPRINT.md` にて Phase25〜25.9 の情報欠落がないことを目視確認。

### 2026-02-16（作業/PR: phase26.1-stage1-bugfixes）
**背景**
- Phase26.1 Stage1 の必須修正（JST表示統一 / Exportブラウザダウンロード / Learning Curves 表示）を、既存GUI基盤を維持したまま収束させる必要があった。

**変更内容**
- `src/veldra/gui/app.py`
  - Runジョブ一覧テーブルの created 時刻を JST 表示へ統一。
  - Run詳細の Created/Started/Finished を `_format_jst_timestamp()` 経由に統一し、欠損時は `n/a` 表示へ統一。
  - Results Export を「ジョブ投入 + `dcc.Store` 保持 + `dcc.Interval` ポーリング」へ拡張。
  - Export ジョブ完了時に `dcc.send_file()` で HTML/Excel をブラウザダウンロードするコールバックを追加。
  - `_cb_update_result_extras()` の learning history 参照を `art.training_history` 優先に修正し、metadata/file fallback 依存を削除。
  - Runs テーブルデータに `started_at_utc` / `finished_at_utc` を追加し、JST 表示へ統一。
- `src/veldra/gui/pages/results_page.py`
  - `result-report-download` / `result-export-job-store` / `result-export-poll-interval` を追加。
- `src/veldra/gui/pages/runs_page.py`
  - Runs DataTable に `started` / `finished` 列を追加。
- `src/veldra/gui/services.py`
  - `_export_output_path()` の出力ファイル名 timestamp を JST 基準へ変更。
- テスト:
  - 新規: `tests/test_gui_phase26_1.py`（Stage1受け入れ観点を集約）
  - 更新: `tests/test_gui_runs_page.py`, `tests/test_gui_results_enhanced.py`

**決定事項**
- Decision: confirmed（確定）
  - 内容: Phase26.1 の正式定義は `DESIGN_BLUEPRINT.md` の 13.1 を唯一の正とする。
  - 理由: 旧サブフェーズ表と 13.1 の解釈ズレを解消し、実装単位を固定するため。
  - 影響範囲: docs / phase planning
- Decision: confirmed（確定）
  - 内容: Phase26.1 は Stage1（コード修正）/Stage2（設計成果物）を 2PR 分割で運用する。
  - 理由: 回帰確認と設計レビューを分離し、変更責務を明確化するため。
  - 影響範囲: delivery process / docs
- Decision: confirmed（確定）
  - 内容: Stage2 成果物（対応マトリクス・ギャップ一覧・Phase26.2ドラフト）は `DESIGN_BLUEPRINT.md` に集約する。
  - 理由: 設計判断と次フェーズ実装計画を単一参照元に固定するため。
  - 影響範囲: docs

**検証結果**
- `uv run ruff check src/veldra/gui tests` を通過。
- `uv run pytest -q tests/test_gui_phase26_1.py tests/test_gui_runs_page.py tests/test_gui_results_enhanced.py tests/test_gui_export_excel.py tests/test_gui_export_html.py` を通過。
- `uv run pytest -q -m "gui"` を通過。

### 2026-02-16（作業/PR: phase26.1-stage2-usecase-design）
**背景**
- Phase26.2 実装に向け、現行GUIのユースケース対応状況を設計文書として固定し、ギャップと優先度を明文化する必要があった。

**変更内容**
- `DESIGN_BLUEPRINT.md` の 13.1 配下へ以下を追加:
  - 現行GUI対応状況マトリクス（全ユースケース対象）
  - 優先度付きギャップ一覧（P0/P1/P2）
  - 画面単位の Phase26.2 実装計画ドラフト（実装順/依存/完了条件/テスト観点）
- `DESIGN_BLUEPRINT.md` の旧サブフェーズ依存表に注記を追加し、
  `13.1` を Phase26.1 の正定義として明示。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 26.1 旧依存表は履歴として保持し、現在運用の意思決定は 13.1 を正とする。
  - 理由: 履歴保全と現行運用の明確化を両立するため。
  - 影響範囲: docs

**検証結果**
- 文書更新の相互参照（旧依存表注記 / 13.1 / Stage2成果物）を目視確認。

### 2026-02-17（作業/PR: phase26.2-usecase-guided-ui-implementation）
**背景**
- Phase26.1 Stage2 で確定した P0-P2 ギャップを、Core/API 非変更のまま GUI adapter 層で解消する必要があった。

**変更内容**
- Step0: `notebooks/phase26_2_ux_audit.ipynb` を追加し、UC-1〜UC-10 の監査テンプレートと優先度固定欄を整備。
- Step1: `src/veldra/gui/components/help_ui.py` / `src/veldra/gui/components/help_texts.py` を追加し、共通ヘルプUI基盤を導入。
- Step2-4: `target_page` / `validation_page` / `train_page` にガイドUI、推奨表示、プリセット、objective説明を追加。
- Step5: `run_page` に action manual override + pre-run guardrail 表示を追加。`runs_page` に Export/Re-evaluate ショートカット列を追加。`results_page` に再評価前 feature schema 診断とショートカットハイライトを追加。
- `src/veldra/gui/app.py` の callback を拡張し、`workflow-state` に `results_shortcut_focus` / `run_action_override` を追加。
- テスト追加/更新: `tests/test_gui_help_ui.py`, `tests/test_gui_run_presets.py`, `tests/test_gui_target_page.py`, `tests/test_gui_validation_page.py`, `tests/test_gui_train_page.py`, `tests/test_gui_runs_page.py`, `tests/test_gui_results_enhanced.py`。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Phase26.2 は 13.2 の Step0-5 を一括で実装し、ヘルプUI基盤を全画面で再利用する。
  - 理由: 画面ごとの文言・導線を重複実装すると保守性が下がるため。
  - 影響範囲: gui components/pages/app/tests
- Decision: confirmed（確定）
  - 内容: Run の実行可否は「必須入力チェック + guardrail error」で制御し、warning/info は実行をブロックしない。
  - 理由: 過剰ブロックを避けつつ、重大な設定ミスのみ事前停止するため。
  - 影響範囲: run page callbacks / UX

**検証結果**
- `uv run pytest -q tests/test_gui_help_ui.py tests/test_gui_target_page.py tests/test_gui_validation_page.py tests/test_gui_train_page.py tests/test_gui_run_presets.py tests/test_gui_runs_page.py tests/test_gui_results_enhanced.py tests/test_gui_e2e_flow.py` を実施。
- `uv run pytest -q -m "gui"` で GUI 回帰確認を実施。

### 2026-02-17（作業/PR: phase26.2-correction-notebook-evidence-and-gui-parity）
**背景**
- Phase26.2 の Step0 は監査テンプレート整備までに留まり、UC-1〜UC-10 の実行証跡と GUI 同等性（到達+成果物一致）の証跡が不足していた。
- 完了条件を補正し、Notebook 実行証跡・Playwright E2E・parity report を追加する必要があった。

**変更内容**
- Notebook 補正:
  - `notebooks/phase26_2_ux_audit.ipynb` を監査ハブ化（UC索引・A-D分類・優先度固定）。
  - UC別 Notebook を 10 冊追加。
    - `notebooks/phase26_2_uc01_regression_fit_evaluate.ipynb`
    - `notebooks/phase26_2_uc02_binary_tune_evaluate.ipynb`
    - `notebooks/phase26_2_uc03_frontier_fit_evaluate.ipynb`
    - `notebooks/phase26_2_uc04_causal_dr_estimate.ipynb`
    - `notebooks/phase26_2_uc05_causal_drdid_estimate.ipynb`
    - `notebooks/phase26_2_uc06_causal_dr_tune.ipynb`
    - `notebooks/phase26_2_uc07_artifact_evaluate.ipynb`
    - `notebooks/phase26_2_uc08_artifact_reevaluate.ipynb`
    - `notebooks/phase26_2_uc09_export_python_onnx.ipynb`
    - `notebooks/phase26_2_uc10_export_html_excel.ipynb`
  - 実行証跡を `notebooks/phase26_2_execution_manifest.json` に固定。
- Notebook 契約テスト追加:
  - `tests/test_notebook_phase26_2_uc_structure.py`
  - `tests/test_notebook_phase26_2_execution_evidence.py`
  - `tests/test_notebook_phase26_2_paths.py`
- GUI parity（Playwright）追加:
  - `tests/e2e_playwright/conftest.py`
  - `tests/e2e_playwright/test_uc01_regression_flow.py`
  - `tests/e2e_playwright/test_uc02_binary_tune_flow.py`
  - `tests/e2e_playwright/test_uc03_frontier_flow.py`
  - `tests/e2e_playwright/test_uc04_causal_dr_flow.py`
  - `tests/e2e_playwright/test_uc05_causal_drdid_flow.py`
  - `tests/e2e_playwright/test_uc06_causal_tune_flow.py`
  - `tests/e2e_playwright/test_uc07_evaluate_existing_artifact_flow.py`
  - `tests/e2e_playwright/test_uc08_reevaluate_flow.py`
  - `tests/e2e_playwright/test_uc09_export_python_onnx_flow.py`
  - `tests/e2e_playwright/test_uc10_export_html_excel_flow.py`
- ドキュメント/設定更新:
  - `docs/phase26_2_parity_report.md` を追加。
  - `DESIGN_BLUEPRINT.md` 13.2 に補正フェーズ完了状況を追記。
  - `pyproject.toml` / `tests/conftest.py` に `gui_e2e` / `gui_smoke` marker を追加。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase26.2 の完了条件を「テンプレート整備」から「Notebook実行証跡 + GUI parity pass（到達+成果物一致）」へ再定義する。
  - 理由: 実行証跡を欠いた状態では UX 改善の実効性を担保できないため。
  - 影響範囲: notebooks / tests / docs / release gate
- Decision: confirmed（確定）
  - 内容: 同等性判定は Notebook と GUI の数値同値比較ではなく、到達導線と成果物生成一致を正とする。
  - 理由: 実行環境差でのノイズを避けつつ、利用者導線の品質保証にフォーカスするため。
  - 影響範囲: parity report / E2E acceptance

**検証結果**
- Notebook 実行証跡を `notebooks/phase26_2_execution_manifest.json` に記録（UC-1〜UC-10）。
- Optional 依存の graceful degradation を記録（例: `openpyxl` 未導入時の Excel export）。
- Playwright E2E は `gui_e2e`/`gui_smoke` marker で手動全件・CIスモーク運用を可能化。
- `.venv/bin/ruff check tests/e2e_playwright tests/test_notebook_phase26_2_uc_structure.py tests/test_notebook_phase26_2_execution_evidence.py tests/test_notebook_phase26_2_paths.py` を通過。
- `.venv/bin/pytest -q tests/test_notebook_phase26_2_uc_structure.py tests/test_notebook_phase26_2_execution_evidence.py tests/test_notebook_phase26_2_paths.py` を実施（`9 passed`）。
- `.venv/bin/pytest -q tests/e2e_playwright --collect-only` を実施（10件収集）。
- `.venv/bin/pytest -q -m "gui"` を実施（`146 passed, 418 deselected`）。
- `.venv/bin/pytest -q -m "gui_e2e and gui_smoke"` を実施（`3 skipped, 561 deselected`）。
- `.venv/bin/pytest -q -m "gui_e2e"` を実施（`10 skipped, 554 deselected`）。

### 2026-02-17（作業/PR: phase26.3-core-diagnostics-and-notebook-delivery）
**背景**
- Phase26.3 で定義したユースケース詳細化（Core拡張 + Notebook完全版 + 証跡運用）を、既存 Stable API 互換を維持したまま実装する必要があった。
- Notebook の重実行を常時CIに載せるとコストが高いため、証跡検証を marker 分離する運用方針を固定する必要があった。

**変更内容**
- Core/Config拡張:
  - `train.metrics` / `tuning.metrics_candidates` を RunConfig に追加し、task/causal method 別バリデーションを実装。
  - tuning objective に `mape` / `multi_logloss` / `multi_error` を追加し、`multi_logloss -> logloss`, `multi_error -> error_rate` の alias 解決を導入。
  - tuning search space で `train.*` プレフィックス指定を解釈し、`TrainConfig` フィールドへ直接注入できるよう拡張。
- Modeling/Artifact拡張:
  - `RegressionTrainingOutput` / `BinaryTrainingOutput` / `MulticlassTrainingOutput` / `FrontierTrainingOutput` に `observation_table` を追加。
  - Artifact save/load に `observation_table.parquet` の optional 永続化を追加（後方互換維持）。
- Causal拡張:
  - `DREstimationOutput` に `nuisance_diagnostics` を追加。
  - DR で nuisance importance と OOF diagnostics を生成し、DR-DiD で `parallel_trends` を summary/diagnostics に追加。
  - `causal.diagnostics` に `compute_ess` / `extreme_weight_ratio` / `overlap_summary` を追加。
- Diagnosticsパッケージ新設:
  - `src/veldra/diagnostics/`（importance/shap_native/metrics/plots/tables/causal_diag）を追加。
- Notebook/証跡:
  - UC-1〜UC-10 Notebook を Phase26.3 契約（`matplotlib.use('Agg')`, `savefig`, `SUMMARY`）へ更新。
  - `phase26_3_uc_multiclass_fit_evaluate.ipynb` / `phase26_3_uc_timeseries_fit_evaluate.ipynb` を追加。
  - `notebooks/phase26_3_execution_manifest.json` を追加。
- テスト追加:
  - diagnostics系: `tests/test_diagnostics_*.py`
  - config/tuning拡張: `tests/test_phase263_config_extensions.py`, `tests/test_phase263_tuning_aliases.py`
  - observation/causal: `tests/test_observation_table.py`, `tests/test_causal_dr.py`, `tests/test_causal_drdid.py`
  - notebook契約: `tests/test_notebook_phase26_3_*.py`（`notebook_e2e` marker 運用）

**決定事項**
- Decision: provisional（暫定）
  - 内容: `config_version` は 1 を維持し、Phase26.3 の設定拡張は optional フィールド追加で収める。
  - 理由: Config migration の破壊的変更を回避し、既存運用への影響を最小化するため。
  - 影響範囲: config / tuning / GUI-config 互換
- Decision: confirmed（確定）
  - 内容: Notebook 証跡は「常時構造テスト + `notebook_e2e` で重証跡検証」のハイブリッド運用とする。
  - 理由: 検証強度と CI 実行コストを両立するため。
  - 影響範囲: pyproject markers / notebook tests / release gate

**検証結果**
- `uv run ruff check`（Phase26.3 対象ファイル群）を通過。
- 主要新規テスト（config/tuning/modeling/causal/diagnostics/notebook構造）を実行対象として追加済み。

### 2026-02-17（作業/PR: phase26.3-residual-notebook-visibility-closure）
**背景**
- Phase26.3 の骨組み実装後、Notebook が placeholder 出力中心で、図表/表/指標の可視性と証跡契約が不足していた。
- `tuning.metrics_candidates` の許可ルールが objective 制約と同一化され、設計テーブル（task別候補）と不整合が残っていた。

**変更内容**
- Config 契約:
  - `tuning.metrics_candidates` を objective 許可セットから分離し、task別許可セットで検証する実装へ修正。
  - 関連テストを拡張（許可/拒否・causal時の既存契約維持）。
- Notebook 実体化:
  - `UC-1〜UC-8` と `UC-11/12` を実データ実行セルへ更新し、実行済み出力（グラフ・表・指標）をコミット。
  - 生成ファイルを `examples/out/phase26_2_*` / `examples/out/phase26_3_*` に再生成し、PNG 実体ファイル化を確認。
  - `notebooks/phase26_3_execution_manifest.json` を実実行値（artifact_path / outputs / metrics）で更新。
- Notebook テスト強化:
  - 構造テストに `placeholder` 禁止、diagnostics import、実行済みセル契約を追加。
  - evidence テストに対象 UC 固定（`UC-1〜8,11,12`）と `artifact_path/metrics/outputs` 契約を追加。
  - outputs テストに PNG signature 検証、CSV列契約、主要指標レンジ検証を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Phase26.3 Notebook は「実行済みで可視化済み」を完了条件にし、placeholder 出力を残さない。
  - 理由: 利用者が Notebook を開いた瞬間に診断結果を確認できる状態を保証するため。
  - 影響範囲: notebooks / execution manifest / notebook_e2e tests
- Decision: confirmed（確定）
  - 内容: `metrics_candidates` は objective と独立した task別候補セットで検証し、causal時は causal objective セットを流用する。
  - 理由: 設計テーブルと実装契約を一致させ、運用時の誤設定を早期検出するため。
  - 影響範囲: config validation / tuning diagnostics

**検証結果**
- `uv run pytest -q tests/test_phase263_config_extensions.py tests/test_runconfig_validation.py tests/test_notebook_phase26_3_uc_structure.py` を通過（`37 passed`）。
- `uv run pytest -q tests/test_notebook_phase26_3_execution_evidence.py tests/test_notebook_phase26_3_outputs.py -m notebook_e2e` を通過（`2 passed`）。
- `uv run pytest -q tests -m "not gui_e2e"` を通過（`584 passed, 10 deselected`）。

### 2026-02-17（作業/PR: phase26.4-notebook-ux-rename-and-test-hardening）
**背景**
- Phase26.4 の実装着手にあたり、`phase26_*` 命名が利用者導線として理解しづらく、Notebook 情報設計の再編が必要だった。
- Notebook 契約テストは構造寄りに偏っており、runner happy path / edge / 数値安定性 / data loader 堅牢性を補完する必要があった。

**変更内容**
- 設計更新:
  - `DESIGN_BLUEPRINT.md` 13.4 を Step0-6 の実装順序、命名再編マップ、互換期限（1リリース）つきで具体化。
- Notebook 再編:
  - `notebooks/tutorials/` と `notebooks/quick_reference/` を新設し、canonical Notebook を用途別に再配置。
  - 新規 tutorial: `tutorial_00_quickstart.ipynb`, `tutorial_07_model_evaluation_guide.ipynb` を追加。
  - `notebooks/reference_index.ipynb` を追加し、Tutorial/Quick Reference の索引を統一。
  - 旧名 Notebook（workflow / phase26 UC / phase26 audit）を `Moved to ...` 明記の互換スタブへ置換。
- 生成/証跡更新:
  - `scripts/generate_phase263_notebooks.py` を `notebooks/quick_reference/` 出力へ更新し、Overview / Learn More / Config Notes / Output Annotation を追加。
  - `notebooks/phase26_3_execution_manifest.json` の `notebook` フィールドを canonical パスへ更新。
- 参照更新:
  - `README.md` の notebook 導線を `notebooks/tutorials/*` へ更新。
  - `docs/phase26_2_parity_report.md` の Notebook 証跡パスを `notebooks/quick_reference/*` へ更新。
  - Notebook テスト群の参照先を canonical へ更新し、互換スタブ検証を追加。
- テスト強化:
  - 追加: `tests/test_runner_fit_happy.py`, `tests/test_runner_evaluate_happy.py`, `tests/test_runner_predict_happy.py`, `tests/test_runner_tune_happy.py`
  - 追加: `tests/test_edge_cases.py`, `tests/test_numerical_stability.py`, `tests/test_config_cross_field.py`, `tests/test_data_loader_robust.py`
  - 追加: `tests/test_notebook_tutorial_catalog.py`
  - 拡張: `tests/conftest.py` に `unbalanced_binary_frame` / `categorical_frame` / `timeseries_frame` / `missing_values_frame` / `outlier_frame` を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Notebook 命名規約は英語スネークケースとし、canonical 配置を `notebooks/tutorials` / `notebooks/quick_reference` の2系統へ固定する。
  - 理由: 学習導線（チュートリアル）と実行証跡（クイックリファレンス）の責務を分離し、利用者が目的別に辿れるようにするため。
  - 影響範囲: notebooks / README / docs / notebook tests / manifest notebook paths
- Decision: provisional（暫定）
  - 内容: 旧名 Notebook は 1リリース互換スタブとして維持し、次期リリースで削除可否を確定する。
  - 理由: 既存リンク・運用手順の破壊を避けながら段階移行するため。
  - 影響範囲: notebooks root 旧名ファイル群 / 移行ガイド

**検証結果**
- `uv run ruff check`（Phase26.4 変更対象の scripts/tests）を通過。
- `uv run pytest -q`（Phase26.4 追加/更新テスト群 25ファイル）を実施し、`70 passed`。
- `uv run pytest -q tests -m "not gui_e2e"` を実施し、`624 passed, 10 deselected`。
- `rg --files notebooks | rg \"phase26_.*\\.ipynb\"` のヒットが互換スタブのみであることを確認。

### 2026-02-17（作業/PR: docs-phase26-blueprint-compression）
**背景**
- `DESIGN_BLUEPRINT.md` の Phase26〜26.2 が詳細手順中心で肥大化し、現状把握に必要な要点を素早く参照しにくくなっていた。

**変更内容**
- `DESIGN_BLUEPRINT.md` の `13 Phase 26` / `13.1 Phase 26.1` / `13.2 Phase 26.2` を要約再編。
- 各セクションを「目的 / 固定方針 / 実装結果 / 完了条件」中心に圧縮し、詳細な手順列挙を削減。
- 26.2 の補正フェーズ（実行証跡 + parity 検証）と成果物（Notebook群、manifest、E2E、parity report）は保持。
- 詳細トレースは `HISTORY.md` を正とする運用メモを `DESIGN_BLUEPRINT.md` に明記。

**決定事項**
- Decision: confirmed（確定）
  - 内容: DESIGN_BLUEPRINT の Phase26〜26.2 は「設計状態の要約」に徹し、実装手順の詳細は HISTORY に集約する。
  - 理由: 設計書の可読性と履歴トレース性を両立するため。
  - 影響範囲: docs

**検証結果**
- `DESIGN_BLUEPRINT.md` の 13〜13.2 が要約形で連続し、`13.3` 以降へ接続されることを確認。

### 2026-02-17（作業/PR: phase26.5-notebook-ab-alignment-and-gui-e2e-hardening）
**背景**
- `DESIGN_BLUEPRINT.md` 13.3 の A/B 契約（固定学習パラメーター / tuning search space）が、Phase26.4 後の canonical Notebook（`quick_reference` / `tutorials`）と不整合だった。
- `tests/e2e_playwright` は hidden input の `visible` 待機に依存し、`gui_e2e` がタイムアウトで失敗していた。

**変更内容**
- 設計/履歴:
  - `DESIGN_BLUEPRINT.md` に `13.5 Phase26.5` を追加（背景/目的/適用範囲/実装ステップ/テスト計画/完了条件/Decision）。
- Notebook A/B 適用:
  - `scripts/generate_phase263_notebooks.py` を更新し、UC-1/2/3/4/5/6/11/12 の train 設定を 13.3 A へ統一。
  - UC-2 objective を `brier` へ更新し、UC-2/UC-6 search space を 13.3 B へ統一。
  - `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_phase263_notebooks.py` を再実行し、`quick_reference` と `phase26_3_execution_manifest.json` を再生成。
  - `notebooks/tutorials/tutorial_01..06.ipynb` の config セルを A/B 準拠へ更新（`tutorial_02` は `train.*` best_params 反映ロジックへ修正）。
- Core/tuning:
  - `src/veldra/modeling/tuning.py` の `standard` preset 既定探索空間を 13.3 B 契約へ更新。
  - `tests/test_tuning_search_space.py` を新契約に合わせて更新。
- E2E 安定化:
  - `tests/e2e_playwright/_helpers.py` の `goto` に `#page-content` 待機を追加。
  - `assert_ids` を `attached/visible` 切替対応へ変更。
  - `tests/e2e_playwright/test_uc01_*`, `test_uc02_*`, `test_uc04_*`, `test_uc05_*`, `test_uc09_*` を hidden input 非依存の操作へ更新。
- 契約テスト:
  - `tests/test_notebook_phase26_5_ab_contract.py` を追加し、canonical Notebook の A/B キー・値を機械検証可能にした。

**決定事項**
- Decision: provisional（暫定）
  - 内容: 13.3 A/B の適用対象は canonical Notebook（`quick_reference` + `tutorials`）とし、legacy stub は対象外とする。
  - 理由: 互換スタブの責務を維持しつつ、実利用導線の契約整合を優先するため。
  - 影響範囲: notebooks / notebook tests / generation script
- Decision: confirmed（確定）
  - 内容: `gui_e2e` の失敗は GUI 実装を変更せず、Playwright テストの待機/操作戦略の見直しで収束させる。
  - 理由: 既存 UI 互換性を保持しながら flaky 要因を除去できるため。
  - 影響範囲: tests/e2e_playwright/*

**検証結果**
- `uv run ruff check scripts/generate_phase263_notebooks.py src/veldra/modeling/tuning.py tests/e2e_playwright tests/test_tuning_search_space.py tests/test_notebook_phase26_5_ab_contract.py` を通過。
- `uv run pytest -q tests/test_tuning_search_space.py tests/test_notebook_phase26_5_ab_contract.py` を実施（`3 passed`）。
- `uv run pytest -q tests/test_notebook_phase26_2_uc_structure.py tests/test_notebook_phase26_3_uc_structure.py` を実施（`5 passed`）。
- `uv run pytest -q tests/e2e_playwright -m gui_e2e` を実施（`10 passed`）。
- `uv run pytest -q -m "not gui_e2e"` を実施（`626 passed, 10 deselected`）。

### 2026-02-18（作業/PR: phase26.6-legacy-notebook-manifest-cleanup）
**背景**
- Phase26.4 で導入した legacy notebook 互換スタブの運用期限が終了し、canonical notebook への一本化が必要だった。
- notebook 証跡が `notebooks/phase26_*_execution_manifest.json` に分散しており、`examples/out/*` 実体との二重管理を解消する必要があった。

**変更内容**
- legacy notebook を削除し、`notebooks/tutorials/*` / `notebooks/quick_reference/*` / `notebooks/reference_index.ipynb` のみを canonical として残した。
- `notebooks/phase26_2_execution_manifest.json` と `notebooks/phase26_3_execution_manifest.json` を削除し、証跡参照を `examples/out/phase26_*/summary.json` + outputs 実体へ移行した。
- `tests/e2e_playwright/conftest.py` の fixture を manifest 依存から固定パス解決（`latest_artifact_path.txt`, `reeval_missing_col.csv` 等）へ置換し、`test_uc07/08/10` を新 IF に更新した。
- `tests/test_notebook_phase26_2_execution_evidence.py` を削除し、`tests/test_notebook_execution_evidence.py` を追加。`tests/test_notebook_phase26_3_execution_evidence.py` / `tests/test_notebook_phase26_3_outputs.py` を summary ベースへ更新した。
- `scripts/generate_phase263_notebooks.py` から manifest 生成ロジックを削除し、quick reference 生成 + 実行に責務を限定した。
- `README.md` と `DESIGN_BLUEPRINT.md` を更新し、manifest 前提・legacy stub 維持前提・phase26.2 parity report 依存を撤去した。
- `docs/phase26_2_parity_report.md` と未使用 `src/veldra/postprocess/__init__.py`（および空ディレクトリ）を削除した。

**決定事項**
- Decision: confirmed（確定）
  - 内容: legacy notebook 互換スタブ運用を終了し、canonical notebook のみを正とする。
  - 理由: ドキュメント導線とテスト契約を一本化し、保守対象を縮小するため。
  - 影響範囲: notebooks / notebook tests / DESIGN_BLUEPRINT / README
- Decision: confirmed（確定）
  - 内容: notebook 証跡は manifest ファイルを廃止し、`examples/out/phase26_*/summary.json` と outputs 実体で管理する。
  - 理由: 生成物の実体を単一の真実源にして、証跡の二重管理を防ぐため。
  - 影響範囲: tests/e2e_playwright / notebook_e2e tests / scripts/generate_phase263_notebooks.py / docs

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run ruff check README.md DESIGN_BLUEPRINT.md tests/e2e_playwright tests/test_notebook_tutorial_catalog.py tests/test_notebook_phase26_2_uc_structure.py tests/test_notebook_phase26_2_paths.py tests/test_notebook_execution_evidence.py tests/test_notebook_phase26_3_execution_evidence.py tests/test_notebook_phase26_3_outputs.py scripts/generate_phase263_notebooks.py` を通過。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_notebook_tutorial_catalog.py tests/test_notebook_phase26_2_uc_structure.py tests/test_notebook_phase26_2_paths.py tests/test_notebook_execution_evidence.py` を実施（`12 passed`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_notebook_phase26_3_uc_structure.py tests/test_notebook_phase26_3_execution_evidence.py tests/test_notebook_phase26_3_outputs.py -m notebook_e2e` を実施（`2 passed, 1 deselected`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/e2e_playwright -m gui_e2e` を実施（`10 skipped`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "not gui_e2e"` を実施（`626 passed, 10 deselected`）。

### 2026-02-18（作業/PR: phase26.6-test-quality-rename-and-coverage）
**背景**
- Phase26.6 の計画（命名整理 + カバレッジ強化）に対し、notebook テスト命名と実体の不一致、および一部コアモジュールの直接ユニットテスト不足を解消する必要があった。
- `tests/test_notebook_*.py` は実測 18 ファイルで、設計上の対象数と差分があり、実態同期が必要だった。

**変更内容**
- Stage A（命名整理）
  - Tutorial テストを `test_tutorial_01_*`〜`test_tutorial_06_*` へリネーム。
  - `test_notebook_phase26_3_uc_structure.py` を `test_quickref_structure.py` へリネームし、`test_notebook_phase26_2_uc_structure.py` の独自検証（reference_index / legacy notebook 削除）を統合した上で削除。
  - `test_notebook_phase26_2_paths.py` を `test_quickref_paths.py` へリネーム。
  - `test_notebook_phase26_3_execution_evidence.py` を削除し、`test_notebook_execution_evidence.py` を単一の証跡テストとして維持。
  - `test_notebook_phase26_3_outputs.py` を `test_notebook_execution_outputs.py` へリネーム。
  - `test_notebook_phase26_5_ab_contract.py` を `test_notebook_reference_ab_contract.py` へリネーム。
- Stage B（カバレッジ強化）
  - 新規: `tests/test_artifact_store.py`
  - 新規: `tests/test_config_io.py`
  - 新規: `tests/test_causal_diagnostics_unit.py`
  - 拡張: `tests/test_numerical_stability.py`（NaN 伝播契約と clipping 境界）
  - 新規: `tests/test_binary_edge_cases.py`, `tests/test_regression_edge_cases.py`, `tests/test_frontier_edge_cases.py`, `tests/test_multiclass_edge_cases.py`, `tests/test_tune_edge_cases.py`
  - リネーム + 強化: `tests/test_data_loader_edge.py`（旧 `test_data_loader_robust.py`）, `tests/test_split_time_series.py`（旧 `test_time_series_splitter_additional.py`）
- ドキュメント
  - `DESIGN_BLUEPRINT.md` 13.6 にベースライン（18ファイル）と実装完了状況を追記し、Decision を confirmed 化した。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase26.6 は 3PR 粒度（命名整理 → Critical+数値安定 → edge/data/split）で段階実施する。
  - 理由: テスト資産の大規模リネームとカバレッジ追加を分離し、レビュー/回帰リスクを制御するため。
  - 影響範囲: tests / DESIGN_BLUEPRINT
- Decision: confirmed（確定）
  - 内容: notebook テスト命名は phase 番号依存を廃止し、対象種別ベース（tutorial / quickref / notebook_execution / notebook_reference）へ統一する。
  - 理由: notebook 再編時でも命名規約を安定化し、テスト責務を明確化するため。
  - 影響範囲: `tests/test_tutorial_*.py`, `tests/test_quickref_*.py`, `tests/test_notebook_execution_*.py`, `tests/test_notebook_reference_ab_contract.py`
- Decision: confirmed（確定）
  - 内容: Phase26.6 の完了条件は Stage A/B 実装 + `-m \"not gui_e2e and not notebook_e2e\"` 回帰グリーンで固定する。
  - 理由: GUI E2E / notebook E2E の実行コストを分離しつつ、日常回帰の品質ゲートを維持するため。
  - 影響範囲: tests / CI 運用

**検証結果**
- `uv run ruff check tests/test_tutorial_*.py tests/test_quickref_*.py tests/test_notebook_execution_*.py tests/test_notebook_reference_ab_contract.py tests/test_notebook_tutorial_catalog.py` を通過。
- `uv run pytest -q tests/test_tutorial_*.py tests/test_quickref_*.py tests/test_notebook_execution_evidence.py tests/test_notebook_execution_outputs.py tests/test_notebook_reference_ab_contract.py tests/test_notebook_tutorial_catalog.py` を実施（`34 passed`）。
- `uv run ruff check tests/test_artifact_store.py tests/test_config_io.py tests/test_causal_diagnostics_unit.py tests/test_numerical_stability.py` を通過。
- `uv run pytest -q tests/test_artifact_store.py tests/test_config_io.py tests/test_causal_diagnostics_unit.py tests/test_numerical_stability.py` を実施（`20 passed`）。
- `uv run ruff check tests/test_*edge*.py tests/test_split_time_series.py tests/test_data_loader_edge.py tests/test_tune_edge_cases.py` を通過。
- `uv run pytest -q tests/test_binary_edge_cases.py tests/test_regression_edge_cases.py tests/test_frontier_edge_cases.py tests/test_multiclass_edge_cases.py tests/test_tune_edge_cases.py tests/test_data_loader_edge.py tests/test_split_time_series.py` を実施（`31 passed, 1 warning`）。
- `uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施（`658 passed, 11 deselected, 1 warning`）。

### 2026-02-18（作業/PR: phase26.7-core-refactor-step1-3）
**背景**
- modeling 4タスク実装・RunConfig cross-field 検証・causal DR nuisance learner に構造的重複が残り、保守/拡張コストが高止まりしていた。
- Stable API と Artifact 契約を維持したまま、段階的リファクタ（Step1/2/3）を実施する必要があった。

**実施計画**
- 3PRで実施（Step1: RunConfig分離、Step2: causal learner抽象化、Step3: CVループ統合）。
- Step3 は baseline capture + 完全一致パリティテストでゲートする。

**変更内容**
- Step1（RunConfigバリデーション分離）:
  - `src/veldra/config/models.py` の `SplitConfig`/`TrainConfig` に `@model_validator(mode=\"after\")` を追加。
  - `_validate_cross_fields` から split/train の自己完結検証を除去し、task横断・サブコンフィグ横断検証に集約。
- Step2（causal learner 抽象化）:
  - `src/veldra/causal/learners.py` を新設し、`PropensityLearner`/`OutcomeLearner` Protocol と default factory を追加。
  - `src/veldra/causal/dr.py` の LightGBM 直接生成を factory 経由へ変更し、`run_dr_estimation(..., *, propensity_factory=None, outcome_factory=None)` を後方互換で拡張。
- Step3（CVループ統合）:
  - `src/veldra/modeling/_cv_runner.py` を新設し、`TaskSpec` + `run_cv_training` + `booster_iteration_stats` を実装。
  - `src/veldra/modeling/{regression,binary,multiclass,frontier}.py` の CV fold loop / final model 学習 / training_history 組み立てを共通ランナーへ移行。
  - private helper 互換として各 task module に `_booster_iteration_stats` を薄い委譲として維持。
- 完全一致パリティ:
  - baseline を `tests/fixtures/phase267_parity/*.pkl` に保存（4 task）。
  - `tests/test_phase267_output_parity.py` を追加し、`metrics/cv_results/training_history/observation_table/feature_schema/model_text`（binary は `threshold/calibration_curve` を追加）を完全一致比較。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase26.7 は 3PR 分割で実施し、同等性判定は「完全一致」を採用する（非決定メタ項目は除外）。
  - 理由: 大規模リファクタの回帰リスクを局所化し、既存契約を機械的に担保するため。
  - 影響範囲: config/models, causal/dr, modeling/*, parity tests, DESIGN_BLUEPRINT
- Decision: confirmed（確定）
  - 内容: Step1→Step2→Step3 の順序、3PR分割、完全一致ゲートを実装し、Stable API/Artifact 契約を維持したまま完了する。
  - 理由: 設計上の保守性改善と既存運用の安全性を両立できるため。
  - 影響範囲: `src/veldra/config/models.py`, `src/veldra/causal/{dr.py,learners.py}`, `src/veldra/modeling/{_cv_runner.py,binary.py,multiclass.py,regression.py,frontier.py}`, `tests/test_phase267_output_parity.py`, `tests/fixtures/phase267_parity/`

**検証結果**
- `uv run ruff check src/veldra/config/models.py tests/test_runconfig_validation.py tests/test_config_train_fields.py tests/test_config_cross_field.py tests/test_tune_validation.py tests/test_early_stopping_validation.py` を通過。
- `uv run pytest -q tests/test_runconfig_validation.py tests/test_config_train_fields.py tests/test_config_cross_field.py tests/test_tune_validation.py tests/test_early_stopping_validation.py` を実施（`55 passed`）。
- `uv run ruff check src/veldra/causal/learners.py src/veldra/causal/dr.py tests/test_dr_internal.py tests/test_causal_dr.py tests/test_dr_validation.py tests/test_drdid_validation.py tests/test_drdid_binary_validation.py tests/test_drdid_smoke_panel.py` を通過。
- `uv run pytest -q tests/test_dr_internal.py tests/test_causal_dr.py tests/test_dr_validation.py tests/test_drdid_validation.py tests/test_drdid_binary_validation.py tests/test_drdid_smoke_panel.py` を実施（`28 passed`）。
- `uv run ruff check src/veldra/modeling/_cv_runner.py src/veldra/modeling/regression.py src/veldra/modeling/binary.py src/veldra/modeling/multiclass.py src/veldra/modeling/frontier.py tests/test_phase267_output_parity.py` を通過。
- `uv run pytest -q tests/test_regression_internal.py tests/test_binary_internal.py tests/test_multiclass_internal.py tests/test_frontier_internal.py tests/test_observation_table.py tests/test_training_history.py tests/test_top_k_precision.py tests/test_num_boost_round.py tests/test_binary_class_weight.py tests/test_multiclass_class_weight.py tests/test_phase267_output_parity.py tests/test_early_stopping_validation.py` を実施（`48 passed`）。
- `uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施（`704 passed, 11 deselected`）。
