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
