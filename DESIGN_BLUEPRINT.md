# DESIGN_BLUEPRINT

最終更新: 2026-02-12

## 1. 目的
VeldraML は、LightGBM ベースの分析機能を RunConfig 駆動で統一的に実行するためのライブラリです。対象領域は以下です。
- 学習: `fit`
- 推論: `predict`
- 評価: `evaluate`
- 最適化: `tune`
- シミュレーション: `simulate`
- 配布: `export`
- 因果推論: `estimate_dr`

## 2. 設計原則
1. Core は GUI/CLI/API を知らない。
2. 共通入口は RunConfig。
3. Artifact を受け渡し単位にする。
4. Stable API (`veldra.api.*`) のシグネチャを壊さない。
5. 再現性を優先する（seed/split/config/schema/version を保存）。
6. 仕様未確定は仮実装しない（provisional -> confirmed の履歴運用）。

## 3. 現在の実装能力（Current State）

### 3.1 API x Task Capability Matrix
| API | regression | binary | multiclass | frontier | 備考 |
| --- | --- | --- | --- | --- | --- |
| `fit` | Yes | Yes | Yes | Yes | CV 学習 + Artifact 保存 |
| `predict` | Yes | Yes | Yes | Yes | Artifact 経由 |
| `evaluate(artifact, data)` | Yes | Yes | Yes | Yes | 指標は task ごとに定義済み |
| `evaluate(config, data)` | Yes | Yes | Yes | Yes | 一時学習で評価（永続化なし） |
| `tune` | Yes | Yes | Yes | Yes | Optuna/Resume/SQLite 対応 |
| `simulate` | Yes | Yes | Yes | Yes | DSL: `set/add/mul/clip` |
| `export(format="python")` | Yes | Yes | Yes | Yes | validation report 生成 |
| `export(format="onnx")` | Yes | Yes | Yes | Yes | optional dependency |
| `estimate_dr` | Yes | Yes | No | No | `causal.method=dr` |
| `estimate_dr (dr_did)` | Yes | Yes | N/A | N/A | 2時点 `panel/repeated_cross_section` |

### 3.2 タスク別の主要契約
- regression:
  - predict: `np.ndarray`
  - evaluate: `rmse`, `mae`, `r2`
- binary:
  - predict: `p_cal`, `p_raw`, `label_pred`
  - evaluate: `auc`, `logloss`, `brier`, `accuracy`, `f1`, `precision`, `recall`, `threshold`
  - OOF 確率校正（既定: platt）
- multiclass:
  - predict: `label_pred`, `proba_<class>`
  - evaluate: `accuracy`, `macro_f1`, `logloss`
- frontier:
  - predict: `frontier_pred`（target があれば `u_hat`）
  - evaluate: `pinball`, `mae`, `mean_u_hat`, `coverage`

### 3.3 因果推論の実装範囲
- DR (`causal.method="dr"`): ATT 既定、ATE 任意
- DR-DiD (`causal.method="dr_did"`): 2時点 MVP
  - `design="panel"`
  - `design="repeated_cross_section"`
  - `task.type="binary"` は Risk Difference ATT で解釈（estimand は ATT のみ）
- propensity は校正後確率（既定: `platt`）を使用

## 4. 実装済みフェーズ要約
- Phase 1-4: 基本 API, Artifact, regression + examples
- Phase 5-7: binary/multiclass + OOF 校正 + threshold opt-in
- Phase 8-12: tune/simulate/export/frontier/timeseries 拡張
- Phase 13-18: ONNX 拡張, evaluate(config, data), notebook整備
- Phase 19-20: DR, DR-DiD, causal tuning objective
- Phase 21: Dash GUI MVP（Config編集 + Run実行 + Artifact閲覧）
- Phase 22: Config migration utility MVP（`veldra config migrate`）
- Phase 25: GUI運用強化（async job queue + config migrate統合）

## 5. 未実装ギャップ（優先度付き）

### P0 (スキップした機能)
0. ONNX graph optimization
   - 本件は「当面スキップ（非優先）」とする。
   - 既存の dynamic quantization は opt-in で利用可能。

### P1（次に着手すべき）
1. Causal 高度化
   - multi-period / staggered adoption 対応。

### 実装済み（新規）
- Config migration utility（MVP）
  - Python API: `migrate_run_config_payload`, `migrate_run_config_file`
  - CLI: `veldra config migrate --input ... [--output ...]`
  - v1正規化のみ対応、上書き禁止、未知キーは明示エラー

## 6. GUI実装（Dash, 運用強化MVP）

### 6.0 実装状況
- Phase 21 で Dash ベースの GUI アダプタを実装済み。
- Phase 25 で運用強化MVPを追加:
  - 非同期ジョブキュー（SQLite永続、single worker）
  - best-effort cancel（queuedは即時cancel、runningはcancel_requested）
  - Config migrate統合（preview/diff/apply）
- GUI は adapter 層として `veldra.api.runner` を呼び出すのみで、Coreロジックは保持しない。

### 6.1 目的
- RunConfig 共通入口原則を GUI でも成立させる。
- GUI 層は `veldra.api.runner` を呼ぶだけにして Core 非依存を維持する。

### 6.2 MVP 機能
1. Config Editor
   - RunConfig 作成
   - バリデーション
   - YAML 保存/読込
2. Run Console
   - 実行対象: `fit`, `evaluate`, `tune`, `simulate`, `export`, `estimate_dr`
   - 非同期 enqueue / polling / cancel / history
   - 実行ログ表示
3. Artifact Explorer
   - Artifact 一覧
   - metrics 表示
   - predict/evaluate 再実行

### 6.3 画面構成
- `/config`
- `/run`
- `/artifacts`

### 6.4 非機能要件
- 単一ユーザー/ローカル実行を前提
- 構造化ログを画面に表示（`run_id`, `artifact_path`, `task_type`）
- ジョブ履歴は `VELDRA_GUI_JOB_DB_PATH`（既定 `.veldra_gui/jobs.sqlite3`）で永続化

### 6.5 非スコープ（MVPではやらない）
- 認証・権限管理
- 分散実行
- ジョブキュー高度化

### 6.6 次段階（GUI）
- ジョブ優先度と並列worker（現状はsingle worker固定）
- 実行中ジョブの協調キャンセル改善（action単位）
- 可視化コンポーネントの強化（KPIカード、時系列比較）

## 7. 将来の拡張ポイント
1. Simulation DSL の高度化
   - 現在は `set/add/mul/clip` のみ。
   - `allocate_total`、制約最適化（group/period など）は未実装。
2. Binary 閾値最適化の高度化
   - 現在は F1 固定。
   - cost/constraint ベース最適化は未実装。
3. Export 配布要素の拡張
   - Python export の Dockerfile 自動生成などは未実装。

## 8. 公開API互換ポリシー
- 既存の公開シグネチャは維持する。
- 新機能は原則 opt-in で追加する。
- 破壊的変更が必要な場合は config_version と migration 方針を必須化する。

## 9. HISTORYとの関係
- 詳細な意思決定と時系列ログは `HISTORY.md` を正とする。
- 本書は「現時点の設計状態」を示す要約ドキュメントとして維持する。

## 10. Phase 23（DR-DiD Binary + 最小診断）
- Proposal:
  - `causal.method="dr_did"` の `task.type="binary"` を追加し、効果を Risk Difference ATT として扱う。
  - 診断として `overlap_metric`, `smd_max_unweighted`, `smd_max_weighted` を返却契約へ追加する。
- Implemented:
  - `estimate_dr` の DR-DiD 経路は `regression|binary` を許可。
  - panel/repeated_cross_section の両設計で binary DR-DiD を実行可能化。
  - `CausalResult.metadata` に `outcome_scale` / `binary_outcome` を追加。

## 11. Phase 24（Causal Tune Balance-Priority）
- Proposal:
  - DR/DR-DiD の因果チューニングを、SE中心から balance-priority へ拡張する。
  - 既定 objective を `dr_balance_priority` / `drdid_balance_priority` に変更する。
  - balance判定は `smd_max_weighted <= causal_balance_threshold`（既定 0.10）を利用する。
- Implemented:
  - `tuning.objective` に balance-priority objective を追加。
  - `tuning.causal_balance_threshold` を追加（causal時のみカスタム可）。
  - DR/DR-DiD の metrics/summary で `overlap_metric`, `smd_max_unweighted`,
    `smd_max_weighted` を統一返却。
  - tune trial attributes に balance violation と objective stage を保存。

## 12. Phase 25（完了）: GUI運用強化

### 完了内容
- 非同期ジョブ実行（SQLite永続 + single worker + best-effort cancel）
- `/config` への config migrate 統合（preview/diff/apply）
- GUI callback の堅牢化
  - uploadデータのbase64デコード例外をユーザー向けエラーに変換
  - callback外実行でも破綻しない互換フォールバックを追加
  - 旧テスト契約（引数/戻り値）との後方互換を維持

### 検証結果
- `pytest -q`: **405 passed, 0 failed**
- GUI主要フロー（Data → Config → Run → Results）で callback 破綻が発生しないことを確認

### Notes
- `ruff check` はリポジトリ全体で既存違反が残っており、Phase25スコープ外として別途整理する。

## 12.5 Phase25.5: テスト改善計画（DRY / 対称性 / API化）

### Context（2026-02-14 時点）
- テストスイートは約145ファイル。
- データ生成ロジック（`_binary_frame` など）が42ファイルに分散し、関連ヘルパー定義は56箇所ある。
- タスク別の契約テストは regression だけ `fit_smoke / predict_contract / evaluate_metrics / artifact_roundtrip` が不足している。
- `*_internal` テストの一部が private関数に直結し、リファクタリング耐性を下げている。

### 目的
- DRY原則に沿ってテストデータ生成を共通化する。
- task間で同じ種類の契約テストを揃える。
- private実装依存のテストを公開ユーティリティ/公開API検証へ移す。

### Phase 1: データ生成ロジックの共通化（DRY）
- `tests/conftest.py` に以下を追加する。
- `binary_frame(rows, seed, coef1, coef2, noise)`
- `multiclass_frame(rows_per_class, seed, scale)`
- `regression_frame(rows, seed, coef1, coef2, noise)`
- `frontier_frame(rows, seed)`
- `panel_frame(n_units, seed)`
- `config_payload(task_type, **overrides)`
- `FakeBooster`
- 42ファイルをWave方式で段階移行する。
- Wave1: smoke/contract系 + internal系（優先）
- Wave2: tune/simulate/export 系
- Wave3: examples/補助テスト系
- 再発防止として、対象Waveでローカル `def _*frame` を禁止する契約テストを追加する。

### Phase 2: テスト対称性の確保（regression補完）
- 新規作成:
- `tests/test_regression_fit_smoke.py`
- `tests/test_regression_predict_contract.py`
- `tests/test_regression_evaluate_metrics.py`
- `tests/test_regression_artifact_roundtrip.py`
- 既存の binary/frontier 契約テストをテンプレートにし、以下を検証する。
- fitでartifactと主要ファイルが生成されること
- predict契約（出力shape、特徴量順序、欠損特徴量エラー）
- evaluate契約（`rmse`, `mae`, `r2`）
- save/load往復で予測整合と `feature_schema` 維持

### Phase 3: internalテストの公式API化
- 3.1 CV splitユーティリティの公開化。
- 新規: `src/veldra/split/cv.py` に `iter_cv_splits(config, data, x, y=None)` を追加
- `src/veldra/split/__init__.py` でexport
- `binary/multiclass/regression/frontier` の private `_iter_cv_splits` を削除し公開関数利用へ置換
- 新規: `tests/test_split_cv.py`
- 3.2 因果診断メトリクスの公開化。
- 新規: `src/veldra/causal/diagnostics.py`
- 公開関数: `overlap_metric`, `max_standardized_mean_difference`
- `src/veldra/causal/__init__.py` でexport
- `dr.py` と `dr_did.py` の重複実装を当該公開関数へ置換
- 新規: `tests/test_causal_diagnostics.py`
- `tests/test_drdid_internal.py` の該当private依存テストを置換
- 3.3 private維持方針。
- 公開化しない: `_train_single_booster`, `_to_python_scalar`, `_normalize_proba`
- 将来検討: `_default_search_space`

### 検証コマンド
- `uv run pytest tests -x --tb=short`
- `uv run pytest tests/test_binary_fit_smoke.py tests/test_multiclass_fit_smoke.py -v`
- `uv run pytest tests/test_regression_fit_smoke.py tests/test_regression_predict_contract.py tests/test_regression_evaluate_metrics.py tests/test_regression_artifact_roundtrip.py -v`
- `uv run pytest tests/test_split_cv.py tests/test_causal_diagnostics.py -v`
- `uv run pytest tests/test_binary_internal.py tests/test_regression_internal.py -v`
- `uv run ruff check src/veldra/split src/veldra/causal tests`

### 完了条件
- regressionの契約テスト4種が追加され、task間の対称性ギャップが解消される。
- 優先Waveの重複データ生成ロジックがconftestファクトリーへ移行される。
- CV split/causal diagnostics が公開ユーティリティとしてテストされる。
- Stable API（`veldra.api.*`）の互換性は維持される。

## 12.6 Phase25.6: GUI UXポリッシュ（CSS/HTML限定）

### 目的
- ダークテーマ上の可読性を改善する。
- データプレビューの操作性（縦スクロール時の列ヘッダー参照性）を改善する。
- Configページ内の主要導線を視覚的に統一し、操作迷いを減らす。

### 非スコープ
- 機能追加（新しいコールバック、データ処理ロジック、永続化仕様の変更）。
- 公開API/Artifact/RunConfig契約の変更。

### 主要成果物
- `text-muted-readable` クラス導入（`#cbd5e1`）による補助テキストのコントラスト改善。
- データプレビュー表の `thead th` sticky固定（インラインstyle + CSS fallback）。
- `ID Columns (Optional - for Group K-Fold)` の表示を削除（内部互換のため関連IDは非表示維持）。
- Export preset (`cfg-export-dir-preset`) の初期選択を `artifacts` に固定。
- ワークフロー進行ボタン色を `primary` に統一。
- ステッパー活性状態に glow、完了済みコネクタに成功色を反映。
- `Split Type=Time Series` 時に `Time Column` 必須であることを GUI 上で明示する警告表示を追加。
- Data Settings から `Categorical Columns (Optional override)` を非表示化し、導線を簡素化。

### 検証コマンド
- `uv run pytest tests/test_gui_app_callbacks_internal.py tests/test_gui_app_pure_callbacks.py tests/test_gui_app_job_flow.py -v`
- `uv run pytest tests/test_gui_pages_logic.py tests/test_gui_pages_and_init.py tests/test_gui_app_callbacks_config.py tests/test_gui_app_additional_branches.py -v`
- `uv run pytest tests -x --tb=short`

## 13 Phase 26: ジョブキュー強化 & 優先度システム

**目的:** 優先度ベースのジョブスケジューリングと並列worker実行により、スループットとユーザー制御を向上させる。

---

### 主要成果物
* `GuiJobRecord` に `priority` フィールド追加（high/normal/low、既定 normal）
* `GuiJobStore.claim_next_job()` で優先度考慮のクレーム処理
* 設定可能なworkerプール（`--worker-count` CLI引数、既定1で後方互換）
* Run ページでのジョブ優先度設定UI
* queuedジョブの並び替え機能
* Graceful shutdown対応のworkerプール管理

### 技術アプローチ
* **SQLiteスキーマ拡張:** `ALTER TABLE jobs ADD COLUMN priority INTEGER DEFAULT 50`
* **ロジック変更:** `claim_next_job()` を `ORDER BY priority DESC, created_at_utc ASC` に変更
* **新規クラス:** `GuiWorkerPool` クラス作成（複数 `GuiWorker` インスタンス管理）
* **排他制御:** データベースロックでthread-safeなジョブクレーム調整
* **UI変更:** Run ページに優先度ドロップダウン追加（Low/Normal/High）

### テスト要件
1.  優先度ベースのクレームロジック単体テスト
2.  マルチworker並行実行統合テスト
3.  ジョブクレーム競合条件テスト
4.  優先度オーバーライドテスト

### 成功基準
* 複数workerが競合なく並行実行可能
* 高優先度ジョブが低優先度より先にクレームされる
* Worker数が設定可能でGraceful shutdownが動作
* 並行負荷下でジョブロスや破損が発生しない

---

### 対象ファイル
* `src/veldra/gui/job_store.py` - 優先度フィールド、優先度考慮クレーム
* `src/veldra/gui/worker.py` - Workerプール管理実装
* `src/veldra/gui/types.py` - RunInvocation/GuiJobRecord拡張
* `src/veldra/gui/pages/run_page.py` - 優先度選択UI追加
* `src/veldra/gui/server.py` - worker-count設定

## 14 Phase 27: リアルタイム進捗追跡 & ストリーミングログ

**目的:** ジョブ実行中のリアルタイムフィードバックとして、進捗インジケーターとストリーミングログを提供する。

---

### 主要成果物
* **フィールド追加:** Job storeに `progress_pct` と `current_step` フィールドを追加
* **進捗管理:** Runner層での進捗コールバック機構の実装
* **ログ管理:** ストリーミングログのキャプチャと永続化
* **UIコンポーネント:** 進捗バー付きリアルタイム進捗UIの提供
* **ビューア:** Job詳細パネルに展開可能なログビューアを追加

### 技術アプローチ
* **スキーマ拡張:** `GuiJobRecord` に `progress_pct: float`, `current_step: str | None` を追加
* **メソッド追加:** `GuiJobStore.update_progress(job_id, pct, step)` を実装
* **コンテキスト管理:** `run_action()` 内に進捗追跡用コンテキストマネージャを作成
* **ログ永続化:** 実行中の構造化ログをキャプチャし、`job_logs` テーブルに保存
* **ポーリング:** Dash `dcc.Interval` を使用して実行中ジョブの進捗更新を監視
* **UI実装:** シンタックスハイライト付きの折りたたみ式ログビューアを作成

### テスト要件
1.  **アトミック性:** 進捗更新処理のトランザクション/アトミック性テスト
2.  **パフォーマンス:** ログストリーミング時のシステム負荷テスト
3.  **応答性:** 大量（1万行超）のログ出力時におけるUIの動作検証
4.  **整合性:** ジョブ状態と進捗表示の同期テスト

### 成功基準
* ユーザーがリアルタイムで進捗率（％）と現在のステップを確認できること
* ログ出力がジョブ実行をブロック（遅延）させずにUIへ反映されること
* 進捗更新がアトミックに実行され、ジョブ状態の破損が発生しないこと
* ログビューアが10k行以上のデータでもパフォーマンス劣化なく動作すること

---

### 対象ファイル
* `src/veldra/gui/job_store.py` - 進捗追跡スキーマと更新メソッド
* `src/veldra/gui/services.py` - `run_action` 内の進捗コールバック実装
* `src/veldra/gui/pages/run_page.py` - 進捗バーコンポーネントの組み込み
* `src/veldra/gui/app.py` - 進捗ポーリング用コールバックの追加
* `src/veldra/gui/components/` - 新規 `progress_viewer` コンポーネントの作成

## 15 Phase 28: キャンセル強化 & エラーリカバリ

**目的:** Action単位の協調キャンセル改善と、失敗ジョブのリトライ機構を追加することで、システムの堅牢性と操作性を向上させる。

---

### 主要成果物
* **キャンセル処理:** 長時間実行操作へのキャンセルチェックポイント実装
* **ポリシー設定:** リトライポリシー（最大リトライ数、バックオフ）の設定機能
* **自動リトライ:** 一時的な失敗に対する自動リトライ機構
* **手動操作:** 失敗ジョブに対する「手動リトライ」ボタンのUI追加
* **診断機能:** 実用的な診断提案（Next Step）付きのエラーメッセージ改善

### 技術アプローチ
* **チェックポイント:** CVループ、tuningトライアル、データロード内に `check_cancellation()` 呼び出しを挿入
* **スキーマ拡張:** `RunInvocation` に `RetryPolicy`（max_retries, retry_on）を追加、`GuiJobRecord` に `retry_count` フィールドを追加
* **スケジューリング:** Worker内に「指数バックオフ」を用いたリトライスケジューラを実装
* **エラー解析:** 一般的な問題（データ未検出、メモリエラー等）のパターンマッチング処理
* **UI/UX:** エラーメッセージに具体的な診断ヒントと解決策を追加

### テスト要件
1.  **レスポンス:** キャンセル要求から停止までの応答時間テスト（目標 5秒以内）
2.  **ロジック検証:** 指数バックオフを含むリトライロジックの正確性テスト
3.  **エラー分類:** 「一時的失敗」と「恒久的失敗」の分類ロジックの妥当性検証
4.  **整合性:** 手動リトライ時のジョブ状態遷移テスト

### 成功基準
* 実行中のジョブがキャンセルリクエストから5秒以内に正常に停止すること
* 一時的な失敗が設定された上限回数まで自動でリトライされること
* UIから失敗したジョブをワンクリックで手動リトライできること
* エラーメッセージに「次のステップ」を含む有用なヒントが表示されること

---

### 対象ファイル
* `src/veldra/gui/services.py` - `run_action` 内へのキャンセルチェックポイント挿入
* `src/veldra/gui/job_store.py` - リトライポリシー用フィールドおよび管理メソッドの実装
* `src/veldra/gui/worker.py` - 指数バックオフを伴うリトライスケジューラの実装
* `src/veldra/gui/types.py` - `RetryPolicy` データクラスの定義追加
* `src/veldra/api/runner.py` - キャンセルチェックフックの注入（共通パターン参照）

## 16 Phase 29: Config管理 & テンプレートライブラリ

**目的:** Config テンプレート、バリデーション、バージョン管理を追加し、Config作成の効率化と人為的ミスの削減を実現する。

---

### 主要成果物
* **テンプレート群:** 汎用タスク（regression, binary, causal, tuning）のプリセットライブラリ作成
* **バリデーション:** UI上でのジョブ投入前Configチェック機能
* **永続化:** ブラウザ `localStorage` を活用したConfigの保存・読込・クローン機能
* **比較ツール:** テンプレートカスタマイズ用の Config diff ビューアの実装
* **UX改善:** 初心者向けのクイックスタートウィザードの提供

### 技術アプローチ
* **テンプレート定義:** `src/veldra/gui/templates/` に YAML 形式でスキーマを定義
* **即時フィードバック:** ジョブ投入前に `validate_config()` を呼び出し、エラー箇所をハイライト表示
* **クライアント保存:** `localStorage` を使用し、ユーザーのConfig履歴（最新10件）を管理
* **UI統合:** Config ページにテンプレート選択ドロップダウンと YAML diff ビューアを実装
* **ウィザード実装:** マルチステップ形式（タスク選択 → データ指定 → 詳細設定）のコンポーネント構築

### テスト要件
1.  **整合性:** 提供する全テンプレートがバリデーションを通過することの確認
2.  **永続性:** ブラウザ再起動後も `localStorage` のデータが保持されているかのテスト
3.  **フロー検証:** ウィザード形式での入力からジョブ投入までのエンドツーエンドテスト
4.  **エッジケース:** 不正なYAML入力時のエラー表示とリカバリの検証

### 成功基準
* 5種類以上の本番利用可能な標準テンプレートが提供されていること
* Configエラーがジョブ投入前に明確な解決策（ガイダンス）と共に表示されること
* ユーザーが独自のカスタムConfigを保存し、いつでも再読込できること
* 新規ユーザーがウィザードを使用して2分以内に有効なConfigを完成させられること

---

### 対象ファイル
* `src/veldra/gui/templates/` - 新規ディレクトリ、各タスク用 YAML テンプレート
* `src/veldra/gui/pages/config_page.py` - バリデーション、テンプレート選択、ウィザードUIの実装
* `src/veldra/gui/app.py` - テンプレート読込およびバリデーション用コールバック
* `src/veldra/gui/components/` - 新規 `config_wizard` コンポーネントの作成

## 17 Phase 30: 高度可視化 & Artifact比較

**目的:** 時系列比較、因果診断、マルチArtifact分析を導入し、モデルの性能評価と意思決定の精度を向上させる。

---

### 主要成果物
* **時系列分析:** 学習履歴およびCV（交差検証）フォルドごとのパフォーマンス推移追跡
* **因果診断:** 因果推論結果の妥当性を評価する可視化（SMDプロット、オーバーラップメトリクス）
* **比較ビュー:** 複数（2個以上）のArtifactを並列で比較・分析するインターフェース
* **探索的分析:** インタラクティブな特徴量重要度（Feature Importance）のドリルダウン機能
* **レポート出力:** 外部共有可能な出版クオリティのPDF/HTMLレポート生成機能

### 技術アプローチ
* **データ拡張:** Artifact metadata に fold-level メトリクスを保存し、時系列プロットに活用
* **因果可視化:** Plotly をベースに SMD（Standardized Mean Difference）散布図を実装
* **比較ロジック:** 差分（デルタ）計算機能を備えたマルチArtifact比較テーブルの構築
* **UIインタラクション:** 特徴量クリックで詳細な分布を表示するドリルダウン機能を実装
* **エクスポート:** `jinja2` や `pdfkit` 等を用いた、構造化データのレポート出力エンジン実装

### テスト要件
1.  **レンダリング:** 追加された全チャートタイプ（SMD、時系列等）の描画テスト
2.  **比較正確性:** 複数Artifact間のメトリクス差分計算が数学的に正しいかの検証
3.  **エクスポート:** 生成されたPDF/HTMLレポートのレイアウト崩れおよびデータ整合性テスト
4.  **負荷検証:** 大規模な特徴量セットを持つArtifactのドリルダウン操作時のレスポンス検証

### 成功基準
* ユーザーがCVフォルド間の学習進捗やバラつきを視覚的に把握できること
* 因果推論において、SMDプロットとオーバーラップ診断による共変量バランスの確認ができること
* 最大5つのArtifactを同一画面上で並列比較できること
* 出力されたレポートが、そのままプレゼンや報告書として利用可能な品質であること

---

### 対象ファイル
* `src/veldra/gui/components/charts.py` - 時系列チャート、SMDプロットの新規実装
* `src/veldra/gui/pages/results_page.py` - Artifact比較UI、レポートエクスポート機能の統合
* `src/veldra/gui/app.py` - 比較ロジックおよびレポート生成用のコールバック実装
* `src/veldra/artifact/` - fold-levelメトリクス保存のためのスキーマ拡張（必要に応じて）

## 18 Phase 31: パフォーマンス最適化 & スケーラビリティ

**目的:** ページネーション、遅延読込、データベースチューニングを導入し、大規模データ運用時におけるGUIの応答性能とスケーラビリティを確保する。

---

### 主要成果物
* **ページネーション:** ジョブ履歴およびArtifactリストへの `OFFSET/LIMIT` ベースのページング追加
* **遅延読込:** 大規模データプレビューにおける仮想スクロール（Virtual Scrolling）の実装
* **DB最適化:** 頻出クエリに対するインデックスの再設計と最適化
* **データ管理:** ジョブ/Artifactの自動アーカイブおよびクリーンアップポリシーの実装
* **監視機能:** システムパフォーマンスモニタリングとスロークエリの分析機能

### 技術アプローチ
* **クエリ改善:** `list_jobs()` と `list_artifacts()` にページネーションロジックを統合
* **高速レンダリング:** Dash AG Grid を採用し、クライアント側のメモリ消費を抑えた仮想スクロールを実装
* **インデックス構築:** `(status, priority, created_at_utc)` 等の複合インデックスを作成し検索を高速化
* **ライフサイクル管理:** 完了から一定期間（例: 30日）経過したジョブを別テーブルへ移動するアーカイブ処理の実装
* **リソース管理:** SQLite 用のコネクションプーリングを導入し、並行アクセス時の効率を向上
* **プロファイリング:** 実行計画（EXPLAIN QUERY PLAN）を用いたクエリ最適化の実施

### テスト要件
1.  **高負荷テスト:** 10,000件以上のジョブおよび1,000件以上のArtifactが存在する環境でのレスポンス検証
2.  **ベンチマーク:** 主要なDB操作（検索、ソート、フィルタリング）の実行時間計測
3.  **データ整合性:** アーカイブおよびリストア（復元）処理におけるデータ破損の有無を確認
4.  **UI応答性:** 10万行を超えるデータセット表示時のスクロールの滑らかさとメモリ使用量の監視

### 成功基準
* ジョブ履歴が10,000件を超えてもUIがフリーズせず、軽快に動作すること
* データプレビューにおいて10万行以上のデータセットをスムーズに閲覧できること
* すべてのジョブリスト取得クエリが 100ms 以内に完了すること
* アーカイブ処理がバックグラウンドで実行され、UIの操作を妨げないこと

---

### 対象ファイル
* `src/veldra/gui/job_store.py` - ページネーション、インデックス、アーカイブロジックの実装
* `src/veldra/gui/pages/run_page.py` - ページネーション対応ジョブリストUIへの刷新
* `src/veldra/gui/pages/results_page.py` - ページネーション対応ArtifactリストUIへの刷新
* `src/veldra/gui/pages/data_page.py` - 仮想スクロールを用いたデータプレビュー機能の導入

## 19 Phase 32: 洗練 & プロダクション対応

**目的:** 最終的なシステムの洗練、ドキュメントの整備、およびプロダクション環境へのデプロイを容易にする機能の実装。

---

### 主要成果物
* **ドキュメント:** スクリーンショット付きのGUIユーザーガイド作成
* **モニタリング:** 運用監視用のヘルスチェックエンドポイント実装
* **堅牢性:** 欠落している依存関係（GPU/ONNX等）に対する優雅な退化（Graceful Degradation）の実装
* **コンテナ化:** 本番デプロイ用の Docker テンプレート一式の提供
* **分析機能:** 改善のための匿名利用テレメトリ（Opt-in方式）の導入
* **操作性向上:** パワーユーザー向けのキーボードショートカット実装

### 技術アプローチ
* **マニュアル作成:** 実際のワークフローを網羅したガイドを `/docs/gui_guide.md` に作成
* **監視API:** Workerの稼働状況とDB接続状態を返す `/health` エンドポイントを実装
* **例外処理:** オプショナルなライブラリが不在でも、基本機能を損なわずに警告を表示する処理を追加
* **デプロイ資産:** GUI環境を包含した `Dockerfile` および `docker-compose.yml` の作成
* **メトリクス:** ジョブ数やタスクタイプ等の匿名データを収集するテレメトリ機能の実装
* **UIショートカット:** `Ctrl + 1-4`（ページ移動）、`Ctrl + Enter`（ジョブ投入）などのナビゲーション実装

### テスト要件
1.  **ドキュメントレビュー:** 記述内容と最新のUI仕様に乖離がないかの完全性確認
2.  **ヘルスチェック:** 異常（DBダウン、Worker停止）を正確に検知できるかの信頼性テスト
3.  **スモークテスト:** Docker環境でコンテナ起動からジョブ完了まで一貫して動作するかの検証
4.  **互換性テスト:** 最小構成の依存関係環境でシステムがクラッシュせずに動作するかの確認

### 成功基準
* 新規ユーザーが外部の助けを借りず、ガイドのみで一連の解析ワークフローを完遂できること
* ヘルスエンドポイントがシステムの状態（正常・異常）を正確に報告すること
* Dockerデプロイが設定変更なしで即座に動作すること
* キーボードショートカットの導入により、パワーユーザーの操作効率が 20% 以上向上すること

---

### 対象ファイル
* `docs/gui_guide.md` - 新規ユーザーガイドドキュメントの作成
* `src/veldra/gui/server.py` - ヘルスチェック用 API エンドポイントの実装
* `Dockerfile`, `docker-compose.yml` - プロダクションデプロイ用設定ファイルの追加
* `src/veldra/gui/app.py` - キーボードショートカットのイベント処理およびテレメトリ送信ロジック

## 20 Phase 33: GUIメモリ再最適化 & テスト分離（提案）

### Proposal
- GUI adapter（`veldra.gui.app` / `veldra.gui.services`）で、重量級依存
  （`veldra.api.runner`, `veldra.api.artifact`, `veldra.data`）の eager import を再度廃止し、
  callback 実行時に遅延解決する。
- pytest 側で GUI テストを `gui` marker で分離し、coverage 実行を
  `core` と `gui` の2段階で実行可能にする。

### 理由
- GUI import だけで高いRSSを消費すると、`coverage run -m pytest` 実行時に
  環境次第でOOMに到達しやすくなる。
- テストを論理分離することで、メモリ制約環境でも再現性を維持した運用がしやすくなる。

### 互換性方針
- `veldra.api.*` の公開シグネチャは変更しない。
- GUI機能契約（callback I/O、RunConfig共通入口）は維持する。
