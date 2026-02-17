# DESIGN_BLUEPRINT

最終更新: 2026-02-17

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
- Phase 26: UX/UI改善（7画面再構成 + Export + Learning Curves）
- Phase 26.1: UI改善（バグ修正3件 + ユースケース駆動UI再構成設計）← **実装中**
- Phase 26.2: ユースケース駆動UI改善（ヘルプUI基盤 + 画面別ガイド強化）
- Phase 26.3: ユースケース詳細化（diagnostics ライブラリ + observation_table + Notebook 完全版 + 実行証跡）← **完了**
- Phase 26.4: Notebook 教育化 & テスト品質強化 ← **計画策定済み**

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

### 要約
- 非同期ジョブ実行基盤を GUI に導入（SQLite 永続 + single worker + best-effort cancel）。
- `/config` に config migrate（preview/diff/apply）を統合。
- GUI callback の堅牢化を実施（アップロード例外のユーザー向け変換、callback 外フォールバック、旧テスト契約互換維持）。

### 検証結果
- `pytest -q`: **405 passed, 0 failed**
- Data → Config → Run → Results の主要導線で callback 破綻なし。

### Notes
- `ruff check` の既存違反は Phase25 スコープ外として別管理。

## 12.5 Phase25.5: テスト改善計画（DRY / 対称性 / API化）(完了)

### 要約
- 課題: テストデータ生成の重複（42ファイル）、regression 契約テスト不足、private 関数依存テストの存在。
- 実施:
  - `tests/conftest.py` に共通ファクトリー群（`binary_frame` / `multiclass_frame` / `regression_frame` / `frontier_frame` / `panel_frame` / `config_payload` / `FakeBooster`）を追加し、Wave 方式で移行。
  - regression の契約テスト 4 種を補完（fit/predict/evaluate/artifact roundtrip）。
  - 公開ユーティリティ化:
    - `src/veldra/split/cv.py`: `iter_cv_splits`
    - `src/veldra/causal/diagnostics.py`: `overlap_metric`, `max_standardized_mean_difference`
  - task 実装の private `_iter_cv_splits` と causal 内重複ロジックを公開関数利用へ置換。
- 非公開維持: `_train_single_booster`, `_to_python_scalar`, `_normalize_proba`。

### 完了条件の達成
- regression の契約対称性を解消。
- DRY 化と公開 API 寄せを完了。
- Stable API（`veldra.api.*`）互換性を維持。

## 12.6 Phase25.6: GUI UXポリッシュ（CSS/HTML限定）(完了)

### 要約
- スコープを CSS/HTML に限定し、機能追加や API/Artifact/RunConfig 契約変更は未実施。
- 主な改善:
  - 可読性改善（`text-muted-readable`）。
  - データプレビューのヘッダー sticky 化。
  - `Split Type=Time Series` 時の `Time Column` 必須警告を明示。
  - Data Settings の簡素化（Categorical Columns 非表示化）。
  - 不要表示削除（`ID Columns (Optional - for Group K-Fold)` を非表示維持）。
  - workflow ボタン色/ステッパー状態の視覚統一、export preset 初期値を `artifacts` に固定。

### 検証
- GUI 関連テスト群および全体スモーク（`uv run pytest tests -x --tb=short`）で確認。

## 12.7 Phase25.7: LightGBMの機能強化（完了・検証済み）

### 目的
- 学習契約を RunConfig 駆動で強化（`num_boost_round`、Early Stopping 分割、class weight、自動 split、学習履歴保存、GUI/migrate 連携）。

### 実装要点（Step1-8 完了）
- `TrainConfig` 拡張:
  - `num_boost_round`, `early_stopping_validation_fraction`, `auto_class_weight`, `class_weight` を追加。
  - cross-field validation を追加（task 制約、競合指定、値域）。
- 学習ループ:
  - 全 task で `num_boost_round` を 300 固定から設定値駆動へ移行。
  - CV/最終学習ともに train 部分から ES 用 validation を自動分割（OOF を ES 監視に使わない）。
  - `timeseries` は末尾 N%、分類は stratified、回帰/frontier は shuffle split。
- クラス不均衡対応:
  - binary: `is_unbalance` 自動、手動 `class_weight` から `scale_pos_weight` 適用。
  - multiclass: 自動/手動 sample weight を `Dataset(weight=...)` へ反映。
- 学習履歴:
  - `record_evaluation` を保存し、Artifact に `training_history.json` を永続化。
- GUI + migration:
  - `Num Boost Round` 表示、`Auto Class Weight` / `Class Weight` 入力追加。
  - `lgb_params.n_estimators` → `train.num_boost_round` を migrate。

### 検証結果（2026-02-16）
- Phase25.7 関連: **31 passed, 0 failed**
- Stable API 互換性: 維持確認済み。

## 12.8 Phase25.8: LightGBMのパラメーター追加（実装・検証完了）

### 目的
- パラメーター拡張を RunConfig/GUI/Tuning/Evaluate/Artifact に一貫適用。

### 実装要点（Step1-9 完了）
- `TrainConfig` 拡張:
  - `auto_num_leaves`, `num_leaves_ratio`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`, `feature_weights`, `top_k` を追加。
  - 競合・値域・task 制約（`top_k` は binary のみ）を検証。
- modeling utils:
  - `resolve_auto_num_leaves`, `resolve_ratio_params`, `resolve_feature_weights` を追加し、全4 task 学習器へ適用。
- Binary Top-K:
  - `precision_at_k` を `feval` として fit へ統合。
  - tune objective（`precision_at_k`）と evaluate 返却、`training_history` 記録に連携。
- GUI 拡張:
  - Advanced Training Parameters に 25.8 項目を追加。
  - `auto_num_leaves=True` 時は YAML から `lgb_params.num_leaves` を除外。
  - binary tune objective 候補に `brier` / `precision_at_k` を追加。
- tuning search space:
  - standard に `lambda_l1`, `lambda_l2`, `path_smooth`, `min_gain_to_split` を追加。
- docs/運用:
  - RunConfig reference 生成スクリプトを更新し README へ反映。

### Decision（provisional）
- `train.top_k` 指定時、ES 監視は `precision_at_{k}` 優先。
- `train.feature_weights` は未知特徴量キーを許容せず、学習前に validation error。

### 検証結果（2026-02-16）
- `uv run ruff check .`: passed
- 主要テスト群（config/lgb resolver/top_k/tuning/evaluate/gui）通過。
- `uv run pytest -q -m "not gui"`: **385 passed**
- `uv run pytest -q -m "gui"`: **100 passed**

## 12.9 Phase25.9: LightGBM機能強化の不足テスト補完（完了）

### 目的
- Phase25.7/25.8 の未カバー領域をテストで閉じ、必要時は同フェーズ内で最小本体修正まで実施。

### 実装方針（固定）
- 不足テスト追加で差分が出た場合は最小本体修正を同時適用。
- Causal は `split.group_col` に加えて `causal.unit_id_col` 経路を検証。
- `best_iteration` は monkeypatch 契約検証を主方式にして CI 安定性を優先。

### 追加/更新テスト
- 新規:
  - `tests/test_auto_num_leaves.py`
  - `tests/test_ratio_params.py`
  - `tests/test_feature_weights.py`
  - `tests/test_tuning_search_space.py`
  - `tests/test_objective_override.py`
  - `tests/test_artifact_param_roundtrip.py`
- 既存拡張:
  - `tests/test_num_boost_round.py`（既定値 300 後方互換）
  - `tests/test_early_stopping_validation.py`（`best_iteration` 記録契約）
  - `tests/test_dr_internal.py`（`unit_id_col` 経路の GroupKFold / fallback）

### 最小本体修正
- `src/veldra/modeling/regression.py`
- `src/veldra/modeling/binary.py`
- `src/veldra/modeling/multiclass.py`
- `src/veldra/modeling/frontier.py`
- `feature_weights` 指定時に `params["feature_pre_filter"] = False` を付与し、適用契約を固定。

### Decision（confirmed）
- 不足テスト + 必要時の最小本体修正を同一フェーズで閉じる方針を確定。
- Causal GroupKFold は `group_col` 維持 + `unit_id_col` 補完で確定。
- 早期停止 `best_iteration` は monkeypatch 契約検証を正式方式として確定。

### 検証結果（2026-02-16）
- 追加テスト群: **6 passed / 4 passed / 26 passed**（3バッチ）
- `uv run ruff check .`: passed
- `uv run pytest -q -m "not gui"`: **399 passed, 100 deselected**

## 13 Phase 26: UX/UI 改善

### 目的
- 初学者が Data 取り込みから学習・評価・出力までを迷わず完遂できる GUI を提供する。
- GUI を RunConfig / Artifact 中心の運用に揃え、比較・再評価・エクスポートの実務導線を強化する。

### 実装固定方針（confirmed, 2026-02-16）
- ロールアウトは Stage A/B/C の段階導入とする。
- Export は Excel/HTML を標準導線にする（SHAP は `export-report` extra 導入時のみ有効）。
- `/config` は 1 フェーズ互換を維持しつつ、主導線は `/target` へ移行する。

### 実装結果（要約）
- 画面構成を `Data -> Target -> Validation -> Train -> Run -> Results` + `Runs/Compare` に再編。
- `workflow-state` 拡張 + `_build_config_from_state()` により、GUI 入力から RunConfig を再構築可能にした。
- Results を `Overview / Feature Importance / Learning Curves / Config` に拡張した。
- Export を非同期ジョブ（`export_excel` / `export_html_report`）化し、Results から実行可能にした。
- ガードレール表示を各画面に統合し、実行前診断を強化した。
- Stable API (`veldra.api.*`) と RunConfig/Artifact 契約は維持した。

### 非スコープ
- 特徴量エンジニアリング自動化、認証/権限管理、分散実行。

### 完了条件（要約）
1. 6段階メインフロー + Runs/Compare 導線が機能すること。
2. 主要ユースケース（学習、チューニング、因果推論、再評価、export）を GUI で完遂できること。
3. 互換性（Stable API / RunConfig / Artifact）を維持すること。

## 13.1 Phase 26.1: UI改善

### 目的
- Phase26 の安定化として、先行バグ修正とユースケース駆動の再構成設計を完了する。

### 実装固定方針（confirmed, 2026-02-17）
- Stage 1（バグ修正）と Stage 2（設計整理）の 2 段階で進める。

### Stage 1: バグ修正（完了）
- JST 表示統一:
  - ユーザー向け時刻表示と export 出力名タイムスタンプを JST で統一。
- Export ダウンロード導線:
  - Results に `dcc.Download` ベースのブラウザダウンロードを追加（HTML/Excel）。
- Learning Curves 取得ロジック:
  - `training_history` を Artifact オブジェクトから直接参照する実装へ修正。

### Stage 2: ユースケース駆動設計（完了）
- UC マトリクス、優先度付きギャップ（P0-P2）、26.2 実装計画を確定。
- 重点ギャップ:
  - task/split/objective の説明不足
  - causal 入力ガイド不足
  - run action 手動切替導線不足
  - frontier alpha / results precheck / runs-results ショートカット不足

### 26.2 への引き継ぎ
- 26.2 は Stage 1 の修正を前提に、GUI adapter のみで Step0-5 を実装する。
- Core/API は非変更を維持する。

## 13.2 Phase 26.2: ユースケース駆動UI改善

### 目的
- 26.1 Stage2 で特定したギャップを解消し、UC-1〜UC-10 の操作完遂性を高める。
- Core/API を変更せず、GUI adapter 層の改善に限定する。

### 前提
- 26.1 Stage1 完了（JST / Export DL / Learning Curves 修正済み）。

### 実装固定方針（confirmed, 2026-02-17）
- Step0（Notebook 監査）-> Step1〜5（GUI 改修）の順で実施する。

### 実装結果（要約）
- Step0: 監査基盤
  - `notebooks/phase26_2_ux_audit.ipynb` で UC 監査枠組みを整備。
- Step1: 共通ヘルプUI
  - `help_ui.py` / `help_texts.py` を追加し、説明表示の再利用基盤を導入。
- Step2: Target 強化
  - task/causal/frontier ガイドと causal 必須入力チェックを追加。
- Step3: Validation 強化
  - split 比較ガイド、timeseries 補助説明、推奨バッジを追加。
- Step4: Train 強化
  - 主要パラメータ help、Conservative/Balanced プリセット、objective 説明カードを追加。
- Step5: Run/Runs/Results 導線強化
  - run action manual override、pre-run guardrail 可視化、runs shortcut、results 再評価前 precheck を追加。

### 補正フェーズ（2026-02-17）
- 完了基準を「テンプレート整備」から「実行証跡 + parity 検証」へ補正した。
- 追加成果物:
  - UC別 Notebook 10 本（`phase26_2_uc01`〜`phase26_2_uc10`）
  - 実行証跡 `notebooks/phase26_2_execution_manifest.json`
  - Notebook 契約テスト（構造/証跡/パス）
  - Playwright E2E（UC-1〜UC-10）と `gui_e2e` / `gui_smoke` marker
  - 同等性レポート `docs/phase26_2_parity_report.md`

### 完了条件（要約）
1. Step0-5 の GUI 改修が反映され、UC-1〜UC-10 の到達導線が確認できること。
2. Notebook 実行証跡と parity レポートが更新されていること。
3. Stable API / RunConfig / Artifact 契約を維持していること。

### 運用メモ
- Phase26〜26.2 の詳細な時系列・判断理由・テスト実績は `HISTORY.md` を正とする。

## 13.3 Phase26.3: ユースケース詳細化

### 目的

Phase 26.2 で作成した骨格 Notebook（UC-1〜UC-10）を、実務レベルの診断・可視化・CSV 出力を含む完全版ユースケースに仕上げる。併せて、各 Notebook が依存する **診断計算ライブラリ** (`veldra.diagnostics`) を新設し、Notebook セルから 1 行で呼べる高レベル API を提供する。

### 前提

- SHAP 算出は LightGBM の **内蔵 SHAP**（`booster.predict(data, pred_contrib=True)`）を使用する。外部 `shap` ライブラリに依存しない。
- Feature Importance は `booster.feature_importance(importance_type='split' | 'gain')` で取得。
- Notebook はすべて **ヘッドレス実行可能**（`matplotlib.use('Agg')` + `plt.savefig`）とする。
- 可視化は `matplotlib` のみ。追加描画ライブラリに依存しない。

---

### 要件仕様

#### A. LightGBM 固定パラメーター

Notebook 内の学習設定は以下の値を使用する:

| パラメーター | 値 |
|---|---|
| `epochs` | 2000 |
| `patience` | 200 |
| `learning_rate` | 0.01 |
| `validation_ratio` | 0.2 |
| `max_bin` | 255 |
| `auto_num_leaves` | True |
| `num_leaves_ratio` | 1 |
| `min_data_in_leaf_ratio` | 0.01 |
| `min_data_in_bin_ratio` | 0.01 |
| `max_depth` | 10 |
| `feature_fraction` | 1 |
| `bagging_fraction` | 1 |
| `bagging_freq` | 0 |
| `lambda_l1` | 0 |
| `lambda_l2` | 0.000001 |
| `min_child_samples` | 20 |
| `first_metric_only` | True |

タスクタイプ別 metrics:

| タスクタイプ | metrics |
|---|---|
| Regression | `[rmse, mae]` |
| Binary | `[logloss, auc]` |
| Multiclass | `[multi_logloss, multi_error]` |

---

#### B. パラメーター最適化 Search Space

| パラメーター | 範囲 | 型 |
|---|---|---|
| `learning_rate` | 0.01〜0.1 | log uniform |
| `num_leaves_ratio` | 0.5〜1.0 | float |
| `validation_ratio` | 0.1〜0.3 | float |
| `max_bin` | 127〜255 | int |
| `min_data_in_leaf_ratio` | 0.01〜0.1 | float |
| `min_data_in_bin_ratio` | 0.01〜0.1 | float |
| `max_depth` | 3〜15 | int |
| `feature_fraction` | 0.5〜1.0 | float |
| `bagging_fraction` | 1.0（固定） | float |
| `bagging_freq` | 0（固定） | int |
| `lambda_l1` | 0（固定） | float |
| `lambda_l2` | 0.000001〜0.1 | float |

タスクタイプ別 tuning objective:

| タスクタイプ | objective |
|---|---|
| Regression | `[mape]` |
| Binary | `[brier]` |
| Multiclass | `[multi_logloss]` |
| Causal DR | `[dr_balance_priority, dr_std_error, dr_overlap_penalty]` |
| Causal DR-DiD | `[drdid_balance_priority, drdid_std_error, drdid_overlap_penalty]` |

タスクタイプ別 tuning metrics:

| タスクタイプ | metrics 候補 |
|---|---|
| Regression | `[rmse]`, `[huber]`, `[mae]` |
| Binary | `[logloss]`, `[auc]` |
| Multiclass | `[multi_logloss]`, `[multi_error]` |

---

#### C. タスクタイプ別 期待アウトプット

##### C-1. Regression 予測モデル

| # | アウトプット | 説明 |
|---|---|---|
| 1 | 誤差分布ヒストグラム | In-sample / Out-of-sample の残差分布を重ねて表示。過学習度合いの比較 |
| 2 | 評価指標テーブル | MAE, MAPE, RMSE, R² を In/Out 別に併記 |
| 3 | Feature Importance | Split / Gain の棒グラフ（上位 20 特徴量） |
| 4 | SHAP（全特徴量） | LightGBM 内蔵 SHAP。bee swarm 風の散布図 |
| 5 | 詳細テーブル（CSV） | 元データ + `fold_id` + `in_out_label` + `prediction` + `residual` |

##### C-2. Binary 予測モデル

| # | アウトプット | 説明 |
|---|---|---|
| 1 | ROC Chart | In-sample / Out-of-sample 別の ROC 曲線。過学習比較 |
| 2 | 評価指標テーブル | AUC, Brier, Average Precision, Logloss を In/Out 別に併記 |
| 3 | Lift Chart | OOF 予測のリフトカーブ（全体の予測力確認） |
| 4 | Feature Importance | Split / Gain の棒グラフ |
| 5 | SHAP（全特徴量） | LightGBM 内蔵 SHAP |
| 6 | 詳細テーブル（CSV） | 元データ + `fold_id` + `in_out_label` + `score` |

##### C-3. Multiclass 予測モデル

| # | アウトプット | 説明 |
|---|---|---|
| 1 | NLL ヒストグラム | サンプル別 Negative Log Likelihood を In/Out 別に比較 |
| 2 | 正解クラス確率ヒストグラム | p(true_class) を In/Out 別に比較 |
| 3 | 評価指標テーブル | 多クラスAUC, 多クラスBrier, Multi-logloss, Multi-error を In/Out 別に併記 |
| 4 | Feature Importance | Split / Gain の棒グラフ |
| 5 | SHAP | 最大確率ラベルについての SHAP（全特徴量） |
| 6 | 詳細テーブル（CSV） | 元データ + `fold_id` + `in_out_label` + クラス別スコア列 |

##### C-4. 時系列予測モデル

| # | アウトプット | 説明 |
|---|---|---|
| 1 | 時系列プロット（実測 vs 予測） | X軸: 時系列、Y軸: 目的変数 + 予測値。In/Out 境界を垂直線で明示 |
| 2 | 残差時系列プロット | X軸: 時系列、Y軸: 残差。In/Out 境界を明示 |
| 3 | 評価指標テーブル | MAE, MAPE, RMSE, R² を In/Out 別に併記 |
| 4 | Feature Importance | Split / Gain の棒グラフ |
| 5 | SHAP（全特徴量） | LightGBM 内蔵 SHAP |
| 6 | 詳細テーブル（CSV） | 元データ + `fold_id` + `in_out_label` + `prediction` + `residual` |

##### C-5. Frontier（分位点回帰）

前提:
- 分位点はユーザーで変更可能
- 特徴量に対して単調制約を設定できる
- Group CV ですべてのデータに対して OOF としての予測値が得られている
- Output Oriented の効率を算出: **相対到達度** `eff = y / q_hat_tau(x)`（1 に近いほど良い。1 超えはフロンティア超え扱い）

| # | アウトプット | 説明 |
|---|---|---|
| 1 | Pinball loss ヒストグラム | サンプル別 Pinball loss を In/Out 別に比較 |
| 2 | 評価指標テーブル | Pinball Loss, coverage, exceedance rate を In/Out 別に併記 |
| 3 | フロンティア散布図 | 予測値 vs 実測の散布図（45 度線 + フロンティア線） |
| 4 | Feature Importance | Split / Gain の棒グラフ |
| 5 | SHAP（全特徴量） | LightGBM 内蔵 SHAP |
| 6 | 詳細テーブル（CSV） | 元データ + `fold_id` + `prediction` + `efficiency` |

##### C-6. DR（Doubly Robust: ATE/ATT 推定）

前提:
- 推定対象: ATE / ATT（どちらか明示。両方出すなら両方）
- Cross-fitting（group K-fold）で nuisance を学習し、全データで OOF 予測を保持
- nuisance 構成: Propensity model `e(x) = P(D=1|X=x)`（キャリブレーション付き）、Outcome model `μ1(x), μ0(x)`
- 推定量: AIPW / DR（標準誤差は influence function ベース or ブートストラップ）

**最終推定（因果効果）:**

| # | アウトプット | 説明 |
|---|---|---|
| 1 | 推定サマリ | 推定値（ATE/ATT）、標準誤差、95%CI、p 値 |
| 2 | IF 分布ヒストグラム | Influence function / pseudo-outcome の分布（全体） |
| 3 | IF 外れ値一覧 | 上位 1% の IF 値を持つサンプルと、それらに多い特徴 |

**Overlap / 傾向スコア診断:**

| # | アウトプット | 説明 |
|---|---|---|
| 4 | 傾向スコア分布 | Treated / Control 別ヒストグラム |
| 5 | IPW 重み分布 | ATE/ATT 定義に応じた重み w のヒストグラム |
| 6 | Overlap 指標 | e(x) の min/max/分位点、極端値比率（<0.01, >0.99）、有効標本サイズ（ESS） |

**バランスチェック:**

| # | アウトプット | 説明 |
|---|---|---|
| 7 | Love plot | Unweighted vs Weighted の SMD（標準化差）比較 |
| 8 | SMD 要約 | 中央値/最大値、\|SMD\|>0.1 の割合 |

**nuisance モデル健全性:**

| # | アウトプット | 説明 |
|---|---|---|
| 9 | Propensity 診断 | InFold / OOF の ROC, AUC, Logloss, Brier, Average Precision |
| 10 | Outcome 診断 | InFold / OOF の誤差分布ヒストグラム + MAE, RMSE, R² |
| 11 | Feature Importance / SHAP | Propensity・Outcome 各モデルの Split/Gain + SHAP（全特徴量） |
| 12 | Overlap 崩壊警告 | AUC が極端に高い場合の注意喚起メッセージ |

**ロバストネス:**

| # | アウトプット | 説明 |
|---|---|---|
| 13 | トリミング比較 | 重みの 1%/99% クリッピングごとの推定結果比較表 |

**詳細テーブル（CSV）:**

| 列 | 説明 |
|---|---|
| 元データ全列 | 入力特徴量 |
| `fold_id` | CV fold 番号 |
| `D` | 処置フラグ |
| `Y` | アウトカム |
| `e_x` | OOF 傾向スコア |
| `mu1_x`, `mu0_x` | OOF outcome 予測 |
| `weight` | ATE/ATT 定義に応じた重み |
| `pseudo_outcome` | pseudo-outcome / IF 成分 |
| `trimmed` | トリミング適用フラグ |

##### C-7. DR-DiD（Doubly Robust Difference-in-Differences: ATT）

前提:
- 推定対象: ATT（Average Treatment effect on the Treated）
- データ構造: Panel / Repeated cross-section を明示
- 時点: Pre / Post を明示
- Cross-fitting（Group K-fold）で nuisance を学習し、全データで OOF 予測を保持
- nuisance: 傾向スコア `e(x) = P(D=1|X)`（キャリブレーション付き）、結果モデル `E[Y_t | D, X]` or `E[ΔY | D, X]`
- 標準誤差: クラスタロバスト（panel なら個体 ID）or ブートストラップ

**最終推定（ATT）:**

| # | アウトプット | 説明 |
|---|---|---|
| 1 | 推定サマリ | ATT 推定値、標準誤差、95%CI、p 値 |
| 2 | IF 分布ヒストグラム | DR-DiD の IF / pseudo-outcome 分布 |
| 3 | IF 外れ値一覧 | 上位 1% |

**DiD 前提（並行トレンド）診断:**

| # | アウトプット | 説明 |
|---|---|---|
| 4 | Placebo DID | Pre 期間のみの placebo DID（効果=0 を期待）。リードがゼロ付近か確認 |
| 5 | 平均推移プロット | Treated vs Control の平均アウトカム推移（Unweighted + Weighted） |

**Overlap / 傾向スコア診断:**

| # | アウトプット | 説明 |
|---|---|---|
| 6 | 傾向スコア分布 | Treated / Control 別ヒストグラム |
| 7 | 重み分布 | DR-DiD 定義に合わせた w のヒストグラム |
| 8 | Overlap 指標 | e(x) の min/max/分位点、極端値比率、ESS、極端重み比率（p99 超） |

**バランスチェック（Pre の X に対して）:**

| # | アウトプット | 説明 |
|---|---|---|
| 9 | Love plot | Unweighted vs Weighted の SMD 比較 |
| 10 | SMD 要約 | 中央値/最大値、\|SMD\|>0.1 の割合 |

**nuisance モデル健全性:**

| # | アウトプット | 説明 |
|---|---|---|
| 11 | Propensity 診断 | InFold / OOF の ROC, AUC, Logloss, Brier, Average Precision |
| 12 | Outcome 診断 | Pre/Post 別（or ΔY）の InFold / OOF 誤差分布 + MAE, RMSE, R² |
| 13 | Feature Importance / SHAP | Propensity・Outcome 各モデルの Split/Gain + SHAP |
| 14 | Overlap 崩壊警告 | AUC 極端時の注意喚起 |

**ロバストネス:**

| # | アウトプット | 説明 |
|---|---|---|
| 15 | トリミング比較 | 重みの 1%/99% クリッピングごとの ATT 比較表 |
| 16 | Placebo outcome（任意） | 影響しないはずの目的変数で効果ゼロ確認 |

**詳細テーブル（CSV）:**

| 列 | 説明 |
|---|---|
| 元データ全列 | 入力特徴量 |
| `fold_id` | CV fold 番号 |
| `unit_id`（Panel のみ） | 個体 ID |
| `time`, `pre_post` | 時点、Pre/Post フラグ |
| `D` | treated indicator |
| `Y` | アウトカム |
| `e_x` | OOF 傾向スコア |
| `mu_d_pre_x`, `mu_d_post_x` or `delta_mu_d_x` | OOF outcome nuisance |
| `weight` | DR-DiD 定義に応じた重み |
| `pseudo_outcome` | pseudo-outcome / IF 成分 |
| `trimmed` | トリミング適用フラグ |

---

### 実装ステップ

#### Step 1: 診断計算ライブラリ新設 (`veldra.diagnostics`)

**目的**: Notebook セルから 1 行で呼べる高レベル診断 API を提供する。

**新規ファイル**:

| ファイル | 内容 |
|---|---|
| `src/veldra/diagnostics/__init__.py` | 公開 API の re-export |
| `src/veldra/diagnostics/importance.py` | Feature importance（Split/Gain）取得ユーティリティ |
| `src/veldra/diagnostics/shap_native.py` | LightGBM 内蔵 SHAP の算出・整形 |
| `src/veldra/diagnostics/metrics.py` | In/Out 別メトリクス算出（Regression, Binary, Multiclass, Frontier, TimeSeries） |
| `src/veldra/diagnostics/plots.py` | matplotlib ベースの可視化関数群 |
| `src/veldra/diagnostics/tables.py` | 詳細テーブル（CSV）生成ユーティリティ |
| `src/veldra/diagnostics/causal_diag.py` | 因果推定固有の診断（IF 分布、Overlap、Balance、Love plot、トリミング比較） |

**主要 API**:

```python
# importance.py
def compute_importance(booster, importance_type='gain', top_n=20) -> pd.DataFrame

# shap_native.py
def compute_shap(booster, X: pd.DataFrame) -> pd.DataFrame
# Multiclass の場合: 最大確率ラベルの SHAP を返す
def compute_shap_multiclass(booster, X, predictions, n_classes) -> pd.DataFrame

# metrics.py
def regression_metrics(y_true, y_pred, label='overall') -> dict
def binary_metrics(y_true, y_score, label='overall') -> dict
def multiclass_metrics(y_true, y_proba, label='overall') -> dict
def frontier_metrics(y_true, y_pred, alpha, label='overall') -> dict
def split_in_out_metrics(metric_fn, y_true, y_pred, fold_ids, eval_fold_ids) -> pd.DataFrame

# plots.py
def plot_error_histogram(residuals_in, residuals_out, metrics_in, metrics_out, save_path)
def plot_roc_comparison(y_true_in, y_score_in, y_true_out, y_score_out, save_path)
def plot_lift_chart(y_true, y_score, save_path)
def plot_nll_histogram(nll_in, nll_out, save_path)
def plot_true_class_prob_histogram(prob_in, prob_out, save_path)
def plot_timeseries_prediction(time_index, y_true, y_pred, split_point, save_path)
def plot_timeseries_residual(time_index, residuals, split_point, save_path)
def plot_pinball_histogram(pinball_in, pinball_out, save_path)
def plot_frontier_scatter(y_true, y_pred, save_path)
def plot_feature_importance(importance_df, importance_type, save_path)
def plot_shap_summary(shap_df, X, save_path)

# causal_diag.py
def plot_propensity_distribution(propensity, treatment, save_path)
def plot_weight_distribution(weights, save_path)
def compute_overlap_stats(propensity, treatment) -> dict
def compute_balance_smd(covariates, treatment, weights=None) -> pd.DataFrame
def plot_love_plot(smd_unweighted, smd_weighted, save_path)
def compute_trimming_comparison(estimate_fn, observation_table, trim_levels=[0.01, 0.05]) -> pd.DataFrame
def plot_if_distribution(if_values, save_path)
def get_if_outliers(if_values, observation_table, percentile=99) -> pd.DataFrame
def plot_parallel_trends(means_treated, means_control, time_labels, save_path)

# tables.py
def build_regression_table(X, y, fold_ids, predictions, in_out_labels) -> pd.DataFrame
def build_binary_table(X, y, fold_ids, scores, in_out_labels) -> pd.DataFrame
def build_multiclass_table(X, y, fold_ids, class_probas, in_out_labels) -> pd.DataFrame
def build_frontier_table(X, y, fold_ids, predictions, efficiency) -> pd.DataFrame
def build_dr_table(observation_table) -> pd.DataFrame
def build_drdid_table(observation_table) -> pd.DataFrame
```

**テスト**: `tests/test_diagnostics_importance.py`, `tests/test_diagnostics_shap.py`, `tests/test_diagnostics_metrics.py`, `tests/test_diagnostics_plots.py`, `tests/test_diagnostics_tables.py`, `tests/test_diagnostics_causal.py`

---

#### Step 2: 既存モデリング層の拡張（CV 結果に In/Out 情報を保持）

**目的**: 各 `train_*_with_cv()` の返り値に、サンプル別の fold_id・in/out ラベル・予測値を含める。

**変更ファイル**:

| ファイル | 変更内容 |
|---|---|
| `src/veldra/modeling/regression.py` | `RegressionTrainingOutput` に `observation_table: pd.DataFrame` を追加（fold_id, in_out, prediction, residual） |
| `src/veldra/modeling/binary.py` | `BinaryTrainingOutput` に `observation_table: pd.DataFrame` を追加（fold_id, in_out, score） |
| `src/veldra/modeling/multiclass.py` | `MulticlassTrainingOutput` に `observation_table: pd.DataFrame` を追加（fold_id, in_out, class 別 proba） |
| `src/veldra/modeling/frontier.py` | `FrontierTrainingOutput` に `observation_table: pd.DataFrame` を追加（fold_id, prediction, efficiency） |
| `src/veldra/api/artifact.py` | `Artifact` に `observation_table` を保持（persist/load 対応） |

**制約**: 既存の `RunResult`, `EvalResult` の公開シグネチャは変更しない（後方互換維持）。

**テスト**: 既存テスト群の拡張 + `tests/test_observation_table.py`（新規: observation_table の列・行数・型の検証）

---

#### Step 3: 因果推定層の拡張（nuisance 診断情報の充実）

**目的**: `DREstimationOutput` に nuisance モデルの Feature Importance / SHAP / InFold メトリクスを追加。

**変更ファイル**:

| ファイル | 変更内容 |
|---|---|
| `src/veldra/causal/dr.py` | nuisance 学習ループで InFold 予測を保持。`DREstimationOutput` に `nuisance_diagnostics: dict` を追加（propensity_importance, outcome_importance, infold_metrics） |
| `src/veldra/causal/dr_did.py` | 同上。加えて `parallel_trends: dict` に placebo DID 推定値と平均推移データを追加 |
| `src/veldra/causal/diagnostics.py` | `compute_ess()`, `extreme_weight_ratio()`, `overlap_summary()` を追加 |

**テスト**: `tests/test_causal_dr.py`, `tests/test_causal_drdid.py` の拡張（新規フィールドの存在と型の検証）

---

#### Step 4: Notebook 詳細化（UC-1〜UC-6）

**目的**: Phase26.2 の骨格 Notebook を完全版に差し替える。

**対象 Notebook と内容**:

| Notebook | 内容 |
|---|---|
| `notebooks/phase26_2_uc01_regression_fit_evaluate.ipynb` | Setup → fit（固定パラメーター） → 誤差ヒストグラム → 指標テーブル → Feature Importance → SHAP → CSV 出力 |
| `notebooks/phase26_2_uc02_binary_tune_evaluate.ipynb` | Setup → tune（Search Space 適用） → fit → ROC 比較 → Lift → 指標テーブル → Importance → SHAP → CSV |
| `notebooks/phase26_2_uc03_frontier_fit_evaluate.ipynb` | Setup → fit → Pinball ヒストグラム → 指標テーブル → 散布図 → Importance → SHAP → CSV |
| `notebooks/phase26_2_uc04_causal_dr_estimate.ipynb` | Setup → estimate_dr → 推定サマリ → IF 分布 → Overlap 診断 → Balance → nuisance 診断 → Importance/SHAP → トリミング比較 → CSV |
| `notebooks/phase26_2_uc05_causal_drdid_estimate.ipynb` | Setup → estimate_dr_did → 推定サマリ → IF 分布 → 並行トレンド → Overlap → Balance → nuisance 診断 → Importance/SHAP → トリミング比較 → CSV |
| `notebooks/phase26_2_uc06_causal_dr_tune.ipynb` | Setup → tune（Causal objective） → estimate_dr → 診断一式 |

**追加 Notebook**:

| Notebook | 内容 |
|---|---|
| `notebooks/phase26_3_uc_multiclass_fit_evaluate.ipynb` | 新規: Multiclass fit → NLL ヒストグラム → 正解クラス確率 → 指標テーブル → Importance → SHAP → CSV |
| `notebooks/phase26_3_uc_timeseries_fit_evaluate.ipynb` | 新規: TimeSeries fit → 時系列プロット → 残差プロット → 指標テーブル → Importance → SHAP → CSV |

**各 Notebook の共通構造**:

```
1. Setup（import, OUT_DIR, matplotlib.use('Agg')）
2. データ読み込み + config 定義（固定パラメーター使用）
3. fit / tune / estimate 実行
4. 診断セクション（diagnostics API 呼び出し）
   - 可視化: plt.savefig(OUT_DIR / 'plot_name.png')
   - 指標: pd.DataFrame 表示 + CSV 保存
5. 詳細テーブル CSV 出力
6. SUMMARY dict（status, artifact_path, outputs リスト, metrics）
```

**テスト**: `tests/test_notebook_phase26_3_uc_structure.py`（Notebook セル構造・import・SUMMARY 形式の検証）

---

#### Step 5: Notebook 詳細化（UC-7〜UC-10: 既存アーティファクト評価・エクスポート）

**目的**: 評価・エクスポート系 Notebook にも診断出力を追加。

| Notebook | 追加内容 |
|---|---|
| `notebooks/phase26_2_uc07_artifact_evaluate.ipynb` | artifact.load → evaluate → タスクタイプに応じた診断一式（Step 4 と同じ可視化セット） |
| `notebooks/phase26_2_uc08_artifact_reevaluate.ipynb` | 別データでの evaluate → In/Out 比較（学習時 vs 再評価時）の指標並置 |
| `notebooks/phase26_2_uc09_export_python_onnx.ipynb` | 変更なし（エクスポートは診断対象外） |
| `notebooks/phase26_2_uc10_export_html_excel.ipynb` | 変更なし（エクスポートは診断対象外） |

**テスト**: Step 4 のテストで併せて検証

---

#### Step 6: 実行証跡の更新と契約テスト

**目的**: 全 Notebook のヘッドレス実行で出力ファイルが生成されることを確認し、execution manifest を更新。

**成果物**:

| ファイル | 内容 |
|---|---|
| `notebooks/phase26_3_execution_manifest.json` | 各 Notebook の実行結果（status, outputs リスト, 出力ファイルのハッシュ） |
| `tests/test_notebook_phase26_3_execution_evidence.py` | manifest の整合性テスト（全 UC が passed、outputs が存在） |
| `tests/test_notebook_phase26_3_outputs.py` | 各 Notebook の出力ファイル検証（PNG 画像の存在、CSV の列名・行数、指標の妥当範囲） |

---

### 対象ファイル一覧

| ファイル | Step | 変更種別 |
|---|---|---|
| `src/veldra/diagnostics/__init__.py` | 1 | 新規 |
| `src/veldra/diagnostics/importance.py` | 1 | 新規 |
| `src/veldra/diagnostics/shap_native.py` | 1 | 新規 |
| `src/veldra/diagnostics/metrics.py` | 1 | 新規 |
| `src/veldra/diagnostics/plots.py` | 1 | 新規 |
| `src/veldra/diagnostics/tables.py` | 1 | 新規 |
| `src/veldra/diagnostics/causal_diag.py` | 1 | 新規 |
| `src/veldra/modeling/regression.py` | 2 | 変更: observation_table 追加 |
| `src/veldra/modeling/binary.py` | 2 | 変更: observation_table 追加 |
| `src/veldra/modeling/multiclass.py` | 2 | 変更: observation_table 追加 |
| `src/veldra/modeling/frontier.py` | 2 | 変更: observation_table 追加 |
| `src/veldra/api/artifact.py` | 2 | 変更: observation_table persist/load |
| `src/veldra/causal/dr.py` | 3 | 変更: nuisance_diagnostics 追加 |
| `src/veldra/causal/dr_did.py` | 3 | 変更: nuisance_diagnostics + parallel_trends 追加 |
| `src/veldra/causal/diagnostics.py` | 3 | 変更: ESS, extreme_weight_ratio 追加 |
| `notebooks/phase26_2_uc01_regression_fit_evaluate.ipynb` | 4 | 変更: 診断セクション追加 |
| `notebooks/phase26_2_uc02_binary_tune_evaluate.ipynb` | 4 | 変更: 診断セクション追加 |
| `notebooks/phase26_2_uc03_frontier_fit_evaluate.ipynb` | 4 | 変更: 診断セクション追加 |
| `notebooks/phase26_2_uc04_causal_dr_estimate.ipynb` | 4 | 変更: 診断セクション追加 |
| `notebooks/phase26_2_uc05_causal_drdid_estimate.ipynb` | 4 | 変更: 診断セクション追加 |
| `notebooks/phase26_2_uc06_causal_dr_tune.ipynb` | 4 | 変更: 診断セクション追加 |
| `notebooks/phase26_3_uc_multiclass_fit_evaluate.ipynb` | 4 | 新規 |
| `notebooks/phase26_3_uc_timeseries_fit_evaluate.ipynb` | 4 | 新規 |
| `notebooks/phase26_2_uc07_artifact_evaluate.ipynb` | 5 | 変更: 診断セクション追加 |
| `notebooks/phase26_2_uc08_artifact_reevaluate.ipynb` | 5 | 変更: 比較指標追加 |
| `notebooks/phase26_3_execution_manifest.json` | 6 | 新規 |
| `tests/test_diagnostics_importance.py` | 1 | 新規 |
| `tests/test_diagnostics_shap.py` | 1 | 新規 |
| `tests/test_diagnostics_metrics.py` | 1 | 新規 |
| `tests/test_diagnostics_plots.py` | 1 | 新規 |
| `tests/test_diagnostics_tables.py` | 1 | 新規 |
| `tests/test_diagnostics_causal.py` | 1 | 新規 |
| `tests/test_observation_table.py` | 2 | 新規 |
| `tests/test_notebook_phase26_3_uc_structure.py` | 4 | 新規 |
| `tests/test_notebook_phase26_3_execution_evidence.py` | 6 | 新規 |
| `tests/test_notebook_phase26_3_outputs.py` | 6 | 新規 |

---

### テスト計画

| テスト | 内容 | ファイル |
|---|---|---|
| Feature Importance | `compute_importance` が正しい shape の DataFrame を返すこと | `tests/test_diagnostics_importance.py` |
| SHAP 算出 | `compute_shap` が feature 数と一致する列を返すこと、合計が予測値と一致すること | `tests/test_diagnostics_shap.py` |
| メトリクス計算 | 各タスクタイプの In/Out 別メトリクスが正しい範囲の値を返すこと | `tests/test_diagnostics_metrics.py` |
| 可視化 | 各 plot 関数が PNG ファイルを生成し、サイズ > 0 であること | `tests/test_diagnostics_plots.py` |
| テーブル生成 | 各 `build_*_table` が期待列を含む DataFrame を返すこと | `tests/test_diagnostics_tables.py` |
| 因果診断 | ESS、SMD、トリミング比較が正しい型・範囲で返ること | `tests/test_diagnostics_causal.py` |
| Observation table | 各 TrainingOutput の observation_table が fold_id, in_out 列を含むこと | `tests/test_observation_table.py` |
| Notebook 構造 | 全 Notebook が SUMMARY セル、diagnostics import、savefig 呼び出しを含むこと | `tests/test_notebook_phase26_3_uc_structure.py` |
| 実行証跡 | manifest の全 UC が passed で outputs がファイルシステム上に存在すること | `tests/test_notebook_phase26_3_execution_evidence.py` |
| 出力ファイル検証 | PNG の存在、CSV の列名一致、指標値の妥当範囲（例: 0 ≤ AUC ≤ 1） | `tests/test_notebook_phase26_3_outputs.py` |
| 後方互換 | 既存テスト群（`tests/test_*.py`）が全パス | 既存テスト群 |

### 検証コマンド

```bash
# Step 1: 診断ライブラリ
uv run pytest tests/test_diagnostics_*.py -v

# Step 2: observation table
uv run pytest tests/test_observation_table.py -v

# Step 3: 因果拡張
uv run pytest tests/test_causal_dr.py tests/test_causal_drdid.py -v

# Step 4-5: Notebook 構造
uv run pytest tests/test_notebook_phase26_3_uc_structure.py -v

# Step 6: 実行証跡 + 出力検証
uv run pytest tests/test_notebook_phase26_3_execution_evidence.py tests/test_notebook_phase26_3_outputs.py -v

# 全体回帰テスト
uv run pytest tests -x --tb=short
```

---

### 完了条件

1. **Step 1**: `veldra.diagnostics` パッケージが作成され、全 API のユニットテストがパスすること。
2. **Step 2**: 各 `*TrainingOutput` に `observation_table` が追加され、既存テストが全パスすること（後方互換維持）。
3. **Step 3**: `DREstimationOutput` に `nuisance_diagnostics` が追加され、因果テストがパスすること。
4. **Step 4**: UC-1〜UC-6 の Notebook が完全版に更新され、Multiclass / TimeSeries の新規 Notebook が追加されていること。
5. **Step 5**: UC-7, UC-8 の Notebook に診断出力が追加されていること。
6. **Step 6**: `phase26_3_execution_manifest.json` が生成され、全 Notebook の出力ファイルが検証済みであること。
7. **後方互換**: `veldra.api.*` の公開シグネチャが未変更であること。`RunResult`, `EvalResult`, `CausalResult` の既存フィールドが維持されていること。
8. **依存制約**: 外部ライブラリの追加なし（matplotlib は既存依存）。SHAP は LightGBM 内蔵のみ使用。

### Decision（provisional）
- 内容: Phase26.3 は `config_version=1` を維持し、`train.metrics` / `tuning.metrics_candidates` を optional 追加で拡張する。
- 理由: Stable API と既存 Config 運用の互換性を保持しつつ、Notebook 詳細化に必要な設定表現力を増やすため。
- 影響範囲: `src/veldra/config/models.py` / tuning objective validation / GUI-config serialization

### Decision（confirmed）
- 内容: Notebook 証跡はハイブリッド運用とし、構造契約テストは常時実行、重い証跡検証は `notebook_e2e` marker で分離する。
- 理由: CI 負荷と実行時間を抑制しつつ、Phase26.3 の成果物検証を維持するため。
- 影響範囲: `pyproject.toml` marker / `tests/test_notebook_phase26_3_*` / execution manifest 運用

### Decision（confirmed）
- 内容: Phase26.3 の Notebook は `UC-1〜UC-8 + UC-11/12` を実行済み状態でコミットし、placeholder 出力を撤廃する。`UC-9/10` は export 中心の最小更新を維持する。
- 理由: Notebook を開いた時点で図表・表・指標を確認可能にし、実行証跡の再現性を担保するため。
- 影響範囲: `notebooks/phase26_2_uc0*.ipynb` / `notebooks/phase26_3_uc_*.ipynb` / `notebooks/phase26_3_execution_manifest.json`

### Decision（confirmed）
- 内容: `tuning.metrics_candidates` は tuning objective 許可セットとは独立した task 別許可セットで検証する（regression: `rmse/huber/mae`, binary: `logloss/auc`, multiclass: `multi_logloss/multi_error`）。
- 理由: モニタリング/診断用 metrics 候補と最適化 objective の責務を分離し、設計意図と実装契約を一致させるため。
- 影響範囲: `src/veldra/config/models.py` / `tests/test_phase263_config_extensions.py` / `tests/test_runconfig_validation.py`

### Phase26.3 実装評価（2026-02-17）

#### 完了状況

Phase26.3 の全 6 Step を完了。テスト **27 passed, 0 failed**。

| Step | 内容 | 状況 |
|---|---|---|
| Step 1 | `veldra.diagnostics` パッケージ新設（7 モジュール） | 完了 |
| Step 2 | `observation_table` を全 TrainingOutput + Artifact に追加 | 完了 |
| Step 3 | 因果推定層の nuisance 診断拡張 | 完了 |
| Step 4 | Notebook 詳細化（UC-1〜UC-6 + UC-11/UC-12 新規） | 完了 |
| Step 5 | UC-7/UC-8 の診断出力追加 | 完了 |
| Step 6 | 実行証跡 manifest + 契約テスト | 完了 |

#### 残課題（Phase26.4 へ引き継ぎ）

1. **Notebook の教育的品質が低い**: 自動生成 Notebook（UC-1〜UC-12）は実行証跡としては十分だが、学習教材としては不十分（概念説明なし、パラメーター未解説、結果解釈なし）。
2. **手書き Workflow Notebook との品質格差**: `*_analysis_workflow.ipynb` 群は教育的品質 7-8/10 だが、Phase26 UC Notebook は 0-2/10。
3. **テストカバレッジの偏り**: 統合テスト 90% 超だが、ユニットテスト（特に core API happy path、エッジケース、数値安定性）が不足。

---

## 13.4 Phase26.4: Notebook 教育化 & テスト品質強化

### 目的

Phase26.3 で完成した診断・可視化基盤の上に、**Notebook を初学者向け教材**として再構成し、併せて**テストカバレッジの構造的ギャップ**を解消する。

### 前提

- Notebook の2系統（自動生成 UC 系 / 手書き Workflow 系）は統合せず、役割を明確に分離する。
- UC 系は「実行証跡 + クイックリファレンス」として維持し、教育的補強を最小限に留める。
- Workflow 系を「公式チュートリアル」として位置づけ、重点的に教育品質を向上させる。
- テスト追加は既存の公開 API シグネチャを変更しない。
- 命名規約は **英語スネークケース** を採用し、ユーザー向け名称から `phase26` 識別子を除去する。
- Notebook 配置は `notebooks/tutorials` と `notebooks/quick_reference` の 2 系統に分離する。
- 旧ファイル名は **1リリース互換** として root 配下に残し、`Moved to ...` を明記した互換スタブへ置換する。

---

### 実装ステップ（Step 0-6）

#### Step 0: 設計記録更新（先行）

- 本節（13.4）に命名再編方針、実装順序、互換期限を明記する。
- `HISTORY.md` に以下の Decision を記録する。
  - `Decision: confirmed` 命名規約（英語スネークケース）と配置規約（tutorials / quick_reference）
  - `Decision: provisional` 旧名スタブ撤去時期（次期リリースで確定）

#### Step 1: Notebook 配置再編と互換スタブ整備

##### 1-1. Workflow → Tutorials（canonical）

| 旧ファイル | 新ファイル（canonical） |
|---|---|
| `notebooks/regression_analysis_workflow.ipynb` | `notebooks/tutorials/tutorial_01_regression_basics.ipynb` |
| `notebooks/binary_tuning_analysis_workflow.ipynb` | `notebooks/tutorials/tutorial_02_binary_classification_tuning.ipynb` |
| `notebooks/frontier_analysis_workflow.ipynb` | `notebooks/tutorials/tutorial_03_frontier_quantile_regression.ipynb` |
| `notebooks/simulate_analysis_workflow.ipynb` | `notebooks/tutorials/tutorial_04_scenario_simulation.ipynb` |
| `notebooks/lalonde_dr_analysis_workflow.ipynb` | `notebooks/tutorials/tutorial_05_causal_dr_lalonde.ipynb` |
| `notebooks/lalonde_drdid_analysis_workflow.ipynb` | `notebooks/tutorials/tutorial_06_causal_drdid_lalonde.ipynb` |
| （新規） | `notebooks/tutorials/tutorial_00_quickstart.ipynb` |
| （新規） | `notebooks/tutorials/tutorial_07_model_evaluation_guide.ipynb` |

##### 1-2. UC 実行証跡 → Quick Reference（canonical）

| 旧ファイル | 新ファイル（canonical） |
|---|---|
| `notebooks/phase26_2_uc01_regression_fit_evaluate.ipynb` | `notebooks/quick_reference/reference_01_regression_fit_evaluate.ipynb` |
| `notebooks/phase26_2_uc02_binary_tune_evaluate.ipynb` | `notebooks/quick_reference/reference_02_binary_tune_evaluate.ipynb` |
| `notebooks/phase26_2_uc03_frontier_fit_evaluate.ipynb` | `notebooks/quick_reference/reference_03_frontier_fit_evaluate.ipynb` |
| `notebooks/phase26_2_uc04_causal_dr_estimate.ipynb` | `notebooks/quick_reference/reference_04_causal_dr_estimate.ipynb` |
| `notebooks/phase26_2_uc05_causal_drdid_estimate.ipynb` | `notebooks/quick_reference/reference_05_causal_drdid_estimate.ipynb` |
| `notebooks/phase26_2_uc06_causal_dr_tune.ipynb` | `notebooks/quick_reference/reference_06_causal_dr_tune.ipynb` |
| `notebooks/phase26_2_uc07_artifact_evaluate.ipynb` | `notebooks/quick_reference/reference_07_artifact_evaluate.ipynb` |
| `notebooks/phase26_2_uc08_artifact_reevaluate.ipynb` | `notebooks/quick_reference/reference_08_artifact_reevaluate.ipynb` |
| `notebooks/phase26_2_uc09_export_python_onnx.ipynb` | `notebooks/quick_reference/reference_09_export_python_onnx.ipynb` |
| `notebooks/phase26_2_uc10_export_html_excel.ipynb` | `notebooks/quick_reference/reference_10_export_html_excel.ipynb` |
| `notebooks/phase26_3_uc_multiclass_fit_evaluate.ipynb` | `notebooks/quick_reference/reference_11_multiclass_fit_evaluate.ipynb` |
| `notebooks/phase26_3_uc_timeseries_fit_evaluate.ipynb` | `notebooks/quick_reference/reference_12_timeseries_fit_evaluate.ipynb` |
| `notebooks/phase26_2_ux_audit.ipynb` | `notebooks/reference_index.ipynb` |

##### 1-3. 互換スタブ

- 旧名 Notebook は root 配下に残し、冒頭セルを `Moved to ...` + 互換期限（1リリース）へ差し替える。
- `rg --files notebooks | rg "phase26_.*\\.ipynb"` のヒットは互換スタブのみを許容条件とする。

#### Step 2: Tutorials の教育強化（Part A-1 + A-3）

- 対象: `tutorial_01`〜`tutorial_06` + 新規 `tutorial_00`, `tutorial_07`
- 全 tutorial に共通して以下を追加する。
  - Concept primer
  - Config 解説表
  - 結果解釈セル
  - If-then 感度分析セル
  - よくある失敗
  - Further reading

#### Step 3: Quick Reference 12本の整備（Part A-2）

- `scripts/generate_phase263_notebooks.py` の出力先を `notebooks/quick_reference` に変更する。
- 各 notebook に以下を追加する。
  - 冒頭 3行の Overview
  - Config コメント
  - 出力注釈
  - 対応 tutorial へのリンク
- `notebooks/phase26_3_execution_manifest.json` の `notebook` フィールドは canonical パスへ更新する。
- `examples/out/phase26_2_*` / `examples/out/phase26_3_*` は互換維持のため変更しない。

#### Step 4: テスト品質強化（Part B）

##### P0
- `tests/test_runner_fit_happy.py`
- `tests/test_runner_evaluate_happy.py`
- `tests/test_runner_predict_happy.py`
- `tests/test_runner_tune_happy.py`
- `tests/test_edge_cases.py`

##### P1
- `tests/test_numerical_stability.py`
- `tests/test_config_cross_field.py`
- `tests/conftest.py` fixture 追加
  - `unbalanced_binary_frame`
  - `categorical_frame`
  - `timeseries_frame`
  - `missing_values_frame`
  - `outlier_frame`

##### P2
- `tests/test_data_loader_robust.py`

#### Step 5: 既存参照の全面更新

- `README.md` の notebook 導線を canonical 名へ更新。
- `docs/phase26_2_parity_report.md` の notebook パスを canonical 名へ更新。
- Notebook 構造/契約テスト（`tests/test_notebook_*`）の対象パスを canonical 名へ更新。
- `tests/e2e_playwright/conftest.py` の manifest ファイル名（`phase26_2_execution_manifest.json`）は据え置く。

#### Step 6: 受け入れ確認

```bash
uv run pytest tests/test_runner_fit_happy.py tests/test_runner_evaluate_happy.py tests/test_runner_predict_happy.py tests/test_runner_tune_happy.py -v
uv run pytest tests/test_edge_cases.py tests/test_numerical_stability.py tests/test_config_cross_field.py tests/test_data_loader_robust.py -v
uv run pytest tests/test_notebook_phase26_2_uc_structure.py tests/test_notebook_phase26_2_paths.py tests/test_notebook_phase26_2_execution_evidence.py tests/test_notebook_phase26_3_uc_structure.py -v
uv run pytest tests/test_notebook_phase26_3_execution_evidence.py tests/test_notebook_phase26_3_outputs.py -m notebook_e2e -v
uv run pytest tests -x --tb=short
```

### 完了条件

1. `notebooks/tutorials` / `notebooks/quick_reference` の canonical Notebook が全件存在する。
2. `notebooks/reference_index.ipynb` が tutorial / quick reference を全件リンクする。
3. 旧名 notebook が互換スタブとして残り、移行先と互換期限を明記している。
4. quick reference 12本が `Setup / Workflow / Result Summary / SUMMARY` を維持している。
5. tutorial 8本が教育セクション（Concept primer など）を含む。
6. `phase26_3_execution_manifest.json` の `notebook` パスが canonical を指し、対象 UC は `passed` で outputs が実在する。
7. Part B の新規テストがパスし、`veldra.api.*` の公開シグネチャ互換を維持している。


## 14 Phase 27: ジョブキュー強化 & 優先度システム

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

## 15 Phase 28: リアルタイム進捗追跡 & ストリーミングログ

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

## 16 Phase 29: キャンセル強化 & エラーリカバリ

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

## 17 Phase 30: Config管理 & テンプレートライブラリ

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

## 18 Phase 31: 高度可視化 & Artifact比較

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

## 19 Phase 32: パフォーマンス最適化 & スケーラビリティ

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

## 20 Phase 33: 洗練 & プロダクション対応

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

## 21 Phase 34: GUIメモリ再最適化 & テスト分離（提案）

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
