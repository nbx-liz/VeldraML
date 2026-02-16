# DESIGN_BLUEPRINT

最終更新: 2026-02-16

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
初学者が指示のもと、初回で**学習→評価**まで完遂でき、結果が適切に保存され後で比較・エクスポートできる GUI を実現する。
現行4画面（Data / Config / Run / Results）を、ユーザージャーニーに基づいた7画面＋2補助画面に再構成し、操作性とユーザー満足度を向上させる。

### 実装固定方針（2026-02-16）
- ロールアウトは 3 段階分割（Stage A/B/C）で実施する。
- Export 機能は Phase26 で Excel + HTML を実装し、SHAP は `export-report` extra が導入済みの場合のみ有効化する。
- `/config` 導線は 1 フェーズ互換を維持し、GUI 上は `/target` へ誘導する。

### 実装状況（2026-02-16）
- Stage A/B/C を完了し、`/target` `/validation` `/train` `/runs` `/compare` を追加済み。
- `workflow-state` 拡張と `_build_config_from_state` 導入により、Run ページは state 駆動 YAML で実行可能。
- Results は `Overview / Feature Importance / Learning Curves / Config` タブ構成へ拡張済み。
- Export は非同期ジョブ（`export_excel` / `export_html_report`）として実装済み。
- `/config` は互換経路として維持し、旧テスト契約（`cfg-*` callback）を温存している。

### 対象ユーザー
- 初学者（SQLは理解、Python/MLは初心者）
- 利用頻度：毎日、1回あたり1時間以内で作業完了したい
- サポート役が存在（レビューとエクスポート受領をしたい）
- 個人専用ツール（誰が実行したかの監査は不要）

### スコープ外
- 特徴量エンジニアリング、データマート構築（データ準備は外部前提）
- 認証・権限管理
- 分散実行
- ただしバックエンドで自動化できる範囲の前処理は実施可（欠損補完、カテゴリエンコード等）

---

### 画面一覧と遷移図

```
                    ┌──────────────────────────────────────┐
                    │          Sidebar (常設)                │
                    │  [Data] [Target] [Validation] [Train] │
                    │  [Run] [Results]                      │
                    │  ──────                               │
                    │  [Runs] [Compare]                     │
                    └──────────────────────────────────────┘

メインフロー（Wizard型、上部ステッパー連動）:

  ① Data ──→ ② Target ──→ ③ Validation ──→ ④ Train ──→ ⑤ Run ──→ ⑥ Results
     │                                                        │
     │  サイドバーからいつでもジャンプ可能                        │
     │                                                        ↓
     │                                              ⑦ Runs（履歴一覧）
     │                                                   │
     │                                                   ↓
     │                                              ⑧ Compare（2Run比較）
     │
     └── 既存 Artifact を選択して Results へ直接遷移も可能

Export はフロー内の機能（Results / Compare 内の Export ボタン）として提供。
独立画面は設けない（操作導線の複雑化を避ける）。
```

### 画面設計の基本方針

| 方針 | 説明 |
|------|------|
| **Wizard＋Studio ハイブリッド** | メインフローは Wizard 型ステッパーでガイドするが、サイドバーから任意画面にジャンプ可能。初学者は Wizard に従い、経験者はショートカットを使う |
| **既定値で動く** | 全画面で最低限の入力（データパス＋目的変数＋タスクタイプ）があれば、残りは全てスマートデフォルトで実行可能 |
| **段階的開示** | 必須入力を上部に、高度オプションは Accordion/Collapse で隠す。CV種別・Optuna・時系列オプション等は「詳細設定」内 |
| **ガードレール優先** | 各画面で実行前に自動診断を実行し、問題と修正案をインライン表示 |
| **1画面1責務** | 現行 Config ページの50+入力を Target / Validation / Train の3画面に分割し、認知負荷を軽減 |

---

### Step 1: 画面再構成 — Config ページの3分割

**現状**: Config ページ（`config_page.py`, 1,107行）に50+入力が集中し、認知負荷が高い。
**変更**: Config ページを廃止し、以下の3つの専用ページに分割する。

#### 1a: Target ページ（新規: `src/veldra/gui/pages/target_page.py`）

**目的**: 目的変数とタスクタイプの選択。データの性質に基づくタスクタイプの自動推定。

**レイアウト**:
```
┌──────────────────────────────────────────────────┐
│ ② Target                                         │
├──────────────────────────────────────────────────┤
│ [データプレビュー: 先頭5行、選択列ハイライト]       │
│                                                  │
│ ★ 目的変数 ────────────── [ドロップダウン]          │
│   推定タスクタイプ: Binary (ユニーク値=2)  [自動]   │
│                                                  │
│ ★ タスクタイプ ─────────── (●) Regression          │
│                           ( ) Binary              │
│                           ( ) Multiclass          │
│                           ( ) Frontier            │
│                                                  │
│ ▸ 因果推論設定（折りたたみ）                        │
│   [ ] 因果推論を有効化                             │
│   メソッド: DR / DR-DiD                           │
│   Treatment列: [ドロップダウン]                    │
│   Unit ID列: [ドロップダウン]                      │
│                                                  │
│ ▸ 除外列（折りたたみ）                             │
│   [チェックリスト: 全列表示、スクロール可能]         │
│                                                  │
│ ⓘ ガードレール診断                                │
│   ✅ 目的変数にNULLなし                            │
│   ⚠️ ユニーク値=2: Binary推奨                      │
│                                                  │
│        [← Back: Data]  [Next: Validation →]       │
└──────────────────────────────────────────────────┘
```

**自動推定ロジック**（`services.py` に追加）:
- ユニーク値 = 2 → Binary 推奨
- ユニーク値 3-20 かつ整数型 → Multiclass 推奨
- それ以外 → Regression 推奨
- 推定結果はラジオボタンの初期値として設定（ユーザーは上書き可能）

**ガードレール**:
- 目的変数の NULL 率チェック（> 5% で警告）
- Binary タスクでユニーク値 ≠ 2 のとき警告
- Multiclass タスクでクラス数 > 50 のとき警告
- 除外列に目的変数が含まれる場合エラー

#### 1b: Validation ページ（新規: `src/veldra/gui/pages/validation_page.py`）

**目的**: データ分割戦略の設定。タスクタイプに基づくスマートデフォルトの適用。

**レイアウト**:
```
┌──────────────────────────────────────────────────┐
│ ③ Validation                                     │
├──────────────────────────────────────────────────┤
│ ★ 分割タイプ ─────── [ドロップダウン]               │
│   推奨: Stratified K-Fold (Binary タスク)  [自動]  │
│                                                  │
│   選択肢:                                        │
│   - K-Fold（既定: regression/frontier）            │
│   - Stratified K-Fold（既定: binary/multiclass）   │
│   - Group K-Fold                                 │
│   - Time Series                                  │
│   - Custom (fold_id 列指定)                       │
│                                                  │
│ ★ Fold数 ──────────── [5] (スピナー, 2-20)        │
│                                                  │
│ ▸ Group K-Fold 設定（split_type=Group 時のみ表示） │
│   Group列: [ドロップダウン]                        │
│                                                  │
│ ▸ Time Series 設定（split_type=TimeSeries 時のみ）│
│   Time列: [ドロップダウン]                         │
│   モード: expanding / blocked                     │
│   テストサイズ / ギャップ / エンバーゴ              │
│                                                  │
│ ▸ Custom 設定（split_type=Custom 時のみ）         │
│   Fold ID列: [ドロップダウン]                      │
│                                                  │
│ ⓘ ガードレール診断                                │
│   ✅ Stratified K-Fold: クラス分布が保たれます      │
│   ⚠️ TimeSeries 選択時: Time列未指定 → 必須です    │
│   ⚠️ 未来情報リーク警告（※後述）                   │
│                                                  │
│        [← Back: Target]  [Next: Train →]          │
└──────────────────────────────────────────────────┘
```

**スマートデフォルト**:
- `binary` / `multiclass` → Stratified K-Fold を自動選択
- `regression` / `frontier` → K-Fold を自動選択
- `causal` 有効時 → Group K-Fold を推奨（`unit_id_col` を group_col に自動設定）
- ユーザーが明示変更した場合はそれを尊重

**未来情報リークガードレール**（`services.py` に追加）:
- Time Series 分割選択時に、除外されていない日付/時刻型列の存在を検出し警告
- Group K-Fold 選択時に group 列の指定漏れを検出
- Custom fold 選択時に fold_id 列の値域チェック（連番であること）

#### 1c: Train ページ（新規: `src/veldra/gui/pages/train_page.py`）

**目的**: LightGBM パラメーターとチューニング設定。既定値で動くことを前提に、高度設定は折りたたみ。

**レイアウト**:
```
┌──────────────────────────────────────────────────┐
│ ④ Train                                          │
├──────────────────────────────────────────────────┤
│ ★ 基本パラメーター                                │
│   学習率: [0.05] (スライダー, 0.001-0.3)           │
│   Num Boost Round: [300] (スピナー)               │
│   Num Leaves: [31] (スピナー)                     │
│                                                  │
│ ▸ Early Stopping（折りたたみ、既定: ON）           │
│   Early Stopping Rounds: [100]                   │
│   Validation Fraction: [0.1] (スライダー)          │
│                                                  │
│ ▸ クラス不均衡対策（binary/multiclass 時のみ表示）  │
│   [✓] Auto Class Weight (既定: ON)               │
│   手動 Class Weight: [JSON入力]                   │
│                                                  │
│ ▸ 高度パラメーター（折りたたみ）                    │
│   max_depth / min_child_samples / subsample       │
│   colsample_bytree / reg_alpha / reg_lambda       │
│   auto_num_leaves / num_leaves_ratio              │
│   min_data_in_leaf_ratio / min_data_in_bin_ratio  │
│   feature_weights / top_k (binary のみ)           │
│   path_smooth / cat_l2 / cat_smooth 等            │
│                                                  │
│ ▸ ハイパラチューニング（Optuna）（折りたたみ）     │
│   [ ] チューニングを有効化                         │
│   プリセット: Fast / Standard                     │
│   N Trials: [50] (スピナー)                       │
│   Objective: [ドロップダウン]                      │
│   ▸ カスタム探索空間（さらに折りたたみ）            │
│                                                  │
│ ▸ Artifact 出力先                                 │
│   ディレクトリ: [artifacts] (プリセット選択)        │
│                                                  │
│ ⓘ 設定サマリー                                   │
│   タスク: Binary | 分割: Stratified 5-Fold         │
│   学習率: 0.05 | Rounds: 300 | Leaves: 31         │
│   チューニング: OFF                               │
│                                                  │
│        [← Back: Validation]  [Next: Run →]        │
└──────────────────────────────────────────────────┘
```

**設定サマリーカード**: Train ページ下部に、Target / Validation / Train の主要設定を1行ずつ表示。設定漏れを実行前に一覧できる。

---

### Step 2: ステッパーの6段階化

**現状**: 4段階ステッパー（Data → Config → Run → Results）
**変更**: 6段階ステッパー（Data → Target → Validation → Train → Run → Results）

**対象**: `src/veldra/gui/app.py` の `_stepper_bar()` 関数

```python
STEPPER_STEPS = [
    {"path": "/data",       "label": "Data",       "number": 1},
    {"path": "/target",     "label": "Target",     "number": 2},
    {"path": "/validation", "label": "Validation", "number": 3},
    {"path": "/train",      "label": "Train",      "number": 4},
    {"path": "/run",        "label": "Run",        "number": 5},
    {"path": "/results",    "label": "Results",    "number": 6},
]
```

**ステッパー完了判定ロジック**:
- Data: `workflow-state.data_path` が設定済み
- Target: `workflow-state.target_col` と `workflow-state.task_type` が設定済み
- Validation: `workflow-state.split_config` が設定済み（既定値でも可）
- Train: `workflow-state.train_config` が設定済み（既定値でも可）
- Run: 最後のジョブが `SUCCEEDED`
- Results: Artifact が存在

---

### Step 3: サイドバー拡張 — Runs / Compare の追加

**現状**: サイドバーに Data / Config / Run / Results の4項目
**変更**: メインフロー6項目 + セパレーター + 補助2項目

**対象**: `src/veldra/gui/app.py` の `_sidebar()` 関数

```
── メインフロー ──
  📊 Data
  🎯 Target
  ✂️ Validation
  ⚙️ Train
  ▶️ Run
  📋 Results
── 分析 ──
  📜 Runs
  🔀 Compare
```

---

### Step 4: Runs ページ（新規: `src/veldra/gui/pages/runs_page.py`）

**目的**: 過去の全 Run 履歴を一覧表示し、複製・比較・削除を行う。

**レイアウト**:
```
┌──────────────────────────────────────────────────┐
│ Runs                                             │
├──────────────────────────────────────────────────┤
│ [フィルター: タスクタイプ | ステータス | 日付範囲]   │
│ [検索: Run ID / Artifact パス]                    │
│                                                  │
│ ┌────────────────────────────────────────────┐   │
│ │ ☐ │ Status │ Action │ Task │ Created │ ID  │   │
│ ├────────────────────────────────────────────┤   │
│ │ ☐ │ ✅     │ fit    │ bin  │ 02-16   │ a1  │   │
│ │ ☐ │ ✅     │ tune   │ reg  │ 02-15   │ b2  │   │
│ │ ☐ │ ❌     │ fit    │ mc   │ 02-15   │ c3  │   │
│ └────────────────────────────────────────────┘   │
│                                                  │
│ [Compare Selected (2)] [Clone] [Delete] [Export]  │
│                                                  │
│ ▸ 選択中 Run の詳細                               │
│   Run ID: a1b2c3                                 │
│   Artifact: artifacts/20260216_120000_binary/     │
│   Config YAML: (折りたたみ表示)                    │
│   Metrics: auc=0.85, logloss=0.42                │
│                                                  │
│ [View Results] [Clone Config to Train]            │
└──────────────────────────────────────────────────┘
```

**データソース**: `GuiJobStore` の既存 `list_jobs()` + Artifact メタデータ

**アクション**:
- **Compare Selected**: チェックボックスで2件選択し Compare ページへ遷移
- **Clone**: 選択した Run の Config を Train ページにコピーして新規実行準備
- **Delete**: 選択した Run の Job レコードを削除（Artifact は保持、確認ダイアログ付き）
- **View Results**: Results ページへ遷移（選択 Artifact をセット）
- **Export**: 選択 Run の Config + Metrics を JSON / YAML でダウンロード

---

### Step 5: Compare ページ（新規: `src/veldra/gui/pages/compare_page.py`）

**目的**: 2つの Run を並列比較し、指標差分・設定差分・予測差分を表示する。

**レイアウト**:
```
┌──────────────────────────────────────────────────┐
│ Compare                                          │
├──────────────────────────────────────────────────┤
│ Run A: [ドロップダウン: Artifact 一覧]              │
│ Run B: [ドロップダウン: Artifact 一覧]              │
│                                                  │
│ ⓘ 比較可能性チェック                              │
│   ✅ 同一タスクタイプ (binary)                     │
│   ⚠️ データソースが異なります → 行単位差分は不可     │
│                                                  │
│ ── Metrics 差分 ──                                │
│ ┌─────────────────────────────────────────┐      │
│ │ Metric  │ Run A  │ Run B  │ Δ    │ 判定 │      │
│ ├─────────────────────────────────────────┤      │
│ │ auc     │ 0.850  │ 0.823  │+0.027│ ✅   │      │
│ │ logloss │ 0.420  │ 0.445  │-0.025│ ✅   │      │
│ │ f1      │ 0.780  │ 0.765  │+0.015│ ──   │      │
│ └─────────────────────────────────────────┘      │
│ [Grouped Bar Chart: Run A vs Run B]              │
│                                                  │
│ ── Config 差分 ──                                 │
│ [Side-by-side YAML diff view]                    │
│ 変更箇所のみハイライト表示                         │
│                                                  │
│ ── 予測差分 ──（同一データ時のみ表示）              │
│ [散布図: Run A predicted vs Run B predicted]      │
│ [残差ヒストグラム: Δ prediction]                   │
│                                                  │
│ [Export Comparison Report (HTML)]                 │
└──────────────────────────────────────────────────┘
```

**比較可能性チェック**（`services.py` に追加）:
- 同一タスクタイプか → 異なる場合は警告（メトリクス名が異なるため差分表示に制約あり）
- 同一データソースか → 異なる場合は行単位予測差分を無効化し、集約メトリクス比較のみ
- 同一分割戦略か → 異なる場合は情報表示（比較は許可するが注意喚起）

**Config 差分**: `deepdiff` ライブラリまたは自前の dict diff で YAML 差分を生成し、変更箇所を色分け表示

---

### Step 6: Results ページの強化

**現状**: メトリクス KPI カード + 棒グラフ + 特徴量重要度の2タブ
**変更**: 以下のタブ構成に拡張

#### 6a: タブ構成

| タブ | 内容 | 新規/既存 |
|------|------|-----------|
| Overview | KPI カード + メトリクスサマリー | 既存拡張 |
| Feature Importance | 特徴量重要度棒グラフ（既存） | 既存 |
| Learning Curves | フォールドごとの学習曲線（`training_history.json` を可視化） | **新規** |
| Config | 使用した RunConfig の YAML 表示 | **新規** |

#### 6b: Overview タブの拡張

- 既存の KPI カード + 棒グラフに加え、以下を追加:
  - **データスキーマ要約**: 特徴量数、行数、カテゴリ変数数、欠損率
  - **分割戦略要約**: 分割タイプ、フォールド数
  - **学習パラメーター要約**: 学習率、ラウンド数、早期停止設定

#### 6c: Learning Curves タブ（新規）

**対象**: `src/veldra/gui/components/charts.py` に追加

- Artifact 内の `training_history.json` を読み込み
- フォールドごとのメトリクス推移を折れ線グラフで表示
- X軸: イテレーション数、Y軸: メトリクス値
- 最終モデルの `best_iteration` をマーカーで表示
- 複数フォールドを半透明で重ね、平均線を太線で表示

#### 6d: Export ボタンの追加

**Results ページ上部に Export セクションを追加**:

| ボタン | 動作 | 実装 |
|--------|------|------|
| Export Excel | 非同期ジョブとして Excel 生成（特徴量 + 予測値 + 残差 + SHAP） | `services.py` に `export_excel()` 追加 |
| Export HTML Report | 非同期ジョブとして HTML レポート生成 | `services.py` に `export_html_report()` 追加 |
| Download Config | RunConfig YAML のダウンロード | クライアントサイド `dcc.Download` |

**Excel Export の実装方針**:
- 非同期ジョブキューに投入（大量データの場合数分かかるため）
- SHAP 値の算出は `shap` ライブラリ（optional dependency として追加）
- `openpyxl` で Excel ファイルを生成
- 出力シート構成: データ+予測+残差 | SHAP値 | メトリクスサマリー | Config
- 完了後 `dcc.Download` でブラウザダウンロード

**HTML Report の実装方針**:
- Jinja2 テンプレートベース
- 内容: メトリクスサマリー + 特徴量重要度チャート（Plotly の静的画像） + 学習曲線 + Config + データスキーマ
- 自己完結型 HTML（外部依存なし、インラインCSS + Base64画像）
- ファイルサイズ制限: チャート画像は SVG、データ表は先頭100行まで

---

### Step 7: ガードレール診断システム

**目的**: 各画面で実行前に自動診断を実行し、問題と修正案をインライン表示する。

**対象**: `src/veldra/gui/services.py` に `GuardRailChecker` クラスを新設

```python
@dataclass
class GuardRailResult:
    level: Literal["error", "warning", "info", "ok"]
    message: str
    suggestion: str | None = None

class GuardRailChecker:
    def check_target(self, data: pd.DataFrame, target_col: str, task_type: str) -> list[GuardRailResult]: ...
    def check_validation(self, data: pd.DataFrame, split_config: dict, task_type: str) -> list[GuardRailResult]: ...
    def check_train(self, config: dict) -> list[GuardRailResult]: ...
    def check_pre_run(self, config: dict, data_path: str) -> list[GuardRailResult]: ...
```

**診断項目**:

| 画面 | チェック | レベル | メッセージ例 |
|------|---------|--------|-------------|
| Target | 目的変数 NULL > 5% | warning | 「目的変数に {n}% の欠損があります。欠損行は学習時に除外されます」 |
| Target | Binary で ユニーク値 ≠ 2 | error | 「Binary タスクですが目的変数のユニーク値が {n} です。Multiclass を検討してください」 |
| Target | クラス不均衡 (少数クラス < 5%) | warning | 「少数クラスが {pct}% です。Auto Class Weight が有効です」 |
| Validation | TimeSeries 時 time 列未指定 | error | 「Time Series 分割には Time 列の指定が必須です」 |
| Validation | Group K-Fold 時 group 列未指定 | error | 「Group K-Fold には Group 列の指定が必須です」 |
| Validation | フォールド数 > データ行数 / 10 | warning | 「フォールド数が多すぎます。各フォールドの学習データが少なくなります」 |
| Validation | 未来情報リーク疑い | warning | 「日付型列 '{col}' が特徴量に含まれています。Time Series 分割または除外を検討してください」 |
| Train | num_boost_round > 5000 | warning | 「学習ラウンド数が非常に多いです。Early Stopping が有効か確認してください」 |
| Train | learning_rate > 0.3 | warning | 「学習率が高めです。過学習のリスクがあります」 |
| Pre-Run | データファイルが存在しない | error | 「データファイルが見つかりません: {path}」 |
| Pre-Run | Config バリデーションエラー | error | RunConfig のバリデーション結果をそのまま表示 |

**表示コンポーネント**: `src/veldra/gui/components/guardrail.py`（新規）
- `dbc.Alert` ベースでレベル別色分け（error=赤, warning=黄, info=青, ok=緑）
- 修正案がある場合は「修正する」ボタンを表示（該当フィールドへの自動スクロール）

---

### Step 8: workflow-state の拡張

**現状**: `workflow-state` は `data_path`, `config_yaml`, `target_col`, `last_run_artifact` を保持
**変更**: 画面分割に伴い、以下のキーを追加

```python
workflow_state = {
    # 既存
    "data_path": str | None,
    "target_col": str | None,
    "last_run_artifact": str | None,
    # 新規
    "task_type": str | None,          # "regression" | "binary" | "multiclass" | "frontier"
    "exclude_cols": list[str],         # 除外列リスト
    "causal_config": dict | None,      # 因果推論設定
    "split_config": dict,              # 分割戦略設定（既定値含む）
    "train_config": dict,              # 学習パラメーター（既定値含む）
    "tuning_config": dict | None,      # チューニング設定
    "artifact_dir": str,               # Artifact 出力先
    # config_yaml は各 state から動的に生成（_build_config_from_state()）
}
```

**Config YAML 生成**: `_build_config_from_state(state: dict) -> str`
- workflow-state の各キーから RunConfig 互換の YAML を動的に生成
- Run ページでのジョブ投入時にこの関数を呼び出し

---

### Step 9: YAML Source タブの移設

**現状**: Config ページの Tab 2 に YAML Source エディタ
**変更**: Train ページの Tab 2 に移設

- Train ページを2タブ構成にする: **Builder** | **YAML Source**
- YAML Source は `_build_config_from_state()` の出力をリアルタイム表示
- ユーザーが YAML を直接編集した場合は state を上書き（Import to Builder 機能）
- Load / Save / Validate 機能は維持

---

### Step 10: Config Migration タブの移設

**現状**: Config ページの Tab 3 に Migration ツール
**変更**: 専用モーダルまたは Runs ページ内のアクションとして提供

- Runs ページで古い Config の Run を選択した場合に「Migrate Config」ボタンを表示
- クリックでモーダル表示（Preview / Diff / Apply）
- 機能自体は既存実装をそのまま流用

---

### Step 11: Data ページの拡張

**現状**: ファイルアップロード + 統計 KPI + 先頭10行プレビュー
**変更**: 以下を追加

- **データ品質サマリー**: 列ごとの欠損率・ユニーク値数・型の一覧表
- **列型インジケーター**: numeric / categorical / datetime のバッジ表示
- **データ品質警告**: 高欠損率列（> 50%）、定数列、高カーディナリティ列の検出と警告
- **行数警告**: 100万行超のデータに対するメモリ使用量推定表示

---

### Step 12: 既存ファイルの変更一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/veldra/gui/app.py` | ルーティング更新（6画面 + 2補助）、ステッパー6段階化、サイドバー拡張、Config 関連コールバックの Target/Validation/Train への分割、workflow-state 拡張、`_build_config_from_state()` 追加 |
| `src/veldra/gui/pages/data_page.py` | データ品質サマリー追加、列型インジケーター追加 |
| `src/veldra/gui/pages/config_page.py` | **廃止**（Target / Validation / Train に分割） |
| `src/veldra/gui/pages/run_page.py` | Config YAML を `_build_config_from_state()` から取得するよう変更 |
| `src/veldra/gui/pages/results_page.py` | Learning Curves タブ追加、Config タブ追加、Export ボタン追加 |
| `src/veldra/gui/services.py` | `GuardRailChecker` 追加、タスクタイプ自動推定、Excel/HTML Export ロジック |
| `src/veldra/gui/components/charts.py` | 学習曲線チャート追加、予測差分散布図追加、残差ヒストグラム追加 |
| `src/veldra/gui/components/guardrail.py` | **新規**: ガードレール診断表示コンポーネント |
| `src/veldra/gui/assets/style.css` | 新画面用スタイル追加、ガードレールアラートスタイル |

### 新規ファイル一覧

| ファイル | 内容 |
|---------|------|
| `src/veldra/gui/pages/target_page.py` | Target ページ |
| `src/veldra/gui/pages/validation_page.py` | Validation ページ |
| `src/veldra/gui/pages/train_page.py` | Train ページ |
| `src/veldra/gui/pages/runs_page.py` | Runs ページ |
| `src/veldra/gui/pages/compare_page.py` | Compare ページ |
| `src/veldra/gui/components/guardrail.py` | ガードレール診断表示 |
| `src/veldra/gui/components/config_summary.py` | 設定サマリーカード |
| `src/veldra/gui/components/yaml_diff.py` | YAML 差分ビューア |

---

### 依存関係の追加

```toml
[project.optional-dependencies]
gui = [
    # 既存
    "dash>=2.18.2",
    "plotly>=6.0.0",
    "dash-bootstrap-components>=1.6.0",
    # 新規追加
    "openpyxl>=3.1.0",     # Excel Export
    "jinja2>=3.1.0",       # HTML Report
]
export-report = [
    "shap>=0.46.0",        # SHAP 値算出（optional、なくても Excel は生成可能）
]
```

---

### 実装順序（依存関係順）

| Sub-Phase | 内容 | 依存 |
|-----------|------|------|
| 26.1 | workflow-state 拡張 + `_build_config_from_state()` | なし |
| 26.2 | Target ページ新設 + タスクタイプ自動推定 | 26.1 |
| 26.3 | Validation ページ新設 + スマートデフォルト | 26.1 |
| 26.4 | Train ページ新設（Config ページの移植） + YAML Source タブ移設 | 26.1 |
| 26.5 | ステッパー6段階化 + サイドバー拡張 + ルーティング更新 + Config ページ廃止 | 26.2, 26.3, 26.4 |
| 26.6 | ガードレール診断システム（`GuardRailChecker` + 表示コンポーネント） | 26.2, 26.3, 26.4 |
| 26.7 | Data ページ拡張（品質サマリー、列型インジケーター） | なし |
| 26.8 | Results ページ強化（Learning Curves + Config タブ） | なし |
| 26.9 | Runs ページ新設 + Compare ページ新設 | 26.5 |
| 26.10 | Export 機能（Excel + HTML Report） | 26.8 |
| 26.11 | テスト + 統合検証 | 全 Sub-Phase |

> 注記（2026-02-16）:
> この依存表は Phase26 実装時の旧サブフェーズ定義を保持する履歴である。
> **Phase26.1 の正式定義は `13.1 Phase 26.1: UI改善` を唯一の正**とし、
> Stage 1（バグ修正）/Stage 2（UI再構成設計）の2段階で運用する。

---

### テスト計画

| テスト | 内容 | ファイル |
|--------|------|----------|
| Target ページレイアウト | タスクタイプ自動推定、列ドロップダウン、因果設定の条件表示 | `tests/test_gui_target_page.py` |
| Validation ページレイアウト | スマートデフォルト、分割タイプ別条件表示 | `tests/test_gui_validation_page.py` |
| Train ページレイアウト | パラメーター入力、YAML Source タブ、チューニング設定 | `tests/test_gui_train_page.py` |
| Runs ページ | 履歴一覧、フィルタリング、Clone/Delete アクション | `tests/test_gui_runs_page.py` |
| Compare ページ | 2Run 比較、Metrics 差分、Config 差分、比較可能性チェック | `tests/test_gui_compare_page.py` |
| ガードレール | `GuardRailChecker` の各診断項目 | `tests/test_gui_guardrail.py` |
| workflow-state | 状態管理、`_build_config_from_state()` の YAML 生成 | `tests/test_gui_workflow_state.py` |
| Config 生成 | 分割された3画面からの Config YAML が既存 Builder と同等であること | `tests/test_gui_config_from_state.py` |
| ステッパー | 6段階の完了判定ロジック | `tests/test_gui_stepper.py` |
| Export Excel | Excel ファイル生成、シート構成、データ整合性 | `tests/test_gui_export_excel.py` |
| Export HTML | HTML レポート生成、自己完結性 | `tests/test_gui_export_html.py` |
| Results 拡張 | Learning Curves タブ、Config タブの表示 | `tests/test_gui_results_enhanced.py` |
| Data 拡張 | データ品質サマリー、列型インジケーター | `tests/test_gui_data_enhanced.py` |
| 後方互換 | 既存の GUI テスト（`test_gui_*`）が全パス | 既存テスト群 |
| E2E フロー | Data → Target → Validation → Train → Run → Results の一連フロー | `tests/test_gui_e2e_flow.py` |

### 検証コマンド
- `uv run pytest tests/test_gui_target_page.py tests/test_gui_validation_page.py tests/test_gui_train_page.py -v`
- `uv run pytest tests/test_gui_runs_page.py tests/test_gui_compare_page.py -v`
- `uv run pytest tests/test_gui_guardrail.py tests/test_gui_workflow_state.py tests/test_gui_config_from_state.py -v`
- `uv run pytest tests/test_gui_export_excel.py tests/test_gui_export_html.py -v`
- `uv run pytest tests/test_gui_stepper.py tests/test_gui_results_enhanced.py tests/test_gui_data_enhanced.py -v`
- `uv run pytest tests/test_gui_e2e_flow.py -v`
- `uv run pytest tests -x --tb=short`

---

### 完了条件

1. **画面再構成**: 現行4画面（Data / Config / Run / Results）が7画面＋2補助画面（Data / Target / Validation / Train / Run / Results / Runs / Compare）に再構成され、Config ページが廃止されていること。
2. **ステッパー**: 6段階ステッパーが正しく動作し、各段階の完了判定が workflow-state に基づいて行われること。
3. **スマートデフォルト**: データパス＋目的変数＋タスクタイプの最低限入力で、残りは全てスマートデフォルトが適用され実行可能であること。
4. **ガードレール**: 各画面で自動診断が実行され、問題と修正案がインライン表示されること。特に未来情報リーク、分割ミス、ターゲット不整合が実行前に検知されること。
5. **Runs ページ**: 過去の全 Run 履歴が一覧表示され、Clone / Delete / Compare Selected / View Results のアクションが動作すること。
6. **Compare ページ**: 2つの Run の指標差分・設定差分が表示されること。同一データの Run 同士では予測差分も表示されること。異なるデータの Run 同士では比較制約が明示されること。
7. **Export**: Results ページから Excel（特徴量 + 予測値 + 残差 + SHAP）と HTML レポートが非同期生成・ダウンロード可能であること。
8. **Learning Curves**: Results ページに学習曲線タブが追加され、`training_history.json` のフォールドごとメトリクス推移が可視化されること。
9. **後方互換**: 既存の GUI テストが全パスし、Stable API（`veldra.api.*`）の互換性が維持されること。workflow-state の既存キーは維持されること。
10. **E2E フロー**: Data → Target → Validation → Train → Run → Results の一連フローが、初学者が指示のもとで初回完遂可能であること。

## 13.1 Phase 26.1: UI改善

### 目的
ユーザーの使い方に合わせたUI改善。バグ修正3件 + ユースケース駆動のGUI再構成を行う。

---

### 実装固定方針（2026-02-17）

Phase 26.1 は **Stage 1（バグ修正）** と **Stage 2（UI再構成設計）** の2段階で実施する。

---

### Stage 1: バグ修正（3件）

#### 1-A: Queue時刻の日本時間表示

**現状**: `job_store.py` は UTC ISO 形式で保存。`app.py:643` の `_format_jst_timestamp()` で JST 変換して表示している。
ただし Runs テーブル等で JST 変換が適用されていない箇所がある可能性がある。

**対応内容**:
- Runs テーブル（`runs_page.py`）、Run ページのジョブステータス表示、Compare ページ等、全てのユーザー向けタイムスタンプ表示箇所で `_format_jst_timestamp()` が適用されていることを確認・修正する。
- Export ファイル名に使用されるタイムスタンプ（`services.py` の `_export_output_path()`）も JST に統一する。

**対象ファイル**:
- `src/veldra/gui/app.py` — Runs テーブルデータ生成コールバック内のタイムスタンプ変換確認
- `src/veldra/gui/services.py` — `_export_output_path()` のタイムスタンプを JST に変更

#### 1-B: HTMLレポートのブラウザダウンロード対応

**現状**: HTMLレポート生成は `export_html_report` アクションとしてジョブキューに投入される（`app.py:2604`）。
ジョブ完了後、レポートファイルは `{artifact}/reports/` ディレクトリに保存されるが、
ブラウザへの自動ダウンロードトリガーが存在しない。ユーザーはファイルシステムから手動でアクセスする必要がある。

**対応内容**:
1. Results ページに `dcc.Download(id="result-report-download")` コンポーネントを追加する。
2. HTMLレポートジョブ完了時に、`dcc.Interval` ポーリングで完了を検知し、`dcc.send_file()` でブラウザダウンロードをトリガーする。
3. Excel レポートも同様に `dcc.send_file()` でダウンロード可能にする。
4. ジョブ完了前はステータス表示（「生成中...」）、完了後は「ダウンロード」ボタンに切り替える。

**具体的な実装方針**:
- `results_page.py`: `dcc.Download` コンポーネントと「ダウンロード」ボタンを追加
- `app.py`: エクスポートジョブ投入後、ジョブIDを `dcc.Store` に保持し、`dcc.Interval` でジョブ完了をポーリング。
  完了検知時に `GuiRunResult.result_json` からファイルパスを取得し、`dcc.send_file()` を返す新規コールバックを追加
- `services.py`: `export_html_report()` / `export_excel_report()` の戻り値（ファイルパス）が `GuiRunResult.result_json` に格納されていることを確認

**対象ファイル**:
- `src/veldra/gui/pages/results_page.py` — `dcc.Download` + ダウンロードボタン追加
- `src/veldra/gui/app.py` — ポーリング + ダウンロードトリガーのコールバック追加

#### 1-C: 学習曲線が表示されないバグの修正

**原因特定**: `app.py:2572-2578` の `_cb_update_result_extras()` にバグがある。

```python
# 現在のコード（バグあり）
metadata = getattr(art, "metadata", {}) or {}
history = metadata.get("training_history")  # ← Artifact に metadata 属性は存在しない → 常に None
if history is None:
    history_path = Path(artifact_path) / "training_history.json"  # ← ファイル fallback
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
```

`Artifact` クラスには `metadata` 属性が存在しない（`artifact.py:34-58`）。
`training_history` は `art.training_history` として直接保持されている。
そのため `metadata.get("training_history")` は常に `None` となり、ファイル fallback に頼る。
しかし `Artifact.load()` → `store.load_artifact()` で `training_history.json` は既に読み込まれ
`art.training_history` に格納済みであるため、ファイルパスの二重読み込みは冗長であり、
`artifact_path` がディレクトリでない場合などに失敗する可能性がある。

**修正内容**:
```python
# 修正後
history = art.training_history
curve_fig = plot_learning_curves(history if isinstance(history, dict) else {})
```

**対象ファイル**:
- `src/veldra/gui/app.py` — `_cb_update_result_extras()` の `training_history` 取得ロジック修正

---

### Stage 2: ユースケース駆動UI再構成設計

**方針**:
- ユーザーの使い方に基づくUI改善を設計する。
- 下記の全ユースケースパターンを分析し、現行GUI画面遷移との差分を特定する。
- 具体的なUI変更は Stage 2 完了後に Phase 26.2 として実装する。

**成果物**:
- 各ユースケースパターンの現行GUI対応状況マトリクス
- ギャップ一覧（現行GUIで対応できていない操作フロー）
- Phase 26.2 実装計画のドラフト

**分析対象ユースケースパターン**:
- ユーザーがどのような順番で操作するのが自然かを考慮してGUIを再構成する
  - Regression/Binary/Multiclassモデル学習・評価パターン
    - データ指定
    - 目的変数の指定
    - タスクタイプの指定（Regression/Binary/Multiclassは自動判定、ユーザーが指定することもできる）
    - 特徴量指定（目的変数以外全使用+除外変数指定）
    - 分割方法の指定（KFold / StratifiedKFold / TimeSeriesSplit 等）
    - 学習パラメーターの指定（LightGBMパラメーター）
    - 学習実行
    - モデル評価（タスクタイプに応じて自動的に選択される、詳細は後述）
    - HTMLレポートダウンロード

  - Regression/Binary/Multiclassモデル学習（パラメーター最適化）・評価パターン
    - データ指定
    - 目的変数の指定
    - タスクタイプの指定
    - 特徴量指定
    - 分割方法の指定
    - チューニング設定の指定（探索回数、時間制限、探索空間）
    - パラメーター最適化 & 学習実行
    - モデル評価（タスクタイプに応じて自動的に選択される、詳細は後述）
    - HTMLレポートダウンロード

  - 学習済みRegression/Binary/Multiclassモデルの予測値算出パターン
    - 学習済みモデル（アーティファクト）指定
    - 予測対象データ指定
    - 予測実行
    - 予測値ダウンロード

  - 学習済みRegression/Binary/Multiclassモデルの別データによるモデル再評価パターン
    - 学習済みモデル（アーティファクト）指定
    - 評価対象データ指定（評価用Columnを含むかチェック、特徴量が一致するかチェック）
    - 予測実行
    - モデル評価（タスクタイプに応じて自動的に選択される、詳細は後述）
    - HTMLレポートダウンロード

  - Frontier（分位点回帰）モデル学習・評価パターン
    - データ指定
    - 目的変数の指定
    - タスクタイプの指定（frontier）
    - **分位点（Alpha）の指定**（例: 0.90, 0.50 等）
    - 特徴量指定
    - 分割方法の指定
    - 学習パラメーターの指定
    - 学習実行
    - モデル評価（タスクタイプに応じて自動的に選択される、詳細は後述）
    - HTMLレポートダウンロード

  - Frontier（分位点回帰）モデル学習（パラメーター最適化）・評価パターン
    - データ指定
    - 目的変数の指定
    - タスクタイプの指定（frontier）
    - 分位点（Alpha）の指定
    - 特徴量指定
    - 分割方法の指定
    - チューニング設定の指定（探索回数、時間制限、探索空間）
    - パラメーター最適化 & 学習実行
    - モデル評価（タスクタイプに応じて自動的に選択される、詳細は後述）
    - HTMLレポートダウンロード

  - 因果推論（Doubly Robust法）効果推定パターン
    - データ指定
    - 目的変数（Outcome）の指定
    - タスクタイプの指定（causal）
    - **推定対象（Estimand）の指定**（ATE: 平均処置効果 / ATT: 処置群平均処置効果、DefaultはATT）
    - **処置変数（Treatment）の指定**（バイナリ変数、自動チェック）
    - **個体ID変数の指定**（パネルデータの場合のCross-fitting用）
    - 共変量指定（目的変数と処置変数と個体ID変数以外をデフォルトで指定、ユーザーが手動で除外指定）
    - 傾向スコア・Outcomeモデルの設定
      - **クロスフィッティングの有無**（DefaultはTrue：True/False）
      - **傾向スコアモデルのハイパーパラメーター**（LightGBM: num_leaves, learning_rate 等）
      - **結果モデルのハイパーパラメーター**（LightGBM: num_leaves, learning_rate 等）
      - **傾向スコアのキャリブレーション手法**（Platt Scaling / Isotonic Regression）
      - **傾向スコアのクリッピング閾値**（極端な重みを防ぐための下限・上限カット：例 0.01）
    - 学習・推定実行
    - 診断・評価（**Overlap / SMD: 標準化平均差** 等のバランス確認）
    - 推定結果レポート確認（ATE/ATTの推定値、点推定値、信頼区間）
    - HTMLレポートダウンロード

- 因果推論（Doubly Robust法）効果推定パターン＋パラメーター最適化
    - データ指定
    - 目的変数（Outcome）の指定
    - タスクタイプの指定（causal）
    - **推定対象（Estimand）の指定**（ATE: 平均処置効果 / ATT: 処置群平均処置効果、DefaultはATT）
    - **処置変数（Treatment）の指定**（バイナリ変数、自動チェック）
    - **個体ID変数の指定**（パネルデータの場合のCross-fitting用）
    - 共変量指定（目的変数と処置変数と個体ID変数以外をデフォルトで指定、ユーザーが手動で除外指定）
    - モデル学習・探索の基本設定
      - **クロスフィッティングの有無**（DefaultはTrue：True/False）
      - **傾向スコアのキャリブレーション手法**（Platt Scaling / Isotonic Regression）
      - **傾向スコアのクリッピング閾値**
    - チューニング設定（最適化）
      - **チューニング実行の有無**（Enabled=True）
      - **試行回数（n_trials）の指定**（例: 30回）
      - **探索プリセットの指定**（Standard / Fast：探索範囲の広さを選択）
      - **最適化ターゲット（Objective）の指定**
        - **Balance Priority (dr_balance_priority)**: 共変量バランス（SMD）の確保を最優先し、その制約下で精度を最適化（Default）
        - **Standard Error (dr_std_error)**: 推定量の標準誤差の最小化（精度の最大化）のみを目指す
        - **Overlap Penalty (dr_overlap_penalty)**: 傾向スコアの重なり（Overlap）の確保を重視する
      - **バランス閾値（Balance Threshold）の指定**（Balance Priority選択時、許容するSMDの最大値。例: 0.10）
      - **ペナルティ重み（Penalty Weight）の指定**（制約違反時のペナルティ強度）
    - 最適化 & 推定実行
    - 診断・評価（最適化されたモデルでの **Overlap / SMD** 確認）
    - 推定結果レポート確認（ATE/ATTの推定値、点推定値、信頼区間）
    - HTMLレポートダウンロード

  - 因果推論（Doubly Robust DiD法）効果推定パターン
    - データ指定
    - 目的変数（Outcome）の指定
    - **処置変数（Treatment）の指定**（バイナリ変数、自動チェック）
    - **時点変数の指定**
      - **時点識別カラム（Time Col）**（例: year, date 等）
      - **処置後フラグ（Post Col）**（バイナリ変数、自動チェック）
    - **データ構造（Design）の指定**
      - **パネルデータ（panel）**: 同一個体を複数時点で観測
      - **繰り返しクロスセクション（repeated_cross_section）**: 時点ごとに異なる個体を観測
    - **個体ID変数の指定**（パネルデータの場合は**必須**。各個体を識別するキー）
    - **推定対象（Estimand）の指定**
      - **ATT**（処置群における平均処置効果）：DiDの標準的な推定対象（※目的変数がBinaryの場合はATTのみサポート）
    - 共変量指定（目的変数と処置変数と個体ID変数以外をデフォルトで指定、ユーザーが手動で除外指定）
    - 傾向スコア・結果モデルの設定（DR法と同様）
      - **クロスフィッティングの有無**（DefaultはTrue：True/False）
      - **傾向スコアモデルのハイパーパラメーター**（LightGBM設定）
      - **結果モデルのハイパーパラメーター**（LightGBM設定。※パネルの場合は「差分」に対して学習されます）
      - **傾向スコアのキャリブレーション手法**（Platt / Isotonic）
      - **傾向スコアのクリッピング閾値**
    - 学習・推定実行
    - 診断・評価（**Overlap / SMD: 標準化平均差** 等のバランス確認）
    - 推定結果レポート確認（ATTの点推定値、信頼区間）
    - HTMLレポートダウンロード

- 因果推論（Doubly Robust DiD法）効果推定パターン＋パラメーター最適化
    - データ指定
    - 目的変数（Outcome）の指定
    - **処置変数（Treatment）の指定**（バイナリ変数、自動チェック）
    - **時点変数の指定**
      - **時点識別カラム（Time Col）**（例: year, date 等）
      - **処置後フラグ（Post Col）**（バイナリ変数、自動チェック）
    - **データ構造（Design）の指定**
      - **パネルデータ（panel）**: 同一個体を複数時点で観測
      - **繰り返しクロスセクション（repeated_cross_section）**: 時点ごとに異なる個体を観測
    - **個体ID変数の指定**（パネルデータの場合は**必須**。各個体を識別するキー）
    - **推定対象（Estimand）の指定**
      - **ATT**（処置群における平均処置効果）：DiDの標準的な推定対象（※目的変数がBinaryの場合はATTのみサポート）
    - 共変量指定（目的変数と処置変数と個体ID変数以外をデフォルトで指定、ユーザーが手動で除外指定）
    - モデル学習・探索の基本設定
      - **クロスフィッティングの有無**（DefaultはTrue：True/False）
      - **傾向スコアのキャリブレーション手法**（Platt Scaling / Isotonic Regression）
      - **傾向スコアのクリッピング閾値**
    - チューニング設定（最適化）
      - **チューニング実行の有無**（Enabled=True）
      - **試行回数（n_trials）の指定**（例: 30回）
      - **探索プリセットの指定**（Standard / Fast）
      - **最適化ターゲット（Objective）の指定**
        - **DR-DiD Balance Priority (drdid_balance_priority)**: 共変量バランス（SMD）の確保を最優先し、その制約下で精度を最適化（Default）
        - **DR-DiD Standard Error (drdid_std_error)**: 推定量の標準誤差の最小化のみを目指す
        - **DR-DiD Overlap Penalty (drdid_overlap_penalty)**: 傾向スコアの重なり（Overlap）の確保を重視する
      - **バランス閾値（Balance Threshold）の指定**（Balance Priority選択時、許容するSMDの最大値。例: 0.10）
      - **ペナルティ重み（Penalty Weight）の指定**
    - 最適化 & 推定実行
    - 診断・評価（最適化されたモデルでの **Overlap / SMD** 確認）
    - 推定結果レポート確認（ATTの点推定値、信頼区間）
    - HTMLレポートダウンロード

  - シミュレーション（Counterfactual）実行パターン
    - 学習済みRegression/Binary/Multiclassモデル（アーティファクト）指定
    - ベースラインデータ指定
    - **シナリオ定義**（特定の変数に対する増減・固定などの操作定義）
    - シミュレーション実行
    - 結果比較（ベースライン予測値 vs シナリオ予測値の差分）
    - 結果ダウンロード

  - モデルエクスポートパターン
    - 学習済みモデル（アーティファクト）指定
    - **出力フォーマット指定**（Python Native / ONNX）
    - エクスポート実行
    - 検証レポート確認（推論結果の一致性確認）
    - モデルファイルダウンロード

---

### テスト要件

#### Stage 1 テスト

1. **1-A: JST時刻表示**
   - Runs テーブルの全タイムスタンプカラム（created_at, started_at, finished_at）が JST 形式で表示されること。
   - Export ファイル名のタイムスタンプが JST であること。
   - UTC タイムスタンプが `None` の場合に "n/a" が表示されること（既存テスト契約維持）。

2. **1-B: レポートダウンロード**
   - HTMLレポートのエクスポートジョブ完了後、ブラウザダウンロードがトリガーされること。
   - Excelレポートのエクスポートジョブ完了後、同様にダウンロードがトリガーされること。
   - ジョブ失敗時にエラーメッセージが表示され、ダウンロードがトリガーされないこと。
   - `dcc.Download` コンポーネントがレイアウトに存在すること。

3. **1-C: 学習曲線表示**
   - `training_history` が存在する Artifact を選択した場合に学習曲線が描画されること。
   - `training_history` が `None` の Artifact（因果推論等）でエラーにならず空グラフが表示されること。
   - フォルドごとの曲線と平均曲線が正しくプロットされること。

#### Stage 2 テスト
- ユースケース分析の成果物（マトリクス、ギャップ一覧）が作成されていること。
- Phase 26.2 実装計画ドラフトが DESIGN_BLUEPRINT に追記されていること。

### 成功基準

1. **Stage 1**: 3件のバグ修正が完了し、全既存テストがパスすること。
2. **Stage 2**: 全ユースケースパターンの現行GUI対応状況が文書化されていること。
3. **後方互換**: 既存の GUI テストが全パスし、Stable API の互換性が維持されること。

### Stage 2 成果物（2026-02-16）

#### A. 現行GUI対応状況マトリクス

| ユースケース | 現行GUI対応 | 現行導線 | 主なギャップ | Phase26.2 での扱い |
|---|---|---|---|---|
| Regression/Binary/Multiclass 学習・評価 | 部分対応 | Data → Target → Validation → Train → Run → Results | Target/Validation の初学者向けガード説明が不足 | 導線説明カードとガードレールUI統合を追加 |
| Regression/Binary/Multiclass チューニング学習・評価 | 部分対応 | Train（tuning有効化）→ Run → Results | チューニング設定の推奨値プリセット説明不足 | Objective別テンプレートと推奨値ヘルプ追加 |
| 学習済みモデルで予測値算出 | 部分対応 | Run（evaluate） | 予測専用ウィザード不在、出力ダウンロード導線が弱い | Evaluate専用プリセット導線を Results/Runs から追加 |
| 学習済みモデルの別データ再評価 | 部分対応 | Results（Re-evaluate）/Run（evaluate） | 特徴量一致チェックの可視化が不足 | 事前診断結果を実行前に表示 |
| Frontier 学習・評価 | 部分対応 | Target/Train で frontier 設定 → Run | Alpha設定の意味説明不足 | Frontier専用ヘルプとデフォルト候補を追加 |
| Frontier チューニング学習・評価 | 部分対応 | Train（tuning）→ Run | pinball最適化意図の説明不足 | Objective説明と推奨探索範囲を追加 |
| DR 効果推定 | 部分対応 | Target（causal）→ Validation/Train → Run | estimand/treatment/unit_id の入力ガイド不足 | Causal専用フォームセクション分離 |
| DR 効果推定 + 最適化 | 部分対応 | Train（tuning objective）→ Run | balance/objectiveの選択基準が不明瞭 | 目的別プリセットと警告文を追加 |
| DR-DiD 効果推定 | 部分対応 | Target/Validation/Train → Run | panel/repeated_cs 条件の入力ミス防止が弱い | design別必須項目の動的ガード追加 |
| DR-DiD 効果推定 + 最適化 | 部分対応 | Train（tuning）→ Run | objective差の説明不足 | DR-DiD objectiveの比較ヘルプ追加 |
| Counterfactual シミュレーション | 部分対応 | Run（simulate） | シナリオ作成UIがなくファイルパス手入力依存 | Scenario DSL 補助UIは別Phaseで設計（26.2は導線整備まで） |
| モデルエクスポート（Python/ONNX） | 部分対応 | Run（export） | Export設定と成果物参照の導線分断 | Runs/Results から export action へショートカット追加 |
| HTML/Excel レポート出力 | 対応 | Results（export） | ジョブ完了後のブラウザDLが未実装（Stage1で修正） | Stage1で解消、26.2では文言/再実行導線を改善 |

#### B. ギャップ一覧（優先度付き）

| Priority | ギャップ | 影響範囲 | 解消方針（Phase26.2） |
|---|---|---|---|
| P0 | Causal系（DR/DR-DiD）の必須項目ガイド不足 | Target / Validation / Train / Run | Causal専用セクション化 + 必須入力のインライン診断 |
| P0 | Evaluate/Export/Simulate が Run 画面の自由入力に依存 | Run / Results / Runs | ユースケース別プリセット導線を追加し手入力を削減 |
| P1 | チューニング objective と指標の関係が不明瞭 | Train / Results | objective説明、推奨値、代表指標マッピング表示 |
| P1 | Frontier の Alpha 設定支援不足 | Target / Train | Alpha の用途ヘルプと推奨プリセット追加 |
| P1 | 実行前チェック結果の視認性不足 | Validation / Run | ガードレール結果を severity順に固定表示 |
| P2 | Compare/Runs から再学習導線が弱い | Runs / Compare / Train | Clone時に「次にやること」を自動表示 |
| P2 | Export/Artifact 操作結果のフィードバック不足 | Results / Runs | 成功/失敗ステータスと再試行導線を統一 |

#### C. Phase 26.2 実装計画ドラフト

1. 実装順序（画面単位）
   - Step 1: Target 画面のユースケース別補助UI（task type / causal分岐）
   - Step 2: Validation 画面の必須条件ガード強化（timeseries/causal）
   - Step 3: Train 画面の objective/パラメータ説明テンプレート追加
   - Step 4: Run 画面に action プリセット導線（fit/tune/evaluate/simulate/export）
   - Step 5: Runs/Results から再実行・再評価のショートカット導線統一

2. 依存関係
   - 26.2 は 26.1 Stage 1 完了（JST/Export DL/Learning Curves 修正）を前提とする。
   - Core/API 非変更。GUI adapter 層のみ更新。

3. 完了条件
   - 主要ユースケース（学習/チューニング/再評価/因果推定/エクスポート）で、手入力必須項目が現状比で減少していること。
   - ガードレール警告が実行前に視認可能で、修正導線が明示されること。
   - 既存GUIテスト + 26.2追加テストが全てパスすること。

4. テスト観点（26.2追加予定）
   - `tests/test_gui_target_page.py`: causal/frontier 分岐の必須項目ガイド
   - `tests/test_gui_validation_page.py`: split/causal の必須条件表示
   - `tests/test_gui_train_page.py`: objective説明と推奨値のUI契約
   - `tests/test_gui_run_presets.py`（新規予定）: actionプリセットで必要入力が自動補完されること
   - `tests/test_gui_e2e_flow.py`: 主要ユースケースの完遂導線

### 対象ファイル（Stage 1）

| ファイル | 変更内容 |
|----------|----------|
| `src/veldra/gui/app.py` | 1-A: JST変換適用箇所確認、1-B: ダウンロードコールバック追加、1-C: `training_history` 取得ロジック修正 |
| `src/veldra/gui/pages/results_page.py` | 1-B: `dcc.Download` + ダウンロードボタン追加 |
| `src/veldra/gui/services.py` | 1-A: Export ファイル名タイムスタンプ JST 化 |
| `tests/test_gui_phase26_1.py` | 新規: Stage 1 の3件に対するユニットテスト |

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
