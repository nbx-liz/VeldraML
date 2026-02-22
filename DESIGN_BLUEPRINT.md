# DESIGN_BLUEPRINT

最終更新: 2026-02-22

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
  - evaluate: `rmse`, `mae`, `r2`, `huber`
- binary:
  - predict: `p_cal`, `p_raw`, `label_pred`
  - evaluate: `auc`, `logloss`, `brier`, `accuracy`, `f1`, `precision`, `recall`, `threshold`, `top5_pct_positive`
  - `train.top_k` 設定時は `precision_at_k` を追加返却
  - OOF 確率校正（既定: platt）
- multiclass:
  - predict: `label_pred`, `proba_<class>`
  - evaluate（基本）: `accuracy`, `macro_f1`, `logloss`, `balanced_accuracy`, `brier_macro`
  - evaluate（条件付き）: `ovr_roc_auc_macro`, `average_precision_macro`
    - 評価データで一部クラスが欠落し計算不能な場合は返却を省略（評価自体は継続）
- frontier:
  - predict: `frontier_pred`（target があれば `u_hat`）
  - evaluate: `pinball`, `mae`, `mean_u_hat`, `coverage`
- timeseries（`split.type="timeseries"` 時の特記事項）:
  - CV 先頭区間は OOF 未スコアになり得る（partial OOF 許容）
  - calibration / threshold 最適化 / mean metrics は OOF 有効行のみで計算
  - `training_history` に `oof_total_rows`, `oof_scored_rows`, `oof_coverage_ratio` を記録
  - 非 timeseries（kfold/stratified/group）は OOF 欠損をエラーとする従来挙動を維持

### 3.3 因果推論の実装範囲
- DR (`causal.method="dr"`): ATT 既定、ATE 任意
- DR-DiD (`causal.method="dr_did"`): 2時点 MVP
  - `design="panel"`
  - `design="repeated_cross_section"`
  - `task.type="binary"` は Risk Difference ATT で解釈（estimand は ATT のみ）
- propensity は校正後確率（既定: `platt`）を使用

## 4. 実装済みフェーズ要約

### 4.1 全体サマリー（Phase 1〜22）
- Phase 1-4: 基本 API, Artifact, regression + examples
- Phase 5-7: binary/multiclass + OOF 校正 + threshold opt-in
- Phase 8-12: tune/simulate/export/frontier/timeseries 拡張
- Phase 13-18: ONNX 拡張, evaluate(config, data), notebook整備
- Phase 19-20: DR, DR-DiD, causal tuning objective
- Phase 21: Dash GUI MVP（Config編集 + Run実行 + Artifact閲覧）
- Phase 22: Config migration utility MVP（`veldra config migrate`）

### 4.2 Phase 23〜35 実装サマリー（完了）
- Phase 23: DR-DiD Binary + 最小診断を実装し、2時点 ATT 推定契約を確立。
- Phase 24: Causal tuning を balance-priority 既定へ移行し、balance 閾値契約を導入。
- Phase 25〜30: GUI 運用基盤を段階実装（非同期ジョブ、cancel/retry枠、config migrate、テンプレート）。
- Phase 31〜34: GUIの可視化・性能・Studio UX を実装し、運用導線を安定化。
- Phase 35: quick reference 01〜13 本線化、timeseries partial OOF 契約、評価系テストギャップ（G1/G2/G3）を解消。

## 5. GUI実装（要約）
- GUI は adapter 層限定（Core 非依存、`veldra.api.runner` 呼び出し専用）を維持。
- Phase21〜34 で以下を実装済み:
  - Config編集 / Run実行 / Artifact閲覧
  - 非同期ジョブ管理（永続化、cancel/retry枠）
  - テンプレート運用、比較可視化、Studio/Guided導線
- 運用契約:
  - SQLite 永続ジョブ
  - 構造化ログ（`run_id`, `artifact_path`, `task_type`）
  - `gui` / `gui_e2e` / `notebook_e2e` marker 分離

## 6. 公開API互換ポリシー
- 既存の公開シグネチャは維持する。
- 新機能は原則 opt-in で追加する。
- 破壊的変更が必要な場合は config_version と migration 方針を必須化する。

## 7. HISTORYとの関係
- 詳細な意思決定と時系列ログは `HISTORY.md` を正とする。
- 本書は「現時点の設計状態」を示す要約ドキュメントとして維持する。
