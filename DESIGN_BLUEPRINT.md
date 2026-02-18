# DESIGN_BLUEPRINT

最終更新: 2026-02-18

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
- Phase 26.1: UI改善（バグ修正3件 + ユースケース駆動UI再構成設計）← **完了**
- Phase 26.2: ユースケース駆動UI改善（ヘルプUI基盤 + 画面別ガイド強化）← **完了**
- Phase 26.3: ユースケース詳細化（diagnostics ライブラリ + observation_table + Notebook 完全版 + 実行証跡）← **完了**
- Phase 26.4: Notebook 教育化 & テスト品質強化 ← **完了**
- Phase 26.5: 13.3 A/B Notebook適用 + gui_e2e安定化 ← **完了**
- Phase 26.6: テスト品質向上（命名整理 + カバレッジ強化）← **完了**

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
### 要約
- `causal.method="dr_did"` で `task.type="binary"` を正式対応し、推定量は Risk Difference（ATT）として解釈する契約を追加。
- `panel` / `repeated_cross_section` の両設計で実行可能。
- 診断返却を `overlap_metric`, `smd_max_unweighted`, `smd_max_weighted` で統一。

### 互換性
- `estimate_dr` の公開導線は維持し、`CausalResult.metadata` に追加情報（`outcome_scale`, `binary_outcome`）を付与する拡張に限定。

## 11. Phase 24（Causal Tune Balance-Priority）
### 要約
- Causal tuning の既定目的を SE 偏重から balance-priority（`dr_balance_priority` / `drdid_balance_priority`）へ移行。
- `tuning.causal_balance_threshold`（既定 0.10）を導入し、`smd_max_weighted <= threshold` を主要判定に採用。
- trial attributes に balance violation と objective stage を保存し、監査可能性を強化。

### 互換性
- 追加は opt-in 設定と既定値更新に限定し、Stable API は非破壊。

## 12. Phase 25（完了）: GUI運用強化
### 要約
- GUI に非同期ジョブ実行（SQLite 永続 + single worker + best-effort cancel）を導入。
- `/config` に config migrate（preview/diff/apply）を統合。
- callback の例外処理とフォールバックを整理し、Data→Run→Results 導線を安定化。

### 検証スナップショット
- 当時の全体回帰は `pytest -q` で green（詳細件数は `HISTORY.md` を正とする）。

## 12.5 Phase25.5: テスト改善計画（DRY / 対称性 / API化）(完了)
### 要約
- `tests/conftest.py` に共通 fixture/factory を集約し、重複を段階移行で削減。
- regression の契約テスト（fit/predict/evaluate/artifact roundtrip）を補完。
- CV split と causal diagnostics の重複ロジックを公開ユーティリティへ寄せた。

### 互換性
- `veldra.api.*` と Artifact 契約は維持。

## 12.6 Phase25.6: GUI UXポリッシュ（CSS/HTML限定）(完了)
### 要約
- スコープを CSS/HTML に限定し、機能契約は変更なし。
- 可読性、split 警告、workflow 導線、表示ノイズ削減を中心に UI を改善。

## 12.7 Phase25.7: LightGBMの機能強化（完了）
### 要約
- `TrainConfig` を拡張（`num_boost_round`, `early_stopping_validation_fraction`, `auto_class_weight`, `class_weight`）。
- 全 task で `num_boost_round` を設定駆動化し、ES 用 validation 分割を導入（OOF 監視は不使用）。
- `training_history.json` を Artifact へ永続化し、GUI/migration と連携。

### 互換性
- 既存シグネチャは維持し、後方互換の既定値（300 boost round）を保持。

## 12.8 Phase25.8: LightGBMのパラメーター追加（完了）
### 要約
- `TrainConfig` に `auto_num_leaves`, ratio 系、`feature_weights`, `top_k` を追加。
- `precision_at_k` を binary fit/tune/evaluate に統合。
- tuning `standard` 空間を拡張し、GUI と RunConfig reference を同期。

### Decision（要点）
- `train.top_k` 指定時の ES 監視は `precision_at_{k}` を優先。
- `train.feature_weights` は未知特徴量キーを validation error とする。

## 12.9 Phase25.9: LightGBM機能強化の不足テスト補完（完了）
### 要約
- 25.7/25.8 の不足テストを追加し、必要最小限の本体修正まで同フェーズで閉じた。
- `feature_weights` 適用時は `feature_pre_filter=False` を明示して契約を固定。
- causal の group 取り扱いは `group_col` 維持 + `unit_id_col` 補完で確定。

## 13 Phase 26: UX/UI 改善
### 要約
- 画面構成を `Data -> Target -> Validation -> Train -> Run -> Results (+ Runs/Compare)` に再編。
- GUI 入力から RunConfig を再構築できる状態管理を導入。
- Results に learning curves / config / export 導線を統合。

### 互換性
- Core 非依存（adapter 層）と Stable API / RunConfig / Artifact 契約を維持。

## 13.1 Phase 26.1: UI改善
### 要約
- Stage1: JST 表示統一、Results からのダウンロード導線、learning history 参照修正を実施。
- Stage2: UC ギャップ（説明不足・導線不足）を整理し、26.2 実装計画を確定。

## 13.2 Phase 26.2: ユースケース駆動UI改善
### 要約
- GUI adapter のみを対象に Step0-5（監査基盤、help UI、Target/Validation/Train/Run 強化）を実施。
- UC-1〜UC-10 実行証跡、Notebook 契約テスト、Playwright E2E を整備。
- legacy notebook/stub を撤去し、canonical 導線へ集約。

### 運用
- 実行証跡は `examples/out/phase26_*/summary.json` と生成物実体で管理。

## 13.3 Phase26.3: ユースケース詳細化
### 要約
- `veldra.diagnostics`（importance / shap / metrics / plots / tables / causal_diag）を新設。
- 全 task の CV 出力に `observation_table` を導入し、Artifact 永続化を対応。
- DR/DR-DiD に nuisance diagnostics を追加。
- Quick Reference notebook を実務向け（診断・可視化・CSV）に拡張。

### Decision（要点）
- `config_version=1` 維持のまま optional 拡張で表現力を追加。
- notebook 証跡検証は常時契約テスト + `notebook_e2e` 分離運用を確定。
- `tuning.metrics_candidates` は objective と独立した task 別許可セットで検証。

## 13.4 Phase26.4: Notebook 教育化 & テスト品質強化
### 要約
- notebook を `tutorials`（教材）と `quick_reference`（実行証跡）に分離。
- 命名を英語スネークケースへ統一し、`reference_index` を canonical ハブ化。
- 教材品質（概念説明・解釈・失敗例）とテスト品質改善の方針を固定。

## 13.5 Phase26.5: 13.3 A/B Notebook適用 + gui_e2e 安定化
### 要約
- 13.3 A/B 契約を canonical notebook（`quick_reference` / `tutorials`）へ再適用。
- tuning `standard` 既定探索空間を 13.3 B 準拠へ更新。
- GUI 実装は変えず、Playwright 側の待機/操作戦略を堅牢化して flaky を低減。

### Decision（confirmed）
- A/B 適用対象は canonical notebook のみに限定。
- `gui_e2e` 不安定性はテスト層修正で収束させる。

## 13.6 Phase26.6: テスト品質向上（命名整理 + カバレッジ強化）
### 要約
- notebook テスト命名を phase 番号依存から責務ベース（tutorial/quickref/execution/reference）へ再編。
- 重複テストを統合し、execution evidence 検証を一本化。
- Critical モジュール（artifact store/config io/causal diagnostics）と edge/numerical/data-split テストを拡充。

### 検証スナップショット
- `-m "not gui_e2e and not notebook_e2e"` 回帰は green（件数・詳細は `HISTORY.md` を正とする）。

### Decision（confirmed）
- 命名規約は対象種別 + 連番を採用し、phase 依存を廃止。
- Stage A/B を段階実施し、非 E2E 回帰グリーンを完了条件として固定。

### 補足
- Phase23〜26.6 の詳細仕様（対象ファイル一覧、テストコマンド、時系列判断）は `HISTORY.md` 各エントリを正とする。

## 13.7 Phase26.7: コアロジック構造改善リファクタリング

最終更新: 2026-02-18

### 目的
コアロジックの長期的な保守性・拡張性・テスト容易性を向上させる。GUI および大規模データ処理（分散処理等）はスコープ外とし、モデリング層・設定層・因果推論層の構造改善に焦点を当てる。

### 背景
モデリング層の4学習ファイル（binary/regression/multiclass/frontier）間で90%以上のコード重複が確認された。`_booster_iteration_stats`は4箇所で完全コピー、CVループ本体・最終モデル学習ブロック・OOF初期化も構造的に同一。また`config/models.py`の`_validate_cross_fields`が310行に肥大化し、`causal/dr.py`ではLightGBMが直接インスタンス化されている。

当初提案7項目を批判的に精査し、3項目を採用、4項目を却下/延期とした。

### 却下・延期した提案

**却下: Artifact保存DTO化** — `Artifact`クラス自体が既にDTOとして機能。中間層は不要。Step 3完了後に`getattr`フォールバックが自然解消する。

**却下: 前処理パイプライン化** — `_build_feature_frame`の共通部分は約15-20行。タスク固有のバリデーションと戻り値型が異なる（2-tupleと3-tuple）。Step 3のCVRunner統合で共通部分は自然に吸収されるため独立パッケージは過剰。

**却下: 定数のEnum化** — Pydantic `Literal`型が既にPydantic v2のイディオムに沿い型安全性を提供済み。`StrEnum`変換は全型注釈の書き換え・YAML直列化変更・安定API契約破壊を伴い、コスト対効果が合わない。

**延期: ユーティリティ関数の再配置** — Step 3完了後に`_cv_runner.py`の内部詳細となるため、先に移動すると二重作業になる。

### Step 1: RunConfigバリデーションの分離（低リスク・小規模）

**対象ファイル:** `src/veldra/config/models.py`

**方針:** `_validate_cross_fields`（L209-518, 310行）から、単一サブコンフィグ内で完結するバリデーションを各モデルの`@model_validator(mode="after")`へ移動する。クロスコンフィグ検証（task×split, task×tuning等）はRunConfigに残す。

**`SplitConfig`へ移動（現L214-257, 約44行）:**
- timeseries → time_col必須、group → group_col必須
- timeseries固有: gap>=0, embargo>=0, test_size>=1, blocked→train_size必須
- 非timeseries: timeseries専用フィールドがデフォルト以外なら拒否

**`TrainConfig`へ移動（現L273-326から自己完結部分, 約30行）:**
- num_boost_round >= 1
- early_stopping_validation_fractionの範囲 (0, 1)
- auto_num_leaves有効時のnum_leaves_ratio範囲 (0, 1]
- min_data_in_leaf_ratio / min_data_in_bin_ratioの範囲とlgb_params競合チェック
- feature_weightsの値 > 0

**RunConfigに残すもの:** task.type依存の検証全て（metrics許可リスト、top_k、class_weight、postprocess、tuning、causal）

**効果:** `_validate_cross_fields`が約310行→約230行に縮小。バリデーションがフィールド定義の近くに移動し可読性向上。

**検証:**
```bash
pytest tests/test_config*.py tests/test_api_surface.py -v
```

### Step 2: 因果推論の学習器抽象化（低リスク・小規模）

**対象ファイル:**
- 新規: `src/veldra/causal/learners.py`
- 修正: `src/veldra/causal/dr.py`

**方針:** `typing.Protocol`でnuisanceモデルのインターフェースを定義し、ファクトリ関数でデフォルト実装を提供する。ABCではなくProtocolを使用（コードベース全体が構造的型付けを採用しているため）。

```python
# src/veldra/causal/learners.py
class PropensityLearner(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

class OutcomeLearner(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

def default_propensity_factory(seed: int, params: dict) -> lgb.LGBMClassifier: ...
def default_outcome_factory(seed: int, params: dict) -> lgb.LGBMRegressor: ...
```

**変更箇所:**
- `_fit_propensity_model`（dr.py L144-163）: ファクトリ関数経由でインスタンス化
- `_fit_outcome_model`（dr.py L172-192）: 同上
- `run_dr_estimation`シグネチャに`propensity_factory`/`outcome_factory`オプション引数を追加（デフォルトは現行動作を保持）
- `dr_did.py`は`dr.py`に委譲しているため自動的に恩恵を受ける

**効果:** テスト時に軽量ダミーモデルを注入可能。将来のアルゴリズム差し替え（XGBoost, Ridge等）が容易に。

**検証:**
```bash
pytest tests/test_causal*.py -v
```

### Step 3: CVループの統合（高リスク・大規模）

**対象ファイル:**
- 新規: `src/veldra/modeling/_cv_runner.py`
- 修正: `src/veldra/modeling/binary.py`
- 修正: `src/veldra/modeling/regression.py`
- 修正: `src/veldra/modeling/multiclass.py`
- 修正: `src/veldra/modeling/frontier.py`
- 修正: `src/veldra/modeling/utils.py`

**設計方針:** ABC継承ではなく、**dataclassベースのStrategy + 共通ランナー関数**を採用する。理由: コードベースにABCは一切なく、フラットなdataclass + 関数が一貫したパターンであるため。

```python
# src/veldra/modeling/_cv_runner.py
@dataclass(slots=True)
class TaskSpec:
    build_features: Callable[[RunConfig, pd.DataFrame], tuple]
    build_lgb_params: Callable[[RunConfig, int], dict]
    init_oof: Callable[[int, ...], np.ndarray]    # 1D or 2D
    extract_fold_pred: Callable[[Any, pd.DataFrame, int], np.ndarray]
    compute_fold_metrics: Callable[[np.ndarray, np.ndarray, ...], dict]
    build_output: Callable[..., Any]               # タスク固有のTrainingOutput

def run_cv_training(config: RunConfig, data: pd.DataFrame, spec: TaskSpec) -> Any:
    """全タスク共通のCV学習フロー"""
    # 1. timeseries sort (共通)
    # 2. spec.build_features() でX, y（+α）取得
    # 3. iter_cv_splits() でfold分割
    # 4. OOF配列初期化 (spec.init_oof)
    # 5. fold loop: early stopping split → _train_single_booster → predict → metrics
    # 6. final model training
    # 7. training_history組み立て
    # 8. spec.build_output() でタスク固有の出力生成
```

**共通化される重複コード:**
- `_booster_iteration_stats`: 4ファイル完全同一 → `_cv_runner.py`へ統合
- `_train_single_booster`: 4ファイルほぼ同一（multiclassのnum_classのみ差分） → パラメータ化して統合
- CVループ本体: fold iteration, early stopping split, 推論, 履歴記録
- 最終モデル学習ブロック: 4ファイル95%同一
- `_build_feature_frame`の共通部分（target検証, drop_cols除外, feature選択）

**タスク固有のロジック（各ファイルに残る）:**
- `binary.py`: TaskSpec定義 + calibration/threshold後処理
- `multiclass.py`: TaskSpec定義（2D OOF, num_class, softmax正規化）
- `frontier.py`: TaskSpec定義（alpha, quantile metrics）
- `regression.py`: TaskSpec定義（最もシンプル）

**見込み:** 各タスクファイルが260-436行 → 60-100行に削減。共通ランナーは約300-400行。

**リスク軽減策:**
1. 実装前に全テスト（28+ファイル）をパスすることを確認
2. golden output比較テストを追加: 現行コードの出力をキャプチャし、リファクタ後に完全一致を検証
3. 段階的に実装: まずregressionのみ統合→テスト→他タスクを順次移行

**検証:**
```bash
pytest tests/ -v --tb=short
```
全テストパス。特にsmoke test（binary/regression/multiclass/frontier）のmetrics値・OOF形状・training_history構造が完全一致すること。

### 互換性方針
- `veldra.api.*` の公開シグネチャは変更しない。
- 各`TrainingOutput`のフィールド構造は維持する。
- Artifact の保存形式・読み込み互換性は変更しない。

### 成功基準
- 全既存テスト（234テスト）がパスすること
- `_validate_cross_fields`が310行→230行以下に縮小
- `_booster_iteration_stats`の重複が4箇所→1箇所に統合
- CVループの構造的重複が解消されること
- `causal/dr.py`のLightGBM直接インスタンス化が排除されること

### 実装計画確定（2026-02-18）
- 実装順序は Step1 → Step2 → Step3 で固定する。
- 実行粒度は 3PR（Step1/Step2/Step3 分離）で固定する。
- Step3 の同等性判定は「完全一致」を採用し、`run_id` / `artifact_path` / ファイル時刻など非決定項目のみ比較対象外とする。
- PR3 着手前に baseline capture を実施し、`tests/test_phase267_output_parity.py` で4 task（regression/binary/multiclass/frontier）を一括検証する。

### Decision（provisional）
- 内容: Phase26.7 は 3PR + 完全一致ゲートで実装し、各PRで ruff/対象pytest、最終で `-m "not gui_e2e and not notebook_e2e"` 回帰を必須化する。
- 理由: 大規模リファクタの回帰リスクを段階分離しつつ、既存契約の非破壊を機械的に担保するため。
- 影響範囲: `src/veldra/config/models.py`, `src/veldra/causal/dr.py`, `src/veldra/causal/learners.py`, `src/veldra/modeling/_cv_runner.py`, `src/veldra/modeling/{binary,regression,multiclass,frontier}.py`, `tests/test_phase267_output_parity.py`, `HISTORY.md`

### Decision（confirmed）
- 内容: Step1/2/3 を実装し、`run_dr_estimation` の後方互換を維持したまま learner factory 注入を導入、modeling のCV共通化を `_cv_runner.py` へ集約した。
- 理由: コード重複削減と拡張性向上を達成しつつ、Stable API と Artifact 契約を非破壊で維持できたため。
- 影響範囲: `src/veldra/config/models.py`, `src/veldra/causal/{dr.py,learners.py}`, `src/veldra/modeling/{_cv_runner.py,binary.py,regression.py,multiclass.py,frontier.py}`, `tests/test_phase267_output_parity.py`

### 実装結果（2026-02-18）
- Step1: Split/Train の自己完結バリデーションを各 config へ移管し、`_validate_cross_fields` をクロス検証中心へ縮小。
- Step2: nuisance learner Protocol + default factory を追加し、DR の LightGBM 直接生成を排除。
- Step3: CVループ/履歴/final学習を共通ランナーへ統合し、4 task の出力契約を維持。
- 検証: `uv run pytest -q -m "not gui_e2e and not notebook_e2e"` で `704 passed, 11 deselected`。


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
* **優先度マップ固定:** `high=90`, `normal=50`, `low=10`（任意数値priorityは非サポート）
* **公平性方針:** strict priority（Aging/比率制御なし）
* **新規クラス:** `GuiWorkerPool` クラス作成（複数 `GuiWorker` インスタンス管理）
* **排他制御:** データベースロックでthread-safeなジョブクレーム調整
* **UI変更:** Run ページに優先度ドロップダウン追加（Low/Normal/High）
* **並び替え方式:** queued ジョブの priority 変更で順序を制御（Move Up/Down, D&D は非採用）
* **変更制約:** priority変更対象は `status=\"queued\"` のみ

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
* queued以外のpriority変更が拒否され、ユーザーに明示メッセージを返す

### 実装結果（2026-02-18）
* `RunInvocation` / `GuiJobRecord` に `priority`（`low|normal|high`）を追加し、既定値を `normal` で固定。
* `GuiJobStore` に priority 列の後方互換 migration（`PRAGMA table_info` + `ALTER TABLE`）を実装。
* `claim_next_job()` を `priority DESC, created_at_utc ASC` に変更し、strict priority を適用。
* `GuiWorkerPool` を導入し、`--worker-count`（既定1）で並列worker数を設定可能化。
* Run 画面に投入priority選択（`run-priority`）と queued再優先付け（`run-queue-priority` + `run-set-priority-btn`）を追加。

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

### 実装結果（2026-02-18）
* `GuiJobRecord` に `progress_pct` / `current_step` を追加し、`jobs` テーブルへ後方互換 migration を実装。
* `job_logs` テーブル（`seq`順）を追加し、`append_job_log` / `list_job_logs` / `update_progress` を `GuiJobStore` に実装。
* `run_action()` は `job_id` + `job_store` を受ける内部経路を追加し、action別ステップ進捗と失敗ログを永続化。
* `tune` 実行時は runner の trial 完了ログ（`n_trials_done`）を取り込み、進捗率を段階更新。
* `GuiWorker` は started/completed のジョブログ記録と進捗初期化を実装。
* Run 画面は Progress 列を追加し、Job detail を `progress_viewer` ベース（進捗バー + level色分けログ）へ更新。
* ログ上限は 10,000 行/job を採用し、超過時は古いログを削除して `log_retention_applied` を記録。

### Decision（要点）
* Decision: confirmed
  * 内容: Phase28 は polling 継続（`run-jobs-interval`）+ SQLite 永続化で進捗/ログ可視化を実装し、WebSocket/SSE は導入しない。
  * 理由: 既存GUI契約を維持しつつ、最小変更で運用可視性を高めるため。
* Decision: confirmed
  * 内容: ログ保持上限は 10,000 行/job とし、超過時は FIFO で古い行を削除する。
  * 理由: UI応答性とDB肥大抑制を両立するため。

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

### 実装結果（2026-02-18）
* `RetryPolicy` を `RunInvocation` に追加し、`GuiJobRecord` に `retry_count` / `retry_parent_job_id` / `last_error_kind` を追加。
* `jobs` テーブルへ後方互換 migration（`retry_count`, `retry_parent_job_id`, `last_error_kind`）を実装。
* `GuiJobStore` に `is_cancel_requested` / `mark_canceled_from_request` / `create_retry_job` を追加。
* `run_action()` に協調キャンセルチェックポイント（設定読込/データ読込/runner前後）を追加し、`CanceledByUser` を導入。
* `veldra.api.runner.fit/tune/estimate_dr` に `cancellation_hook`（keyword-only, optional）を追加し、既存呼び出し互換を維持。
* `GuiWorker` は `cancel_requested` 競合を `canceled` 終端で処理し、`RetryPolicy` を使う自動リトライ枠（既定0回）を実装。
* Runページに `Retry Task` ボタンを追加し、failed/canceled ジョブの手動再投入をサポート。
* 失敗ジョブ詳細に `next_steps` を表示し、エラー分類（validation/file/permission/memory/timeout/cancel等）を payload 化。

### Decision（要点）
* Decision: confirmed
  * 内容: `RetryPolicy` は GUI adapter の `RunInvocation` にのみ持たせ、Core `RunConfig` には追加しない。
  * 理由: Core/Stable API 契約を維持しつつ、GUI運用機能を最小侵襲で拡張するため。
* Decision: confirmed
  * 内容: 自動リトライ既定は `max_retries=0`（手動リトライ中心）とし、将来有効化可能なバックオフ枠のみ先行実装する。
  * 理由: 不要な自動再実行リスクを避けつつ、運用拡張余地を確保するため。
* Decision: confirmed
  * 内容: 失敗時の Next Step 提示は既知分類のみを対象とし、未知エラーは原文優先とする。
  * 理由: ノイズを抑え、誤誘導を避けるため。

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

### 実装結果（2026-02-18）
* `src/veldra/gui/templates/` に 5 テンプレート（`regression_baseline`, `binary_balanced`, `multiclass_standard`, `causal_dr_panel`, `tuning_standard`）を追加。
* `src/veldra/gui/template_service.py` を新設し、テンプレート読込/検証、localStorage スロット（max 10, save/load/clone, LRU）管理、YAML 変更キー数カウントを実装。
* `src/veldra/gui/components/config_library.py` と `src/veldra/gui/components/config_wizard.py` を追加し、`/train` と `/config` 双方に同等導線を統合。
* `workflow-state` に `template_id`, `template_origin`, `custom_config_slots`, `wizard_state`, `last_validation`, `config_diff_base_yaml` を追加。
* `validate_config_with_guidance()` を導入し、Validate 時のパス付きエラー表示と next-step ガイダンスを提供。
* `run-execute` 前に RunConfig 形式の YAML を検証し、不正時は投入をブロック（既存モック契約と両立する条件付きゲート）。

### Decision（要点）
* Decision: confirmed
  * 内容: テンプレート/保存/ウィザード機能は GUI adapter 内で完結し、Core RunConfig schema は `config_version=1` を維持する。
  * 理由: Stable API と Artifact 契約への影響を避けつつ運用改善を達成するため。
* Decision: confirmed
  * 内容: `/train` と `/config` は UI を分けつつ同一コールバック関数を再利用し、機能差分を禁止する。
  * 理由: 導線互換と保守性（重複実装回避）を両立するため。

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

## 22. Phase35: GUI改善
### 目的
- GUIのユーザビリティと安定性を向上させ、ユーザーがモデリング・チューニングプロセスをより直感的に理解し操作できるようにする。
- 画面遷移が多すぎるため、ユーザビリティが悪化している。モデルの学習・評価を1画面で完結させ、必要に応じてモーダルやドロップダウンで詳細を表示する構成に変更する。
- 学習済みモデルの読込と予測・評価を1画面で完結させる。
### 2. 画面草案
A. 学習モード (Train Mode)
目的: データの探索、パラメータ調整、モデル学習の高速な反復。
+-----------------------------------------------------------------------------------+
| [ Veldra Studio ]    モード: [ ● 学習 | ○ 推論 ]    [ 📁 モデル管理 ]           |
+-----------------------+---------------------------+-------------------------------+
| 1. データと目的 (SCOPE) | 2. 戦略設定 (STRATEGY)    | 3. 実行と結果 (ACTION)        |
+-----------------------+---------------------------+-------------------------------+
| ■ データソース        | [ Validation ][ Model ]   | ステータス: 準備完了 (READY)  |
| +-------------------+ | +-----------------------+ | +---------------------------+ |
| | housing_data.csv  | | モデル選択 (Estimator)  | | [ ▶ 実験を開始 (RUN) ]      | |
| | (1460 rows, 81col)| | [ LightGBM (GBDT) (v) ] | +---------------------------+ |
| +-------------------+ | |                       |                               |
|                       | | 学習率 (Learning Rate)| 実行ログ (Console):           |
| ■ ターゲット列        | | 0.01 [======|====] 0.5| > Waiting for command...      |
| [ price (v) ]         | | [ 0.05 ]              | >                             |
|                       | |                       | >                             |
| ■ タスクタイプ        | | 木の深さ (Max Depth)  | >                             |
| [ 回帰 (Regression) ] | | [ -1 ] (自動)         | >                             |
| (自動判定: 数値型)    | |                       | >                             |
|                       | | 早期終了 (Early Stop) |                               |
|                       | | [ 100 ] rounds        |                               |
|                       | +-----------------------+ | 速報メトリクス (Quick Result):|
|                       |                         | +---------------------------+ |
|                       |                         | | (実行後にここに表示)      | |
|                       |                         | +---------------------------+ |
+-----------------------+---------------------------+-------------------------------+

---

B. 推論モード (Inference Mode)
目的: 過去に作成したモデルのロード、新規データに対する予測、結果の出力。
+-----------------------------------------------------------------------------------+
| [ Veldra Studio ]    モード: [ ○ 学習 | ● 推論 ]    [ 📁 モデル管理 ]           |
+-----------------------+---------------------------+-------------------------------+
| 1. モデルと入力 (SCOPE)| 2. モデル仕様 (SPEC)      | 3. 予測と出力 (ACTION)        |
+-----------------------+---------------------------+-------------------------------+
| ■ ロード中モデル情報  | ■ 学習時設定 (Read-Only)  | ステータス: 準備完了 (READY)  |
| +-------------------+ | タスク: 回帰              | +---------------------------+ |
| | ID: model_2023...A| | モデル: LightGBM          | | [ ▶ 予測を開始 (PREDICT) ]  | |
| | Target: price     | | Best RMSE: 24500        | +---------------------------+ |
| +-------------------+ |                           |                               |
|                       | ■ 必要な入力特徴量リスト  | 予測結果プレビュー (Preview): |
| ■ 推論用データ入力    | (この列が入力に必須です)  | +---------------------------+ |
| +-------------------+ | +-----------------------+ | | (実行後にここに表示)      | |
| | new_houses.csv    | | - GrLivArea (数値)      | | ID, GrLivArea, ..., **Pred**| |
| | (50 rows)         | | - YearBuilt (数値)      | | 01, 1500, ..., **145000** | |
| +-------------------+ | | - OverallQual (数値)    | | 02, 2100, ..., **210000** | |
|                       | | - GarageCars (数値)     | +---------------------------+ |
| ■ (任意) 正解ラベル列 | | - GarageCars (数値)     | +---------------------------+ |
| [ (未選択) (v) ]      | +-----------------------+ | [ ⬇️ CSVダウンロード ]        |
| (精度評価を行う場合)  |                           |                               |
+-----------------------+---------------------------+-------------------------------+

### 3. 機能要件詳細
3.1 ヘッダー・グローバル制御
- モード切替スイッチ: dbc.RadioItems 等で実装。「学習」と「推論」を切り替える。
- モデル管理ボタン (Model Hub): クリックすると右側から dbc.Offcanvas (サイドバー) が出現。
  - モデル一覧: dash-ag-grid で過去の Artifact を一覧表示。
  - Load: 選択したモデルをロードし、強制的に「推論モード」へ遷移させる。
  - Delete: 不要なモデルの削除。

---

3.2 パネル別機能仕様 (モード依存)
パネル,コンポーネント,学習モード (Train),推論モード (Inference)
Left,Scope,Input Data: 学習データアップロードTarget: ターゲット列選択Task: タスクタイプ推定/指定,Model Info: ロード中モデルID/ターゲット名Predict Data: 推論用データアップロードEval Target: (任意) 正解ラベル列選択
Center,Strategy / Spec,Tabs: 設定タブ切り替え - Valid: 分割法 (KFold/Group) - Model: GBDTパラメータ - Tuning: Optuna設定,Spec View: (Read-only) - 学習時のConfig要約 - Feature Schema: 必須特徴量リスト - (欠損列の警告表示など)
Right,Action,Run: 学習ジョブ投入Log: リアルタイムログ出力Quick KPI: 学習完了後の主要指標,Predict: 推論ジョブ投入Preview: 結果テーブル (先頭100行)Export: 全件CSVダウンロード

### 4. 技術実装計画

#### 4.1 フロントエンド構成 (`src/veldra/gui`)

* **`pages/studio_page.py`**: 新規作成。3ペインのグリッドレイアウトを定義。
* **`components/studio_parts.py`**: モードごとに切り替わる部品群（学習用フォーム、推論用ビュー）をコンポーネント化して実装。
* **State Management (`dcc.Store`)**:
    * `store-studio-mode`: `"train"` | `"inference"`
    * `store-loaded-artifact`: ロード中の Artifact ID / Path
    * `store-input-meta`: アップロードデータの列情報（学習・推論共通）

---

#### 4.2 バックエンド API 拡張 (`src/veldra/api`)

推論モードのUX向上のため、以下のAPIまたはヘルパー関数を整備する。

* **`get_artifact_spec(path)`**
    * モデルが必要とする特徴量リスト（名前、型）と、学習時のConfig要約を返す軽量API。
    * Artifact全体をロードせずにメタデータのみを取得する。
* **`validate_prediction_data(artifact, dataframe)`**
    * 推論実行前に、入力データに必要な列が含まれているかチェックし、不足があればエラー/警告を返すロジック。

---

#### 4.3 開発フェーズ

1.  **Phase 1: レイアウト移行**
    * 現行のページ遷移を廃止し、学習モードの3ペイン画面 (`studio_page`) を実装。
2.  **Phase 2: モデル管理と推論**
    * モデル管理サイドバーと推論モードの実装。モード切替ロジックの結合。
3.  **Phase 3: マイグレーション**
    * 古いページ (`target_page`, `train_page` 等) の完全削除。
