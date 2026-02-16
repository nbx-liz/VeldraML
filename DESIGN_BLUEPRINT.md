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

## 12.5 Phase25.5: テスト改善計画（DRY / 対称性 / API化）(完了)

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

## 12.6 Phase25.6: GUI UXポリッシュ（CSS/HTML限定）(完了)

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

## 12.7 Phase25.7: LightGBMの機能強化（完了・検証済み）

### 目的
- 目的変数の自動判定機能
- バリデーションデータの適切な自動設定
- ImbalanceデータにWeightを自動適用する機能
- `num_boost_round` の設定可能化（現在300にハードコード）
- 学習曲線の早期停止機能（Learning Curve 監視 および Early Stopping）
- 学習曲線データの記録・Artifact保存（可視化はPhase30で対応）
- GUI対応（新パラメーター、ラベル修正）
- Config migration（`lgb_params.n_estimators` → `train.num_boost_round`）

### 現状分析

| 機能 | 現状 | 対応 |
|------|------|------|
| `num_boost_round` | 300にハードコード。GUIの `n_estimators` は `lgb_params` に格納されるが `lgb.train()` では無視される | `TrainConfig` に昇格 |
| 目的関数 | タスクごとにハードコード（`binary`, `regression` 等）。自動判定は既存で機能している | ユーザー上書きは `lgb_params.objective` 経由で既に可能。GUIドロップダウン追加 |
| クラス不均衡 | 未実装 | `is_unbalance` / sample weight の自動・手動設定 |
| バリデーション分割 | CVフォールドから適切に生成されている。ただしタスクに応じた分割タイプの自動選択は未実装 | タスク/設定に応じた分割タイプの自動適用を実装 |
| 早期停止 | CVフォールドではOOFデータをES監視に使用（OOFの独立性にリーク）。最終モデルは `x_valid=x, y_valid=y` でES実質無効 | CVフォールド・最終モデルの両方で、train部分からES用バリデーションを自動分割。OOFは純粋にOOF予測専用 |
| 学習曲線 | イテレーションごとのメトリクス保存なし | `record_evaluation` callback + Artifact保存。可視化はPhase30で対応 |

---

### Step 1: `TrainConfig` スキーマ拡張

**対象**: `src/veldra/config/models.py`

```python
class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lgb_params: dict[str, Any] = Field(default_factory=dict)
    early_stopping_rounds: int | None = 100
    early_stopping_validation_fraction: float = 0.1  # NEW: 最終モデルの early stopping 用バリデーション比率
    num_boost_round: int = 300                    # NEW
    seed: int = 42
    auto_class_weight: bool = True                # NEW: opt-out
    class_weight: dict[str, float] | None = None  # NEW: ユーザー手動指定
```

**バリデーション追加**（`RunConfig._validate_cross_fields` 内）:
- `auto_class_weight=True` は `binary` / `multiclass` のみ許可（`regression`/`frontier` で True なら ValueError）
- `class_weight` は `binary` / `multiclass` のみ許可
- `auto_class_weight=True` と `class_weight` の同時指定は禁止（手動指定が優先される旨のエラー）
- `num_boost_round >= 1` を検証
- `0.0 < early_stopping_validation_fraction < 1.0` を検証

---

### Step 2: `num_boost_round` の設定可能化 + 最終モデル Early Stopping 用バリデーション分割

**対象**: 全 `_train_single_booster` 関数（4ファイル）+ 全 `train_*_with_cv` 関数（4ファイル）

#### 2a: `num_boost_round` の設定可能化

```python
# Before（全タスク共通）
num_boost_round=300

# After
num_boost_round=config.train.num_boost_round
```

#### 2b: Early Stopping 用バリデーション分割（CVフォールド + 最終モデル共通）

**問題**:
- CVフォールド: 現在OOFデータ（valid部分）を early stopping の監視対象として使用している。これではOOFが完全にOOFでなくなる（early stopping の判断にリークする）。
- 最終モデル: `x_valid=x, y_valid=y`（訓練データ＝バリデーションデータ）のため early stopping が実質無効。

**設計方針**:
- CVフォールド・最終モデルの両方で、**学習に使うデータ（train部分）からさらに early stopping 用バリデーションデータを自動分割**する
- OOFデータ（CVのvalid部分）は early stopping の監視に一切使わず、**純粋にOOF予測専用**として扱う
- 分割比率: `train.early_stopping_validation_fraction`（既定 `0.1`、train部分の10%をearly stopping用バリデーションに使用）
- `early_stopping_rounds=None`（early stopping 無効）の場合は分割せず、従来通り全データで学習する

**タスクに応じた分割タイプの自動適用**:
- `task.type=binary/multiclass` → `sklearn.model_selection.StratifiedShuffleSplit` で層化分割
- `task.type=regression/frontier` → `sklearn.model_selection.ShuffleSplit` でランダム分割
- `split.type=timeseries` → 時系列順で末尾 N% をバリデーションに使用（シャッフルしない）

**CVフォールドの変更**:
```python
# Before
for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
    booster = _train_single_booster(
        x_train=x.iloc[train_idx], y_train=y.iloc[train_idx],
        x_valid=x.iloc[valid_idx], y_valid=y.iloc[valid_idx],  # OOFデータをES監視に使用（リーク）
        config=config,
    )

# After
for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
    x_fold_train, y_fold_train = x.iloc[train_idx], y.iloc[train_idx]
    # train部分からES用バリデーションを分割（OOFは純粋にOOFのまま）
    x_es_train, x_es_valid, y_es_train, y_es_valid = _split_for_early_stopping(
        x_fold_train, y_fold_train, config
    )
    booster = _train_single_booster(
        x_train=x_es_train, y_train=y_es_train,
        x_valid=x_es_valid, y_valid=y_es_valid,  # ES専用バリデーション
        config=config,
    )
    # OOFデータはES非依存で予測のみに使用
    pred = booster.predict(x.iloc[valid_idx], ...)
```

**最終モデルの変更**:
```python
# Before
final_model = _train_single_booster(
    x_train=x, y_train=y,
    x_valid=x, y_valid=y,  # 訓練データ＝バリデーションデータ（ES無効）
    config=config,
)

# After
x_es_train, x_es_valid, y_es_train, y_es_valid = _split_for_early_stopping(
    x, y, config
)
final_model = _train_single_booster(
    x_train=x_es_train, y_train=y_es_train,
    x_valid=x_es_valid, y_valid=y_es_valid,
    config=config,
)
```

**`_split_for_early_stopping` 関数**（`src/veldra/modeling/utils.py` に新設）:
```python
def _split_for_early_stopping(
    x: pd.DataFrame,
    y: pd.Series,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """学習データから early stopping 用バリデーションデータを分割する。

    - early_stopping_rounds=None の場合は分割せず (x, x, y, y) を返す（従来互換）。
    - タスクに応じた分割タイプを自動適用する。
    - CVフォールド内でも最終モデル学習時でも共通で使用する。
    """
```

**`TrainConfig` 追加フィールド**:
```python
early_stopping_validation_fraction: float = 0.1  # NEW: ES用バリデーション比率
```

**対象ファイル**:
- `src/veldra/modeling/utils.py`（新設: `_split_for_early_stopping`）
- `src/veldra/modeling/binary.py`（CVループ + 最終モデル）
- `src/veldra/modeling/multiclass.py`（CVループ + 最終モデル）
- `src/veldra/modeling/regression.py`（CVループ + 最終モデル）
- `src/veldra/modeling/frontier.py`（CVループ + 最終モデル）

---

### Step 3: クラス不均衡の自動検出・重み適用

**対象**: `src/veldra/modeling/binary.py`, `src/veldra/modeling/multiclass.py`

**Binary**（`_train_single_booster` 内）:
- `config.train.class_weight` が指定されている場合:
  - `class_weight` から `scale_pos_weight`（= neg_count / pos_count 相当）を算出し `params` に設定
- `config.train.auto_class_weight=True` かつ `class_weight=None` の場合:
  - `params["is_unbalance"] = True` を自動的に設定

**Multiclass**（`_train_single_booster` 内）:
- `config.train.class_weight` が指定されている場合:
  - `class_weight` dict からサンプルごとの重みを算出し `lgb.Dataset(weight=...)` に渡す
- `config.train.auto_class_weight=True` かつ `class_weight=None` の場合:
  - `sklearn.utils.class_weight.compute_sample_weight("balanced", y_train)` でサンプル重みを自動算出し `lgb.Dataset(weight=...)` に渡す

---

### Step 4: バリデーション分割の自動適用

**対象**: `src/veldra/modeling/binary.py`, `src/veldra/modeling/multiclass.py`, 因果推論経路

**実装方針**（分割タイプをタスク/設定に応じて自動的に適用する）:
- `task.type=binary/multiclass` かつ `split.type=kfold` → 内部で `stratified` 分割を自動適用する
  - ユーザーが明示的に `split.type` を設定している場合はそれを尊重する
- `causal` 設定時 → `split.group_col` または `causal.unit_id_col`（panel）利用可能時に `GroupKFold` を適用し、利用不可時は `KFold` にフォールバックする
- `split.type=timeseries` → 既存実装で時系列分割が適用される（変更なし）

---

### Step 5: 学習曲線データの記録・Artifact保存

**対象**: 全 `_train_single_booster` + 各 `TrainingOutput` + Artifact保存

**実装方針**:
- `_train_single_booster` の戻り値を `tuple[lgb.Booster, dict]` に変更し、`lgb.record_evaluation()` の結果を返す
- 各 `TrainingOutput` dataclass に `training_history: list[dict]` フィールドを追加
- Artifact保存時に `training_history.json` として永続化

**Artifactスキーマ**:
```json
{
  "folds": [
    {
      "fold": 1,
      "num_iterations": 150,
      "best_iteration": 120,
      "eval_history": {"binary_logloss": [0.69, 0.55, 0.42, ...]}
    }
  ]
}
```

**注**: 学習曲線の可視化（GUI/チャート）はPhase30で対応する。本Stepではデータ保存のみ。

---

### Step 6: GUI対応

**対象**: `src/veldra/gui/pages/config_page.py`, `src/veldra/gui/app.py`

**変更点**:
1. `N Estimators` ラベルを `Num Boost Round` に変更し、意味を正確に反映する
2. `Auto Class Weight` トグルスイッチ追加（`binary`/`multiclass` 選択時のみ表示、既定 ON）
3. `Class Weight` 手動入力フィールド追加（`Auto Class Weight=OFF` 時のみ活性化）
4. Config builder で `num_boost_round` を `train.num_boost_round` に正しくマッピング（現在の `lgb_params.n_estimators` のミスマッピングを修正）

---

### Step 7: Config Migration

**対象**: `src/veldra/config/migrate.py`

- `train.lgb_params.n_estimators` が存在する場合 → 値を `train.num_boost_round` に移行
- `n_estimators` キーを `lgb_params` から削除
- migration utility に変換ルールを追加

---

### Step 8: テスト計画

| テスト | 内容 | ファイル |
|--------|------|----------|
| スキーマ検証 | `num_boost_round`, `auto_class_weight`, `class_weight`, `early_stopping_validation_fraction` のバリデーション | `tests/test_config_train_fields.py` |
| Binary不均衡自動 | `auto_class_weight=True` で `is_unbalance` が自動的に設定される | `tests/test_binary_class_weight.py` |
| Binary手動重み | `class_weight` 指定で `scale_pos_weight` が適用される | 同上 |
| Multiclass不均衡自動 | `auto_class_weight=True` で balanced sample weight が自動的に適用される | `tests/test_multiclass_class_weight.py` |
| Multiclass手動重み | `class_weight` 指定で指定重みがサンプルに適用される | 同上 |
| num_boost_round | 設定値が実際の学習に反映される | `tests/test_num_boost_round.py` |
| 分割自動適用 | binary/multiclass で stratified が自動適用される | `tests/test_auto_split_selection.py` |
| 分割自動適用(causal) | causal 設定時に group KFold が自動適用される | 同上 |
| ES用バリデーション分割(CV) | CVフォールドでtrain部分からES用バリデーションが分割され、OOFがES監視に使用されない | `tests/test_early_stopping_validation.py` |
| ES用バリデーション分割(最終モデル) | 最終モデル学習時にも全データからES用バリデーションが分割される | 同上 |
| ES用バリデーション分割(timeseries) | 時系列データで末尾N%がバリデーションに使用される | 同上 |
| ES用バリデーション分割(stratified) | binary/multiclass で層化分割がバリデーション生成に使用される | 同上 |
| ES無効時 | `early_stopping_rounds=None` の場合は分割せず全データで学習される | 同上 |
| 学習曲線 | `training_history` がArtifactに保存される | `tests/test_training_history.py` |
| 早期停止 | `best_iteration` が `training_history` に記録される | 同上 |
| GUI | Config builder が新フィールドを正しく生成する | `tests/test_gui_app_callbacks_config.py` |
| Migration | `lgb_params.n_estimators` → `num_boost_round` 変換 | `tests/test_config_migration.py` |

### 検証コマンド
- `uv run pytest tests/test_config_train_fields.py tests/test_binary_class_weight.py tests/test_multiclass_class_weight.py -v`
- `uv run pytest tests/test_num_boost_round.py tests/test_auto_split_selection.py tests/test_early_stopping_validation.py -v`
- `uv run pytest tests/test_training_history.py -v`
- `uv run pytest tests/test_gui_app_callbacks_config.py tests/test_config_migration.py -v`
- `uv run pytest tests -x --tb=short`

### 実装順序（依存関係順）
1. Step 1: `TrainConfig` スキーマ拡張 + バリデーション
2. Step 2: `num_boost_round` 設定可能化 + 最終モデル Early Stopping 用バリデーション分割（全4ファイル + utils.py 新設）
3. Step 3: クラス不均衡の自動検出・重み適用（binary + multiclass）
4. Step 4: バリデーション分割の自動適用
5. Step 5: 学習曲線データの記録・Artifact保存
6. Step 6: GUI対応
7. Step 7: Config Migration
8. Step 8: テスト

### 完了条件
- 目的変数の自動判定機能が実装され、ユーザーが明示的に指定しなくても適切な目的関数が選択されること。必要に応じてユーザーが設定変更もできること。
- バリデーションデータの適切な設定が自動的に行われ、モデルの過学習を防止するための適切なバリデーションが実施されること。
  - データが時系列の場合は、時系列分割が自動的に適用されること。
  - 目的変数がカテゴリカル（Binary or Multi-class）であれば、層化分割が自動的に適用されること。
  - DR or DR-DiD の傾向スコアモデルとOutcomeモデルには、Group K-Fold 分割が自動的に適用されること。
- CVフォールド・最終モデルの両方で、学習データ（train部分）からタスクに応じた分割タイプで early stopping 用バリデーションデータが自動分割されること。OOFデータ（CVのvalid部分）は early stopping の監視に一切使わず、純粋にOOF予測専用として扱われること。
  - binary/multiclass → 層化分割（StratifiedShuffleSplit）で自動分割されること。
  - regression/frontier → ランダム分割（ShuffleSplit）で自動分割されること。
  - timeseries → 時系列順で末尾 N% がバリデーションに使用されること。
  - `early_stopping_rounds=None`（early stopping 無効）の場合は分割せず、従来通り全データで学習すること。
  - 分割比率は `early_stopping_validation_fraction`（既定 0.1）で制御可能であること。
- ImbalanceデータにWeightを適用する機能が実装され、クラス不均衡なデータセットに対して適切な重み付けが自動的に行われること。
  - Binary分類タスクで、`auto_class_weight=True`（既定）の場合に、LightGBMの `is_unbalance` パラメーターが自動的に設定されること。
    - ユーザーが明示的にクラス重みを `class_weight` で指定できるオプションも提供されること。
  - Multi-class分類タスクで、`auto_class_weight=True`（既定）の場合に、balanced sample weight が自動的に算出・適用されること。
    - ユーザーが明示的にクラス重みを `class_weight` で指定できるオプションも提供されること。
- `num_boost_round` が `TrainConfig` から制御可能で、全タスク（regression/binary/multiclass/frontier）で反映されること。
- 学習曲線の早期停止機能が実装され、ユーザーが `early_stopping_rounds` と `num_boost_round` を設定できること。
- 学習曲線データ（foldごとのイテレーション別メトリクス + best_iteration）が `training_history.json` としてArtifactに保存されること。学習曲線の可視化はPhase30で対応する。
- GUI で `Num Boost Round`、`Auto Class Weight`、`Class Weight` が設定可能であること。
- `lgb_params.n_estimators` → `train.num_boost_round` の Config migration が動作すること。
- 既存テストが全パスし、Stable API（`veldra.api.*`）の互換性が維持されること。

### 検証結果（2026-02-16）
- 全8ステップ（スキーマ拡張 / num_boost_round / ES分割 / クラス不均衡 / 分割自動適用 / 学習曲線 / GUI / Migration）が実装完了。
- Phase25.7 関連テスト: **31 passed, 0 failed**
- Stable API 互換性: 維持確認済み

## 12.8 Phase25.8: LightGBMのパラメーター追加

### 目的
- Top-K Precision の追加（`top_k` パラメーター）により、Binary モデル性能評価の多様化を実現する。
- 特徴量の重み付けの追加（`feature_weights` パラメーター）により、特定特徴量への分割集中を制御する。
- `num_leaves` の自動調整機能（`auto_num_leaves` + `num_leaves_ratio`）により、木構造のデータ適応を自動化する。
- 比率ベースのリーフ制約（`min_data_in_leaf_ratio`, `min_data_in_bin_ratio`）により、過学習防止とモデル安定性を向上する。
- 既存の `lgb_params` 経由パラメーターを GUI で直接設定可能にする。

### 現状分析

| パラメーター | 現状 | 対応 |
|-------------|------|------|
| `top_k` | 未実装。Binary 評価・学習に precision@k がない | `TrainConfig` に追加。LightGBM カスタム metric（`feval`）として学習ループに組込み、ES 監視・tune objective・evaluate 返却で利用可能にする |
| `feature_weights` | 未実装 | `TrainConfig` に追加し、`lgb.Dataset(feature_name=..., weight=...)` とは別に `lgb.Dataset` の `feature_name` パラメータに続けて適用 |
| `auto_num_leaves` | 未実装。GUI で `num_leaves` を手動入力するのみ | `TrainConfig` にフラグ追加。`max_depth` から動的に `num_leaves` を算出 |
| `num_leaves_ratio` | 未実装 | `auto_num_leaves=True` 時の補正係数として `TrainConfig` に追加 |
| `min_data_in_leaf_ratio` | 未実装。`min_child_samples` は絶対値で `lgb_params` 経由 | `TrainConfig` に追加。学習時にデータ行数から `min_data_in_leaf` を動的算出 |
| `min_data_in_bin_ratio` | 未実装。`min_data_in_bin` は `lgb_params` 経由で設定可能 | `TrainConfig` に追加。学習時にデータ行数から `min_data_in_bin` を動的算出 |
| `path_smooth` / `cat_l2` / `cat_smooth` | `lgb_params` 経由で設定可能だが GUI 入力なし | GUI Advanced セクションに入力欄追加 |
| `bagging_freq` / `max_bin` / `max_drop` / `min_gain_to_split` | `lgb_params` 経由で設定可能だが GUI 入力なし | GUI Advanced セクションに入力欄追加 |

---

### Step 1: `TrainConfig` スキーマ拡張

**対象**: `src/veldra/config/models.py`

```python
class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lgb_params: dict[str, Any] = Field(default_factory=dict)
    early_stopping_rounds: int | None = 100
    early_stopping_validation_fraction: float = 0.1
    num_boost_round: int = 300
    auto_class_weight: bool = True
    class_weight: dict[str, float] | None = None
    seed: int = 42
    # --- Phase25.8 追加 ---
    auto_num_leaves: bool = False               # NEW: max_depth から num_leaves を自動算出
    num_leaves_ratio: float = 1.0               # NEW: auto_num_leaves 時の補正係数
    min_data_in_leaf_ratio: float | None = None  # NEW: 行数に対する比率 (例: 0.01 → 1%)
    min_data_in_bin_ratio: float | None = None   # NEW: 行数に対する比率 (例: 0.001)
    feature_weights: dict[str, float] | None = None  # NEW: 特徴量名→重み
    top_k: int | None = None                    # NEW: precision@k の k（binary のみ）
```

**バリデーション追加**（`RunConfig._validate_cross_fields` 内）:
- `auto_num_leaves=True` の場合 `num_leaves_ratio` は `0.0 < ratio <= 1.0` を検証
- `auto_num_leaves=True` かつ `lgb_params` に `num_leaves` が明示指定されている場合は ValueError（競合）
- `min_data_in_leaf_ratio` は `0.0 < ratio < 1.0` を検証
- `min_data_in_bin_ratio` は `0.0 < ratio < 1.0` を検証
- `min_data_in_leaf_ratio` と `lgb_params.min_data_in_leaf` の同時指定は禁止
- `min_data_in_bin_ratio` と `lgb_params.min_data_in_bin` の同時指定は禁止
- `top_k` は `binary` タスクのみ許可、`top_k >= 1` を検証
- `feature_weights` の値は全て `> 0` を検証

---

### Step 2: `auto_num_leaves` の動的算出ロジック

**対象**: `src/veldra/modeling/utils.py`（新規関数追加）

```python
def resolve_auto_num_leaves(config: RunConfig) -> int | None:
    """auto_num_leaves=True の場合に num_leaves を動的に算出する。

    - max_depth が指定されていない場合(-1): num_leaves = 131072（LightGBM上限）
    - max_depth が指定されている場合: num_leaves = clip(2^max_depth, 8, 131072)
    - num_leaves_ratio で補正: num_leaves = clip(ceil(num_leaves * ratio), 8, 131072)
    - auto_num_leaves=False の場合は None を返す（既存の lgb_params.num_leaves が使われる）
    """
```

**適用箇所**: 全4ファイルの `_train_single_booster` 内、`params` dict 構築後に:
```python
# auto_num_leaves の解決
resolved_leaves = resolve_auto_num_leaves(config)
if resolved_leaves is not None:
    params["num_leaves"] = resolved_leaves
```

---

### Step 3: 比率ベースのリーフ・ビン制約

**対象**: `src/veldra/modeling/utils.py`（新規関数追加）

```python
def resolve_ratio_params(config: RunConfig, n_rows: int) -> dict[str, int]:
    """比率ベースのパラメーターを絶対値に変換する。

    - min_data_in_leaf_ratio: n_rows * ratio → min_data_in_leaf（最小 1）
    - min_data_in_bin_ratio: n_rows * ratio → min_data_in_bin（最小 1）
    """
```

**適用箇所**: 全4ファイルの `_train_single_booster` 内、`params` dict 構築後に:
```python
ratio_params = resolve_ratio_params(config, len(x_train))
params.update(ratio_params)
```

---

### Step 4: `feature_weights` の適用

**対象**: 全4ファイルの `_train_single_booster` 内

**実装方針**:
- `config.train.feature_weights` が指定されている場合、特徴量名のリストに対応する重みリストを構築
- `lgb.Dataset` コンストラクタの `feature_name` と合わせて `lgb.train()` の `feature_weights` パラメータを使用
- 指定されていない特徴量のデフォルト重みは `1.0`

```python
# feature_weights の適用
if config.train.feature_weights:
    fw = [config.train.feature_weights.get(col, 1.0) for col in x_train.columns]
else:
    fw = None
# ...
booster = lgb.train(
    params=params,
    train_set=train_set,
    # ...,
    feature_name=list(x_train.columns) if fw else "auto",
)
# feature_weights は lgb.Dataset ではなく lgb.train() の引数ではないため、
# params["feature_fraction_bynode"] 等との組合せで実現するか、
# lgb.Dataset(init_score=...) ではなく params dict に直接設定:
# params["feature_weights"] = fw  ← LightGBM は内部で feature_weights を受け付ける（要確認）
```

**注**: LightGBM の `feature_weights` は `lgb.Dataset` の `set_feature_names` 後に
`dataset.set_feature_names()` とは異なり `params` に直接渡す形式ではない。
実際には `lgb.train()` の `feature_name` パラメータと合わせて
`Dataset(feature_name=..., free_raw_data=False)` 構築後に
`train_set.feature_name` を参照し weight リストを構築する方式を取る。
→ LightGBM 4.x の `feature_pre_filter` と `feature_weights` サポートを確認の上、最適な方法を採用する。

---

### Step 5: Top-K Precision のカスタム metric 実装

**対象**: `src/veldra/modeling/binary.py`, `src/veldra/modeling/tuning.py`, `src/veldra/config/models.py`

#### 5a: LightGBM カスタム評価関数（`feval`）

**新規関数**（`src/veldra/modeling/binary.py`）:
```python
def _make_precision_at_k_feval(k: int):
    """LightGBM の feval 互換の precision@k 評価関数を返す。

    - feval シグネチャ: (y_pred, dataset) -> (name, value, is_higher_better)
    - y_pred を降順ソートし、上位 k 件を抽出
    - 抽出された k 件中の正例数 / k が precision@k
    - k > len(y_true) の場合は全件を使用
    - is_higher_better=True（precision は高いほど良い）
    """
    def _precision_at_k(y_pred, dataset):
        y_true = dataset.get_label()
        order = np.argsort(-y_pred)
        top = order[:min(k, len(y_true))]
        value = float(y_true[top].sum() / len(top))
        return f"precision_at_{k}", value, True
    return _precision_at_k
```

**適用箇所**（`_train_single_booster` 内）:
- `config.train.top_k` が指定されている場合、`_make_precision_at_k_feval(k)` を生成
- `lgb.train()` の `feval` 引数に渡す
- これにより early stopping が `precision_at_{k}` を監視メトリクスとして利用可能になる

```python
feval_funcs = []
if config.train.top_k is not None:
    feval_funcs.append(_make_precision_at_k_feval(config.train.top_k))

return lgb.train(
    params=params,
    train_set=train_set,
    valid_sets=[valid_set],
    num_boost_round=config.train.num_boost_round,
    callbacks=callbacks,
    feval=feval_funcs if feval_funcs else None,
)
```

#### 5b: Tuning objective への統合

**対象**: `src/veldra/config/models.py`

`_TUNE_ALLOWED_OBJECTIVES["binary"]` に `precision_at_k` を追加:
```python
"binary": {"auc", "logloss", "brier", "accuracy", "f1", "precision", "recall", "precision_at_k"},
```

**対象**: `src/veldra/modeling/tuning.py`

Tune trial 内で `precision_at_k` が objective に指定された場合:
- `config.train.top_k` の値を使用して OOF 予測に対する precision@k を算出
- `top_k` 未指定時はエラー

#### 5c: evaluate API での返却

**対象**: `src/veldra/modeling/binary.py`

`_binary_metrics` / `_binary_label_metrics` の呼び出し後に:
- `config.train.top_k` が指定されていれば `precision_at_{k}` をメトリクス dict に追加
- これにより `evaluate` API の返却値にも含まれる

**メトリクスキー**: `precision_at_{k}`（例: `precision_at_100`）

**学習曲線**: `record_evaluation` callback により `training_history.json` にイテレーションごとの `precision_at_{k}` が自動記録される。

---

### Step 6: GUI Advanced Training Parameters の拡充

**対象**: `src/veldra/gui/pages/config_page.py`, `src/veldra/gui/app.py`

**config_page.py の Advanced Training Parameters アコーディオン**に以下を追加:

| GUI コンポーネント | ID | タイプ | デフォルト | 備考 |
|-------------------|-----|--------|-----------|------|
| Auto Num Leaves | `cfg-train-auto-num-leaves` | Switch | OFF | ON 時に Num Leaves 入力を無効化 |
| Num Leaves Ratio | `cfg-train-num-leaves-ratio` | Slider (0.1–1.0) | 1.0 | `auto_num_leaves=ON` 時のみ活性 |
| Min Data In Leaf Ratio | `cfg-train-min-leaf-ratio` | Input (number) | 空 | 設定時は `min_child_samples` より優先 |
| Min Data In Bin Ratio | `cfg-train-min-bin-ratio` | Input (number) | 空 | |
| Feature Weights | `cfg-train-feature-weights` | Textarea (JSON) | 空 | `{"col_name": 2.0, ...}` 形式 |
| Path Smooth | `cfg-train-path-smooth` | Input (number) | 0 | |
| Cat L2 | `cfg-train-cat-l2` | Input (number) | 10 | |
| Cat Smooth | `cfg-train-cat-smooth` | Input (number) | 10 | |
| Bagging Freq | `cfg-train-bagging-freq` | Input (number) | 0 | |
| Max Bin | `cfg-train-max-bin` | Input (number) | 255 | |
| Min Gain To Split | `cfg-train-min-gain` | Input (number) | 0 | |
| Top K (Binary) | `cfg-train-top-k` | Input (number) | 空 | binary 選択時のみ表示。学習 feval + 評価 metric 両用 |

**app.py の `_cb_build_config_yaml` への追加**:
- 上記の各 Input を callback の引数に追加
- `cfg["train"]` および `cfg["train"]["lgb_params"]` に値をマッピング
- 空値（None / 空文字）はスキップし、YAML に含めない
- `auto_num_leaves` / `num_leaves_ratio` / `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` / `feature_weights` は `cfg["train"]` 直下に配置
- `path_smooth` / `cat_l2` / `cat_smooth` / `bagging_freq` / `max_bin` / `min_gain_to_split` は `cfg["train"]["lgb_params"]` に配置

---

### Step 7: Tuning Search Space の拡充

**対象**: `src/veldra/modeling/tuning.py`

**standard プリセット**に以下を追加:
```python
"standard": {
    # 既存 ...
    "lambda_l1": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},  # NEW
    "lambda_l2": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},  # NEW
    "path_smooth": {"type": "float", "low": 0.0, "high": 10.0},              # NEW
    "min_gain_to_split": {"type": "float", "low": 0.0, "high": 1.0},         # NEW
}
```

---

### Step 8: Artifact パラメーター永続化の確認

**現状**: `run_config.yaml` が Artifact に保存されるため、`TrainConfig` に追加された全フィールドと `lgb_params` 内のパラメーターは自動的に永続化される。

**追加対応**:
- `feature_weights` が大量の場合でも `run_config.yaml` に含まれるため、別ファイル化は不要（YAML のまま）
- Artifact からの `predict` 実行時に `feature_weights` が保存された RunConfig から復元されることを確認

---

### Step 9: テスト計画

| テスト | 内容 | ファイル |
|--------|------|----------|
| スキーマ検証 | `auto_num_leaves`, `num_leaves_ratio`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`, `feature_weights`, `top_k` のバリデーション | `tests/test_config_param_fields.py` |
| auto_num_leaves | `auto_num_leaves=True` で `max_depth` から `num_leaves` が動的算出される | `tests/test_auto_num_leaves.py` |
| auto_num_leaves + ratio | `num_leaves_ratio=0.5` で葉数が半減する | 同上 |
| auto_num_leaves 競合 | `auto_num_leaves=True` かつ `lgb_params.num_leaves` 指定でエラー | 同上 |
| min_data_in_leaf_ratio | 比率から絶対値が正しく算出される | `tests/test_ratio_params.py` |
| min_data_in_bin_ratio | 比率から絶対値が正しく算出される | 同上 |
| 比率パラメーター競合 | `min_data_in_leaf_ratio` と `lgb_params.min_data_in_leaf` の同時指定でエラー | 同上 |
| feature_weights | 指定した重みが学習に反映される（モデルが作成可能なこと） | `tests/test_feature_weights.py` |
| feature_weights 不正値 | 重み <= 0 でバリデーションエラー | 同上 |
| top_k feval | `top_k` 指定時に LightGBM カスタム metric として学習ループ内で `precision_at_{k}` が算出される | `tests/test_top_k_precision.py` |
| top_k early stopping | `precision_at_{k}` で early stopping が正しく動作する | 同上 |
| top_k evaluate | Binary 評価で `precision_at_{k}` がメトリクスに含まれる | 同上 |
| top_k tune objective | `precision_at_k` を tune objective に指定して最適化が動作する | 同上 |
| top_k 非 binary | `top_k` を regression で指定してバリデーションエラー | 同上 |
| top_k 学習曲線 | `training_history.json` にイテレーションごとの `precision_at_{k}` が記録される | 同上 |
| GUI | Config builder が新フィールドを正しく YAML に生成する | `tests/test_gui_app_callbacks_config.py`（既存に追加） |
| Tuning search space | standard プリセットに `lambda_l1`, `lambda_l2` 等が含まれる | `tests/test_tuning_search_space.py` |
| Artifact 復元 | 新パラメーター付き Artifact からの predict が成功する | `tests/test_artifact_param_roundtrip.py` |

### 検証コマンド
- `uv run pytest tests/test_config_param_fields.py tests/test_auto_num_leaves.py tests/test_ratio_params.py -v`
- `uv run pytest tests/test_feature_weights.py tests/test_top_k_precision.py -v`
- `uv run pytest tests/test_tuning_search_space.py tests/test_artifact_param_roundtrip.py -v`
- `uv run pytest tests/test_gui_app_callbacks_config.py -v`
- `uv run pytest tests -x --tb=short`

### 実装順序（依存関係順）
1. Step 1: `TrainConfig` / `PostprocessConfig` スキーマ拡張 + バリデーション
2. Step 2: `auto_num_leaves` 動的算出ロジック（`utils.py` + 全4 modeling ファイル）
3. Step 3: 比率ベースパラメーター算出ロジック（`utils.py` + 全4 modeling ファイル）
4. Step 4: `feature_weights` の適用（全4 modeling ファイル）
5. Step 5: Top-K Precision カスタム metric（`binary.py` + `tuning.py` + 評価経路）
6. Step 6: GUI Advanced Training Parameters 拡充
7. Step 7: Tuning Search Space 拡充
8. Step 8: Artifact パラメーター永続化確認
9. Step 9: テスト

### 完了条件
- `auto_num_leaves=True` の場合に `max_depth` から `num_leaves` が動的に算出され、`num_leaves_ratio` で補正可能であること。`auto_num_leaves=True` と `lgb_params.num_leaves` の同時指定でバリデーションエラーとなること。
- `min_data_in_leaf_ratio` が指定された場合に学習データの行数に基づいて `min_data_in_leaf` が動的に算出されること。`min_data_in_bin_ratio` も同様に動作すること。比率パラメーターと対応する `lgb_params` の絶対値パラメーターの同時指定でバリデーションエラーとなること。
- `feature_weights` が指定された場合に、特徴量ごとの重みが LightGBM の学習に反映されること。重み <= 0 の指定でバリデーションエラーとなること。
- `top_k` が指定された場合に precision@k が LightGBM カスタム metric（`feval`）として学習ループに組み込まれ、early stopping の監視指標として動作すること。`precision_at_k` を tune の objective として指定可能であること。`evaluate` API でも `precision_at_{k}` がメトリクスに含まれること。`training_history.json` にイテレーションごとの値が記録されること。`top_k` は `binary` タスクのみ許可され、他タスクではバリデーションエラーとなること。
- GUI の Advanced Training Parameters に `Auto Num Leaves` / `Num Leaves Ratio` / `Min Data In Leaf Ratio` / `Min Data In Bin Ratio` / `Feature Weights` / `Path Smooth` / `Cat L2` / `Cat Smooth` / `Bagging Freq` / `Max Bin` / `Min Gain To Split` / `Top K` の入力欄が追加され、Config YAML に正しく反映されること。
- Tuning の standard プリセットに `lambda_l1` / `lambda_l2` / `path_smooth` / `min_gain_to_split` が追加されること。
- 新パラメーター付きの RunConfig が Artifact に保存され、Artifact からの再利用（predict / evaluate）が成功すること。
- 既存テストが全パスし、Stable API（`veldra.api.*`）の互換性が維持されること。

### Decision（provisional）
- `train.top_k` が指定された場合、Early Stopping は `precision_at_{k}` を優先監視する（`metric=None + feval`）。
- `train.feature_weights` は未知特徴量キーを許容しない。未知キーは学習前に `VeldraValidationError` とする。

### 実装結果（2026-02-16）
- `TrainConfig` に `auto_num_leaves / num_leaves_ratio / min_data_in_leaf_ratio / min_data_in_bin_ratio / feature_weights / top_k` を追加し、cross-field validation を実装。
- `modeling/utils.py` に `resolve_auto_num_leaves / resolve_ratio_params / resolve_feature_weights` を追加し、全4 task 学習器へ適用。
- binary 学習に `precision_at_k` を実装し、`fit / tune / evaluate / training_history` で利用可能にした。
- binary `metrics.mean` に `accuracy/f1/precision/recall` を含めるよう修正し、既存 tuning objective の実行不整合を解消。
- GUI Builder に Phase25.8 パラメータ入力を追加し、`auto_num_leaves=True` 時は YAML から `lgb_params.num_leaves` を除外。
- `_cb_update_tune_objectives("binary")` に `brier` と `precision_at_k` を追加。
- `scripts/generate_runconfig_reference.py` を更新し、README RunConfig Reference を再生成。

### 検証結果（2026-02-16）
- `uv run ruff check .` : passed
- `uv run pytest -q tests/test_config_train_fields.py tests/test_lgb_param_resolution.py tests/test_top_k_precision.py` : **19 passed**
- `uv run pytest -q tests/test_tuning_internal.py tests/test_tune_objective_selection.py tests/test_tune_validation.py tests/test_binary_evaluate_metrics.py` : **17 passed**
- `uv run pytest -q tests/test_gui_app_callbacks_config.py tests/test_gui_pages_and_init.py tests/test_gui_new_layout.py tests/test_gui_app_additional_branches.py` : **28 passed**
- `uv run pytest -q -m "not gui"` : **385 passed**
- `uv run pytest -q -m "gui"` : **100 passed**

## 12.9 Phase25.9: LightGBM機能強化の不足テスト補完

### 目的
Phase25.7/25.8 で実装された LightGBM 機能強化に対し、既存テストでカバーされていないギャップを特定し、不足テストを追加する。

### 実装方針（2026-02-16 更新）
- 不足テスト追加で差分が見つかった場合、Phase25.9 内で最小本体修正まで行う。
- Causal の GroupKFold は既存 `split.group_col` 検証に加えて、`causal.unit_id_col` 経路を追加検証する。
- `best_iteration` は実学習依存を避け、monkeypatch による安定した契約検証を優先する。

### 既存テストカバレッジ分析（2026-02-16 時点）

#### Phase25.7: カバー済み
| テスト対象 | 既存テストファイル | 状態 |
|---|---|---|
| Config バリデーション | `test_config_train_fields.py` | 充足 |
| Binary class weight | `test_binary_class_weight.py` | 充足 |
| Multiclass class weight | `test_multiclass_class_weight.py` | 充足 |
| num_boost_round 反映 | `test_num_boost_round.py` | 充足（全4タスク） |
| 分割自動適用 | `test_auto_split_selection.py` | binary/regression のみ |
| ES用バリデーション分割 | `test_early_stopping_validation.py` | 充足（無効時/timeseries/binary層化/CVでOOF非使用） |
| 学習曲線Artifact保存 | `test_training_history.py` | 充足（save/load + legacy互換） |
| Config migration | `test_config_migrate_file.py` / `test_config_migrate_payload.py` | 充足（`n_estimators` → `num_boost_round`） |

#### Phase25.8: ギャップあり
| テスト対象 | 既存テストファイル | 状態 |
|---|---|---|
| Config バリデーション（25.8分） | `test_config_train_fields.py` | 充足（auto_num_leaves/ratio/feature_weights/top_k） |
| パラメーター解決ロジック | `test_lgb_param_resolution.py` | ユニットテストのみ（`resolve_*` 関数の入出力検証） |
| top_k precision | `test_top_k_precision.py` | 充足（helper/feval/CV metrics/training_history） |
| GUI Config builder（25.8分） | `test_gui_new_layout.py` | 充足（YAML出力に25.8パラメーター含む） |
| **auto_num_leaves 学習適用** | — | **不在**: 学習ループで `params["num_leaves"]` に算出値が渡されるか未検証 |
| **ratio_params 学習適用** | — | **不在**: 学習ループで `params["min_data_in_leaf"]` 等が渡されるか未検証 |
| **feature_weights 学習適用** | — | **不在**: 学習ループで重みリストが渡されるか未検証 |
| **Tuning search space 拡充** | — | **不在**: standard プリセットに `lambda_l1`/`lambda_l2`/`path_smooth`/`min_gain_to_split` が含まれるか未検証 |
| **新パラメーター Artifact roundtrip** | — | **不在**: 新パラメーター付き Artifact → predict 成功の検証なし |
| **num_boost_round 既定値後方互換** | — | **不在**: 未指定時に既定値300で動作するか未検証 |
| **早期停止の best_iteration 動作** | — | **不在**: monkeypatch で `best_iteration` 記録契約（`< num_boost_round`）を固定検証していない |
| **Causal での GroupKFold 自動適用** | `test_dr_internal.py` | **一部充足**: `split.group_col` は検証済み、`causal.unit_id_col` 経路が未検証 |
| **目的関数ユーザー上書き** | — | **不在**: `lgb_params.objective` 指定時の優先動作が未検証 |

---

### 取り組み項目

#### 1. `auto_num_leaves` 学習ループ適用テスト
- **ファイル**: `tests/test_auto_num_leaves.py`（新規）
- `auto_num_leaves=True` + `lgb_params.max_depth=5` + `num_leaves_ratio=0.5` で `lgb.train()` に渡される `params["num_leaves"]` が算出値（16）であることを monkeypatch で検証する。
- `auto_num_leaves=False`（既定）の場合に `params["num_leaves"]` がセットされないことを検証する。
- 代表タスク（regression）で検証し、`resolve_auto_num_leaves` の呼び出しが全4タスクで共通であることは `test_lgb_param_resolution.py` のユニットテストで担保する。

#### 2. `ratio_params` 学習ループ適用テスト
- **ファイル**: `tests/test_ratio_params.py`（新規）
- `min_data_in_leaf_ratio=0.05` 指定時に `params["min_data_in_leaf"]` がデータ行数に基づく算出値であることを monkeypatch で検証する。
- `min_data_in_bin_ratio=0.01` 指定時に `params["min_data_in_bin"]` が算出値であることを検証する。
- 代表タスク（regression）で検証する。

#### 3. `feature_weights` 学習ループ適用テスト
- **ファイル**: `tests/test_feature_weights.py`（新規）
- `feature_weights={"x1": 2.0}` 指定時に `params["feature_pre_filter"]` が `False` になり、`params["feature_weights"]` に正しい重みリストが渡されることを monkeypatch で検証する。
- `feature_weights` 未指定時に `params` に `feature_weights` キーが含まれないことを検証する。
- 代表タスク（regression）で検証する。

#### 4. Tuning search space 拡充テスト
- **ファイル**: `tests/test_tuning_search_space.py`（新規）
- `_default_search_space("regression", "standard")` の返却 dict に `lambda_l1`, `lambda_l2`, `path_smooth`, `min_gain_to_split` キーが含まれることを検証する。
- 各キーの `type`, `low`, `high` が妥当な値であることを検証する。

#### 5. 新パラメーター付き Artifact ラウンドトリップテスト
- **ファイル**: `tests/test_artifact_param_roundtrip.py`（新規）
- `auto_num_leaves=True`, `feature_weights`, `top_k` を設定した RunConfig で Artifact を `save` → `load` し、`run_config.yaml` 内にこれらのフィールドが保持されていることを検証する。
- ロードした Artifact から `predict` が成功すること（predict 自体は RunConfig のパラメーターに依存しないが、config 復元の整合性として検証）を確認する。

#### 6. `num_boost_round` 既定値後方互換テスト
- **ファイル**: `tests/test_num_boost_round.py`（既存に追加）
- `train.num_boost_round` 未指定時に既定値 300 で `lgb.train()` が呼ばれることを monkeypatch で検証する。

#### 7. 早期停止 `best_iteration` 動作テスト
- **ファイル**: `tests/test_early_stopping_validation.py`（既存に追加）
- `lgb.train()` を monkeypatch し、`best_iteration < num_boost_round` の Booster を返したときに `training_history.final_model.best_iteration` へ正しく記録されることを検証する。

#### 8. Causal での GroupKFold 自動適用テスト
- **ファイル**: `tests/test_dr_internal.py`（既存に追加）
- 既存の `split.group_col` 分岐テストを維持しつつ、`split.group_col` 未指定でも `causal.unit_id_col`（panel想定）経路で `GroupKFold` が選択されることを monkeypatch で検証する。
- `unit_id_col` が実質1群など GroupKFold が成立しない場合に `KFold` フォールバックが維持されることを検証する。

#### 9. 目的関数ユーザー上書きテスト
- **ファイル**: `tests/test_objective_override.py`（新規）
- Binary タスクで `lgb_params.objective` を `cross_entropy` に指定した場合に、`lgb.train()` の `params["objective"]` がユーザー指定値になることを monkeypatch で検証する。
- 未指定時にタスク既定値（例: `binary` → `binary`）が使用されることを検証する。

---

### テストファイル一覧

| ファイル | 新規/既存 | テスト数（想定） |
|---|---|---|
| `tests/test_auto_num_leaves.py` | 新規 | 2 |
| `tests/test_ratio_params.py` | 新規 | 2 |
| `tests/test_feature_weights.py` | 新規 | 2 |
| `tests/test_tuning_search_space.py` | 新規 | 1–2 |
| `tests/test_artifact_param_roundtrip.py` | 新規 | 1–2 |
| `tests/test_objective_override.py` | 新規 | 2 |
| `tests/test_num_boost_round.py` | 既存追加 | +1 |
| `tests/test_early_stopping_validation.py` | 既存追加 | +1 |
| `tests/test_dr_internal.py` | 既存追加 | +2 |

### 検証コマンド
- `uv run pytest tests/test_auto_num_leaves.py tests/test_ratio_params.py tests/test_feature_weights.py -v`
- `uv run pytest tests/test_tuning_search_space.py tests/test_artifact_param_roundtrip.py tests/test_objective_override.py -v`
- `uv run pytest tests/test_num_boost_round.py tests/test_early_stopping_validation.py tests/test_dr_internal.py -v`
- `uv run pytest tests -x --tb=short`

### 完了条件
- Phase25.8 の学習ループ適用テスト（auto_num_leaves / ratio_params / feature_weights）が追加され、`resolve_*` 関数のユニットテストだけでなく実際の学習器への適用が検証されること。
- Tuning search space の standard プリセットに追加されたパラメーターの存在が検証されること。
- 新パラメーター付き Artifact の save → load ラウンドトリップが成功すること。
- `num_boost_round` 既定値後方互換、`best_iteration` 記録契約（monkeypatch）、Causal GroupKFold 自動適用（`unit_id_col` 経路を含む）、目的関数ユーザー上書きのテストが追加されること。
- 既存テストが全パスし、Stable API（`veldra.api.*`）の互換性が維持されること。

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
