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

### 2026-02-14 (Session/PR: data-layout-root-cause-fix-row-col-height-and-overflow-scope)
**Context**
- 背景 / 依頼 / 目的：
  - Data画面で `stepper` と `Data` が1画面分下へ押し出される現象に対し、スクロール制御ではなくレイアウト計算の根本要因を除去する。

**Changes**
- 実装変更：
  - `src/veldra/gui/app.py`
    - `_main_layout()` の `dbc.Row` から `minHeight: 100vh` を削除。
    - 右カラム (`main-content-col`) の `minHeight: 100vh` を削除し、通常フローへ戻した。
  - `src/veldra/gui/assets/style.css`
    - グローバルな Bootstrap グリッド上書き
      - `.row, .col, [class*="col-"] { overflow: visible !important; }`
      を撤去。
    - `overflow: visible` は `.glass-card/.card/.card-body/.tab-content/.tab-pane` に局所化。
    - `.data-preview-card` の局所スクロール設定
      - `overflow-x: auto`
      - `overflow-y: auto`
      - `max-height: 420px`
      を維持。

**Decisions**
- Decision: confirmed
  - 内容：
    - レイアウト異常は「高さ固定 + グローバルoverflow上書き」の組み合わせを禁止し、必要箇所への局所適用に統一する。
  - 理由：
    - 画面全体の再フロー副作用を抑え、Data表示時の空白押し下げを防ぐため。
  - 影響範囲：
    - GUI layout / UX

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: remove-scrollto-side-effects-and-restore-data-preview-scroll)
**Context**
- 背景 / 依頼 / 目的：
  - Data画面でのスクロール異常（巨大空白・勝手な移動）と、Data Preview横スクロール消失を根本的に解消する。

**Changes**
- 実装変更：
  - `src/veldra/gui/app.py`
    - Data関連の clientside callback（`window.scrollTo` 実行）を全撤去。
    - `ui-scroll-fix` store を削除。
  - `src/veldra/gui/assets/style.css`
    - `html/body` 全体への `overflow-anchor: none` を撤去。
    - `.data-preview-card` を局所スクロール設定へ修正：
      - `overflow-x: auto`
      - `overflow-y: auto`
      - `max-height: 420px`
    - `#data-inspection-result` / `.data-inspection-zone` の局所アンカー制御は維持。
    - Data領域内の `.glass-card` / `.kpi-card` のアニメーション無効化は維持。

**Decisions**
- Decision: confirmed
  - 内容：
    - Data描画時のスクロール制御はブラウザ標準挙動に任せ、`window.scrollTo` による補正は行わない。
  - 理由：
    - スクロール補正JSがレイアウト再計算と競合し、逆効果の副作用を生んでいたため。
  - 影響範囲：
    - GUI / UX

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: rollback-internal-scroll-shell-to-restore-usability)
**Context**
- 背景 / 依頼 / 目的：
  - 内部スクロール化の副作用でページ操作不能（スクロール不可）が発生したため、可用性を最優先で復旧する。

**Changes**
- 実装変更：
  - `src/veldra/gui/assets/style.css`
    - `body { overflow: hidden; }` を撤回し `overflow-x: hidden` に戻した。
    - `#page-content` の内部スクロール指定を撤回。
    - `#main-content-col` の固定高さ/overflow制御を撤回。
    - sidebarのsticky指定を撤回。

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: app-shell-internal-scroll-refactor)
**Context**
- 背景 / 依頼 / 目的：
  - Data Preview表示時のスクロールジャンプが再発し続けたため、個別対症療法ではなく画面フレーム構造を再設計する。

**Changes**
- 実装変更：
  - `src/veldra/gui/app.py`
    - ルートレイアウトに `#app-shell` / `#main-content-col` を定義。
  - `src/veldra/gui/assets/style.css`
    - `body` のスクロールを禁止し、`#page-content` のみ内部スクロールに変更。
    - サイドバーを desktop で sticky 固定化。
    - メインカラムを `height: 100vh; overflow: hidden;` の固定フレームに変更。
  - `src/veldra/gui/pages/data_page.py`
    - Data Previewテーブルのセル表示を `nowrap` にして横スクロールを復帰。

**Decisions**
- Decision: confirmed
  - 内容：
    - GUIは「ウィンドウ全体スクロール」ではなく「メイン領域内部スクロール」を標準構造とする。
  - 理由：
    - 動的レンダリング時のビューポート位置変動を構造的に抑制できるため。
  - 影響範囲：
    - GUI layout / UX

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: data-inspection-single-trigger-and-animation-isolation)
**Context**
- 背景 / 依頼 / 目的：
  - 狭幅時のData Preview後スクロールジャンプについて、場当たり対応ではなく発火条件とレイアウト挙動を見直して根本対処する。

**Changes**
- 実装変更：
  - `src/veldra/gui/app.py`
    - Data inspection callbackの発火条件を `contents` 中心へ変更（`last_modified` 依存を除外）。
    - `workflow-state` を引き継いで `data_path` を更新する形に修正。
    - ファイル名表示を独立 callback に分離。
  - `src/veldra/gui/assets/style.css`
    - Data inspection領域の `overflow-anchor` を無効化。
    - Data inspection領域内の `.glass-card` / `.kpi-card` のアニメーションとtransformを無効化。
  - `tests/test_gui_app_coverage.py`
    - callback引数更新に合わせてテストを修正。

**Decisions**
- Decision: confirmed
  - 内容：
    - Data inspection結果表示は「単一データ入力トリガー + 領域限定の無アニメーション化」を標準とする。
  - 理由：
    - 狭幅レイアウトでの再アンカー/再フローによるスクロールジャンプを構造的に抑えるため。
  - 影響範囲：
    - GUI / UX / callback stability

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: gui-data-preview-datatable-removal-for-scroll-stability)
**Context**
- 背景 / 依頼 / 目的：
  - 画面幅が狭い時にData Preview後に下へ移動する問題が解消しきれていなかったため、根本原因の再除去を行う。

**Changes**
- 実装変更：
  - `src/veldra/gui/pages/data_page.py`
    - Data Previewの表示を `dash_table.DataTable` から静的HTMLテーブルへ置換。
    - 横/縦スクロールはカード側で維持しつつ、DataTable由来のフォーカス/再レイアウト挙動を排除。

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_pages_logic.py tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: gui-narrow-width-scroll-jump-hardening)
**Context**
- 背景 / 依頼 / 目的：
  - 画面幅が狭い場合にData Preview後にページが下へ移動する再発を解消する。

**Changes**
- 実装変更：
  - `src/veldra/gui/pages/data_page.py`
    - Data Previewカードに専用クラス `data-preview-card` を追加。
  - `src/veldra/gui/assets/style.css`
    - `data-preview-card` に `overflow: hidden !important` を適用し、局所スクロールを強制。
    - `#data-inspection-result` に `overflow-anchor: none` を適用。
  - `src/veldra/gui/app.py`
    - Data inspection後のスクロールリセットを多段タイミング（RAF + timeout）へ強化。

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: gui-data-preview-scroll-jump-regression-fix)
**Context**
- 背景 / 依頼 / 目的：
  - Data Preview表示後にビューが勝手に下へ移動する回帰を解消する。

**Changes**
- 実装変更：
  - `src/veldra/gui/app.py`
    - `data-inspection-result` 更新時にクライアントサイドで `window.scrollTo(0, 0)` を二段で実行。
    - DataTable側の内部挙動に依存せず、Data inspection後の画面位置を安定化。

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: gui-config-quick-run-and-preview-stability)
**Context**
- 背景 / 依頼 / 目的：
  - Data Preview表示をページ内で見やすく維持しつつ表示不具合を解消する。
  - Config画面で最下部までスクロールしなくてもRunへ移動できるUXを実現する。

**Changes**
- 実装変更：
  - `src/veldra/gui/pages/data_page.py`
    - Data Previewのテーブル表示を安定化（固定行ヘッダ指定/contain指定を削除）。
    - 画面内スクロール構成は維持しつつ、列表示幅と折返し挙動を調整。
  - `src/veldra/gui/pages/config_page.py`
    - Builder先頭に sticky な `Quick Actions` バーを追加。
      - `Run Now →`（即 `/run` 遷移）
      - `Jump to Export`（詳細設定継続時の導線）
    - 下部アクションに `Back to Top` を追加し、長い設定画面の往復コストを削減。

**Decisions**
- Decision: confirmed
  - 内容：
    - Config画面のRun導線は「上部即時導線」と「下部完了導線」の二重化で提供する。
  - 理由：
    - すぐ実行したい利用者と詳細調整したい利用者の両方を同時に満たせるため。
  - 影響範囲：
    - GUI / UX

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: gui-inline-preview-readiness-and-jst-labels)
**Context**
- 背景 / 依頼 / 目的：
  - Data Previewをモーダルではなく従来の画面内表示に戻しつつ、表示時のレイアウト変位を抑える。
  - RunのLaunch可否理由をユーザーに明示する。
  - ResultsのArtifact時刻表示を日本時間（JST）に統一する。

**Changes**
- 実装変更：
  - `src/veldra/gui/pages/data_page.py`
    - Data Previewを画面内表示へ戻し、固定高さ + 内部横/縦スクロールへ変更。
    - `contain: layout paint size` を適用してプレビュー領域外へのレイアウト影響を抑制。
  - `src/veldra/gui/pages/run_page.py`
    - `run-launch-status` を追加し、Launch可否の状態を常時表示。
  - `src/veldra/gui/app.py`
    - `run-execute-btn` の有効/無効と説明文を更新する callback を追加。
    - 条件不足時は不足項目（Data Source / Config Source / Artifact Path / Scenarios Path）を表示。
    - Artifact一覧ラベル時刻をJST整形する `_format_jst_timestamp(...)` を追加。
    - Results detail の `Created` もJST表示に統一。

**Decisions**
- Decision: confirmed
  - 内容：
    - Run画面は「実行可否」だけでなく「なぜ実行不可か」を常時表示する。
  - 理由：
    - ユーザーが状態把握のために試行錯誤する必要を減らすため。
  - 影響範囲：
    - GUI / UX

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: gui-preview-modal-and-artifact-autoselect-fallback)
**Context**
- 背景 / 依頼 / 目的：
  - Data Preview展開時のレイアウト変位を防止し、Results遷移時のArtifact未選択を解消する。

**Changes**
- 実装変更：
  - `src/veldra/gui/pages/data_page.py`
    - Data Previewをページ内展開からモーダル表示へ変更。
    - これによりプレビュー表示時もメインレイアウトを不変化。
  - `src/veldra/gui/app.py`
    - `data-preview-open-btn` / `data-preview-close-btn` のモーダル開閉コールバックを追加。
    - Artifact自動選択ロジックを関数化し、優先順位を明確化：
      1) `workflow-state.last_run_artifact` が options に存在すればそれを選択
      2) なければ options 先頭（最新）を自動選択

**Decisions**
- Decision: confirmed
  - 内容：
    - Results初期表示では「未選択状態」を基本的に作らず、最新artifactを必ず選択する。
  - 理由：
    - Run後の確認フローで無駄な選択操作を排除するため。
  - 影響範囲：
    - GUI / UX

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

### 2026-02-14 (Session/PR: gui-ux-autoselect-and-scroll-stabilization)
**Context**
- 背景 / 依頼 / 目的：
  - GUIの残課題として、Data画面のスクロールジャンプとResultsでの手動Artifact選択を解消する。

**Plan**
- Data選択時の処理を自動化したまま、ワイドデータ時の画面ジャンプを抑える。
- Run完了時に最新Artifactを状態保持し、Resultsで自動選択する。
- Run画面の既存挙動を壊さないようGUIテストを更新・再実行する。

**Changes**
- 実装変更：
  - `src/veldra/gui/pages/data_page.py`
    - Data PreviewをAccordion（初期折りたたみ）へ変更。
    - Previewテーブルに`maxHeight` + 内部スクロールを設定し、ワイド列データでの画面変位を抑制。
  - `src/veldra/gui/app.py`
    - Runジョブ更新コールバックで`workflow-state`を更新するよう拡張。
    - `fit`成功時に`result.payload.artifact_path`を`workflow-state.last_run_artifact`へ保存。
    - Results側の自動選択を、`workflow-state`と`artifact-select.options`の両方を見て実行する方式へ強化。
- テスト変更：
  - `tests/test_gui_app_coverage.py`
  - `tests/test_gui_app_callbacks_internal.py`
  - コールバック引数/戻り値変更に追随。

**Decisions**
- Decision: confirmed
  - 内容：
    - Resultsの初期選択は「最後に成功したRunのartifact_path」を優先する。
  - 理由：
    - Run直後の確認フローで手動選択を排除し、操作回数を減らすため。
  - 影響範囲：
    - GUI / UX

**Results**
- 動作確認結果：
  - `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_* tests/test_new_ux.py`
  - `73 passed, 1 warning`

**Risks / Notes**
- `fit`以外のアクションでは`artifact_path`がpayloadに含まれない場合があるため、自動選択更新は行われない。

**Open Questions**
- [ ] Data Previewの既定折りたたみをユーザー設定化するか。

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

### 2026-02-09 (Session/PR: phase4-examples-california-demo)
**Context**
- Add runnable examples for the current regression workflow (fit/predict/evaluate).
- Provide demo scripts that generate concrete output files for onboarding and verification.

**Plan**
- Add `examples/prepare_demo_data.py`, `examples/run_demo_regression.py`, and `examples/evaluate_demo_artifact.py`.
- Add shared helpers in `examples/common.py`.
- Add tests for prepare/run/evaluate example scripts.
- Update design and history documents with the examples contract.

**Changes**
- Added `examples/` scripts and `examples/README.md`.
- Added tests:
  - `tests/test_examples_prepare_demo_data.py`
  - `tests/test_examples_run_demo_regression.py`
  - `tests/test_examples_evaluate_demo_artifact.py`
- Updated `.gitignore` for `.codex/`, `examples/out/`, and generated CSV in `examples/data/`.

**Decisions**
- Decision: confirmed
  - Policy:
    - Example scripts are maintained as adapter-level executable references and do not alter `veldra.api.*` signatures.
  - Reason:
    - Reduces onboarding friction while preserving stable API compatibility.
  - Impact area:
    - API / Docs / Operability

- Decision: confirmed
  - Policy:
    - California Housing is the default demo source; local CSV must be created by `prepare_demo_data.py`.
  - Reason:
    - Makes demo flow explicit and reproducible with a consistent local input contract.
  - Impact area:
    - Data / Docs / Operability

**Results**
- Script-level outputs are produced under `examples/out/<timestamp>/`.
- `uv run ruff check .` passed.
- `uv run pytest -q` passed in this workspace before demo execution (15 passed).
- `uv run python examples/prepare_demo_data.py` failed in this environment due blocked network
  (`WinError 10061` while downloading California Housing), and exited with the expected hint.
- `uv run python examples/run_demo_regression.py` succeeded with local labeled CSV at the default path:
  - `run_id`: `ffff7218fba24318a1eaf4db85342f78`
  - `rmse`: `0.391954`
  - `mae`: `0.314148`
  - `r2`: `0.872247`
- `uv run python examples/evaluate_demo_artifact.py --artifact-path ...` succeeded:
  - `rmse`: `0.186839`
  - `mae`: `0.106562`
  - `r2`: `0.971317`
  - `n_rows`: `500`

**Risks / Notes**
- `prepare_demo_data.py` requires network access at fetch time.
- If download is unavailable, the script exits with a retry/network hint.

**Open Questions**
- [ ] Should the next example phase prioritize binary calibration flow or multiclass baseline first?
- [ ] Should artifact re-evaluation examples accept file-path input directly in addition to DataFrame-only API usage?

### 2026-02-09 (Session/PR: phase5-binary-fit-predict-evaluate-oof-calibration)
**Context**
- Extend MVP from regression-only runtime to binary runtime with OOF-safe calibration.
- Preserve stable API signatures and enforce artifact-based reproducibility.

**Plan**
- Add binary training core with CV and OOF Platt calibration.
- Extend artifact persistence for calibrator/calibration diagnostics/fixed threshold.
- Enable binary predict/evaluate through existing stable API entrypoints.
- Add binary-focused unit tests and update design/history docs.

**Changes**
- Code changes:
  - Added `src/veldra/modeling/binary.py`.
  - Updated `src/veldra/modeling/__init__.py` exports.
  - Updated `src/veldra/api/runner.py` for binary `fit/predict/evaluate`.
  - Updated `src/veldra/api/artifact.py` for binary prediction path.
  - Updated `src/veldra/artifact/store.py` to persist/load binary extras.
  - Updated `src/veldra/config/models.py` for binary calibration/threshold validation.
- Tests:
  - Added `tests/test_binary_fit_smoke.py`
  - Added `tests/test_binary_oof_calibration.py`
  - Added `tests/test_binary_predict_contract.py`
  - Added `tests/test_binary_evaluate_metrics.py`
  - Added `tests/test_binary_artifact_roundtrip.py`
  - Updated `tests/test_api_surface.py`
  - Updated `tests/test_runconfig_validation.py`

**Decisions**
- Decision: confirmed
  - Policy:
    - OOF-only fit is mandatory for probability calibration in binary flow.
  - Reason:
    - Prevent calibration leakage and keep evaluation defensible.
  - Impact area:
    - Modeling / Validation / Reproducibility

- Decision: confirmed
  - Policy:
    - Binary prediction contract is `p_cal` (default), plus `p_raw` and `label_pred`.
  - Reason:
    - Balance operational simplicity with auditability/debuggability.
  - Impact area:
    - API / Artifact / Consumer contract

- Decision: confirmed
  - Policy:
    - Binary evaluation metrics are `auc`, `logloss`, `brier`.
  - Reason:
    - Covers discrimination and probability quality.
  - Impact area:
    - Evaluation / Reporting

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : 35 passed.
- `uv run coverage run -m pytest -q && uv run coverage report -m` : TOTAL 94%.
- Binary sample run (`fit` + `evaluate`) metrics snapshot:
  - fit mean: `auc=0.9534`, `logloss=0.3247`, `brier=0.0955`
  - evaluate: `auc=1.0000`, `logloss=0.1065`, `brier=0.0103`

**Risks / Notes**
- Threshold optimization remains out of scope in this phase (fixed 0.5 only).
- `tune/simulate/export` remain unimplemented.

**Open Questions**
- [ ] Should Phase 6 prioritize binary threshold optimization or multiclass baseline first?
- [ ] Should binary `label_pred` support original class label restoration in API output?

### 2026-02-09 (Session/PR follow-up: phase5-binary-examples)
**Context**
- Extend examples after Phase 5 merge so binary workflow is reproducible end-to-end like regression.

**Changes**
- Added scripts:
  - `examples/prepare_demo_data_binary.py`
  - `examples/run_demo_binary.py`
  - `examples/evaluate_demo_binary_artifact.py`
- Updated docs:
  - `examples/README.md` with binary commands and outputs.
  - `DESIGN_BLUEPRINT.md` with binary examples note.
- Added tests:
  - `tests/test_examples_prepare_demo_data_binary.py`
  - `tests/test_examples_run_demo_binary.py`
  - `tests/test_examples_evaluate_demo_binary_artifact.py`

**Decisions**
- Decision: confirmed
  - Policy:
    - Binary examples are included as first-class adapter scripts in the repository.
  - Reason:
    - Keep onboarding and regression/binary parity for reproducible demo flows.
  - Impact area:
    - Examples / Documentation / QA

### 2026-02-10 (Session planning: phase6-multiclass-fit-predict-evaluate-examples)
**Context**
- Start Phase 6 to extend runtime from regression/binary to multiclass MVP.
- Keep stable API signatures unchanged and preserve artifact-first reproducibility.

**Decisions**
- Decision: provisional
  - Policy:
    - Multiclass prediction output contract is `label_pred` + `proba_<class>`.
  - Reason:
    - Supports both operational classification and probability-level analysis.
  - Impact area:
    - API / Artifact / Consumer contract

- Decision: provisional
  - Policy:
    - Multiclass evaluation metrics are `accuracy`, `macro_f1`, `logloss`.
  - Reason:
    - Balances label-quality and probability-quality checks.
  - Impact area:
    - Evaluation / Reporting

- Decision: provisional
  - Policy:
    - Multiclass examples are implemented in the same phase as core/API.
  - Reason:
    - Keeps onboarding parity across regression, binary, multiclass workflows.
  - Impact area:
    - Examples / Documentation / QA

### 2026-02-10 (Session/PR: phase6-multiclass-fit-predict-evaluate-examples)
**Context**
- Implement multiclass runtime (`fit/predict/evaluate`) and keep stable API signatures unchanged.
- Add runnable multiclass examples and matching tests in the same phase.

**Changes**
- Code changes:
  - Added `src/veldra/modeling/multiclass.py`.
  - Updated `src/veldra/modeling/__init__.py` exports.
  - Updated `src/veldra/api/runner.py` for multiclass `fit/predict/evaluate`.
  - Updated `src/veldra/api/artifact.py` for multiclass prediction contract.
  - Updated `examples/common.py` with multiclass default dataset path.
  - Added multiclass examples:
    - `examples/prepare_demo_data_multiclass.py`
    - `examples/run_demo_multiclass.py`
    - `examples/evaluate_demo_multiclass_artifact.py`
  - Updated `README.md` with multiclass status and commands.
- Tests:
  - Added `tests/test_multiclass_fit_smoke.py`
  - Added `tests/test_multiclass_predict_contract.py`
  - Added `tests/test_multiclass_evaluate_metrics.py`
  - Added `tests/test_multiclass_artifact_roundtrip.py`
  - Added `tests/test_examples_prepare_demo_data_multiclass.py`
  - Added `tests/test_examples_run_demo_multiclass.py`
  - Added `tests/test_examples_evaluate_demo_multiclass_artifact.py`
  - Updated `tests/test_api_surface.py`
  - Updated `tests/test_runconfig_validation.py`

**Decisions**
- Decision: confirmed
  - Policy:
    - Multiclass prediction contract is `label_pred` + `proba_<class>`.
  - Reason:
    - Provides both decision output and probability diagnostics from one API contract.
  - Impact area:
    - API / Artifact / Consumer contract

- Decision: confirmed
  - Policy:
    - Multiclass evaluation metrics are `accuracy`, `macro_f1`, `logloss`.
  - Reason:
    - Covers class-level quality and probability calibration quality.
  - Impact area:
    - Evaluation / Reporting

- Decision: confirmed
  - Policy:
    - Multiclass examples are shipped in the same phase as core/API support.
  - Reason:
    - Keeps runnable onboarding parity across task types.
  - Impact area:
    - Examples / Documentation / QA

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : 53 passed.
- Multiclass demo run (`examples/run_demo_multiclass.py`) snapshot:
  - `accuracy=0.900000`
  - `macro_f1=0.899749`
  - `logloss=0.818215`
- Multiclass artifact re-evaluation (`examples/evaluate_demo_multiclass_artifact.py`) snapshot:
  - `accuracy=0.980000`
  - `macro_f1=0.979998`
  - `logloss=0.163656`
  - `n_rows=150`

### 2026-02-10 (Session/PR: phase7-binary-threshold-optimization-optin)
**Context**
- Add binary threshold optimization while keeping default runtime behavior unchanged.
- Ensure optimization is explicit opt-in and does not interfere with typical usage.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - added `postprocess.threshold_optimization` config
    - added validation for binary-only usage and conflict with fixed threshold
  - Updated `src/veldra/modeling/binary.py`:
    - added optional OOF `p_cal` threshold optimization (F1)
    - added optional threshold curve generation
  - Updated `src/veldra/artifact/store.py` and `src/veldra/api/artifact.py`:
    - added optional `threshold_curve.csv` save/load
  - Updated `src/veldra/api/runner.py`:
    - binary evaluate now includes threshold-dependent metrics
    - binary metadata includes threshold policy/value
  - Updated examples/docs:
    - `examples/run_demo_binary.py` adds `--optimize-threshold` flag
    - `examples/evaluate_demo_binary_artifact.py` prints threshold-dependent metrics
    - `README.md` documents opt-in threshold optimization usage

**Decisions**
- Decision: confirmed
  - Policy:
    - Threshold optimization is opt-in only; default remains fixed threshold.
  - Reason:
    - Avoid changing established behavior in common production usage.
  - Impact area:
    - Compatibility / Operability / API behavior

- Decision: confirmed
  - Policy:
    - Threshold optimization uses OOF calibrated probabilities only.
  - Reason:
    - Prevent leakage and preserve defensible evaluation.
  - Impact area:
    - Modeling / Validation

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (phase7 implementation set).
- Binary default path remained fixed-threshold compatible.
- `--optimize-threshold` example path produced optimized threshold policy and optional curve file.

**Risks / Notes**
- Threshold optimization objective is fixed to F1 in this phase.
- The feature is intentionally binary-only and opt-in to avoid runtime surprise for standard users.

**Open Questions**
- [ ] Should additional threshold objectives (e.g., precision/recall constrained) be added in a future
      phase, or kept out of stable API for now?
- [ ] Should threshold policy be exposed in CLI/GUI presets after API stabilization?

### 2026-02-10 (Session/PR: phase7.1-doc-closure-and-phase8-tune-mvp)
**Context**
- Close remaining doc consistency tasks and implement `runner.tune` MVP.
- Keep stable API signatures unchanged and avoid behavioral regressions in existing paths.

**Plan**
- Phase 7.1:
  - complete unfinished Phase 7 history section
  - add capability matrix and historical clarification in design docs
- Phase 8:
  - implement `tune` runtime for regression/binary/multiclass
  - persist tuning artifacts under `artifacts/tuning/<run_id>/`
  - add full test coverage for tune smoke/validation/artifact outputs

**Changes**
- Code changes:
  - Added `src/veldra/modeling/tuning.py` (Optuna-backed tuning engine).
  - Updated `src/veldra/modeling/__init__.py` exports.
  - Updated `src/veldra/api/runner.py`:
    - `tune` now validates config and executes tuning
    - writes `study_summary.json` and `trials.parquet`
    - returns populated `TuneResult`
- Dependency updates:
  - Added runtime dependency: `optuna==4.0.0`
  - Updated lockfile (`uv.lock`)
- Tests:
  - Added `tests/test_tune_smoke_regression.py`
  - Added `tests/test_tune_smoke_binary.py`
  - Added `tests/test_tune_smoke_multiclass.py`
  - Added `tests/test_tune_validation.py`
  - Added `tests/test_tune_artifacts.py`
  - Updated `tests/test_api_surface.py`
  - Updated `tests/test_runner_additional.py`
- Docs:
  - Updated `DESIGN_BLUEPRINT.md` with capability matrix, historical note, and Phase 8 section.
  - Updated `README.md` to reflect `tune` support and usage.

**Decisions**
- Decision: confirmed
  - Policy:
    - `tune` requires `tuning.enabled=true`; otherwise validation error.
  - Reason:
    - Prevent accidental tuning execution from standard training configs.
  - Impact area:
    - API behavior / Operability

- Decision: confirmed
  - Policy:
    - Tuning objectives are fixed by task in MVP:
      regression=`rmse`(min), binary=`auc`(max), multiclass=`macro_f1`(max).
  - Reason:
    - Keep MVP deterministic, simple, and auditable.
  - Impact area:
    - Modeling / Evaluation

- Decision: confirmed
  - Policy:
    - Binary threshold optimization stays fully opt-in for prediction/evaluation flow and is disabled
      in tuning objective evaluation.
  - Reason:
    - Keep tuning comparisons focused on probability quality and preserve non-intrusive default policy.
  - Impact area:
    - Compatibility / Modeling

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed with newly added tune tests.
- `uv run coverage run -m pytest -q` + `uv run coverage report -m` : passed.
- `tune` now returns non-empty `best_params` / `best_score` for regression, binary, and multiclass.
- Tuning artifacts are generated under `artifacts/tuning/<run_id>/`.

**Risks / Notes**
- Search-space DSL is intentionally minimal in MVP; advanced conditional spaces are out of scope.
- `simulate/export/frontier runtime` remain unimplemented by design.

**Open Questions**
- [ ] Should next phase add user-selectable optimization metric per task, or keep fixed objective
      contracts for API stability?
- [ ] Should tuning results be loadable through Artifact-like API, or remain file-based outputs?

**Traceability note**
- The `Decision: provisional` entries in the 2026-02-10 Phase 6 planning section are now resolved by
  the confirmed implementation entry: `2026-02-10 (Session/PR: phase6-multiclass-fit-predict-evaluate-examples)`.

### 2026-02-10 (Session planning: phase8.1-tune-expansion)
**Context**
- Expand tune runtime for practical optimization workflows (objective selection, resume, progress
  logging, and tune examples).

**Decisions**
- Decision: provisional
  - Policy:
    - Objective is selectable with task-specific allowed choices.
  - Reason:
    - Preserve safety and clarity while avoiding free-form invalid objective names.
  - Impact area:
    - Config / Tuning / API behavior

- Decision: provisional
  - Policy:
    - Resume is implemented using Optuna SQLite (`study.db`) in artifact tuning directory.
  - Reason:
    - Ensure trial-level durability and restartability for interrupted runs.
  - Impact area:
    - Operability / Reproducibility

- Decision: provisional
  - Policy:
    - Tune progress logs are emitted during optimization with selectable log level.
  - Reason:
    - Improve observability during long optimization runs.
  - Impact area:
    - Logging / Operations

### 2026-02-10 (Session/PR: phase8.1-tune-expansion-implementation)
**Context**
- Extend tune runtime to support objective selection, resume, progress logging, and runnable examples.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - tuning fields added: `objective`, `resume`, `study_name`, `log_level`
    - task-constrained objective validation added
  - Updated `src/veldra/modeling/tuning.py`:
    - objective-aware direction mapping
    - deterministic/default `study_name` generation
    - SQLite storage support with resume handling
    - trial callback persistence (`study_summary.json`, `trials.parquet`)
  - Updated `src/veldra/api/runner.py`:
    - structured progress logs per trial (`tune trial completed`)
    - log level mapped from config
    - resume/non-resume study behavior and storage path management
  - Updated `src/veldra/modeling/binary.py`:
    - added binary threshold-dependent mean metrics for tune objective compatibility
  - Added `examples/run_demo_tune.py`:
    - task switch, objective override, resume, study-name, log-level, search-space-file
- Tests added:
  - `tests/test_tune_objective_selection.py`
  - `tests/test_tune_resume.py`
  - `tests/test_tune_logging.py`
  - `tests/test_examples_run_demo_tune.py`
- Tests updated:
  - `tests/test_tune_validation.py`
  - `tests/test_tuning_internal.py`

**Decisions**
- Decision: confirmed
  - Policy:
    - Tuning objective is selectable, but constrained by task-specific allowed metrics.
  - Reason:
    - Enable flexibility while preventing invalid objective configurations.
  - Impact area:
    - Config / Tuning / API behavior

- Decision: confirmed
  - Policy:
    - Resume uses Optuna SQLite in `artifacts/tuning/<study_name>/study.db`.
  - Reason:
    - Preserve progress across interruptions and allow controlled continuation.
  - Impact area:
    - Reproducibility / Operability

- Decision: confirmed
  - Policy:
    - Progress logs are emitted per trial with configurable level from `tuning.log_level`.
  - Reason:
    - Improve observability for long-running optimization.
  - Impact area:
    - Logging / Operations

- Decision: confirmed
  - Policy:
    - `tuning.search_space` remains the primary contract to choose target parameters and ranges.
  - Reason:
    - Keep optimization boundary explicit and controllable by users.
  - Impact area:
    - Tuning UX / Reproducibility

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed.

### 2026-02-14 (Session/PR: phase25-gui-operability-completion)
**Context**
- Resume and complete Phase 25 by stabilizing GUI callback behavior and restoring full test pass.

**Plan**
- Reproduce current failures with full test suite.
- Fix callback compatibility and error handling in `src/veldra/gui/app.py`.
- Re-run full `pytest` and update design/history docs.

**Changes**
- Code changes:
  - Updated `src/veldra/gui/app.py`:
    - `_cb_inspect_data` now supports both new and legacy call signatures.
    - Added base64 decode error handling (`ValueError` / `binascii.Error`) for upload parsing.
    - Added legacy/new return-shape compatibility in `_cb_inspect_data`.
    - Added `_PopulateBuilderLegacyResult` for `_cb_populate_builder_options` compatibility:
      - supports modern indexing contract and legacy unpacking contract simultaneously.
- Documentation changes:
  - Updated `DESIGN_BLUEPRINT.md` Phase 25 status from in-progress to completed.
  - Added this session log to `HISTORY.md`.
- Test changes:
  - No test file modifications.
  - Resolved failing tests via callback compatibility fixes.

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI callback helper functions maintain backward-compatible call/return contracts for direct-call tests while preserving Dash runtime outputs.
  - Reason:
    - Prevents regressions across mixed test generations without changing stable API/Core behavior.
  - Impact area:
    - GUI adapter reliability / Test stability

**Results**
- `pytest -q` : **405 passed, 0 failed**.
- Remaining warning:
  - `joblib` serial-mode warning due environment permission (`[Errno 13] Permission denied`), non-blocking.

**Risks / Notes**
- Repository-wide `ruff` violations already exist outside this session's functional scope.

**Open Questions**
- [ ] Should legacy direct-call compatibility paths in GUI callbacks be removed after test suite unification?

### 2026-02-14 (Session/PR: gui-run-default-config-bootstrap-fix)
**Context**
- GUI Run page failed when no existing `configs/gui_run.yaml` file was present.
- User requested a fix with understandable default config values.

**Changes**
- Code changes:
  - Updated `src/veldra/gui/app.py`:
    - added `DEFAULT_GUI_RUN_CONFIG_YAML` template with explicit defaults.
    - added `_ensure_default_run_config(...)` to auto-create missing config file.
    - `_cb_enqueue_run_job(...)` now ensures default config file exists before enqueue.
- Added file:
  - `configs/gui_run.yaml` (default, readable template for GUI Run).
- Docs updated:
  - `README.md` GUI section now clarifies `configs/gui_run.yaml` is auto-created when missing.

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI Run default config path remains `configs/gui_run.yaml`, and the file is auto-bootstrapped with a readable baseline template.
  - Reason:
    - Eliminates first-run failure while keeping operator-visible, editable config state.
  - Impact area:
    - GUI operability / DX

**Results**
- `pytest -q tests/test_gui_app_callbacks_internal.py::test_callback_wrappers_cover_branches tests/test_new_ux.py::test_04_run_page_submission` : passed.

### 2026-02-14 (Session/PR: gui-results-artifact-compat-fix)
**Context**
- GUI Results page failed with:
  - `Error loading artifact: 'Artifact' object has no attribute 'metadata'`

**Changes**
- Code changes:
  - Updated `src/veldra/gui/app.py` result-view callback to support both:
    - legacy/mock attributes: `run_id`, `task_type`, `created_at_utc`, `config`, `metadata`
    - current `Artifact` contract: `manifest`, `run_config`, `feature_schema`
  - Added safe fallback resolution for:
    - run id / task type / created timestamp
    - run config payload display
    - feature importance source (`metadata` or `feature_schema`)

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI adapter must be tolerant to artifact shape differences between runtime objects and test doubles.
  - Reason:
    - Prevent UI hard-failure on non-essential optional fields.
  - Impact area:
    - GUI robustness / Backward compatibility

**Results**
- `pytest -q tests/test_gui_app_callbacks_results.py::test_results_callbacks tests/test_gui_app_coverage_2.py::test_result_view_empty_and_error` : passed.

### 2026-02-14 (Session/PR: gui-data-scroll-and-run-auto-navigation-fix)
**Context**
- GUI issues reported:
  - after data inspect, viewport unexpectedly moved downward.
  - auto-navigation from `/run` to `/results` was inconsistent after job completion.

**Changes**
- Updated `src/veldra/gui/app.py`:
  - Added `dcc.Store(id="ui-scroll-fix")` and clientside callback to reset scroll to top when
    `data-inspection-result` updates.
  - Enhanced `_cb_refresh_run_jobs(...)` auto-navigation logic:
    - existing transition rule (`queued/running -> succeeded`) kept.
    - added first-poll success handling (`old status missing`) with recent-job guard (<=120s).
  - Restored module-scope imports (`Artifact`, `evaluate`, `load_tabular_data`) to keep callback
    monkeypatch compatibility in tests.
- Tests updated:
  - `tests/test_gui_app_coverage.py`
    - added coverage case for first-poll completion auto-navigation to `/results`.

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI auto-navigation must be robust to polling race conditions and should not depend solely on prior status cache.
  - Reason:
    - Prevent intermittent UX behavior when job completion happens between polling intervals.
  - Impact area:
    - GUI operability / Reliability

**Results**
- `pytest -q tests/test_gui_app_coverage.py::test_app_coverage_edge_cases tests/test_gui_app_callbacks_internal.py::test_callback_wrappers_cover_branches` : passed.

### 2026-02-14 (Session/PR: gui-test-regression-compat-restoration)
**Context**
- Full test run regressed after GUI callback/import refactors.
- Failures were test-compat issues, not product behavior issues.

**Changes**
- Updated `src/veldra/gui/services.py`:
  - restored module-level imports used by tests/monkeypatch:
    - `Artifact`
    - runner functions (`fit`, `evaluate`, `tune`, `simulate`, `export`, `estimate_dr`)
    - `load_tabular_data`
  - removed lazy imports inside `inspect_data(...)` and `run_action(...)`.
- Updated `src/veldra/gui/app.py`:
  - added callback-map compatibility bridge for clientside callback entries:
    - ensure each `callback_map` value has `callback` key for legacy tests that iterate map values.

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI adapter modules maintain test-monkeypatchable symbols at module scope.
  - Reason:
    - Keeps regression tests stable while preserving runtime behavior.
  - Impact area:
    - Test stability / Backward compatibility

**Results**
- `pytest -q` : **401 passed, 4 skipped, 0 failed**.

### 2026-02-14 (Session/PR: gui-data-page-upload-state-and-scroll-fix)
**Context**
- Data page issues reported:
  - viewport jumped downward after `Inspect Data` (notably with many preview columns)
  - initial `Error: Data file does not exist: examples/data/california_housing.csv`
  - `No file selected` label was hard to read
  - file label did not update immediately on file selection

**Changes**
- Updated `src/veldra/gui/app.py`:
  - removed no-upload fallback to `examples/data/california_housing.csv` in `_cb_inspect_data(...)`.
  - now returns a clear guidance message when no file is selected.
  - added `_cb_update_selected_file_label(...)` callback:
    - updates `data-selected-file-label` as soon as upload filename changes.
    - clears `data-error-message` on selection.
- Updated `src/veldra/gui/pages/data_page.py`:
  - changed selected-file label class from `text-muted` to `text-light`.
  - set DataTable `cell_selectable=False` to reduce focus-driven scroll jumps after render.
- Tests updated:
  - `tests/test_gui_app_coverage.py`
  - `tests/test_new_ux.py`
  - `tests/test_gui_app_coverage_2.py` (added selected-file-label callback coverage)

**Decisions**
- Decision: confirmed
  - Policy:
    - Data inspection requires explicit file selection; no implicit sample fallback at runtime.
  - Reason:
    - Avoid misleading startup errors and ensure user-visible state reflects actual input selection.
  - Impact area:
    - GUI UX / Operability

**Results**
- `pytest -q tests/test_gui_app_coverage.py::test_app_coverage_edge_cases tests/test_new_ux.py::test_01_data_page_inspection_flow tests/test_gui_app_coverage_2.py::test_inspect_data_upload_unsupported tests/test_gui_app_coverage_2.py::test_update_selected_file_label` : passed.

### 2026-02-14 (Session/PR: gui-config-to-run-state-sync-fix)
**Context**
- User reported Run behaved as regression even after selecting binary config.
- Job history showed all runs used `config_path=configs/gui_run.yaml` with `config_yaml=None`.

**Changes**
- Updated `src/veldra/gui/app.py`:
  - Added `_cb_cache_config_yaml(...)` and callback binding:
    - `Input("config-yaml", "value") -> Output("workflow-state", "data", allow_duplicate=True)`
  - This caches latest Config Builder YAML into workflow state.
- Updated `src/veldra/gui/pages/run_page.py`:
  - `run-config-yaml` textarea now initializes from `workflow-state["config_yaml"]`.
  - Run submission now carries actual config YAML by default.

**Decisions**
- Decision: confirmed
  - Policy:
    - Run page should consume the latest Config Builder YAML through shared workflow state.
  - Reason:
    - Enforces the RunConfig single-entry principle across GUI pages and prevents stale default-config execution.
  - Impact area:
    - GUI consistency / Config correctness

**Results**
- `pytest -q tests/test_new_ux.py::test_04_run_page_submission tests/test_gui_app_callbacks_internal.py::test_callback_wrappers_cover_branches` : passed.
- `pytest -q tests/test_gui_app_layout.py tests/test_gui_pages_and_init.py` : passed.

### 2026-02-14 (Session/PR: gui-results-feature-importance-fallback)
**Context**
- Primary metrics chart rendered, but Feature Importance chart remained empty for real artifacts.

**Changes**
- Updated `src/veldra/gui/app.py`:
  - Added fallback path for feature importance extraction from LightGBM booster when
    `metadata.feature_importance` / `feature_schema.feature_importance` are absent.
  - Uses `feature_schema.feature_names` (or booster feature names) + `importance_type='gain'`.
- Updated tests:
  - `tests/test_gui_app_callbacks_results.py` now covers booster-based feature importance fallback.

**Decisions**
- Decision: confirmed
  - Policy:
    - Results page should derive feature importance from the model artifact when explicit metadata is missing.
  - Reason:
    - Keeps GUI informative without requiring artifact schema migration.
  - Impact area:
    - GUI operability / Artifact backward compatibility

**Results**
- `pytest -q tests/test_gui_app_callbacks_results.py::test_results_callbacks tests/test_gui_app_coverage_2.py::test_result_view_empty_and_error` : passed.

### 2026-02-14 (Session/PR: gui-results-metrics-mean-plot-fix)
**Context**
- Results page no longer errored, but charts were empty for newly created artifacts.

**Changes**
- Updated `src/veldra/gui/app.py`:
  - Added metrics selection logic that supports both:
    - flat numeric metrics (`{"rmse": ..., "mae": ...}`)
    - nested artifact metrics contract (`{"folds": [...], "mean": {...}}`)
  - Results charts/KPI now use `mean` metrics automatically when top-level numeric keys are absent.
  - Added `r2` KPI fallback when `r2_score` key is not present.
- Updated tests:
  - `tests/test_gui_app_callbacks_results.py` now includes nested `mean` metrics coverage.

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI Results visualization uses normalized metrics view rather than raw dict shape assumptions.
  - Reason:
    - Keep graph rendering stable across artifact metric schema variants.
  - Impact area:
    - GUI operability / Artifact compatibility

**Results**
- `pytest -q tests/test_gui_app_callbacks_results.py::test_results_callbacks tests/test_gui_app_coverage_2.py::test_result_view_empty_and_error` : passed.

### 2026-02-14 (Session/PR: gui-results-runconfig-json-serialization-fix)
**Context**
- GUI Results page showed:
  - `Error loading artifact: Object of type RunConfig is not JSON serializable`

**Changes**
- Code changes:
  - Updated `src/veldra/gui/app.py`:
    - added `_to_jsonable(...)` helper for dataclass / Pydantic model safe conversion.
    - `_json_dumps(...)` now serializes via `_to_jsonable(...)` with `default=str`.
    - result detail panel now uses `_json_dumps(config_obj)` for config rendering.

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI detail rendering must not assume plain-dict config objects; it must handle `RunConfig` directly.
  - Reason:
    - Prevents runtime UI failures across artifact versions and object shapes.
  - Impact area:
    - GUI robustness / Operability

**Results**
- `pytest -q tests/test_gui_app_callbacks_results.py::test_results_callbacks tests/test_gui_app_coverage_2.py::test_result_view_empty_and_error` : passed.

### 2026-02-12 (Session planning: phase25-gui-async-jobs-migrate-workflow-mvp)
**Context**
- Prioritize GUI operability over additional DR-DiD scope.
- Keep stable runtime API signatures unchanged and extend GUI adapter behavior only.

**Decisions**
- Decision: provisional
  - Policy:
    - Add async run execution to `/run` using SQLite persistence and a single worker.
  - Reason:
    - Removes UI blocking for long-running `tune`/causal jobs while preserving local simplicity.
  - Impact area:
    - GUI operability / Reliability

- Decision: provisional
  - Policy:
    - Implement best-effort cancel:
      - `queued` -> immediate cancel
      - `running` -> `cancel_requested` marker (completion may still occur)
  - Reason:
    - Avoids unsafe force-kill behavior in MVP.
  - Impact area:
    - Job safety / UX clarity

- Decision: provisional
  - Policy:
    - Integrate config migrate workflow in `/config` (preview + diff + apply).
  - Reason:
    - Reuses Phase 22 migration contract and closes GUI usability gap.
  - Impact area:
    - Config governance / Operator workflow

### 2026-02-12 (Session/PR: phase25-gui-async-jobs-migrate-workflow-mvp)
**Context**
- Enhanced Dash GUI with async job execution and config migrate integration.

**Changes**
- Code changes:
  - Added:
    - `src/veldra/gui/job_store.py`
    - `src/veldra/gui/worker.py`
  - Updated:
    - `src/veldra/gui/types.py`
    - `src/veldra/gui/services.py`
    - `src/veldra/gui/app.py`
    - `src/veldra/gui/pages/run_page.py`
    - `src/veldra/gui/pages/config_page.py`
    - `src/veldra/gui/server.py`
- Tests added:
  - `tests/test_gui_job_store.py`
  - `tests/test_gui_worker.py`
  - `tests/test_gui_run_async.py`
  - `tests/test_gui_config_migrate_workflow.py`
- Tests updated:
  - `tests/test_gui_app_layout.py`
  - `tests/test_gui_app_helpers.py`
  - `tests/test_gui_pages_and_init.py`
  - `tests/test_gui_services_config_validation.py`
  - `tests/test_gui_services_run_dispatch.py`
  - `tests/test_gui_server.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI `/run` now uses async job queue with SQLite persistence (`.veldra_gui/jobs.sqlite3` by default).
  - Reason:
    - Provides restart-safe job history and non-blocking execution with minimal architectural risk.
  - Impact area:
    - GUI runtime behavior / Operability

- Decision: confirmed
  - Policy:
    - best-effort cancel contract is implemented and surfaced in job status transitions.
  - Reason:
    - Keeps cancellation semantics explicit without forcefully terminating model execution.
  - Impact area:
    - Safety / Predictability

- Decision: confirmed
  - Policy:
    - Config migrate workflow is available in GUI with preview/diff/apply and overwrite rejection.
  - Reason:
    - Aligns GUI operations with strict migration guarantees from Phase 22.
  - Impact area:
    - Config management / UX consistency

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed.
- `uv run coverage report -m --include="src/veldra/gui/*,src/veldra/config/migrate.py"` : 90% total.

### 2026-02-12 (Session planning: phase24-causal-tune-balance-priority)
**Context**
- Extend causal tuning from SE/overlap-centered objectives to TWANG-style balance-priority behavior.
- Keep stable public API signatures unchanged.

**Decisions**
- Decision: provisional
  - Policy:
    - Add balance-priority objectives for causal tuning:
      - `dr_balance_priority`
      - `drdid_balance_priority`
  - Reason:
    - Explicitly enforce covariate balance quality before variance minimization.
  - Impact area:
    - Causal tuning quality / Operability

- Decision: provisional
  - Policy:
    - Change causal tuning defaults to balance-priority objectives.
  - Reason:
    - Make safer causal objective behavior the default path for operational use.
  - Impact area:
    - Default behavior / Backward compatibility (explicit old objective still supported)

- Decision: provisional
  - Policy:
    - Introduce `tuning.causal_balance_threshold` with default `0.10`.
  - Reason:
    - Provide an explicit, configurable acceptance boundary for weighted SMD.
  - Impact area:
    - Config contract / Diagnostics governance

### 2026-02-12 (Session/PR: phase24-causal-tune-balance-priority)
**Context**
- Implement balance-priority causal tuning for both DR and DR-DiD.
- Preserve stable API signatures and legacy objective compatibility.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - added objective options:
      - DR: `dr_balance_priority`
      - DR-DiD: `drdid_balance_priority`
    - updated causal default objectives:
      - DR -> `dr_balance_priority`
      - DR-DiD -> `drdid_balance_priority`
    - added `tuning.causal_balance_threshold` (default `0.10`) and validation rules.
  - Updated `src/veldra/causal/dr.py`:
    - added diagnostics in metrics/summary:
      - `overlap_metric`
      - `smd_max_unweighted`
      - `smd_max_weighted`
    - weighted SMD now respects estimand-specific weighting (ATT/ATE).
  - Updated `src/veldra/causal/dr_did.py`:
    - aligned DR-DiD diagnostic contract with DR (`overlap` + `SMD` keys).
  - Updated `src/veldra/modeling/tuning.py`:
    - added balance-priority objective handling.
    - score logic:
      - if balanced: objective = `std_error`
      - if violated: objective = `1_000_000 + penalty_weight * violation`
    - persisted trial components:
      - `estimate`, `std_error`, `overlap_metric`
      - `smd_max_unweighted`, `smd_max_weighted`
      - `balance_threshold`, `balance_violation`
      - `penalty_weight`, `penalty`, `objective_value`, `objective_stage`
  - Updated `src/veldra/api/runner.py`:
    - tune metadata now includes `causal_balance_threshold`.
  - Updated `examples/run_demo_tune.py`:
    - added causal CLI options including `--causal-balance-threshold`.
    - objective-based causal method inference for `dr_*` / `drdid_*`.
- Tests added:
  - `tests/test_tune_causal_balance_priority.py`
  - `tests/test_dr_balance_metrics.py`
  - `tests/test_tune_causal_default_objective.py`
- Tests updated:
  - `tests/test_tune_causal_validation.py`
  - `tests/test_tune_objective_selection.py`
  - `tests/test_tune_dr_smoke.py`
  - `tests/test_tune_drdid_smoke.py`
  - `tests/test_api_surface.py`
  - `tests/test_runconfig_validation.py`
  - `tests/test_tuning_internal.py`
  - `tests/test_examples_run_demo_tune.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Balance-priority causal objectives are implemented for DR and DR-DiD.
  - Reason:
    - Aligns tuning behavior with causal balance quality requirements.
  - Impact area:
    - Causal tuning robustness / Operability

- Decision: confirmed
  - Policy:
    - Causal default objectives are now balance-priority.
  - Reason:
    - Improves default safety while retaining explicit access to legacy objectives.
  - Impact area:
    - Default behavior / Compatibility

- Decision: confirmed
  - Policy:
    - Weighted SMD threshold is configurable via `tuning.causal_balance_threshold`.
  - Reason:
    - Supports policy-specific balance strictness without API expansion.
  - Impact area:
    - Config governance / Diagnostics transparency

### 2026-02-12 (Session planning: phase23-drdid-binary-riskdiff-mvp)
**Context**
- Extend DR-DiD from regression-only to binary outcome while keeping stable API signatures unchanged.
- Add minimum diagnostics for overlap and covariate balance.

**Decisions**
- Decision: provisional
  - Policy:
    - `causal.method='dr_did'` supports `task.type='binary'` in addition to regression.
  - Reason:
    - Close the remaining P1 causal capability gap without adding new public APIs.
  - Impact area:
    - Causal runtime capability / Compatibility

- Decision: provisional
  - Policy:
    - Binary DR-DiD effect is interpreted as Risk Difference ATT.
  - Reason:
    - DR-DiD pseudo outcome targets treated-group probability difference in 2-period setup.
  - Impact area:
    - Causal interpretation contract / Documentation

- Decision: provisional
  - Policy:
    - Add minimum diagnostics: `overlap_metric`, `smd_max_unweighted`, `smd_max_weighted`.
  - Reason:
    - Provide baseline balance/overlap observability for operational review.
  - Impact area:
    - Diagnostics / Operability

### 2026-02-12 (Session/PR: phase23-drdid-binary-riskdiff-mvp)
**Context**
- Implement DR-DiD binary support and minimum diagnostics in a non-breaking way.
- Preserve existing regression DR-DiD behavior and all stable API signatures.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - `causal.method='dr_did'` now allows `task.type='regression'|'binary'`.
  - Updated `src/veldra/api/runner.py`:
    - `estimate_dr` DR-DiD gate expanded to regression/binary.
    - `CausalResult.metadata` now includes `outcome_scale` and `binary_outcome`.
  - Updated `src/veldra/causal/dr_did.py`:
    - binary outcome validation (0/1 strict) for DR-DiD.
    - DR-DiD internal DR execution fixed to regression task for pseudo outcomes.
    - added diagnostics:
      - `smd_max_unweighted`
      - `smd_max_weighted` (ATT weighting)
      - `overlap_metric` (existing, retained)
    - propagated diagnostic fields to metrics and summary outputs.
- Tests added:
  - `tests/test_drdid_binary_smoke_panel.py`
  - `tests/test_drdid_binary_smoke_repeated_cs.py`
  - `tests/test_drdid_binary_validation.py`
  - `tests/test_drdid_binary_metrics_contract.py`
- Tests updated:
  - `tests/test_runconfig_validation.py`
  - `tests/test_drdid_validation.py`
  - `tests/test_drdid_smoke_panel.py`
  - `tests/test_drdid_smoke_repeated_cs.py`
  - `tests/test_drdid_outputs.py`
  - `tests/test_drdid_internal.py`
  - `tests/test_api_surface.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - DR-DiD now supports both regression and binary outcomes for panel/repeated-CS designs.
  - Reason:
    - Completes the intended MVP scope while preserving API compatibility.
  - Impact area:
    - Causal runtime capability / Backward compatibility

- Decision: confirmed
  - Policy:
    - Binary DR-DiD is surfaced as Risk Difference ATT, with explicit metadata flags.
  - Reason:
    - Makes interpretation unambiguous for downstream consumers.
  - Impact area:
    - Analysis contract / Documentation clarity

- Decision: confirmed
  - Policy:
    - Minimum overlap and balance diagnostics are part of DR-DiD metric output.
  - Reason:
    - Improves practical trust and debugging for causal runs.
  - Impact area:
    - Operability / Quality assurance
- Tune now supports:
  - objective selection
  - resume continuation
  - per-trial persisted progress
  - configurable logging level
  - unified tune demo script with CLI overrides

**Risks / Notes**
- Progress persistence writes per trial; very high trial counts can increase artifact I/O.
- For non-resume runs with existing study name, explicit conflict error is returned by design.

**Open Questions**
- [ ] Should pruning strategy (e.g., median pruner) be introduced in a later phase?
- [ ] Should tune results have a dedicated load API (Artifact-like) beyond file outputs?

### 2026-02-10 (Session planning: phase9-frontier-fit-predict-evaluate-mvp)
**Context**
- Implement the next runtime capability for `task.type="frontier"` with minimal MVP scope.
- Preserve stable API signatures and avoid regressions for existing tasks.

**Decisions**
- Decision: provisional
  - Policy:
    - Frontier runtime MVP scope is `fit/predict/evaluate` only.
  - Reason:
    - Delivers practical runtime coverage without coupling to larger simulate/export work.
  - Impact area:
    - API behavior / Modeling / Artifact contract

- Decision: provisional
  - Policy:
    - Frontier default quantile alpha is `0.90`.
  - Reason:
    - Provides a deterministic default while keeping explicit override support.
  - Impact area:
    - Config / Modeling / Evaluation

- Decision: provisional
  - Policy:
    - Frontier prediction output contract is DataFrame with `frontier_pred`, and optional `u_hat`
      when labeled input is provided.
  - Reason:
    - Keeps unlabeled inference simple while supporting immediate inefficiency inspection on labeled
      data.
  - Impact area:
    - API contract / Operability

### 2026-02-10 (Session/PR: phase9-frontier-fit-predict-evaluate-mvp)
**Context**
- Implement frontier runtime as the next minimal production path after tune expansion.
- Keep stable API signatures unchanged and preserve existing task behavior.

**Changes**
- Code changes:
  - Added `src/veldra/modeling/frontier.py`:
    - `train_frontier_with_cv` with quantile objective and CV evaluation
    - frontier metrics: `pinball`, `mae`, `mean_u_hat`, `coverage`
  - Updated `src/veldra/config/models.py`:
    - added `FrontierConfig` (`alpha` default `0.90`)
    - added frontier validation rules (`alpha` bounds, split restrictions, non-frontier guard)
  - Updated `src/veldra/modeling/__init__.py` exports for frontier trainer.
  - Updated `src/veldra/api/runner.py`:
    - `fit/predict/evaluate` now support `task.type="frontier"`
    - evaluate returns frontier metrics and metadata (`frontier_alpha`)
  - Updated `src/veldra/api/artifact.py`:
    - frontier prediction path with `frontier_pred` and optional `u_hat`
  - Updated `examples/common.py` with `DEFAULT_FRONTIER_DATA_PATH`.
  - Added examples:
    - `examples/prepare_demo_data_frontier.py`
    - `examples/run_demo_frontier.py`
    - `examples/evaluate_demo_frontier_artifact.py`
- Tests added:
  - `tests/test_frontier_fit_smoke.py`
  - `tests/test_frontier_predict_contract.py`
  - `tests/test_frontier_evaluate_metrics.py`
  - `tests/test_frontier_artifact_roundtrip.py`
  - `tests/test_frontier_config_validation.py`
  - `tests/test_examples_prepare_demo_data_frontier.py`
  - `tests/test_examples_run_demo_frontier.py`
  - `tests/test_examples_evaluate_demo_frontier_artifact.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_runner_additional.py`
  - `tests/test_artifact_additional.py`
  - `tests/test_runconfig_validation.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Frontier quantile default alpha is `0.90`.
  - Reason:
    - Ensures deterministic behavior while preserving explicit override capability.
  - Impact area:
    - Config / Modeling / Evaluation

- Decision: confirmed
  - Policy:
    - Frontier MVP scope is `fit/predict/evaluate` only.
  - Reason:
    - Delivers runtime value now without coupling to simulate/export backlog.
  - Impact area:
    - API behavior / Delivery risk

- Decision: confirmed
  - Policy:
    - Frontier prediction contract is `frontier_pred` (+ optional `u_hat` when target present).
  - Reason:
    - Supports both unlabeled inference and labeled inefficiency inspection without API branching.
  - Impact area:
    - API contract / Operability

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed with frontier tests included.
- `uv run coverage run -m pytest -q` + `uv run coverage report -m` : passed.

**Risks / Notes**
- `u_hat` output on frontier prediction is available only when target column is present in input.
- `tune(frontier)`, `simulate`, and `export` remain intentionally unimplemented.

**Open Questions**
- [ ] Should frontier `tune` objective support only pinball initially, or include coverage-constrained
      variants from the first release?
- [ ] Should frontier prediction always return `u_hat` when target is absent by allowing explicit
      target argument, or keep current implicit contract?

### 2026-02-10 (Session/PR: phase10-simulate-mvp-scenario-dsl)
**Context**
- Implement `simulate` MVP as the next runtime feature while keeping stable API signatures unchanged.
- Keep delivery non-intrusive: no behavior change in existing `fit/predict/evaluate/tune`.

**Changes**
- Code changes:
  - Added `src/veldra/simulate/engine.py`:
    - scenario normalization (`dict` / `list[dict]`)
    - action application (`set/add/mul/clip`)
    - task-specific simulation output builder
  - Updated `src/veldra/simulate/__init__.py` exports.
  - Updated `src/veldra/api/runner.py`:
    - implemented `simulate(artifact, data, scenarios)` for regression/binary/multiclass/frontier
    - added structured completion log (`simulate completed`)
  - Updated `src/veldra/api/artifact.py`:
    - implemented `Artifact.simulate(df, scenario)` single-scenario shortcut
  - Added `examples/run_demo_simulate.py`.
- Tests added:
  - `tests/test_simulate_engine_actions.py`
  - `tests/test_simulate_runner_regression.py`
  - `tests/test_simulate_runner_binary.py`
  - `tests/test_simulate_runner_multiclass.py`
  - `tests/test_simulate_runner_frontier.py`
  - `tests/test_examples_run_demo_simulate.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_runner_additional.py`
  - `tests/test_artifact_additional.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - `simulate` MVP is delivered for all implemented task runtimes
      (regression/binary/multiclass/frontier).
  - Reason:
    - Closes major runtime gap with minimal API surface change.
  - Impact area:
    - API behavior / Scenario runtime

- Decision: confirmed
  - Policy:
    - Scenario DSL action set is fixed to `set/add/mul/clip` for MVP.
  - Reason:
    - Covers common scenario operations while minimizing complexity and risk.
  - Impact area:
    - Validation / Operability

- Decision: confirmed
  - Policy:
    - Simulation output contract is long-form with shared keys
      (`row_id`, `scenario`, `task_type`) plus task-specific comparison columns.
  - Reason:
    - Keeps downstream processing consistent across tasks.
  - Impact area:
    - API contract / Analytics usability

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (`150 passed`).

**Risks / Notes**
- Simulation operations currently target numeric columns only.
- `export` and `tune(frontier)` remain intentionally unimplemented.

**Open Questions**
- [ ] Should Phase 11 `export` MVP start with python-only inference package, or include ONNX in first release?
- [ ] Should `tune(frontier)` start with pinball-only objective, or include coverage-constrained variants?

### 2026-02-10 (Session planning: phase10-simulate-mvp)
**Context**
- Implement `simulate` MVP as the next runtime capability after frontier and tuning expansions.
- Preserve stable API signatures and keep existing task behavior unchanged.
- Status:
  - Superseded by `Session/PR: phase10-simulate-mvp-scenario-dsl` with `Decision: confirmed`.

**Decisions**
- Decision: provisional
  - Policy:
    - Phase 10 scope is `simulate` only, using one PR with non-intrusive changes.
  - Reason:
    - Close the largest remaining runtime gap while isolating risk from `export` and `tune(frontier)`.
  - Impact area:
    - API behavior / Scenario runtime / Examples

- Decision: provisional
  - Policy:
    - Scenario DSL minimal operations are `set/add/mul/clip`.
  - Reason:
    - Provides practical simulation controls without introducing search/optimization complexity.
  - Impact area:
    - Config/runtime contract / Validation

- Decision: provisional
  - Policy:
    - `SimulationResult.data` is long-form with shared keys (`row_id`, `scenario`, `task_type`) and
      task-specific comparison columns.
  - Reason:
    - Keeps downstream analysis format stable while supporting all implemented task types.
  - Impact area:
    - API contract / Operability

### 2026-02-10 (Session planning: phase11-export-mvp-python-onnx-optional)
**Context**
- Implement `export` MVP as the next runtime capability after `simulate`.
- Keep existing runtime behavior stable and add optional ONNX support without making it mandatory.

**Decisions**
- Decision: provisional
  - Policy:
    - `runner.export` supports `python` and `onnx` formats with unchanged signature.
  - Reason:
    - Closes remaining stable API gap while preserving compatibility.
  - Impact area:
    - API behavior / Artifact distribution

- Decision: provisional
  - Policy:
    - ONNX export is optional-dependency based; missing dependency raises explicit validation error.
  - Reason:
    - Avoids forcing heavy dependencies while keeping ONNX path available.
  - Impact area:
    - Packaging / Operability

- Decision: provisional
  - Policy:
    - Python export is always available and task-agnostic (regression/binary/multiclass/frontier).
  - Reason:
    - Guarantees baseline export usability in constrained environments.
  - Impact area:
    - User experience / Runtime reliability

### 2026-02-10 (Session/PR: phase11-export-mvp-python-onnx-optional)
**Context**
- Implement `export` MVP to close the remaining stable API runtime gap.
- Keep `fit/predict/evaluate/tune/simulate` behavior unchanged.

**Changes**
- Code changes:
  - Added `src/veldra/artifact/exporter.py`:
    - `export_python_package(artifact, out_dir)`
    - `export_onnx_model(artifact, out_dir)`
    - optional ONNX dependency loading with explicit error guidance
  - Updated `src/veldra/api/runner.py`:
    - implemented `export(artifact, format)`
    - supported formats: `python`, `onnx`
    - structured export completion log
  - Added `examples/run_demo_export.py`
  - Updated `pyproject.toml`:
    - added optional dependency group `export-onnx`
- Tests added:
  - `tests/test_export_python_mvp.py`
  - `tests/test_export_onnx_optional.py`
  - `tests/test_export_runner_contract.py`
  - `tests/test_examples_run_demo_export.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_runner_additional.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - `export` supports `python` and `onnx` with unchanged API signature.
  - Reason:
    - Completes the stable API runtime surface with minimal compatibility risk.
  - Impact area:
    - API behavior / Distribution workflow

- Decision: confirmed
  - Policy:
    - ONNX export remains optional-dependency based and non-blocking for python export.
  - Reason:
    - Keeps default setup lightweight while enabling ONNX where required.
  - Impact area:
    - Packaging / Operability

- Decision: confirmed
  - Policy:
    - ONNX export for `frontier` is explicitly unsupported in MVP (`VeldraNotImplementedError`).
  - Reason:
    - Avoids silent behavior ambiguity and keeps conversion contract explicit.
  - Impact area:
    - API contract / Reliability

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed.

**Risks / Notes**
- ONNX conversion success depends on optional toolchain availability and converter compatibility.
- `tune(frontier)` remains intentionally unimplemented.

**Open Questions**
- [ ] Should Phase 12 prioritize `tune(frontier)` or frontier ONNX export support first?

### 2026-02-10 (Session planning: phase12-tune-frontier-mvp)
**Context**
- Implement `tune(frontier)` as the next runtime capability after Phase 11 export MVP.
- Keep existing behavior stable for `fit/predict/evaluate/simulate/export`.
- Status:
  - Superseded by `Session/PR: phase12-tune-frontier-mvp` with `Decision: confirmed`.

**Decisions**
- Decision: provisional
  - Policy:
    - Enable `runner.tune` for `task.type="frontier"` in Phase 12 MVP.
  - Reason:
    - Closes the final stable API runtime gap without broad interface changes.
  - Impact area:
    - API behavior / Tuning runtime

- Decision: provisional
  - Policy:
    - Frontier tuning objective is fixed to `pinball` in MVP.
  - Reason:
    - Minimizes risk and keeps optimization behavior aligned with frontier training metric.
  - Impact area:
    - Objective contract / Reproducibility

- Decision: provisional
  - Policy:
    - Reuse existing tuning infrastructure (`search_space`, SQLite resume, logging, artifacts).
  - Reason:
    - Non-intrusive implementation with lower maintenance cost and consistent UX.
  - Impact area:
    - Operability / Compatibility

### 2026-02-10 (Session/PR: phase12-tune-frontier-mvp)
**Context**
- Implement `tune(frontier)` as the next runtime capability after Phase 11 export MVP.
- Keep existing behavior stable for `fit/predict/evaluate/simulate/export`.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - added frontier tuning objective contract (`pinball`)
    - added frontier tuning default objective (`pinball`)
  - Updated `src/veldra/modeling/tuning.py`:
    - frontier branch in `_score_for_task` using `train_frontier_with_cv`
    - added `pinball -> minimize` direction mapping
    - extended `run_tuning` task support to include frontier
  - Updated `src/veldra/api/runner.py`:
    - removed frontier `NotImplemented` branch in `tune()`
    - enabled `task.type='frontier'` for tune runtime path
  - Updated `examples/run_demo_tune.py`:
    - added `--task frontier`
    - default frontier data path support (`examples/data/frontier_demo.csv`)
    - uses `kfold` for frontier tune split
- Tests added:
  - `tests/test_tune_smoke_frontier.py`
  - `tests/test_tune_frontier_validation.py`
  - `tests/test_tune_resume_frontier.py`
- Tests updated:
  - `tests/test_examples_run_demo_tune.py`
  - `tests/test_tune_validation.py`
  - `tests/test_tune_objective_selection.py`
  - `tests/test_tuning_internal.py`
  - `tests/test_runconfig_validation.py`
  - `tests/test_api_surface.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - `runner.tune` supports `task.type='frontier'` in Phase 12 MVP.
  - Reason:
    - Completes stable API runtime availability for tune with minimal interface change.
  - Impact area:
    - API behavior / Tuning runtime

- Decision: confirmed
  - Policy:
    - Frontier tuning objective is fixed to `pinball` in MVP.
  - Reason:
    - Aligns tune objective with frontier training metric while minimizing risk.
  - Impact area:
    - Objective contract / Reproducibility

- Decision: confirmed
  - Policy:
    - Frontier tuning reuses existing tune infrastructure (`search_space`, resume, logging, artifacts).
  - Reason:
    - Keeps implementation non-intrusive and operationally consistent with existing tasks.
  - Impact area:
    - Operability / Compatibility

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (`178 passed, 1 skipped`).

**Risks / Notes**
- Frontier tuning currently supports only `pinball`; coverage-constrained objectives are not in MVP.
- Frontier ONNX export remains intentionally unsupported in export MVP.

**Open Questions**
- [ ] Should Phase 13 prioritize frontier ONNX export support or frontier tuning objective expansion
      (coverage-constrained variants)?

### 2026-02-10 (Session planning: phase13-frontier-onnx-export-mvp)
**Context**
- Implement frontier ONNX export as the final runtime parity gap in export MVP.
- Keep existing API signatures and runtime behaviors stable.
- Status:
  - Superseded by `Session/PR: phase13-frontier-onnx-export-mvp` with `Decision: confirmed`.

**Decisions**
- Decision: provisional
  - Policy:
    - Enable `export(format="onnx")` for `task.type="frontier"` in Phase 13 MVP.
  - Reason:
    - Completes export task coverage without requiring API changes.
  - Impact area:
    - API behavior / Distribution workflow

- Decision: provisional
  - Policy:
    - Keep ONNX as optional dependency and surface explicit guidance on missing dependencies.
  - Reason:
    - Maintains lightweight default environment while preserving ONNX portability path.
  - Impact area:
    - Packaging / Operability

- Decision: provisional
  - Policy:
    - ONNX converter/runtime failures are handled as explicit validation errors with actionable guidance.
  - Reason:
    - Avoids silent failures and clarifies converter compatibility limits for frontier models.
  - Impact area:
    - Reliability / Troubleshooting

### 2026-02-10 (Session/PR: phase13-frontier-onnx-export-mvp)
**Context**
- Implement frontier ONNX export support in MVP while preserving existing API signatures.
- Keep python export always available and ONNX path optional.

**Changes**
- Code changes:
  - Updated `src/veldra/artifact/exporter.py`:
    - removed frontier-specific `NotImplemented` branch in ONNX export
    - added conversion failure handling with actionable `VeldraValidationError`
    - added metadata field `frontier_alpha` for frontier ONNX export
  - Updated `pyproject.toml`:
    - added explicit optional dependency `onnxconverter-common==1.16.0` to `export-onnx`
  - Updated `uv.lock`:
    - resolved and locked `onnxconverter-common`
- Tests added/updated:
  - Updated `tests/test_exporter_internal.py`:
    - frontier ONNX mocked success path
    - converter failure validation path
  - Updated `tests/test_export_onnx_optional.py`:
    - removed frontier-not-supported expectation
    - added frontier ONNX generation path under optional dependency condition
  - Updated `tests/test_export_runner_contract.py`:
    - runner frontier ONNX export contract via mocked exporter
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - `export(format="onnx")` supports frontier artifacts in MVP.
  - Reason:
    - Completes export task coverage across implemented tasks.
  - Impact area:
    - API behavior / Distribution workflow

- Decision: confirmed
  - Policy:
    - ONNX remains optional dependency based (`export-onnx` extra).
  - Reason:
    - Preserves lightweight default installs while enabling ONNX usage where needed.
  - Impact area:
    - Packaging / Operability

- Decision: confirmed
  - Policy:
    - Converter/runtime failures are surfaced as explicit, guided `VeldraValidationError`.
  - Reason:
    - Improves troubleshooting and avoids silent or ambiguous conversion failures.
  - Impact area:
    - Reliability / User diagnostics

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (`184 passed, 2 skipped`).

**Risks / Notes**
- Frontier ONNX conversion still depends on converter compatibility; explicit failure guidance is retained.
- ONNX graph optimization/quantization remains out of scope.

**Open Questions**
- [ ] Should Phase 14 prioritize ONNX optimization (quantization/graph optimization) or export package
      validation tooling (smoke-runner generation)?

### 2026-02-10 (Session planning: phase14-export-validation-tooling-mvp)
**Context**
- Close remaining operational-quality gap after runtime parity by validating export outputs automatically.
- Keep stable API signatures unchanged and avoid coupling with ONNX optimization.

**Decisions**
- Decision: provisional
  - Policy:
    - Phase 14 prioritizes export validation tooling over ONNX optimization.
  - Reason:
    - Improves reliability immediately with minimal compatibility risk.
  - Impact area:
    - Export operability / QA

- Decision: provisional
  - Policy:
    - `ExportResult.metadata` is extended with validation status/report fields only.
  - Reason:
    - Preserves stable API signatures while exposing actionable validation outcomes.
  - Impact area:
    - API behavior / Compatibility

### 2026-02-10 (Session/PR: phase14-export-validation-tooling-mvp)
**Context**
- Implement export validation reports and runner-level validation metadata/logging.
- Keep `fit/predict/evaluate/tune/simulate` behavior unchanged.

**Changes**
- Code changes:
  - Updated `src/veldra/artifact/exporter.py`:
    - added `_validate_python_export(...)`
    - added `_validate_onnx_export(...)`
    - added `validation_report.json` writer
  - Updated `src/veldra/api/runner.py`:
    - run export validation immediately after export
    - emit `export validation completed` structured event
    - include validation metadata in `ExportResult.metadata`
  - Updated `examples/run_demo_export.py`:
    - print validation status and report path
- Tests added:
  - `tests/test_export_validation_python.py`
  - `tests/test_export_validation_onnx.py`
  - `tests/test_export_runner_validation_contract.py`
- Tests updated:
  - `tests/test_export_runner_contract.py`
  - `tests/test_examples_run_demo_export.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Export writes machine-readable validation reports for both `python` and `onnx` outputs.
  - Reason:
    - Ensures exported artifacts are operationally verifiable, not only structurally present.
  - Impact area:
    - Export reliability / QA

- Decision: confirmed
  - Policy:
    - Validation outcomes are exposed via `ExportResult.metadata` (`validation_passed`,
      `validation_report`, `validation_mode`).
  - Reason:
    - Provides non-breaking visibility to downstream tools and operators.
  - Impact area:
    - API behavior / Operability

**Results**
- `uv run pytest -q tests/test_export_validation_python.py tests/test_export_validation_onnx.py`
  `tests/test_export_runner_validation_contract.py tests/test_export_runner_contract.py`
  `tests/test_examples_run_demo_export.py` : passed (`11 passed`).
- Full regression run planned in this branch before merge.

**Risks / Notes**
- ONNX validation runtime checks depend on optional dependencies (`export-onnx`).
- ONNX graph optimization/quantization remains out of scope for this phase.

**Open Questions**
- [ ] Should Phase 15 prioritize ONNX quantization first or graph optimization first for better
      default trade-off?

### 2026-02-10 (Session planning: phase15-onnx-quantization-mvp)
**Context**
- Add optional ONNX optimization on top of existing export validation foundation.
- Preserve stable API signatures and default export behavior.

**Decisions**
- Decision: provisional
  - Policy:
    - Phase 15 uses quantization-first (`dynamic_quant`) and keeps graph optimization out of scope.
  - Reason:
    - Provides immediate practical optimization with low compatibility risk.
  - Impact area:
    - Export performance / Operability

- Decision: provisional
  - Policy:
    - ONNX optimization is opt-in via `export.onnx_optimization.enabled`.
  - Reason:
    - Guarantees non-invasive defaults for existing users.
  - Impact area:
    - Backward compatibility / Runtime stability

### 2026-02-10 (Session/PR: phase15-onnx-quantization-mvp)
**Context**
- Implement optional ONNX dynamic quantization without changing stable API signatures.
- Keep ONNX optional dependency policy and explicit failure guidance.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - added `OnnxOptimizationConfig`
    - added `export.onnx_optimization` contract and validation
  - Updated `src/veldra/artifact/exporter.py`:
    - added `_optimize_onnx_model(...)` for dynamic quantization
    - added optional `model.optimized.onnx` generation
    - extended ONNX validation report payload with optimization metadata
  - Updated `src/veldra/api/runner.py`:
    - extended `ExportResult.metadata` with optimization fields
    - added structured event `onnx optimization completed`
- Tests added:
  - `tests/test_export_onnx_optimization.py`
  - `tests/test_export_onnx_optimization_errors.py`
- Tests updated:
  - `tests/test_runconfig_validation.py`
  - `tests/test_export_runner_contract.py`
  - `tests/test_export_runner_validation_contract.py`
  - `tests/test_export_validation_onnx.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - ONNX dynamic quantization is opt-in and disabled by default.
  - Reason:
    - Prevents behavior changes in existing export workflows.
  - Impact area:
    - Backward compatibility / Runtime stability

- Decision: confirmed
  - Policy:
    - Export metadata and validation report include optimization result and size comparison.
  - Reason:
    - Makes optimization effects auditable in automation and operations.
  - Impact area:
    - Observability / Export QA

### 2026-02-10 (Session/PR: phase16-timeseries-split-advanced-mvp)
**Context**
- Prioritize leakage-resistant and reproducible time-series evaluation.
- Keep default behavior and stable API signatures unchanged.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - extended `SplitConfig` with `timeseries_mode`, `test_size`, `gap`, `embargo`, `train_size`
    - added cross-field validation for timeseries-only settings and blocked-mode contract
  - Updated `src/veldra/split/time_series.py`:
    - added `mode='expanding'|'blocked'`
    - added `embargo` support
    - added explicit `train_size` contract for blocked mode
    - added future-train exclusion for prior test+embargo windows
  - Updated timeseries split wiring in:
    - `src/veldra/modeling/regression.py`
    - `src/veldra/modeling/binary.py`
    - `src/veldra/modeling/multiclass.py`
    - `src/veldra/modeling/frontier.py`
- Tests updated:
  - `tests/test_splitter_contract.py`
  - `tests/test_time_series_splitter_additional.py`
  - `tests/test_runconfig_validation.py`
  - `tests/test_regression_internal.py`
  - `tests/test_binary_internal.py`
  - `tests/test_multiclass_internal.py`
  - `tests/test_frontier_internal.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Advanced timeseries controls are opt-in and only active for `split.type='timeseries'`.
  - Reason:
    - Improves leakage resistance while preserving existing non-timeseries behavior.
  - Impact area:
    - Split reliability / Backward compatibility

- Decision: confirmed
  - Policy:
    - `blocked` mode requires explicit `train_size`.
  - Reason:
    - Makes blocked-window semantics explicit and reproducible.
  - Impact area:
    - Config contract / Operability

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (`217 passed`).

### 2026-02-10 (Session planning: phase16-timeseries-split-advanced-mvp)
**Context**
- Prioritize time-series split quality after core runtime parity and export/onxx enhancements.
- Improve leakage resistance while keeping existing defaults unchanged.

**Decisions**
- Decision: provisional
  - Policy:
    - Introduce advanced timeseries controls with non-invasive defaults.
  - Reason:
    - Strengthens evaluation reliability across all task trainers without API signature change.
  - Impact area:
    - Data splitting / Reproducibility

- Decision: provisional
  - Policy:
    - `timeseries_mode='blocked'` uses explicit `train_size` contract.
  - Reason:
    - Avoids ambiguous blocked-window behavior and keeps configuration explicit.
  - Impact area:
    - Config contract / Operability

### 2026-02-10 (Session planning: phase17-frontier-tune-coverage-objective-mvp)
**Context**
- Extend frontier tuning objective depth while preserving non-invasive defaults.
- Keep stable API signatures unchanged.

**Decisions**
- Decision: provisional
  - Policy:
    - Frontier tuning keeps `pinball` as default objective and adds opt-in
      `pinball_coverage_penalty`.
  - Reason:
    - Improves operational alignment to target coverage without changing existing workflows.
  - Impact area:
    - Frontier tuning quality / Backward compatibility

- Decision: provisional
  - Policy:
    - Coverage-aware objective uses:
      `pinball + penalty_weight * max(0, abs(coverage - coverage_target) - coverage_tolerance)`.
    - `coverage_target` defaults to `frontier.alpha` when not set.
  - Reason:
    - Keeps configuration concise while making constraint strength explicit.
  - Impact area:
    - Objective design / Operability

### 2026-02-10 (Session/PR: phase17-frontier-tune-coverage-objective-mvp)
**Context**
- Extend frontier tuning objective depth while preserving backward compatibility.
- Keep default frontier tuning objective as `pinball`.

**Changes**
- Code changes:
  - Updated `src/veldra/config/models.py`:
    - frontier objective set expanded to include `pinball_coverage_penalty`
    - added tuning fields:
      - `coverage_target`
      - `coverage_tolerance`
      - `penalty_weight`
    - added frontier/non-frontier validation rules for the new fields
  - Updated `src/veldra/modeling/tuning.py`:
    - added coverage-aware objective calculation:
      `pinball + penalty_weight * max(0, abs(coverage - coverage_target) - coverage_tolerance)`
    - added trial user-attrs persistence for objective components
    - enriched `study_summary.json` / `trials.parquet` with objective component details
  - Updated `src/veldra/api/runner.py`:
    - added frontier coverage objective metadata to `TuneResult.metadata`
    - extended tune logging context with coverage tuning fields
  - Updated `examples/run_demo_tune.py`:
    - added CLI options:
      - `--coverage-target`
      - `--coverage-tolerance`
      - `--penalty-weight`
- Tests added:
  - `tests/test_tune_frontier_objective_selection.py`
  - `tests/test_tune_frontier_coverage_penalty.py`
  - `tests/test_tune_frontier_validation_additional.py`
- Tests updated:
  - `tests/test_tune_smoke_frontier.py`
  - `tests/test_examples_run_demo_tune.py`
  - `tests/test_tune_artifacts.py`
  - `tests/test_tune_objective_selection.py`
  - `tests/test_runconfig_validation.py`
  - `tests/test_tuning_internal.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Frontier tuning keeps `pinball` as default and adds opt-in
      `pinball_coverage_penalty`.
  - Reason:
    - Preserves existing workflows while enabling stronger operational alignment.
  - Impact area:
    - Backward compatibility / Frontier tuning quality

- Decision: confirmed
  - Policy:
    - `coverage_target` defaults to `frontier.alpha` when omitted.
    - Coverage penalty uses `coverage_tolerance` and `penalty_weight` as explicit controls.
  - Reason:
    - Keeps defaults simple but allows deterministic objective shaping when needed.
  - Impact area:
    - Config contract / Operability

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (`230 passed`).

### 2026-02-10 (History hygiene: superseded planning tails)
**Context**
- Some planning entries remained at the end of the file after corresponding implementation entries
  were already confirmed.

**Changes**
- Added superseded mapping notes (append-only; no historical deletion):
  - `Session planning: phase16-timeseries-split-advanced-mvp`
    - superseded by `Session/PR: phase16-timeseries-split-advanced-mvp` (`Decision: confirmed`)
  - `Session planning: phase17-frontier-tune-coverage-objective-mvp`
    - superseded by `Session/PR: phase17-frontier-tune-coverage-objective-mvp` (`Decision: confirmed`)

### 2026-02-10 (Session planning: phase18-evaluate-config-input-mvp)
**Context**
- Close the remaining `RunConfig` entry gap by implementing `evaluate(config, data)`.
- Keep `evaluate(artifact, data)` behavior unchanged and non-invasive.

**Decisions**
- Decision: provisional
  - Policy:
    - `evaluate(artifact_or_config, data)` accepts `RunConfig | dict` in addition to `Artifact`.
    - Config mode runs ephemeral training in memory and does not persist artifacts.
  - Reason:
    - Aligns runtime behavior with the "RunConfig as common entrypoint" design principle.
  - Impact area:
    - API behavior / Operability / Compatibility

- Decision: provisional
  - Policy:
    - Config-mode evaluation metadata includes:
      - `evaluation_mode`
      - `train_source_path`
      - `ephemeral_run`
  - Reason:
    - Keeps execution context explicit for debugging and audit trails without changing signatures.
  - Impact area:
    - Observability / Diagnostics

### 2026-02-10 (Session/PR: phase18-evaluate-config-input-mvp)
**Context**
- Implement `evaluate(config, data)` for all task types while preserving stable API signatures.
- Keep artifact-path evaluation behavior unchanged.

**Changes**
- Code changes:
  - Updated `src/veldra/api/runner.py`:
    - added config-input branch for `evaluate(artifact_or_config, data)`.
    - added ephemeral model build path via existing task trainers.
    - refactored shared evaluation logic into an internal helper.
    - added metadata fields:
      - `evaluation_mode`
      - `train_source_path`
      - `ephemeral_run`
    - extended `evaluate completed` structured log context with config/ephemeral markers.
- Tests added:
  - `tests/test_evaluate_config_path.py`
  - `tests/test_evaluate_config_validation.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_binary_evaluate_metrics.py`
  - `tests/test_multiclass_evaluate_metrics.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - `evaluate` accepts `Artifact` and `RunConfig | dict`; config mode trains ephemerally in memory.
  - Reason:
    - Restores alignment with the common RunConfig entrypoint principle without introducing new APIs.
  - Impact area:
    - API behavior / Operability / Compatibility

- Decision: confirmed
  - Policy:
    - Config-mode metadata includes `evaluation_mode`, `train_source_path`, `ephemeral_run`.
  - Reason:
    - Makes runtime path and provenance explicit while preserving response shape.
  - Impact area:
    - Observability / Diagnostics

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed (`238 passed`).

### 2026-02-10 (Session/PR: phase19-causal-dr-mvp-att-calibrated)
**Context**
- Add DR causal estimation as an additive capability while keeping stable runtime APIs intact.
- Use ATT-first defaults and calibrated propensity workflow for practical operational use.

**Changes**
- Code changes:
  - Added `src/veldra/causal/dr.py` and `src/veldra/causal/__init__.py`:
    - DR estimation core (`ATT` default, `ATE` optional)
    - OOF nuisance estimation with optional cross-fitting
    - propensity calibration (`platt` default, `isotonic` optional)
    - summary + observation-level output payload
  - Updated `src/veldra/config/models.py`:
    - added `CausalConfig`
    - added optional `RunConfig.causal`
    - added causal validation (`propensity_clip`, treatment/target collision)
  - Updated `src/veldra/config/__init__.py` exports:
    - added `CausalConfig`
  - Updated `src/veldra/api/types.py`:
    - added `CausalResult`
  - Updated `src/veldra/api/runner.py`:
    - added `estimate_dr(config)` entrypoint
    - writes causal outputs under `artifacts/causal/<run_id>/`
    - logs `dr estimation completed` with structured payload
  - Updated `src/veldra/api/__init__.py`:
    - exported `estimate_dr` and `CausalResult`
  - Added `examples/generate_data_dr.py`:
    - synthetic confounded DR validation dataset generator
- Tests added:
  - `tests/test_dr_smoke.py`
  - `tests/test_dr_validation.py`
  - `tests/test_dr_propensity_calibration.py`
  - `tests/test_dr_formula.py`
  - `tests/test_dr_outputs.py`
  - `tests/test_examples_generate_data_dr.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_runconfig_validation.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - DR MVP default estimand is `ATT`; `ATE` remains opt-in.
  - Reason:
    - ATT is the most common production estimand for treatment-on-treated analysis.
  - Impact area:
    - Causal inference behavior / Backward compatibility

- Decision: confirmed
  - Policy:
    - DR uses calibrated propensity scores (`platt` default, `isotonic` optional).
  - Reason:
    - Improves probability reliability and stabilizes inverse propensity weighting terms.
  - Impact area:
    - Statistical robustness / Reproducibility

- Decision: confirmed
  - Policy:
    - DR-specific hyperparameter tuning is deferred to the next phase.
  - Reason:
    - Keeps Phase 19 focused and low-risk while delivering a complete DR runtime path.
  - Impact area:
    - Delivery scope / Risk control

### 2026-02-10 (Session planning: phase19-causal-dr-mvp-att-calibrated)
**Context**
- Add DR causal estimation while preserving stable API contracts.
- Prioritize ATT-first estimation and calibrated propensity workflow.

**Decisions**
- Decision: provisional
  - Policy:
    - Add RunConfig-driven DR with default `estimand=att`.
    - Support optional `estimand=ate` without changing stable signatures.
  - Reason:
    - ATT is the most common operational estimand for treatment-on-treated analysis.
  - Impact area:
    - Causal inference capability / Compatibility

- Decision: provisional
  - Policy:
    - DR uses calibrated propensity scores with OOF estimation path by default.
    - Default propensity calibration is `platt`.
  - Reason:
    - Improves probability reliability and stabilizes inverse propensity weighting behavior.
  - Impact area:
    - Statistical robustness / Reproducibility

- Decision: provisional
  - Policy:
    - DR-specific tuning is deferred to the next phase.
  - Reason:
    - Keep MVP focused and non-intrusive to existing tuning contracts.
  - Impact area:
    - Delivery scope / Risk control

### 2026-02-11 (Session planning: phase19.1-lalonde-dr-analysis-notebook)
**Context**
- Add a practical Lalonde DR notebook with scenario-driven interpretation.
- Keep DR runtime API unchanged and use `estimate_dr(config)` as the single entrypoint.

**Decisions**
- Decision: provisional
  - Policy:
    - Lalonde data ingestion is URL-based with local cache-first rerun behavior.
  - Reason:
    - Keeps repository lean while preserving notebook reproducibility after first fetch.
  - Impact area:
    - Notebook operability / Data access

- Decision: provisional
  - Policy:
    - Notebook explicitly uses ATT default and propensity calibration (`platt`) in config.
  - Reason:
    - Makes causal assumptions and defaults visible for analysts.
  - Impact area:
    - Causal analysis transparency / Reproducibility

### 2026-02-11 (Session/PR: phase19.1-lalonde-dr-analysis-notebook)
**Context**
- Add a scenario-driven Lalonde DR notebook on top of the existing Phase 19 causal runtime.
- Keep stable API signatures unchanged and focus on analysis workflow quality.

**Changes**
- Added notebook:
  - `notebooks/lalonde_dr_analysis_workflow.ipynb`
  - URL ingestion from Rdatasets Lalonde source with local cache reuse:
    - `examples/out/notebook_lalonde_dr/lalonde_raw.parquet`
  - Explicit DR config defaults in notebook:
    - `estimand='att'`
    - `propensity_calibration='platt'`
  - Diagnostics included:
    - Naive/IPW/DR comparison table + CI plot
    - propensity distribution plots (`e_raw`, `e_hat`)
    - balance diagnostics (SMD unweighted vs ATT-weighted)
  - Notebook summary output:
    - `examples/out/notebook_lalonde_dr/lalonde_analysis_summary.json`
- Added tests:
  - `tests/test_notebook_lalonde_structure.py`
  - `tests/test_notebook_lalonde_paths.py`
- Updated docs:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Lalonde notebook uses URL ingestion with cache-first reruns.
  - Reason:
    - Keeps repository lightweight while preserving reproducibility after first fetch.
  - Impact area:
    - Notebook operability / Data access

- Decision: confirmed
  - Policy:
    - Notebook explicitly shows ATT/platt defaults even though they are runtime defaults.
  - Reason:
    - Makes causal assumptions transparent to analysts and reviewers.
  - Impact area:
    - Causal analysis transparency / Reproducibility

**Results**
- `uv run --no-sync ruff check .` : passed.
- Notebook contract tests:
  - `tests/test_notebook_regression_structure.py`
  - `tests/test_notebook_regression_paths.py`
  - `tests/test_notebook_frontier_structure.py`
  - `tests/test_notebook_frontier_paths.py`
  - `tests/test_notebook_simulate_structure.py`
  - `tests/test_notebook_binary_tune_structure.py`
  - `tests/test_notebook_lalonde_structure.py`
  - `tests/test_notebook_lalonde_paths.py`
  - result: `15 passed`
- Full regression run:
  - `uv run --no-sync pytest -q`
  - result: `264 passed, 4 skipped, 1 warning`

### 2026-02-11 (History hygiene: superseded planning tails)
**Context**
- The Phase 19.1 planning entry now has an implementation counterpart in the same file.

**Changes**
- Added append-only superseded note:
  - `Session planning: phase19.1-lalonde-dr-analysis-notebook`
    - superseded by `Session/PR: phase19.1-lalonde-dr-analysis-notebook` (`Decision: confirmed`)

### 2026-02-11 (Session planning: phase20.1-lalonde-drdid-analysis-notebook)
**Context**
- Add a dedicated Lalonde DR-DiD notebook on top of the Phase 20 runtime.
- Keep single-period DR notebook and DR-DiD notebook as separate references.

**Decisions**
- Decision: provisional
  - Policy:
    - Notebook scenario uses panel DR-DiD with pre=`re75` and post=`re78`.
  - Reason:
    - Matches the intended policy question (earnings growth impact) and DR-DiD assumptions.
  - Impact area:
    - Causal analysis workflow / Reproducibility

- Decision: provisional
  - Policy:
    - Notebook uses URL ingestion with local cache-first reruns.
  - Reason:
    - Keeps repository lightweight while ensuring deterministic reruns after initial fetch.
  - Impact area:
    - Notebook operability / Data access

- Decision: provisional
  - Policy:
    - ATT and platt defaults are shown explicitly in notebook config.
  - Reason:
    - Makes assumptions visible to analysts and reviewers.
  - Impact area:
    - Causal analysis transparency

### 2026-02-11 (Session/PR: phase20.1-lalonde-drdid-analysis-notebook)
**Context**
- Add a dedicated Lalonde DR-DiD notebook to demonstrate the newly implemented DR-DiD runtime path.
- Keep the existing Lalonde DR notebook untouched for single-period reference.

**Changes**
- Added notebook:
  - `notebooks/lalonde_drdid_analysis_workflow.ipynb`
  - URL ingestion + cache-first behavior:
    - `examples/out/notebook_lalonde_drdid/lalonde_raw.parquet`
  - panel transformation cache:
    - `examples/out/notebook_lalonde_drdid/lalonde_panel.parquet`
  - explicit DR-DiD config:
    - `causal.method='dr_did'`
    - `causal.design='panel'`
    - `estimand='att'`
    - `propensity_calibration='platt'`
  - diagnostics:
    - Naive/IPW/DR/DR-DiD comparison
    - propensity distributions (`e_raw`, `e_hat`)
    - overlap summary
    - SMD balance diagnostics
  - notebook summary output:
    - `examples/out/notebook_lalonde_drdid/lalonde_drdid_summary.json`
- Added tests:
  - `tests/test_notebook_lalonde_drdid_structure.py`
  - `tests/test_notebook_lalonde_drdid_paths.py`
- Updated docs:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Lalonde DR-DiD notebook uses panel design with pre=`re75` and post=`re78`.
  - Reason:
    - Aligns the scenario with a growth-impact interpretation for ATT.
  - Impact area:
    - Causal analysis workflow / Reproducibility

- Decision: confirmed
  - Policy:
    - URL ingestion is used with local cache-first reruns.
  - Reason:
    - Preserves lightweight repository contents while enabling deterministic reruns.
  - Impact area:
    - Notebook operability / Data access

- Decision: confirmed
  - Policy:
    - ATT and platt defaults are explicitly shown in notebook config.
  - Reason:
    - Improves analyst transparency and reviewability.
  - Impact area:
    - Causal analysis transparency

### 2026-02-11 (Session planning: phase20-drdid-and-causal-tune-mvp)
**Context**
- Extend causal support from single-period DR to DR-DiD.
- Add tune support for causal workflows (`dr` and `dr_did`) without changing stable API signatures.

**Decisions**
- Decision: provisional
  - Policy:
    - Add `causal.method="dr_did"` MVP for 2-period data with `design="panel"| "repeated_cross_section"`.
  - Reason:
    - Closes the next major causal capability gap while keeping scope controlled.
  - Impact area:
    - Causal runtime capability / Compatibility

- Decision: provisional
  - Policy:
    - Add causal tuning objectives:
      - DR: `dr_std_error`, `dr_overlap_penalty`
      - DR-DiD: `drdid_std_error`, `drdid_overlap_penalty`
  - Reason:
    - Enables nuisance-quality optimization without introducing new public APIs.
  - Impact area:
    - Tuning capability / Operability

- Decision: provisional
  - Policy:
    - Preserve defaults: `estimand='att'`, `propensity_calibration='platt'`.
  - Reason:
    - Maintains current behavior and minimizes migration risk.
  - Impact area:
    - Backward compatibility / Reproducibility

### 2026-02-11 (Session/PR: phase20-drdid-and-causal-tune-mvp)
**Context**
- Implement DR-DiD estimation and causal tune objectives in a non-breaking way.
- Keep all existing stable API signatures unchanged.

**Changes**
- Code changes:
  - Added `src/veldra/causal/dr_did.py`:
    - DR-DiD estimation for 2-period panel and repeated cross-section designs.
  - Updated `src/veldra/causal/__init__.py`:
    - exported `run_dr_did_estimation`.
  - Updated `src/veldra/config/models.py`:
    - extended `CausalConfig` with `method/design/time_col/post_col/unit_id_col`.
    - added causal tune objective constraints and defaults.
    - added `tuning.causal_penalty_weight`.
  - Updated `src/veldra/api/runner.py`:
    - `estimate_dr` dispatch now supports `dr` and `dr_did`.
    - `CausalResult.metadata` includes design/time diagnostics fields.
    - tune metadata/logging includes causal context when causal config is set.
  - Updated `src/veldra/modeling/tuning.py`:
    - added causal scoring paths:
      - `dr_std_error`, `dr_overlap_penalty`
      - `drdid_std_error`, `drdid_overlap_penalty`
    - trial artifacts include causal objective components.
  - Added `examples/generate_data_drdid.py`:
    - synthetic DR-DiD validation datasets for panel and repeated CS.
- Tests added:
  - `tests/test_drdid_smoke_panel.py`
  - `tests/test_drdid_smoke_repeated_cs.py`
  - `tests/test_drdid_validation.py`
  - `tests/test_drdid_outputs.py`
  - `tests/test_tune_dr_smoke.py`
  - `tests/test_tune_drdid_smoke.py`
  - `tests/test_tune_causal_resume.py`
  - `tests/test_tune_causal_validation.py`
  - `tests/test_examples_generate_data_drdid.py`
- Tests updated:
  - `tests/test_api_surface.py`
  - `tests/test_runconfig_validation.py`
  - `tests/test_tune_objective_selection.py`
  - `tests/test_tuning_internal.py`
  - `tests/test_dr_validation.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - DR-DiD MVP supports exactly 2-period panel/repeated-CS in `task.type='regression'`.
  - Reason:
    - Delivers the needed capability with bounded complexity.
  - Impact area:
    - Causal runtime capability / Risk control

- Decision: confirmed
  - Policy:
    - Causal tuning is available for both DR and DR-DiD through objective selection.
  - Reason:
    - Reuses existing tune infrastructure and avoids API surface expansion.
  - Impact area:
    - Tuning capability / Maintainability

- Decision: confirmed
  - Policy:
    - Defaults remain `ATT` and `platt` calibration.
  - Reason:
    - Ensures non-intrusive migration and behavioral continuity.
  - Impact area:
    - Backward compatibility / Reproducibility

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed.

### 2026-02-11 (Session/PR: phase20.2-design-blueprint-reorg-and-gap-audit)
**Context**
- Reconcile `DESIGN_BLUEPRINT.md` with the current runtime implementation.
- Reorganize the blueprint into readable Japanese and clarify remaining gaps.
- Add a concrete GUI planning section (Dash MVP) without changing runtime behavior.

**Decisions**
- Decision: provisional
  - Policy:
    - Rebuild `DESIGN_BLUEPRINT.md` as a current-state document in Japanese, and separate historical notes from current capability.
  - Reason:
    - Prevent mismatch between old phase notes and current implementation.
  - Impact area:
    - Documentation quality / Team alignment

- Decision: provisional
  - Policy:
    - Define GUI direction as Dash MVP with three pages: Config, Run, Artifacts.
  - Reason:
    - Keeps Core/API boundaries intact while enabling practical local operations.
  - Impact area:
    - Product roadmap / Adapter layer design

**Changes**
- Updated `DESIGN_BLUEPRINT.md`:
  - Reorganized fully in Japanese.
  - Added API x Task capability matrix (current implementation).
  - Added prioritized missing-feature inventory (P1/P2/P3).
  - Added explicit Dash GUI MVP plan (`/config`, `/run`, `/artifacts`).
  - Marked ONNX graph optimization as intentionally skipped (non-priority).
- Updated `README.md`:
  - Aligned backlog with current priorities (GUI, config migration, causal/simulation/export enhancements).
  - Added status note clarifying ONNX graph optimization is skipped, while dynamic quantization remains available.

**Decisions**
- Decision: confirmed
  - Policy:
    - `DESIGN_BLUEPRINT.md` serves as the current-state summary; detailed chronology remains in `HISTORY.md`.
  - Reason:
    - Improves readability and reduces ambiguity in implementation status.
  - Impact area:
    - Documentation governance

- Decision: confirmed
  - Policy:
    - GUI plan is fixed to Dash MVP scope: Config編集 + Run実行 + Artifact閲覧.
  - Reason:
    - Matches existing architecture principles (RunConfig入口 / Core non-UI).
  - Impact area:
    - GUI roadmap / Architectural consistency

### 2026-02-11 (Session planning: phase21-dash-gui-mvp)
**Context**
- Implement the top-priority adapter feature: Dash GUI MVP.
- Keep stable API signatures unchanged and use `veldra.api.runner` only as GUI backend.

**Decisions**
- Decision: provisional
  - Policy:
    - GUI MVP scope is fixed to three pages: Config Editor, Run Console, Artifact Explorer.
  - Reason:
    - Provides a practical local operator surface without expanding Core responsibility.
  - Impact area:
    - Adapter architecture / Operability

- Decision: provisional
  - Policy:
    - GUI is delivered as optional dependency (`uv sync --extra gui`) with launcher `veldra-gui`.
  - Reason:
    - Keeps base runtime lightweight while enabling opt-in interactive workflows.
  - Impact area:
    - Packaging / Developer experience

### 2026-02-11 (Session/PR: phase21-dash-gui-mvp)
**Context**
- Added a non-intrusive Dash adapter for local RunConfig-driven operations.

**Changes**
- Code changes:
  - Added GUI package:
    - `src/veldra/gui/server.py`
    - `src/veldra/gui/app.py`
    - `src/veldra/gui/services.py`
    - `src/veldra/gui/types.py`
    - `src/veldra/gui/pages/config_page.py`
    - `src/veldra/gui/pages/run_page.py`
    - `src/veldra/gui/pages/artifacts_page.py`
  - Added optional GUI dependencies and launcher in `pyproject.toml`:
    - `gui` extra (`dash`, `plotly`, `dash-bootstrap-components`)
    - `veldra-gui` entrypoint
- Tests added:
  - `tests/test_gui_app_layout.py`
  - `tests/test_gui_services_config_validation.py`
  - `tests/test_gui_services_run_dispatch.py`
  - `tests/test_gui_artifact_listing.py`
  - `tests/test_gui_error_mapping.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - GUI adapter is implemented as Dash MVP with `/config`, `/run`, `/artifacts`.
  - Reason:
    - Satisfies immediate usability needs while preserving adapter-only responsibility.
  - Impact area:
    - Product capability / Architecture consistency

- Decision: confirmed
  - Policy:
    - GUI remains optional and does not alter existing stable API signatures.
  - Reason:
    - Maintains non-intrusive rollout and backward compatibility.
  - Impact area:
    - Compatibility / Packaging policy

### 2026-02-12 (Session planning: phase22-config-migrate-mvp)
**Context**
- Implement `veldra.config.migrate` MVP as the next P1 item.
- Keep runtime API signatures unchanged and provide a strict, non-destructive migration entrypoint.

**Decisions**
- Decision: provisional
  - Policy:
    - Provide migration as both Python API and CLI (`veldra config migrate`).
  - Reason:
    - Supports both automation and operator workflows.
  - Impact area:
    - Config operability / DX

- Decision: provisional
  - Policy:
    - MVP scope is `config_version=1` normalization only; target_version is fixed to 1.
  - Reason:
    - Prioritizes safe foundation before multi-version transforms.
  - Impact area:
    - Compatibility / Risk control

- Decision: provisional
  - Policy:
    - Output is non-destructive by default (separate file, overwrite prohibited).
  - Reason:
    - Prevents accidental data loss in configuration workflows.
  - Impact area:
    - Safety / Operability

### 2026-02-12 (Session/PR: phase22-config-migrate-mvp)
**Context**
- Added a strict migration utility for RunConfig normalization and validation.

**Changes**
- Code changes:
  - Added `src/veldra/config/migrate.py`
    - `MigrationResult`
    - `migrate_run_config_payload(...)`
    - `migrate_run_config_file(...)`
  - Updated `src/veldra/config/__init__.py`
    - exports migration APIs
  - Updated `src/veldra/__init__.py`
    - implemented CLI subcommand:
      - `veldra config migrate --input <path> [--output <path>] [--target-version 1]`
    - success event logging:
      - `event: config migrate completed`
- Tests added:
  - `tests/test_config_migrate_payload.py`
  - `tests/test_config_migrate_file.py`
  - `tests/test_config_migrate_version.py`
  - `tests/test_cli_config_migrate.py`
- Tests updated:
  - `tests/test_package_root.py`
- Docs updated:
  - `README.md`
  - `DESIGN_BLUEPRINT.md`
  - `HISTORY.md`

**Decisions**
- Decision: confirmed
  - Policy:
    - Migration utility is delivered as API + CLI with strict validation.
  - Reason:
    - Ensures deterministic normalization and consistent error behavior.
  - Impact area:
    - Config governance / Reliability

- Decision: confirmed
  - Policy:
    - Only v1->v1 normalization is supported in this phase.
  - Reason:
    - Keeps the MVP bounded and safe while enabling future version migrations.
  - Impact area:
    - Scope control / Forward compatibility

- Decision: confirmed
  - Policy:
    - Existing output files are never overwritten by migration.
  - Reason:
    - Maintains non-destructive operator defaults.
  - Impact area:
    - Operational safety

**Results**
- `uv run ruff check .` : passed.
- `uv run pytest -q` : passed.

### 2026-02-14 (Session/PR: gui-memory-investigation-and-lazy-import)
**Context**
- `tests/test_gui_app_callbacks_config.py` 実行時のメモリ消費が大きいという報告に対し、原因を切り分けて実運用上のピークメモリを低減する。

**Plan**
- import経路のプロファイルを取り、`create_app()` 実行コストと import 初期化コストを分離する。
- 重量級依存（runner/artifact/lightgbm/optuna）を不要時に読み込まないよう lazy import 化する。
- GUI関連テストとメモリ再計測で回帰有無を確認する。

**Changes**
- 実装変更：
  - `src/veldra/api/__init__.py`
    - eager import を廃止し、`__getattr__` ベースの lazy export に変更。
    - `import veldra.api.exceptions` で runner/artifact を巻き込まないように修正。
  - `src/veldra/config/__init__.py`
    - eager import を廃止し、`io` / `migrate` / `models` の lazy export に変更。
    - `RunConfig` 参照時に不要な `migrate -> api` 連鎖 import を避ける構成へ修正。
  - `src/veldra/gui/services.py`
    - `run_action` / `inspect_data` 内で必要時のみ `Artifact` / `runner` / `load_tabular_data` を import する形へ変更。
  - `src/veldra/gui/app.py`
    - top-level の `Artifact` / `evaluate` / `load_tabular_data` import を除去。
    - `results` 関連 callback 内で必要時 import へ変更。
- ドキュメント変更：
  - `HISTORY.md` に本エントリを追記。
- テスト変更：
  - なし（既存テストで検証）。

**Decisions**
- Decision: confirmed
  - 内容：
    - API/Config/GUIの公開契約を維持したまま、package export と GUI callback 実装で lazy import を採用する。
  - 理由：
    - Stable API を壊さずに、不要な重量級依存の読み込みを避け、テスト時・対話時のピークメモリを下げるため。
  - 影響範囲：
    - API / GUI / 性能

**Results**
- 動作確認結果：
  - メモリ計測（`uv run --extra gui python`）
    - 修正前（調査時）: `import veldra.gui.app` 後 約 `248.6MB`
    - 修正後: `import veldra.gui.app` 後 約 `143.2MB`
  - `tests/test_gui_app_callbacks_config.py` 実行時ピークRSS
    - 修正前（調査時）: 約 `259328 KB`
    - 修正後: 約 `145976 KB`
  - `uv run --extra gui pytest -q tests/test_gui_app_layout.py tests/test_gui_services_config_validation.py`
    - `5 passed`

**Risks / Notes**
- `tests/test_gui_app_callbacks_config.py::test_migration_apply` の `KeyError: 'callback'` は今回のメモリ最適化とは独立の既存失敗として継続。
- `ruff check` は `src/veldra/gui/app.py` の既存スタイル違反が多数あり、今回スコープ外。

**Open Questions**
- [ ] GUI callback_map のDashバージョン差分に依存しないテスト実装へ更新するか。

### 2026-02-14 (Session/PR: gui-lazy-import-reapply-and-test-separation)
**Context**
- `uv run coverage run -m pytest -q` 実行時のメモリ逼迫を受け、GUI経路の重量import回帰を解消しつつ、GUIテストを分離運用できる状態に戻す。

**Plan**
- `src/veldra/gui/app.py` と `src/veldra/gui/services.py` の eager import を再び lazy import 化する。
- pytest に `gui` marker を導入し、GUIテストを明示的に分離実行できるようにする。
- README のテスト/coverage手順を core と gui の分割運用に更新する。

**Decisions**
- Decision: provisional
  - 内容：
    - LightGBM学習ロジック（反復実行・パラメータ）は変更せず、GUI import経路とテスト実行戦略のみでメモリ対策を行う。
  - 理由：
    - 学習挙動の仕様は維持しつつ、OOMリスクを低減するため。
  - 影響範囲：
    - GUI / テスト運用 / 性能

**Changes**
- 実装変更：
  - `src/veldra/gui/app.py`
    - `Artifact` / `evaluate` / `load_tabular_data` の eager import を除去。
    - `_ensure_runtime_imports()` を追加し、結果表示/評価callback実行時に遅延解決。
  - `src/veldra/gui/services.py`
    - `runner` / `artifact` / `data` の eager import を除去。
    - 初期実装の `_ensure_runtime_imports()`（一括解決）を廃止。
    - action/用途単位の解決（`_get_runner_func`, `_get_load_tabular_data`, `Artifact` proxy）へ変更し、
      `inspect_data` 等の軽処理で `veldra.api.runner` を読み込まないように修正。
  - `tests/conftest.py`
    - `gui` marker を登録。
    - `test_gui_*` および `test_new_ux.py` を自動で `gui` marker 付与。
  - `tests/test_gui_services_unit.py`
    - `MagicMock` 多用箇所を軽量スタブ/実ファイルベースへ置換し、
      モック呼び出し履歴の蓄積を抑制。
  - `tests/test_gui_app_coverage.py`
    - `Artifact` モックを `MagicMock` から軽量ローダークラスへ置換。
  - `tests/test_gui_app_coverage_2.py`
    - `builtins.open` の差し替えを `MagicMock` から `mock_open` へ変更。
  - `tests/test_new_ux.py`
    - 単純戻り値モックを `MagicMock` から `SimpleNamespace` へ変更。
  - `src/veldra/gui/app.py`
    - `_to_jsonable` に再帰安全化を追加：
      - `unittest.mock.Mock` を直接 `repr` 化（`model_dump` 連鎖回避）
      - 循環参照検出（`<cycle>`）
      - 深さ上限（`<max_depth_reached>`）
      - `model_dump()` が self を返す場合の停止ガード
  - `tests/test_gui_jsonable_safety.py`
    - `MagicMock` 入力時と循環参照入力時に `_json_dumps` が停止する回帰テストを追加。
  - `pyproject.toml`
    - pytest marker 定義に `gui` を追加。
- ドキュメント変更：
  - `README.md` に core/gui 分離の pytest/coverage 実行手順を追記。
  - `DESIGN_BLUEPRINT.md` に Phase 33 提案を追記。

**Decisions**
- Decision: confirmed
  - 内容：
    - GUIテストは marker 分離（`-m "not gui"` / `-m "gui"`）で運用し、coverage も2段階実行を標準手順とする。
  - 理由：
    - 低メモリ環境でも実行可能性を高め、OOMリスクを抑えるため。
  - 影響範囲：
    - テスト運用 / 性能

**Results**
- 動作確認結果：
  - `.venv/bin/python -m pytest -q tests/test_gui_app_helpers.py tests/test_gui_services_unit.py tests/test_gui_app_callbacks_results.py`
    - `7 passed`
  - marker 分離確認（collect-only）：
    - `-m "gui"`: `70/405 tests collected`
    - `-m "not gui"`: `335/405 tests collected`
  - import時メモリ再計測（Linux RSS）：
    - `import veldra.gui.app`: `164712 KB`
    - `import veldra.gui.services`: `115940 KB`
  - lazy import 挙動確認：
    - `inspect_data` 実行後も `sys.modules` に `veldra.api.runner` が載らないことを確認
    - `run_action(action='fit')` 実行時に初めて `veldra.api.runner` が解決されることを確認
  - GUI関連の回帰テスト：
    - `tests/test_gui_services_unit.py`: `5 passed`
    - `tests/test_gui_services_run_dispatch.py`: `3 passed`
    - `tests/test_gui_app_coverage.py tests/test_gui_app_coverage_2.py tests/test_new_ux.py`: `12 passed`
    - `tests/test_gui_jsonable_safety.py`: `2 passed`

**Risks / Notes**
- この実行環境では `uv run` が `uv-build` 解決時に外部ネットワーク到達不可で失敗するため、検証は `.venv/bin/python -m pytest` で実施。

### 2026-02-14 (Session/PR: ruff-repo-wide-cleanup)
**Context**
- リポジトリ全体で `uv run ruff check .` が多数失敗しており、品質ゲート通過を阻害していた。

**Plan**
- `ruff --fix` と `ruff format` を先に適用し、機械修正可能な差分を一括で解消する。
- 残件の `E501` / `E722` / `E402` / `F401` / `F821` を手動で最小差分修正する。
- marker 分離済みの pytest 運用に合わせて `gui` / `not gui` を分割実行する。

**Changes**
- 実装変更:
  - `src/veldra/gui/app.py`
    - import 並びを整理し、`toast` import を module top へ移動（`E402` 解消）。
    - 長い文字列・f-string を分割して `E501` を解消。
    - `except:` を `except Exception:` に変更して `E722` を解消。
    - status badge の色判定を事前変数化し、可読性と行長を改善。
  - `src/veldra/gui/pages/run_page.py`
    - 長い literal / className 文字列を分割し `E501` を解消。
  - `tests/debug_imports.py`
    - 未使用 import を避けるため `importlib.import_module` ベースに変更。
  - `tests/test_gui_app_callbacks_internal.py`
    - 長大コメントを簡潔化して `E501` を解消。
  - `tests/test_gui_app_coverage.py`
    - `typing.Any` を追加し `F821` を解消。
- 一括整形:
  - `uv run ruff check . --fix`
  - `uv run ruff check . --fix --unsafe-fixes`
  - `uv run ruff format .`

**Decisions**
- Decision: confirmed
  - 内容:
    - `pyproject.toml` の Ruff ルールは緩和せず、コード修正のみで全体通過させる。
  - 理由:
    - Stable API/Artifact 互換を維持しながら、静的品質ゲートを再び信頼可能にするため。
  - 影響範囲:
    - GUI adapter / tests / 開発品質

**Results**
- `uv run ruff check .`: **passed**
- `uv run pytest -q -m "not gui"`: **335 passed, 73 deselected**
- `uv run pytest -q -m "gui"`: **73 passed, 335 deselected**
