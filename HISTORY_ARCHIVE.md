# HISTORY_ARCHIVE.md
（再編前の原本退避。参照専用）

- 退避日: 2026-02-14
- 取り扱い: 編集禁止（参照専用）。最新運用は `HISTORY.md` を正本とする。

---

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

### 2026-02-15 (Session/PR: api-doc-contract-and-algorithm-docs)
**Context**
- 公開APIで docstring 欠落と契約記述不足（Parameters/Returns/Raises）があり、利用者向け参照性が低かった。
- 主要アルゴリズム関数の説明が短く、設計意図と安全性（リーク防止等）が追いにくかった。

**Plan**
- `veldra.api.*` の公開面を NumPy 形式 docstring へ統一し、公開関数の契約を明文化する。
- 主要アルゴリズム関数に `Notes` を追加し、フロー・前提・失敗条件を説明する。
- README に API サマリとアルゴ概要を追記し、品質ゲートとして doc 契約テストを追加する。

**Changes**
- 実装変更:
  - `src/veldra/api/runner.py`
    - `fit/tune/estimate_dr/evaluate/predict/simulate/export` に NumPy 形式 docstring を追加/統一。
  - `src/veldra/api/types.py`
    - 公開 dataclass (`RunResult` など) に `Attributes` 付き docstring を追加。
  - `src/veldra/api/artifact.py`
    - `Artifact` と公開メソッド（`from_config/load/save/predict/simulate`）の契約 docstring を追加。
  - `src/veldra/api/logging.py`
    - `build_log_payload/log_event` を NumPy 形式 docstring に更新。
  - 主要アルゴリズム関数へ `Notes` 追加:
    - `src/veldra/modeling/regression.py`
    - `src/veldra/modeling/binary.py`
    - `src/veldra/modeling/multiclass.py`
    - `src/veldra/modeling/frontier.py`
    - `src/veldra/modeling/tuning.py`
    - `src/veldra/causal/dr.py`
    - `src/veldra/causal/dr_did.py`
    - `src/veldra/simulate/engine.py`
  - `README.md`
    - `API Reference (Summary)` と `Algorithm Overview` を追加。
  - `tests/test_doc_contract.py`（新規）
    - `veldra.api.__all__` の docstring 有無
    - 公開API関数の `Parameters/Returns/Raises` セクション有無
    - 主要アルゴ関数の `Notes` セクション有無

**Decisions**
- Decision: confirmed
  - 内容:
    - docstring フォーマットは NumPy 形式を採用する。
    - 品質ゲートは公開APIと主要アルゴ関数に限定して導入する。
  - 理由:
    - 公開契約の明確化を優先しつつ、過剰なテストコストを避けるため。
  - 影響範囲:
    - API ドキュメント / README / テスト品質

**Results**
- `uv run ruff check .`: passed
- `uv run pytest -q tests/test_doc_contract.py`: passed
- `uv run pytest -q -m "not gui"`: passed
- `uv run pytest -q -m "gui"`: passed

### 2026-02-15 (Session/PR: readme-why-runbook-runconfig-reference)
**Context**
- README において、RunConfig の完全参照、Quick Start から実運用への導線、プロジェクトの位置づけ表現が不足していた。

**Plan**
- Features 前に `Why VeldraML?` を追加して課題/解決/差別化を明示する。
- `From Quick Start to Production` runbook を追加して、実運用までの段階手順と成功判定を記載する。
- `RunConfig Reference (Complete)` を README 内に追加し、`models.py` から自動生成 + 整合テストで維持する。

**Changes**
- ドキュメント変更:
  - `README.md`
    - `## Why VeldraML?` を追加（課題→解決→位置づけ）。
    - `## From Quick Start to Production` を追加（6-step runbook + checklist）。
    - `## RunConfig Reference (Complete)` を追加。
    - 参照生成ブロックマーカーを追加:
      - `<!-- RUNCONFIG_REF:START -->`
      - `<!-- RUNCONFIG_REF:END -->`
- 生成/保守機構:
  - `scripts/generate_runconfig_reference.py`（新規）
    - `src/veldra/config/models.py` からフィールド表、objective行列、主要交差制約、最小テンプレートを生成。
    - `--write` で README ブロック更新、`--check` で差分検出。
- テスト追加:
  - `tests/test_readme_runconfig_reference.py`（新規）
    - 必須見出し（Why/Runbook/RunConfig Complete）存在確認。
    - 生成ブロック整合（`--check`）確認。

**Decisions**
- Decision: confirmed
  - 内容:
    - README は単一の正本として維持し、RunConfig リファレンスは README 内に埋め込みで管理する。
    - 将来の仕様追随漏れ対策として「自動生成 + 整合テスト」を採用する。
  - 理由:
    - 参照性と運用性を両立し、手動同期漏れリスクを抑えるため。
  - 影響範囲:
    - README / 開発運用 / ドキュメント品質

**Results**
- `uv run ruff check .`: passed
- `uv run pytest -q tests/test_readme_runconfig_reference.py`: passed
- `uv run pytest -q -m "not gui"`: passed
- `uv run pytest -q -m "gui"`: passed

---

## Snapshot 2026-02-22 16:42:06Z (source: HISTORY.md before 1/5 compression)

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

### 2026-02-18（作業/PR: docs-blueprint-compression-before-phase26-7）
**背景**
- `DESIGN_BLUEPRINT.md` の `Phase26.7` 以前（特に 13.3〜13.6）が詳細仕様中心で肥大化し、現行設計の参照コストが高くなっていた。
- 詳細な時系列・検証実績は `HISTORY.md` に既に保持されているため、Blueprint 側は要約へ寄せる余地があった。

**変更内容**
- `DESIGN_BLUEPRINT.md` の `## 10`〜`## 13.6` を圧縮し、重要契約（互換性、Decision要点、成果物）を残した要約版へ再編。
- 冗長な詳細（対象ファイル一覧、実装ステップ詳細、個別テストコマンド群）は削減し、参照先を `HISTORY.md` に統一。
- 冒頭の最終更新日を `2026-02-18` に更新し、Phase要約の進捗表記（26.1/26.2/26.4）を完了状態へ整合。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Blueprint は「現時点の設計状態を短く示す文書」として維持し、Phase23〜26.6 の詳細は `HISTORY.md` を一次情報源とする。
  - 理由: 重要情報を保持したまま、設計参照とレビュー時の認知負荷を下げるため。
  - 影響範囲: `DESIGN_BLUEPRINT.md`, `HISTORY.md`

**検証結果**
- ドキュメント整合性を目視確認（`## 13.6` から `## 13.7` への接続、見出し整合、更新日反映）。
- コード変更なしのため、テスト実行は未実施。

### 2026-02-18（作業/PR: phase27-priority-queue-worker-pool）
**背景**
- GUI ジョブキューは single worker + FIFO で運用しており、優先度制御と並列実行が未対応だった。
- queued ジョブの実運用上の順序変更手段がなく、重要ジョブを先行させる手段が必要だった。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase27 は strict priority（`high=90`, `normal=50`, `low=10`）を採用し、queued 並び替えは priority 変更操作で実現する。
  - 理由: 既存契約を壊さず最小変更で制御性とスループットを向上できるため。
  - 影響範囲: `src/veldra/gui/{types.py,job_store.py,worker.py,server.py,services.py,app.py,pages/run_page.py,components/task_table.py}`, `tests/test_gui_*`, `tests/e2e_playwright/test_uc01_regression_flow.py`, `DESIGN_BLUEPRINT.md`
- Decision: confirmed（確定）
  - 内容: strict priority + worker pool + queued reprioritize UI を実装し、既定 `worker_count=1` の後方互換を維持した。
  - 理由: スループットと運用制御を改善しつつ、既存GUI契約と安定動作を維持できたため。
  - 影響範囲: `src/veldra/gui/{types.py,job_store.py,worker.py,server.py,services.py,app.py,pages/run_page.py,components/task_table.py}`, `tests/test_gui_job_store.py`, `tests/test_gui_worker_pool.py`, `tests/test_gui_server.py`, `tests/test_gui_run_page.py`, `tests/test_gui_pages_and_init.py`, `tests/test_gui_app_job_flow.py`, `tests/test_new_ux.py`, `tests/test_gui_services_edge_cases.py`, `tests/e2e_playwright/test_uc01_regression_flow.py`, `README.md`, `DESIGN_BLUEPRINT.md`

**変更内容**
- `RunInvocation` / `GuiJobRecord` に `priority` を追加（`low|normal|high`, default `normal`）。
- `GuiJobStore` に priority 列 migration（`PRAGMA table_info` + `ALTER TABLE`）と `(status, priority DESC, created_at_utc ASC)` インデックスを追加。
- `claim_next_job()` を `ORDER BY priority DESC, created_at_utc ASC` へ変更し、`set_job_priority()`（queued限定）を実装。
- `GuiWorkerPool` を追加し、`GuiWorker` 複数管理による並列実行と graceful shutdown を実装。
- `veldra-gui` に `--worker-count` / `VELDRA_GUI_WORKER_COUNT` を追加し、起動ログに `worker_count` を記録。
- Runページに `run-priority`（投入時）と `run-queue-priority` + `run-set-priority-btn`（queued再優先付け）を追加。
- Runジョブテーブルに Priority 列を追加し、queued を priority順で表示。
- `services.set_run_job_priority()` と `app._cb_set_job_priority()` を追加。

**検証結果**
- `uv run ruff check src/veldra/gui/types.py src/veldra/gui/job_store.py src/veldra/gui/services.py src/veldra/gui/worker.py src/veldra/gui/server.py src/veldra/gui/pages/run_page.py src/veldra/gui/components/task_table.py src/veldra/gui/app.py tests/test_gui_job_store.py tests/test_gui_worker_pool.py tests/test_gui_server.py tests/test_gui_run_page.py tests/test_gui_pages_and_init.py tests/test_new_ux.py tests/test_gui_app_job_flow.py tests/test_gui_services_edge_cases.py tests/e2e_playwright/test_uc01_regression_flow.py README.md DESIGN_BLUEPRINT.md HISTORY.md` を通過。
- `uv run pytest -q tests/test_gui_job_store.py tests/test_gui_worker.py tests/test_gui_worker_pool.py tests/test_gui_server.py tests/test_gui_run_page.py tests/test_gui_pages_and_init.py tests/test_gui_app_job_flow.py tests/test_gui_services_edge_cases.py tests/test_gui_run_async.py tests/test_gui_app_callbacks_internal.py tests/test_new_ux.py` を実施（`41 passed`）。
- `uv run pytest -q tests/e2e_playwright/test_uc01_regression_flow.py -m gui_e2e` を実施（`1 passed`）。
- `uv run pytest -q tests/test_gui_* tests/test_new_ux.py` を実施（`182 passed`）。

### 2026-02-18（作業/PR: phase28-realtime-progress-and-streaming-logs）
**背景**
- Phase27 時点の GUI run queue は状態遷移（queued/running/succeeded/failed）のみで、実行中ジョブの進捗とログ可視性が不足していた。
- Core/API 契約を維持したまま、GUI adapter 層だけでリアルタイム観測性を強化する必要があった。

**変更内容**
- `GuiJobRecord` に `progress_pct` / `current_step` を追加し、`GuiJobStore` の `jobs` テーブルへ後方互換 migration を実装。
- `job_logs` テーブル（`job_id`, `seq`, `created_at_utc`, `level`, `message`, `payload_json`）を新設し、`append_job_log` / `list_job_logs` / `update_progress` を実装。
- ログ保持上限を 10,000 行/job とし、上限超過時は古いログを FIFO で削除して `log_retention_applied` を記録。
- `run_action` に job 文脈（`job_id`, `job_store`）を導入し、action別ステップ進捗と失敗ログを永続化。
- `tune` 実行では runner trial 完了ログ（`n_trials_done`）を取り込み、`n_trials` ベースで進捗率を更新。
- `GuiWorker` で started/completed ログ記録と進捗初期化を追加。
- Run UI を更新し、ジョブテーブルに Progress 列、Job detail に `progress_viewer`（進捗バー + レベル色分けログ + Load More）を追加。
- `README.md` / `DESIGN_BLUEPRINT.md` を Phase28 実装状態へ同期。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase28 は polling 継続（`run-jobs-interval`）+ SQLite 永続化で実装し、WebSocket/SSE は導入しない。
  - 理由: 既存GUI契約を壊さず最小変更で可視化を強化できるため。
  - 影響範囲: `src/veldra/gui/{job_store.py,services.py,worker.py,app.py,pages/run_page.py,components/progress_viewer.py}`
- Decision: confirmed（確定）
  - 内容: 上記方式で実装を完了し、ログ保持上限は 10,000 行/job（FIFO削除）で固定する。
  - 理由: UI応答性とDB肥大抑制を両立しつつ、運用に必要な履歴を維持できるため。
  - 影響範囲: `src/veldra/gui/{types.py,job_store.py,services.py,worker.py,app.py,pages/run_page.py,components/task_table.py,components/progress_viewer.py}`, `tests/test_gui_*`, `README.md`, `DESIGN_BLUEPRINT.md`

**検証結果**
- `uv run ruff check src/veldra/gui/types.py src/veldra/gui/job_store.py src/veldra/gui/services.py src/veldra/gui/worker.py src/veldra/gui/components/task_table.py src/veldra/gui/components/progress_viewer.py src/veldra/gui/pages/run_page.py src/veldra/gui/app.py tests/test_gui_job_store.py tests/test_gui_worker.py tests/test_gui_worker_pool.py tests/test_gui_app_job_flow.py tests/test_gui_app_additional_branches.py tests/test_gui_phase26_1.py tests/test_gui_run_page.py` を通過。
- `uv run pytest -q tests/test_gui_job_store.py tests/test_gui_worker.py tests/test_gui_worker_pool.py tests/test_gui_app_job_flow.py tests/test_gui_app_additional_branches.py tests/test_gui_phase26_1.py tests/test_gui_run_page.py tests/test_gui_services_run_dispatch.py tests/test_gui_services_core.py` を実施（`51 passed`）。
- `uv run pytest -q tests/test_gui_* tests/test_new_ux.py` を実施（`184 passed`）。

### 2026-02-18（作業/PR: phase29-cancel-retry-and-error-recovery）
**背景**
- Phase28 時点では `cancel_requested` が実行フローへ十分反映されず、running ジョブが成功確定へ上書きされる競合余地が残っていた。
- failed ジョブの再実行は再投入手順が手作業で、ジョブ詳細からの復旧導線と失敗診断（Next Step）が不足していた。

**変更内容**
- `src/veldra/gui/types.py`
  - `RetryPolicy` dataclass を追加し、`RunInvocation.retry_policy` を新設。
  - `GuiJobRecord` に `retry_count` / `retry_parent_job_id` / `last_error_kind` を追加。
- `src/veldra/gui/job_store.py`
  - `jobs` テーブルへ後方互換 migration（`retry_count`, `retry_parent_job_id`, `last_error_kind`）を追加。
  - `is_cancel_requested()` / `mark_canceled_from_request()` / `create_retry_job()` を実装。
  - `mark_failed(..., error_kind=...)` へ拡張。
- `src/veldra/gui/services.py`
  - `CanceledByUser` を導入し、`run_action()` に協調キャンセル checkpoint を追加。
  - `classify_gui_error()` / `build_next_steps()` を追加し、失敗 payload に `error_kind` / `next_steps` を保存。
  - `retry_run_job()` を追加（failed/canceled の手動リトライ）。
- `src/veldra/gui/worker.py`
  - `CanceledByUser` を `canceled` で終端化。
  - 成功直前の cancel 競合時に `canceled` を優先。
  - `RetryPolicy` に基づく自動リトライ枠（指数バックオフ）を追加（既定 `max_retries=0` で無効）。
- `src/veldra/gui/pages/run_page.py` / `src/veldra/gui/app.py`
  - Job Detail に `Retry Task` ボタンを追加。
  - failed/canceled 時のボタン活性制御を追加し、`next_steps` を Job Detail に表示。
- `src/veldra/api/runner.py`
  - `fit` / `tune` / `estimate_dr` に optional keyword-only `cancellation_hook` を追加（後方互換）。

**決定事項**
- Decision: confirmed（確定）
  - 内容: `RetryPolicy` は GUI adapter の `RunInvocation` に限定し、Core `RunConfig` には追加しない。
  - 理由: Core の責務境界と Stable API 互換を維持するため。
  - 影響範囲: `src/veldra/gui/{types.py,job_store.py,services.py,worker.py,app.py,pages/run_page.py}`
- Decision: confirmed（確定）
  - 内容: 自動リトライ既定は `max_retries=0`（手動中心）とし、バックオフ実装は将来有効化に備えて先行導入する。
  - 理由: 不要な自動再実行リスクを抑えるため。
  - 影響範囲: `src/veldra/gui/{types.py,worker.py,services.py,job_store.py}`
- Decision: confirmed（確定）
  - 内容: Next Step は既知エラー分類に限定して表示し、未知エラーは原文優先とする。
  - 理由: 過剰な汎用ヒントによる誤誘導を避けるため。
  - 影響範囲: `src/veldra/gui/services.py`, `src/veldra/gui/app.py`

**検証結果**
- `uv run ruff check src/veldra/gui/types.py src/veldra/gui/job_store.py src/veldra/gui/services.py src/veldra/gui/worker.py src/veldra/gui/app.py src/veldra/gui/pages/run_page.py src/veldra/api/runner.py tests/test_gui_job_store.py tests/test_gui_worker.py tests/test_gui_run_async.py tests/test_gui_services_core.py tests/test_gui_app_job_flow.py tests/test_gui_app_callbacks_internal.py tests/test_gui_phase26_1.py` を通過。
- `uv run pytest -q tests/test_gui_job_store.py tests/test_gui_worker.py tests/test_gui_run_async.py tests/test_gui_services_core.py tests/test_gui_app_job_flow.py tests/test_gui_app_callbacks_internal.py tests/test_gui_phase26_1.py` を実施（`37 passed`）。

### 2026-02-18（作業/PR: phase30-config-template-library-and-wizard）
**背景**
- Phase30 の要件（テンプレート、事前検証、localStorage 保存、diff、クイックスタートウィザード）を GUI adapter 内で一括実装する必要があった。
- `/train` と `/config` の導線差分が残ると運用時の誤操作リスクが増えるため、共通コールバック化が必要だった。

**変更内容**
- `src/veldra/gui/templates/` に 5 種テンプレート（`regression_baseline`, `binary_balanced`, `multiclass_standard`, `causal_dr_panel`, `tuning_standard`）を追加。
- `src/veldra/gui/template_service.py` を新設し、テンプレート検証、localStorage スロット管理（max 10 / save/load/clone / LRU）、YAML 変更キー数算出を実装。
- `src/veldra/gui/components/config_library.py` / `src/veldra/gui/components/config_wizard.py` を追加し、`src/veldra/gui/pages/train_page.py` と `src/veldra/gui/pages/config_page.py` に統合。
- `src/veldra/gui/app.py` に共通 Phase30 コールバック群（template apply/save/load/clone/diff, wizard open/step/apply）を追加し、`custom-config-slots-store`（localStorage）を導入。
- `src/veldra/gui/services.py` に `validate_config_with_guidance` を追加し、パス付きエラー + next-step を返す検証導線を実装。
- Run 投入前ゲートを強化し、RunConfig 形式 YAML のみ厳密検証して invalid 投入をブロック（既存モック互換を保持）。
- ドキュメントを更新（`DESIGN_BLUEPRINT.md`, `README.md`）。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Phase30 機能は GUI adapter に閉じ、Core `RunConfig` schema (`config_version=1`) と `veldra.api.*` の公開契約は非変更とする。
  - 理由: Stable API 互換と運用改善を両立するため。
  - 影響範囲: `src/veldra/gui/*`, `src/veldra/gui/templates/*`, docs
- Decision: confirmed（確定）
  - 内容: `/train` と `/config` は UI 分岐を許容しつつ、機能ロジックは `app.py` の共通コールバックへ集約する。
  - 理由: 乖離防止と保守コスト削減のため。
  - 影響範囲: `src/veldra/gui/app.py`, `src/veldra/gui/pages/{train_page.py,config_page.py}`

**検証結果**
- `uv run ruff check src/veldra/gui/app.py src/veldra/gui/services.py src/veldra/gui/template_service.py src/veldra/gui/components/config_library.py src/veldra/gui/components/config_wizard.py src/veldra/gui/pages/train_page.py src/veldra/gui/pages/config_page.py tests/test_gui_template_service.py tests/test_gui_config_localstore.py tests/test_gui_config_validation_guidance.py tests/test_gui_config_diff.py tests/test_gui_app_callbacks_config_phase30.py` を通過。
- `uv run pytest -q tests/test_gui_template_service.py tests/test_gui_config_localstore.py tests/test_gui_config_validation_guidance.py tests/test_gui_config_diff.py tests/test_gui_app_callbacks_config_phase30.py tests/test_gui_app_callbacks_config.py tests/test_gui_train_page.py tests/test_gui_pages_and_init.py tests/test_gui_app_helpers.py tests/test_gui_app_callbacks_validation_train_edge.py` を実施（`28 passed`）。
- `uv run pytest -q tests/test_gui_* tests/test_new_ux.py` を実施（`201 passed`）。

### 2026-02-18（作業/PR: phase31-advanced-visualization-and-multi-artifact-compare）
**背景**
- Phase31 の要件（fold時系列可視化、因果診断、複数Artifact比較、レポート拡張）を GUI adapter 中心で実装する必要があった。
- Stable API と既存Artifact互換を維持したまま、可視化情報量を増やす設計が必要だった。

**変更内容**
- `src/veldra/api/artifact.py` / `src/veldra/artifact/store.py` / `src/veldra/api/runner.py` を更新し、optional `fold_metrics`（`fold_metrics.parquet`）の保存・読込・fit連携を追加（既存Artifactは後方互換）。
- `src/veldra/gui/components/charts.py` に `plot_fold_metric_timeline` / `plot_causal_smd` / `plot_multi_artifact_comparison` / `plot_feature_drilldown` を追加し、空データ時プレースホルダを統一。
- `src/veldra/gui/pages/results_page.py` / `src/veldra/gui/pages/compare_page.py` を更新し、Results の 3 新タブ（Fold Metrics / Causal Diagnostics / Feature Drilldown）と Compare の multi-select（max 5）+ baseline UI を追加。
- `src/veldra/gui/services.py` を拡張し、`compare_artifacts_multi` / `load_causal_summary` / `export_pdf_report` を追加、`export_html_report` を fold/causal情報込みへ拡張、`run_action` に `export_pdf_report` を統合。
- `src/veldra/gui/app.py` の callback 群を更新し、Results拡張出力・Compare多件比較・PDF export ボタン導線を既存ジョブキュー経由で接続。
- テストを更新/追加（`tests/test_gui_components.py`, `tests/test_gui_compare_page.py`, `tests/test_gui_app_callbacks_results_edge.py`, `tests/test_gui_results_enhanced.py`, `tests/test_gui_app_callbacks_runs_edge.py`, `tests/test_gui_export_pdf.py`）。

**決定事項**
- Decision: confirmed（確定）
  - 内容: Artifact拡張は optional `fold_metrics` の追加に限定し、`manifest_version=1` を維持する。
  - 理由: 過去Artifactの読み込み互換と公開契約の非破壊を両立するため。
  - 影響範囲: `src/veldra/api/artifact.py`, `src/veldra/artifact/store.py`, `src/veldra/api/runner.py`
- Decision: confirmed（確定）
  - 内容: レポートは HTML を基準実装とし、PDF は optional dependency（`weasyprint`）前提で提供する。
  - 理由: 実行環境差による失敗を局所化しつつ、共有可能なレポート品質を確保するため。
  - 影響範囲: `src/veldra/gui/services.py`, `src/veldra/gui/app.py`, `src/veldra/gui/pages/results_page.py`

**検証結果**
- `uv run ruff check src/veldra/api/artifact.py src/veldra/artifact/store.py src/veldra/api/runner.py src/veldra/gui/components/charts.py src/veldra/gui/pages/results_page.py src/veldra/gui/pages/compare_page.py src/veldra/gui/services.py src/veldra/gui/app.py tests/test_gui_compare_page.py tests/test_gui_app_callbacks_results_edge.py tests/test_gui_results_enhanced.py tests/test_gui_components.py tests/test_gui_phase26_1.py tests/test_gui_app_callbacks_runs_edge.py tests/test_gui_export_pdf.py` を通過。
- `uv run pytest -q tests/test_gui_components.py tests/test_gui_compare_page.py tests/test_gui_app_callbacks_results.py tests/test_gui_app_callbacks_results_edge.py tests/test_gui_app_results_flow.py tests/test_gui_results_enhanced.py tests/test_gui_export_html.py tests/test_gui_services_edge_cases.py tests/test_gui_export_excel.py tests/test_gui_phase26_1.py tests/test_gui_app_callbacks_runs_edge.py tests/test_gui_export_pdf.py` を実施（`46 passed`）。
- `uv run pytest -q tests/test_gui_*` を実施（`200 passed`）。

### 2026-02-18（作業/PR: phase32-performance-and-scalability）
**背景**
- 大規模運用（jobs 10,000件、artifacts 1,000件、preview 100,000行）時に、GUI一覧取得とプレビュー描画の応答性能を維持する必要があった。
- 既存実装は全件取得 + クライアント側ページング寄りで、運用規模拡大時の劣化リスクが高かった。

**変更内容**
- `GuiJobStore` に `list_jobs_page(limit, offset, status, action, query)` を追加し、`OFFSET/LIMIT` + total count を提供。
- `GuiJobStore` に `jobs_archive` テーブル、`archive_jobs`、`purge_archived_jobs` を追加し、TTLベース保守の基盤を実装。
- クエリ計測と slow query warning（100ms閾値）を `job_store`/`services` に追加。
- `services` に `PaginatedResult` ベースの `list_run_jobs_page` / `list_artifacts_page` / `load_data_preview_page` を追加。
- Run / Runs / Results ページをサーバー側ページング化（page/page_size/total ストア + Prev/Next UI）。
- Data ページを AG Grid 優先構成へ拡張し、列指定付きの遅延読込 callback を追加。
- housekeeping interval を追加し、archive/purge を定期実行する運用導線を実装。
- `pyproject.toml` の GUI optional dependency に `dash-ag-grid` を追加。
- テストを更新・追加（`test_gui_job_store_perf.py` を新規追加）。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase32 は 2段階（Phase32.1: ページング/DB最適化/AG Grid、Phase32.2: archive/purge/監視）で実施する。
  - 理由: 破壊範囲を段階化し、UI回帰とデータ整合性リスクを制御するため。
  - 影響範囲: `src/veldra/gui/{job_store.py,services.py,app.py,pages/*,components/*}`, tests, docs
- Decision: provisional（暫定）
  - 内容: Data プレビューは Dash AG Grid を標準方式として採用し、遅延読込/仮想スクロールを実装する。
  - 理由: 大規模データ表示時のDOM負荷とメモリ使用量を抑えるため。
  - 影響範囲: `src/veldra/gui/pages/data_page.py`, `src/veldra/gui/app.py`, `src/veldra/gui/services.py`, `pyproject.toml`
- Decision: confirmed（確定）
  - 内容: 上記方針で Phase32.1（ページング + AG Grid）と Phase32.2（archive/purge + 監視）の最小実装を同PR内で完了した。
  - 理由: Stable API 非破壊のまま、運用スケール要件に対応する導線を確立できたため。
  - 影響範囲: `src/veldra/gui/{job_store.py,services.py,app.py,pages/*,components/task_table.py,types.py}`, `pyproject.toml`, `tests/test_gui_*`, `DESIGN_BLUEPRINT.md`

**検証結果**
- `uv run ruff check src/veldra/gui/job_store.py src/veldra/gui/services.py src/veldra/gui/app.py src/veldra/gui/pages/run_page.py src/veldra/gui/pages/results_page.py src/veldra/gui/pages/runs_page.py src/veldra/gui/pages/data_page.py src/veldra/gui/components/task_table.py src/veldra/gui/types.py tests/test_gui_job_store.py tests/test_gui_services_edge_cases.py tests/test_gui_app_job_flow.py tests/test_gui_app_results_flow.py tests/test_gui_app_callbacks_results.py tests/test_gui_phase26_1.py tests/test_gui_runs_page.py tests/test_gui_run_page.py tests/test_gui_pages_and_init.py tests/test_gui_job_store_perf.py` を通過。
- `uv run pytest -q tests/test_gui_job_store.py tests/test_gui_job_store_perf.py tests/test_gui_services_edge_cases.py tests/test_gui_app_job_flow.py tests/test_gui_app_results_flow.py tests/test_gui_app_callbacks_results.py tests/test_gui_phase26_1.py tests/test_gui_runs_page.py tests/test_gui_run_page.py tests/test_gui_pages_and_init.py` を実施（`50 passed`）。
- `uv run pytest -q tests/test_gui_app_callbacks_internal.py tests/test_gui_results_enhanced.py tests/test_gui_pages_logic.py tests/test_gui_data_enhanced.py tests/test_gui_services_unit.py tests/test_gui_app_helpers.py` を実施（`14 passed`）。

### 2026-02-18（作業/PR: phase33-gui-lazy-runtime-contract-and-marker-hardening）
**背景**
- GUI import 時のメモリ回帰を再発させないため、既存 lazy import を「実装依存」から「契約依存」へ固定する必要があった。
- `gui_e2e` が `-m "not gui"` 側に混入しうる収集条件が残っており、core/gui 分離運用を厳密化する必要があった。

**変更内容**
- `src/veldra/gui/_lazy_runtime.py` を新設し、`Artifact` / runner関数 / data loader の遅延解決ロジックを共通化。
- `src/veldra/gui/app.py` / `src/veldra/gui/services.py` の lazy 解決経路を共通ヘルパーへ寄せ、`Artifact` や `fit/evaluate` など既存 monkeypatch ポイントは維持。
- `tests/test_gui_lazy_import_contract.py` を追加し、`import veldra.gui.app` / `import veldra.gui.services` 直後に
  `veldra.api.runner`, `veldra.api.artifact`, `veldra.data`, `lightgbm`, `optuna`, `sklearn` が未ロードであることを subprocess で検証。
- `tests/conftest.py` の収集フックを更新し、`tests/e2e_playwright/*` または `gui_e2e` marker を持つテストへ `gui` marker を自動付与。
- `README.md` の Development セクションを更新し、coverage 実行を
  `core`（`not gui and not notebook_e2e`）→`gui`（`gui and not gui_e2e and not notebook_e2e`）の2段階手順へ標準化。
- `DESIGN_BLUEPRINT.md` の Phase33 を提案から実施計画へ更新し、DoD と provisional decision を明記。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase33 は新規機能追加ではなく、既存 lazy import 契約の固定化と marker 分離の厳密化を主目的とする。
  - 理由: Stable API を維持しつつ、メモリ回帰とテスト運用の曖昧さを同時に抑制するため。
  - 影響範囲: `src/veldra/gui/{_lazy_runtime.py,app.py,services.py}`, `tests/conftest.py`, `tests/test_gui_lazy_import_contract.py`, `README.md`, `DESIGN_BLUEPRINT.md`
- Decision: confirmed（確定）
  - 内容: lazy runtime 共通化、cold import 契約テスト追加、`gui_e2e` の `gui` 内包、coverage 2段階運用の文書化を完了した。
  - 理由: 既存 GUI 互換性を保ったまま、Phase33 DoD を満たせたため。
  - 影響範囲: `src/veldra/gui/{_lazy_runtime.py,app.py,services.py}`, `tests/{conftest.py,test_gui_lazy_import_contract.py}`, `README.md`, `DESIGN_BLUEPRINT.md`, `HISTORY.md`

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run ruff check src/veldra/gui/_lazy_runtime.py src/veldra/gui/app.py src/veldra/gui/services.py tests/conftest.py tests/test_gui_lazy_import_contract.py README.md DESIGN_BLUEPRINT.md HISTORY.md` を通過。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_gui_lazy_import_contract.py tests/test_gui_services_core.py tests/test_gui_app_additional_branches.py tests/test_gui_services_edge_cases.py` を実施（`31 passed`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest --collect-only -q -m "not gui" tests/e2e_playwright` を実施（`no tests collected (14 deselected)`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest --collect-only -q -m "gui"` を実施（`224/754 collected`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "gui and not gui_e2e and not notebook_e2e"` を実施（`210 passed, 544 deselected`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施（`739 passed, 15 deselected`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/e2e_playwright -m "gui_e2e"` を実施（`14 skipped`）。

### 2026-02-18（作業/PR: test-warning-suppression-joblib-serial-mode）
**背景**
- テスト実行環境で `joblib` の multiprocessing 初期化が権限制約により `UserWarning` を出力し、回帰確認ログのノイズになっていた。

**変更内容**
- `pyproject.toml` の `tool.pytest.ini_options.filterwarnings` に
  `joblib will operate in serial mode` の `UserWarning` 抑制ルールを追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 環境依存で非機能的な `joblib` 警告は pytest 設定で抑制する。
  - 理由: 実害のないノイズ警告を除去し、テストログの可読性を維持するため。
  - 影響範囲: `pyproject.toml`, テスト実行ログ

**検証結果**
- `.venv/bin/pytest -q tests/test_multiclass_edge_cases.py tests/test_gui_services_core.py` を実施（`9 passed`、該当 warning 非表示）。

### 2026-02-18（作業/PR: test-io-load-reduction-system-tmp-path）
**背景**
- テスト実行時に CSV/Artifact の書き込みが多く、ワークスペース配下 I/O 負荷が高くなっていた。

**変更内容**
- `tests/conftest.py` の `tmp_path` fixture を更新し、既定書き込み先を system temp dir（`tempfile.gettempdir()`）配下へ変更。
- system temp dir が利用不可な環境向けに、既存の `.pytest_tmp/cases` へフォールバックする分岐を追加。

**決定事項**
- Decision: confirmed（確定）
  - 内容: テストの一時ファイルは system temp dir を優先し、妥当性を維持したまま I/O 負荷を軽減する。
  - 理由: テスト検証内容を変えずに、重い書き込みを高速な一時領域へ寄せるため。
  - 影響範囲: `tests/conftest.py`, テスト実行時の一時ファイル配置

**検証結果**
- `.venv/bin/pytest -q tests/test_runner_fit_happy.py tests/test_runner_tune_happy.py tests/test_gui_job_store.py` を実施（`21 passed`）。
- `.venv/bin/pytest -q tests/test_artifact_store.py tests/test_tune_artifacts.py tests/test_observation_table.py` を実施（`9 passed`）。
- `.venv/bin/ruff check tests/conftest.py` を通過。

### 2026-02-18（作業/PR: phase34-1-studio-train-mode-foundation）
**背景**
- Phase34.1 の目的である Studio 骨格 + 学習モードを、既存 Guided Mode と互換を維持しながら実装する必要があった。
- `/` の既定遷移、Studio 専用 state、RunConfig 自動生成、ジョブ実行/進捗表示を adapter 層で閉じる方針を確定する必要があった。

**変更内容**
- `src/veldra/gui/pages/studio_page.py` を新規追加し、3ペイン構成・Studio専用 `dcc.Store` 群・`studio-run-poll-interval` を実装。
- `src/veldra/gui/components/studio_parts.py` を新規追加し、学習モード pane（Scope/Strategy/Action）と推論プレースホルダ pane を実装。
- `src/veldra/gui/app.py` を更新し、以下を追加。
  - `/studio` ルート追加、`/` の `/studio` リダイレクト、サイドバーに Studio セクションを追加。
  - `/studio` 時のステッパー非表示。
  - Studio callback 群（`_cb_redirect_root_to_studio`, `_cb_studio_mode_switch`, `_cb_studio_upload_train`, `_cb_studio_target_task`, `_build_studio_run_config`, `_cb_studio_run`, `_cb_studio_poll_job`）。
  - `submit_run_job()` による `fit/tune` キュー投入と、`get_run_job()` / `list_run_job_logs()` による進捗/KPI反映。
- `src/veldra/gui/assets/style.css` に Studio レイアウト（3カラム + モバイル1カラム）とサイドバーセクションラベルを追加。
- `src/veldra/gui/pages/__init__.py` に `studio_page` を追加。
- テストを追加/更新。
  - 新規: `tests/test_gui_studio.py`
  - 更新: `tests/test_gui_pages_and_init.py`, `tests/test_gui_app_helpers.py`, `tests/test_gui_app_callbacks_internal.py`
- `DESIGN_BLUEPRINT.md` の Phase34.1 節に実装確定内容を追記。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase34.1 では学習モードを優先実装し、推論モードはプレースホルダ表示に留める。
  - 理由: Phase34.2 で Model Hub / 推論機能を集中的に実装し、段階的に回帰リスクを抑えるため。
  - 影響範囲: `src/veldra/gui/{app.py,pages/studio_page.py,components/studio_parts.py}`
- Decision: confirmed（確定）
  - 内容: `/studio` を既定導線とし、`/` は `/studio` へリダイレクト、Studio 表示時はステッパーを非表示とする。
  - 理由: 高速反復 UX を優先しつつ、既存 Guided Mode の URL と機能を維持するため。
  - 影響範囲: `src/veldra/gui/app.py`, `src/veldra/gui/assets/style.css`
- Decision: confirmed（確定）
  - 内容: Model Hub はヘッダー上に表示のみ行い、Phase34.1 では無効化する。
  - 理由: UI 骨格を固定しながら、未実装操作による誤操作を防止するため。
  - 影響範囲: `src/veldra/gui/components/studio_parts.py`, `src/veldra/gui/pages/studio_page.py`

**検証結果**
- `uv run ruff check src/veldra/gui/app.py src/veldra/gui/pages/__init__.py src/veldra/gui/pages/studio_page.py src/veldra/gui/components/studio_parts.py tests/test_gui_studio.py tests/test_gui_pages_and_init.py tests/test_gui_app_helpers.py tests/test_gui_app_callbacks_internal.py` を通過。
- `uv run pytest -q tests/test_gui_studio.py tests/test_gui_pages_and_init.py tests/test_gui_app_helpers.py tests/test_gui_app_callbacks_internal.py` を実施（`14 passed`）。
- `uv run pytest -q tests/test_gui_* -m "not gui_e2e and not notebook_e2e"` を実施（`214 passed`）。

### 2026-02-19（作業/PR: phase34-2-studio-inference-model-hub）
**背景**
- Phase34.1 でプレースホルダだった推論モードと Model Hub を本実装し、Studio 内で学習と推論の往復を完結させる必要があった。
- Core 非依存と Stable API 互換を維持したまま、`predict` 非同期ジョブと推論前ガードレールを GUI adapter 層へ追加する必要があった。

**変更内容**
- `src/veldra/gui/types.py` に `ArtifactSpec` を追加し、Model Hub/推論モードの read-only 契約型を定義。
- `src/veldra/gui/services.py` を拡張し、`get_artifact_spec()`（Artifact.load 非依存）、`validate_prediction_data()`、`delete_artifact_dir()`、`run_action(action="predict")` を追加。
- `src/veldra/gui/components/studio_parts.py` / `src/veldra/gui/pages/studio_page.py` を更新し、推論 UI（モデルカード、推論アップロード、Spec、guardrail、Predict、CSV download）と Model Hub Offcanvas（Load/Delete/ページング）を実装。
- `src/veldra/gui/app.py` に Phase34.2 callback 群を追加（Model Hub 開閉/Load/Delete確認、推論アップロード、Predict投入、Predict→Evaluate 自動連携ポーリング、CSV download）。
- `src/veldra/gui/pages/runs_page.py` の Action フィルタに `predict` を追加し、運用可視性を整合。
- `tests/test_gui_studio.py`、`tests/test_gui_services_core.py`、`tests/test_gui_services_edge_cases.py`、`tests/test_gui_services_run_dispatch.py` を更新し、Phase34.2 契約を固定。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Model Hub の root は Phase34.2 では `artifacts/` 固定とし、root 切替 UI は導入しない。
  - 理由: UX と破壊的変更リスクを抑えつつ、推論モード本実装に集中するため。
  - 影響範囲: `src/veldra/gui/{app.py,components/studio_parts.py,services.py}`
- Decision: confirmed（確定）
  - 内容: 推論モードで正解ラベル列が指定された場合、Predict 成功後に Evaluate ジョブを自動投入し、評価失敗時も予測プレビュー/CSV は保持する。
  - 理由: 推論結果確認と評価を1画面内で完結させつつ、評価失敗時の再実行性を確保するため。
  - 影響範囲: `src/veldra/gui/app.py`, `src/veldra/gui/services.py`, `tests/test_gui_studio.py`

**検証結果**
- `uv run ruff check src/veldra/gui/types.py src/veldra/gui/services.py src/veldra/gui/components/studio_parts.py src/veldra/gui/pages/studio_page.py src/veldra/gui/pages/runs_page.py src/veldra/gui/app.py tests/test_gui_studio.py tests/test_gui_services_core.py tests/test_gui_services_edge_cases.py tests/test_gui_services_run_dispatch.py DESIGN_BLUEPRINT.md HISTORY.md` を通過。
- `uv run pytest -q tests/test_gui_studio.py tests/test_gui_services_core.py tests/test_gui_services_edge_cases.py tests/test_gui_services_run_dispatch.py` を実施（`32 passed`）。
- `uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施（`753 passed, 15 deselected`）。

### 2026-02-19（作業/PR: phase34-3-guided-mode-clarification）
**背景**
- `/studio` の追加と推論モード実装により高速導線は成立したが、既存ページ群を Guided Mode として明示する UI 契約が未固定だった。
- 既存 URL 互換を維持したまま、初心者導線と高速導線を同時に提供する Phase34.3 の最終整理が必要だった。

**変更内容**
- `src/veldra/gui/components/guided_mode_banner.py` を新規追加し、Guided バナー（英語文言 + `Open Studio` ボタン + ページ別 ID）を共通化した。
- 既存 9 ページ（`data_page.py`, `target_page.py`, `config_page.py`, `validation_page.py`, `train_page.py`, `run_page.py`, `results_page.py`, `runs_page.py`, `compare_page.py`）の見出し直下にバナーを追加した。
- `src/veldra/gui/assets/style.css` に `.guided-mode-banner` の最小スタイル（余白/境界/ボタン強調）を追加した。
- 新規 `tests/test_gui_guided_mode_pages.py` を追加し、9ページでのバナー存在・`/studio` 導線・`/studio` ページ非表示を契約化した。
- `tests/test_gui_app_helpers.py` を更新し、`_sidebar()` の `Studio Mode` / `Guided Mode` / `Operations` 3区分ラベル維持を契約化した。
- `DESIGN_BLUEPRINT.md` の Phase34.3 節へ実装方針確定/実装確定内容を追記した。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Guided バナーは既存 9 ページ（`/data`, `/target`, `/config`, `/validation`, `/train`, `/run`, `/results`, `/runs`, `/compare`）へ共通適用する。
  - 理由: Phase34 の互換性方針（URL維持）を守りつつ、Guided Mode の位置づけを UI 上で一貫して明示するため。
  - 影響範囲: `src/veldra/gui/pages/*`, `src/veldra/gui/components/`, `tests/test_gui_*`, `DESIGN_BLUEPRINT.md`
- Decision: confirmed（確定）
  - 内容: 英語固定の Guided バナーを既存 9 ページに導入し、サイドバー 3 区分（`Studio Mode` / `Guided Mode` / `Operations`）を維持したまま契約テストを追加した。
  - 理由: 既存 URL 互換を維持しつつ、初心者導線と高速導線の役割分離を UI 上で明示できたため。
  - 影響範囲: `src/veldra/gui/components/guided_mode_banner.py`, `src/veldra/gui/pages/*`, `src/veldra/gui/assets/style.css`, `tests/test_gui_guided_mode_pages.py`, `tests/test_gui_app_helpers.py`, `DESIGN_BLUEPRINT.md`, `HISTORY.md`

**検証結果**
- `uv run ruff check src/veldra/gui/pages src/veldra/gui/components/guided_mode_banner.py src/veldra/gui/assets/style.css tests/test_gui_guided_mode_pages.py tests/test_gui_app_helpers.py tests/test_gui_studio.py DESIGN_BLUEPRINT.md HISTORY.md` を実施（`style.css` は Python ではないため `ruff` では `invalid-syntax` となることを確認）。
- `uv run ruff check src/veldra/gui/pages src/veldra/gui/components/guided_mode_banner.py tests/test_gui_guided_mode_pages.py tests/test_gui_app_helpers.py tests/test_gui_studio.py DESIGN_BLUEPRINT.md HISTORY.md` を通過。
- `uv run pytest -q tests/test_gui_guided_mode_pages.py tests/test_gui_app_helpers.py tests/test_gui_studio.py` を実施（`25 passed`）。
- `uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施（`764 passed, 15 deselected`）。

### 2026-02-20（作業/PR: phase35-1-uc1-regression-notebook-enhancement）
**背景**
- quick_reference の UC-1 は1セルに処理が集中しており、手順の可読性と段階実行性が不足していた。
- Phase35.1 の着手として、Notebook 改修前に diagnostics 拡張と生成スクリプト主導の更新方式を固定する必要があった。

**変更内容**
- `src/veldra/diagnostics/metrics.py` の `regression_metrics()` に `huber` 指標を追加した。
- `src/veldra/diagnostics/plots.py` に `plot_learning_curve()` を追加し、UC-1 で利用する `plot_error_histogram` / `plot_feature_importance` / `plot_shap_summary` を Plotly 出力へ移行した（`save_path` は PNG 保存を維持）。
- `src/veldra/diagnostics/__init__.py` に `plot_learning_curve` を公開追加した。
- `scripts/generate_phase263_notebooks.py` に UC-1 専用 31セル構成の生成ロジックを追加し、`notebooks/quick_reference/reference_01_regression_fit_evaluate.ipynb` を再生成した。
- 生成スクリプトの全件実行により、quick_reference 全 Notebook の実行済み出力（execution metadata / outputs）を更新した。
- テストを更新（`tests/test_diagnostics_metrics.py`, `tests/test_diagnostics_plots.py`, `tests/test_quickref_structure.py`, `tests/test_notebook_execution_outputs.py`）。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Plotly 移行は Phase35.1 では UC-1 で利用する描画関数（`plot_error_histogram` / `plot_feature_importance` / `plot_shap_summary` / `plot_learning_curve`）に限定し、他 UC は後続サブフェーズで順次移行する。
  - 理由: 変更範囲を UC-1 に限定して回帰リスクを抑えつつ、Phase35.1 の完了条件（Huber/学習曲線/1セル1処理）を先に達成するため。
  - 影響範囲: `src/veldra/diagnostics/{plots.py,__init__.py}`, `notebooks/quick_reference/reference_01_regression_fit_evaluate.ipynb`
- Decision: provisional（暫定）
  - 内容: Notebook 更新は手編集ではなく `scripts/generate_phase263_notebooks.py` を唯一の生成元として実施する。
  - 理由: 実体 notebook と生成元の乖離を防ぎ、Phase35.2 以降へ同じ更新方式を横展開するため。
  - 影響範囲: `scripts/generate_phase263_notebooks.py`, `notebooks/quick_reference/reference_01_regression_fit_evaluate.ipynb`
- Decision: confirmed（確定）
  - 内容: Phase35.1 は UC-1 の 31セル化、Huber 追加、学習曲線追加、UC-1対象 Plotly 先行移行、生成スクリプト主導更新の方針で完了した。
  - 理由: Stable API を壊さずに DoD（指標・可視化・構造・回帰テスト）を満たしたため。
  - 影響範囲: `src/veldra/diagnostics/{metrics.py,plots.py,__init__.py}`, `scripts/generate_phase263_notebooks.py`, `notebooks/quick_reference/reference_01_regression_fit_evaluate.ipynb`, `tests/test_*`

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run ruff check src/veldra/diagnostics/metrics.py src/veldra/diagnostics/plots.py src/veldra/diagnostics/__init__.py scripts/generate_phase263_notebooks.py tests/test_diagnostics_metrics.py tests/test_diagnostics_plots.py tests/test_quickref_structure.py tests/test_notebook_execution_outputs.py` を通過。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_diagnostics_metrics.py tests/test_diagnostics_plots.py tests/test_quickref_structure.py` を実施（`11 passed`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施（`764 passed, 15 deselected`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_notebook_execution_outputs.py -m notebook_e2e` を実施（`1 passed`）。

### 2026-02-22（作業/PR: phase35-0-frontier-demo-data-generator）
**背景**
- Phase35.0 の要件として、Frontier ノートブックで再利用できる再現可能な DGP データを `data/demo/frontier/` に標準生成する必要があった。
- 仕様文言に `u_s` の扱い（公開列かどうか）と品質判定の向き（high/low どちらが優位か）の曖昧さが残っていた。

**変更内容**
- `scripts/generate_frontier_demo_data.py` を新規追加し、`train_eval.csv`（2023-01〜2024-12, 12,000行）と `latest.csv`（2025-01〜2025-12, 6,000行）を生成する CLI を実装。
- DGP を固定パラメータ化し、組織階層（HQ 5 / DEPT 50 / SEC 500）、月次季節性、ファネルKPI、値引き相殺、`u_s` 非効率を再現できる実装を追加。
- `tests/test_frontier_demo_data.py` を新規追加し、列順・件数・月範囲・`net_sales > 0` 比率・`u_s` 分散・`low-u_s` 優位・seed再現性を契約化。
- `DESIGN_BLUEPRINT.md` の Phase35.0 節を更新し、スコープを「データ生成+テストのみ」に限定、`u_s` を公開列として明記、品質判定を `u_s` 下位10%優位へ修正。
- `data/demo/frontier/train_eval.csv` と `data/demo/frontier/latest.csv` を生成し、実データを配置。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase35.0 の実装範囲は `scripts/generate_frontier_demo_data.py` と `tests/test_frontier_demo_data.py` に限定し、Notebook/Examples の読込経路変更は行わない。
  - 理由: サブフェーズの責務をデータ契約の確立に絞り、回帰範囲を最小化するため。
  - 影響範囲: `scripts/generate_frontier_demo_data.py`, `tests/test_frontier_demo_data.py`, `DESIGN_BLUEPRINT.md`, `HISTORY.md`
- Decision: confirmed（確定）
  - 内容: `u_s` は「課固有の非効率（高いほど不利）」として CSV 公開列に含め、品質判定は `u_s` 下位10%課の売上中央値が `u_s` 上位10%課を上回ることを採用する。
  - 理由: DGP の意味論と品質テストの判定向きを一致させ、後続ノートブックで解釈可能な診断列を保持するため。
  - 影響範囲: `scripts/generate_frontier_demo_data.py`, `tests/test_frontier_demo_data.py`, `DESIGN_BLUEPRINT.md`

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run ruff check scripts/generate_frontier_demo_data.py tests/test_frontier_demo_data.py DESIGN_BLUEPRINT.md HISTORY.md` を実施し、`.md` は ruff 対象外のため Python 対象に絞って再実行した。
- `UV_CACHE_DIR=.uv_cache uv run ruff check scripts/generate_frontier_demo_data.py tests/test_frontier_demo_data.py` を通過。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_frontier_demo_data.py` を実施（`2 passed`）。
- `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_frontier_demo_data.py` を実施し、`data/demo/frontier/train_eval.csv`（12,000行）と `data/demo/frontier/latest.csv`（6,000行）を生成。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施（`766 passed, 15 deselected`）。

### 2026-02-22（作業/PR: phase35-1-1-plus-staged-preview-migration）
**背景**
- Phase35.1 以降は既存 quick_reference を壊さず段階移行する方針が必要で、preview 導線とテスト契約を分離して固定する必要があった。
- Notebook 実行時の外部依存を排除するため、`data/` ローカルCSVの生成・検証フローを先に確定する必要があった。

**変更内容**
- `scripts/generate_phase263_notebooks.py` に `--target {legacy,phase35_preview,all}` を追加し、Phase35 preview 生成を切り替え可能にした。
- `scripts/generate_phase35_preview_notebooks.py` を追加し、`notebooks/quick_reference_phase35/reference_{01..04}_*.ipynb` を生成・実行可能にした（出力先: `examples/out/phase35_*`）。
- `scripts/prepare_phase35_data.py` と `data/{ames_housing,titanic,penguins,bike_sharing}.csv` を追加し、ローカルデータ生成 + 契約検証（列/件数/型）を導入した。
- diagnostics を拡張し、`binary_metrics.top5_pct_positive`、`multiclass_metrics` 4指標、`plot_confusion_matrix`、`plot_roc_multiclass` を追加した。
- Phase35 専用テストを追加（`tests/test_phase35_data_contract.py`, `tests/test_phase35_quickref_structure.py`, `tests/test_phase35_quickref_paths.py`, `tests/test_phase35_notebook_execution_outputs.py`）。
- `notebooks/reference_index.ipynb` と `README.md` に Legacy + Phase35 preview 併存方針、Phase35.5 cutover 方針を追記した。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Phase35.1.1〜35.4 は `notebooks/quick_reference_phase35/` を preview 導線として運用し、Legacy quick_reference は維持する。
  - 理由: 既存運用への影響を最小化しつつ、Notebook 構造・データ契約・可視化刷新を段階適用するため。
  - 影響範囲: notebooks, scripts, tests, README
- Decision: confirmed（確定）
  - 内容: Phase35 preview の 01〜04 Notebook、ローカルCSVデータ契約、診断API拡張、回帰テスト追加を実装し、cutover を Phase35.5 へ先送りする方針を文書化した。
  - 理由: Stable API / legacy quick_reference を維持したまま、実行可能な preview と検証導線を確立できたため。
  - 影響範囲: `scripts/generate_phase263_notebooks.py`, `scripts/generate_phase35_preview_notebooks.py`, `scripts/prepare_phase35_data.py`, `src/veldra/diagnostics/*`, `notebooks/quick_reference_phase35/*`, `tests/test_phase35_*`, `README.md`, `DESIGN_BLUEPRINT.md`, `HISTORY.md`

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run python scripts/prepare_phase35_data.py` を実施し、`data/*.csv` を生成・契約検証した。
- `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_phase263_notebooks.py --target phase35_preview` を実施し、preview Notebook 4本を実行済み出力付きで生成した。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_diagnostics_metrics.py tests/test_diagnostics_plots.py tests/test_phase35_data_contract.py tests/test_phase35_quickref_structure.py tests/test_phase35_quickref_paths.py` を実施し、対象テストを通過した。

### 2026-02-22（作業/PR: phase35-5a-preview-uc05-uc08）
**背景**
- Phase35 preview は 01〜04 まで先行実装済みだったが、frontier / causal / artifact-evaluate の主要導線（05〜08）が未実装だった。
- cutover を遅らせる段階移行方針を維持したまま、preview 側で 05〜08 を完結させる必要があった。

**変更内容**
- `scripts/generate_phase35_preview_notebooks.py` を拡張し、`reference_05_frontier_fit_evaluate.ipynb`、`reference_06_dr_estimate.ipynb`、`reference_07_drdid_estimate.ipynb`、`reference_08_artifact_evaluate.ipynb` を新規生成対象に追加した。
- `reference_05` で `latest_artifact_path.txt` を出力し、`reference_08` の artifact 探索順（marker -> latest artifact）を固定した。
- `scripts/prepare_phase35_data.py` に `--fetch-snapshots` を追加し、`data/lalonde.csv` と `data/cps_panel.csv` の取得・正規化フロー、および `data/phase35_sources.json`（URL/sha/取得時刻等）管理を追加した。
- `data/lalonde.csv`、`data/cps_panel.csv`、`data/phase35_sources.json` を追加し、`check-only` 契約で検証できる状態にした。
- `tests/test_phase35_*` を 05〜08 対応へ拡張し、構造・パス・出力物・データ契約を更新した。
- `notebooks/reference_index.ipynb`、`README.md`、`DESIGN_BLUEPRINT.md`、`HISTORY.md` を更新し、Phase35 preview 進捗（01〜08）と cutover 見送り継続を記録した。

**決定事項**
- Decision: provisional（暫定）
  - 内容: `prepare_phase35_data.py --fetch-snapshots` は公開URL取得を試行し、失敗時は例外で停止する（暗黙フォールバックなし）。
  - 理由: データ出所の監査性を優先し、取得失敗時の不透明な代替データ混入を避けるため。
  - 影響範囲: `scripts/prepare_phase35_data.py`, `data/phase35_sources.json`
- Decision: confirmed（確定）
  - 内容: Phase35 preview の対象を `reference_01`〜`reference_08` へ拡張し、frontier/causal/artifact 再評価の一連シナリオを preview 側で実行可能にした。
  - 理由: Legacy 非破壊を維持しながら、Phase35.5A の実利用導線と検証導線を確立できたため。
  - 影響範囲: `scripts/generate_phase35_preview_notebooks.py`, `notebooks/quick_reference_phase35/*`, `examples/out/phase35_uc0[1-8]_*`, `tests/test_phase35_*`, docs

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run python scripts/prepare_phase35_data.py --check-only` を実施し、`data/{ames_housing,titanic,penguins,bike_sharing,lalonde,cps_panel}.csv` と `data/phase35_sources.json` の契約を確認した。
- `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_phase263_notebooks.py --target phase35_preview` を実施し、`reference_01`〜`reference_08` の実行済み notebook と `examples/out/phase35_uc0[1-8]_*/summary.json` を生成した。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_phase35_data_contract.py tests/test_phase35_quickref_structure.py tests/test_phase35_quickref_paths.py tests/test_phase35_notebook_execution_outputs.py -m notebook_e2e` を実施（対象テストを通過）。

### 2026-02-22（作業/PR: phase35-5b-preview-uc09-uc13）
**背景**
- Phase35 preview は `reference_01`〜`reference_08` まで実装済みで、残る tuning 系シナリオ（09〜13）が未実装だった。
- 段階移行方針を維持したまま、preview 側で 13 本のシナリオを完結させる必要があった。

**変更内容**
- `scripts/generate_phase35_preview_notebooks.py` を拡張し、`reference_09_binary_tune_evaluate.ipynb`、`reference_10_timeseries_tune_evaluate.ipynb`、`reference_11_frontier_tune_evaluate.ipynb`、`reference_12_dr_tune_estimate.ipynb`、`reference_13_drdid_tune_estimate.ipynb` を生成対象に追加した。
- Setup import に `tune` を追加し、各 notebook で `tune -> best_params反映 -> fit/estimate_dr` の 1セル1処理フローを実装した（`n_trials=3`, `resume=true`, `preset=standard`）。
- `reference_08_artifact_evaluate.ipynb` の artifact 探索順を `UC-11 marker -> UC-5 marker -> UC-11 artifacts latest -> UC-5 artifacts latest` へ更新した。
- `tests/test_phase35_quickref_structure.py` を `reference_01`〜`reference_13` 対応へ拡張した。
- `tests/test_phase35_quickref_paths.py` に `reference_09`〜`reference_13` と UC-8 新探索順の断片検証を追加した。
- `tests/test_phase35_notebook_execution_outputs.py` に `UC-9`〜`UC-13` の `summary.json`・CSV列契約（`tuning_trials.csv` を含む）を追加した。
- `notebooks/reference_index.ipynb`、`README.md`、`DESIGN_BLUEPRINT.md` を更新し、Phase35 preview 進捗を `01`〜`13` へ更新した。

**決定事項**
- Decision: provisional（暫定）
  - 内容: tuning 系 notebook（UC-9〜13）の `tuning.n_trials` は一律 `3` に固定し、CI/実行安定性を優先する。
  - 理由: notebook_e2e の時間・ばらつきを抑えつつ、導線契約を先に確定するため。
  - 影響範囲: `scripts/generate_phase35_preview_notebooks.py`, `notebooks/quick_reference_phase35/reference_{09,10,11,12,13}_*.ipynb`
- Decision: confirmed（確定）
  - 内容: Phase35 preview の対象を `reference_01`〜`reference_13` に拡張し、tuning 系を含む全 quick reference を preview 側で生成・実行・検証可能にした。
  - 理由: Legacy 非破壊を維持したまま、Phase35 の全シナリオを単一の生成/テスト契約で運用可能になったため。
  - 影響範囲: `scripts/generate_phase35_preview_notebooks.py`, `notebooks/quick_reference_phase35/*`, `examples/out/phase35_uc0[1-9]_*`, `examples/out/phase35_uc1[0-3]_*`, `tests/test_phase35_*`, docs

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run python scripts/prepare_phase35_data.py --check-only` を実施し、必要データ契約を確認した。
- `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_phase263_notebooks.py --target phase35_preview` を実施し、`reference_01`〜`reference_13` を実行済みで生成した。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施し、回帰ゲートを確認した。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_phase35_notebook_execution_outputs.py -m notebook_e2e` を実施し、Phase35 notebook 出力契約を確認した。

### 2026-02-22（作業/PR: phase35-completion-hard-cutover-timeseries-cs）
**背景**
- Phase35 preview（`quick_reference_phase35/reference_01..13`）は完成していたが、`quick_reference` 本線は phase26 系のままで完了定義を満たしていなかった。
- 時系列 notebook（UC-4/UC-10）の生成設定が `kfold` で、`split.type='timeseries'` 契約と不一致だった。
- timeseries split では先頭学習区間が OOF 未スコアになり得るが、コアが全件 OOF 必須だったためエラー化していた。

**変更内容**
- `src/veldra/modeling/_cv_runner.py` を更新し、`split.type='timeseries'` のみ partial OOF を許容、非timeseriesは従来どおり欠損 OOF をエラー継続にした。
- `training_history` へ `oof_total_rows` / `oof_scored_rows` / `oof_coverage_ratio` を追加した。
- `src/veldra/modeling/binary.py` を更新し、timeseries 時の calibration / threshold 最適化 / mean metrics を OOF 有効行ベースで計算するよう変更した。
- `scripts/generate_phase35_preview_notebooks.py` を更新し、`reference_04` / `reference_10` を `split.type='timeseries'` + `time_col='dteday'` へ修正した。
- 同 generator に出力先切替（`quick_reference` / `quick_reference_phase35`）を追加した。
- `scripts/generate_phase263_notebooks.py` に `--target phase35_main` を追加し、default を `phase35_main` に変更した。
- Hard cutover を実施し、`notebooks/quick_reference/` を Phase35 `reference_01..13` へ置換した。
- 旧 quick reference は `notebooks/quick_reference_legacy/phase26/` へ退避した。
- `tests/test_quickref_*` / `tests/test_notebook_execution_*` / `tests/test_notebook_reference_ab_contract.py` / `tests/e2e_playwright/conftest.py` を本線=Phase35前提へ更新した。
- `README.md` / `DESIGN_BLUEPRINT.md` / `notebooks/reference_index.ipynb` を Phase35 本線化方針へ更新した。

**決定事項**
- Decision: provisional（暫定）
  - 内容: Hard cutover（`quick_reference` 本線化）と timeseries partial OOF 許容を同一フェーズで適用する。
  - 理由: Phase35 DoD（本線01〜13統一 + timeseries split契約）を同時に満たすため。
  - 影響範囲: modeling core, notebook generators, quick reference tests, docs/index
- Decision: confirmed（確定）
  - 内容: 2026-02-22 時点で `quick_reference` は Phase35 01〜13 へ切替完了し、legacy は `quick_reference_legacy/phase26` へ退避済み。timeseries は partial OOF 契約で学習/評価可能になった。
  - 理由: 非timeseriesの厳格契約を維持したまま、時系列 split の実データ挙動と notebook 契約の不一致を解消できたため。
  - 影響範囲: `src/veldra/modeling/{_cv_runner.py,binary.py}`, `scripts/generate_{phase35_preview,phase263}_notebooks.py`, `notebooks/{quick_reference,quick_reference_legacy,reference_index}.ipynb`, `tests/test_quickref_*`, `tests/test_notebook_execution_*`, `tests/test_notebook_reference_ab_contract.py`, `tests/e2e_playwright/conftest.py`, `README.md`, `DESIGN_BLUEPRINT.md`, `HISTORY.md`

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_phase263_notebooks.py --target phase35_preview` を実施。
- `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_phase263_notebooks.py --target phase35_main` を実施。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_split_time_series.py tests/test_regression_internal.py tests/test_binary_internal.py tests/test_multiclass_internal.py tests/test_frontier_internal.py` を実施。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施。

### 2026-02-22（作業/PR: phase35-5c-dedup-english-standardization）
**背景**
- Hard cutover 後も `quick_reference_phase35` の重複導線と関連契約が残っており、canonical 運用の複雑さを増やしていた。
- `reference_01`〜`reference_13` の markdown が日本語混在で、Notebook言語標準が未固定だった。

**変更内容**
- `scripts/generate_phase35_preview_notebooks.py` の section 見出し/表示文言を英語へ統一し、`reference_01`〜`reference_13` の英語Notebook生成を標準化した。
- `scripts/generate_phase263_notebooks.py` で `--target phase35_preview` を deprecated alias とし、内部的に canonical（`quick_reference`）へルーティングするよう変更した。
- `--target all` を `legacy + canonical` のみ生成する挙動へ整理し、重複 preview セット生成を停止した。
- `tests/test_generate_phase263_notebooks_targets.py` を追加し、preview alias の warning 出力と canonical ルーティング契約を固定した。
- `tests/test_quickref_structure.py` に「Phase35 quick_reference markdown は日本語非含有（英語標準）」テストを追加した。
- `notebooks/reference_index.ipynb` を更新し、preview一覧を廃止して compatibility note（deprecated alias）へ置換した。
- `README.md` / `DESIGN_BLUEPRINT.md` を更新し、canonical-only 方針・英語Notebook標準・alias互換ウィンドウを明文化した。

**決定事項**
- Decision: provisional（暫定）
  - 内容: `phase35_preview` alias は Phase35.5C では維持するが、次フェーズ（35.5D）で廃止判断可能な状態まで de-dup を先行する。
  - 理由: 既存運用への影響を抑えつつ、CLI/テスト/ドキュメントの canonical 化を段階適用するため。
  - 影響範囲: generation CLI, notebook index, quickref tests, docs
- Decision: confirmed（確定）
  - 内容: quick reference の canonical は `notebooks/quick_reference/` のみとし、`reference_01`〜`reference_13` は英語Notebook標準を適用した。
  - 理由: 重複資産を必須契約から外し、運用コストを下げながら互換性（legacy archive / preview alias）を維持できるため。
  - 影響範囲: `scripts/generate_{phase35_preview,phase263}_notebooks.py`, `notebooks/quick_reference/reference_{01..13}_*.ipynb`, `notebooks/reference_index.ipynb`, `tests/test_{quickref_structure,generate_phase263_notebooks_targets}.py`, `README.md`, `DESIGN_BLUEPRINT.md`, `HISTORY.md`

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_phase263_notebooks.py --target phase35_preview --no-execute` を実施し、deprecation warning と canonical 生成経路を確認。
- `UV_CACHE_DIR=.uv_cache uv run python scripts/generate_phase263_notebooks.py --target phase35_main` を実施し、`reference_01`〜`reference_13` を実行済み再生成した。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_notebook_execution_outputs.py -m notebook_e2e` を実施。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_phase35_notebook_execution_outputs.py -m notebook_e2e` を実施（retained suite）。

### 2026-02-22（作業/PR: quickref-phase-name-clarification）
**背景**
- `phase` 接頭辞のファイル名が増え、責務（正準/互換/履歴）の判別が難しくなっていた。
- 既存の運用コマンド互換は維持したまま、命名と役割の可読性を上げる必要があった。

**変更内容**
- 正準スクリプトを追加:
  - `scripts/generate_quick_reference_notebooks.py`
  - `scripts/generate_quick_reference_notebook_specs.py`
  - `scripts/prepare_quick_reference_data.py`
- phase-prefix スクリプト名を廃止（正準スクリプトへ一本化）。
- データ manifest を正準名へ移行:
  - `data/quick_reference_sources.json`（正準）
  - `phase35_sources.json` は互換読み取りのみ
- テスト/fixture を目的ベース命名へ整理:
  - `tests/test_quick_reference_generator_targets.py`
  - `tests/test_quick_reference_data_contract.py`
  - `tests/test_quick_reference_output_contracts.py`
  - `tests/test_runconfig_metric_extensions.py`
  - `tests/test_tuning_metric_aliases.py`
  - `tests/test_training_output_parity.py`
  - `tests/fixtures/training_output_parity/`
- Legacy notebook 退避先を `notebooks/quick_reference_legacy/archive_2025/` へ改名。
- GUI関連の phase-prefix テストファイルを意味ベース名へ改名。
- `README.md` と `DESIGN_BLUEPRINT.md` に正準名と互換ラッパーの役割を明記。

**決定事項**
- Decision: confirmed（確定）
  - 内容: phase-prefix の実ファイル名を排除し、quick-reference 系の意味ベース命名へ統一した。
  - 理由: 開発者がファイル名だけで責務を理解できる状態を保証するため。
  - 影響範囲: `scripts/*quick_reference*.py`, `tests/test_*`, `tests/fixtures/*`, `notebooks/quick_reference_legacy/archive_2025/*`, `data/quick_reference_sources.json`, `README.md`, `DESIGN_BLUEPRINT.md`, `HISTORY.md`

### 2026-02-22（作業/PR: phase35-test-gap-closure-g1-g2-g3）
**背景**
- `DESIGN_BLUEPRINT.md` の「23 Phase35 テスト拡充計画」は未実施状態で、G1/G2/G3 の3ギャップが残っていた。
- `evaluate()` は Phase35 で導入したメトリクス関数を内部で再利用しておらず、公開評価結果に新指標が反映されていなかった。

**変更内容**
- `src/veldra/api/runner.py` の `evaluate()` を拡張し、既存キーを維持したまま以下を加算した。
  - regression: `huber`
  - binary: `top5_pct_positive`
  - multiclass: `balanced_accuracy`, `brier_macro`, `ovr_roc_auc_macro`, `average_precision_macro`
- multiclass では評価データのクラス欠落時に評価を継続し、計算不能な追加指標のみ省略する挙動を導入した（基本3指標は維持）。
- G1 テストを追加/更新:
  - `tests/test_regression_evaluate_metrics.py`
  - `tests/test_binary_evaluate_metrics.py`
  - `tests/test_multiclass_evaluate_metrics.py`
  - `tests/test_evaluate_config_path.py`
- G2 テストを追加:
  - `tests/test_diagnostics_plots.py` に package-level import 検証を追加。
- G3 テストを追加:
  - `tests/test_training_history.py` に OOF coverage キー roundtrip 検証を追加。
- examples 評価契約を後方互換のため必須キー包含へ更新:
  - `tests/test_examples_evaluate_demo_artifact.py`
  - `tests/test_examples_evaluate_demo_multiclass_artifact.py`
- `DESIGN_BLUEPRINT.md` Section 23 を実施済みへ更新し、検証結果を同期した。

**決定事項**
- Decision: confirmed（確定）
  - 内容: `evaluate()` の Phase35 指標は加算的に公開し、既存キーは維持する。
  - 理由: Stable API 互換を維持しつつ、Phase35 の指標後退を統合テストで検知可能にするため。
  - 影響範囲: `src/veldra/api/runner.py`, `tests/test_*evaluate*`, `tests/test_evaluate_config_path.py`
- Decision: confirmed（確定）
  - 内容: multiclass の評価データが一部クラス欠落の場合は evaluate を失敗させず、計算不能な追加指標のみ返却から省略する。
  - 理由: 現場の評価サンプル分布差による過剰失敗を防ぎ、既存の基本指標導線を維持するため。
  - 影響範囲: `src/veldra/api/runner.py`, `tests/test_multiclass_evaluate_metrics.py`

**検証結果**
- `UV_CACHE_DIR=.uv_cache uv run ruff check src/veldra/api/runner.py tests/test_regression_evaluate_metrics.py tests/test_binary_evaluate_metrics.py tests/test_multiclass_evaluate_metrics.py tests/test_diagnostics_plots.py tests/test_training_history.py tests/test_examples_evaluate_demo_artifact.py tests/test_examples_evaluate_demo_multiclass_artifact.py tests/test_evaluate_config_path.py` を通過。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_diagnostics_plots.py tests/test_training_history.py` を実施（`5 passed`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_regression_evaluate_metrics.py tests/test_binary_evaluate_metrics.py tests/test_multiclass_evaluate_metrics.py tests/test_evaluate_config_path.py` を実施（`16 passed, 1 warning`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q tests/test_examples_evaluate_demo_artifact.py tests/test_examples_evaluate_demo_binary_artifact.py tests/test_examples_evaluate_demo_multiclass_artifact.py` を実施（`6 passed`）。
- `UV_CACHE_DIR=.uv_cache uv run pytest -q -m "not gui_e2e and not notebook_e2e"` を実施（`776 passed, 16 deselected, 1 warning`）。

### 2026-02-22（作業/PR: design-blueprint-phase23-35-summary-reorg）
**背景**
- `DESIGN_BLUEPRINT.md` の「実装済みフェーズ要約」は Phase23/24 の要点、未実装ギャップ要約、GUI実装要約、将来拡張要約が分散しており、参照時にセクション横断が必要だった。

**変更内容**
- `DESIGN_BLUEPRINT.md` Section 4 を再編し、以下を同一セクションへ統合した。
  - Phase 1〜22 の全体サマリー
  - Phase 23〜35 の実装済みサマリー（時系列）
  - 未実装ギャップ要約（P0/P1 + Phase35 test gap解消状況）
  - GUI実装要約（adapter原則 + 現行機能 + 運用契約）
  - 将来の拡張ポイント要約

**決定事項**
- Decision: confirmed（確定）
  - 内容: 「実装済みフェーズ要約」は Phase23〜35 を主軸に、未実装ギャップ・GUI実装・将来拡張を同居させる統合要約を正とする。
  - 理由: フェーズ進捗と今後の着手点を単一セクションで把握できるようにし、設計書の参照コストを下げるため。
  - 影響範囲: `DESIGN_BLUEPRINT.md` Section 4

**検証結果**
- ドキュメント再編のみ（コード/テスト変更なし）。

### 2026-02-22（作業/PR: design-blueprint-summary-dedup）
**背景**
- Section 4 に追加した要約と Section 5/6/7 の詳細記述が重複し、参照時に同義内容を複数箇所で読む必要があった。

**変更内容**
- `DESIGN_BLUEPRINT.md` Section 4 を再要約し、Phase23〜35 をレンジ単位で簡潔化した。
- Section 4 の「未実装ギャップ/GUI実装/将来拡張」は詳細の再掲をやめ、Section 5/6/7 参照中心に統一した。
- 重複していた詳細列挙（GUI機能一覧・将来拡張一覧）の要約を削除した。

**決定事項**
- Decision: confirmed（確定）
  - 内容: 実装済みフェーズ要約（Section 4）は「進捗把握に必要な最小要約」に限定し、詳細は後続セクションを正とする。
  - 理由: 設計書内の重複を減らし、更新時の整合性維持コストを下げるため。
  - 影響範囲: `DESIGN_BLUEPRINT.md` Section 4

**検証結果**
- ドキュメント再編のみ（コード/テスト変更なし）。

### 2026-02-22（作業/PR: design-blueprint-numbering-and-current-state-sync）
**背景**
- `DESIGN_BLUEPRINT.md` のセクション番号が `4 -> 8 -> 9` へ飛んでおり、参照整合が崩れていた。
- 「現在の実装能力」に evaluate の最新契約（binary `precision_at_k` 条件付き返却、multiclass 追加指標の条件付き返却）が未反映だった。

**変更内容**
- `DESIGN_BLUEPRINT.md` の章構成を再整理し、`5`（未実装ギャップ）/`6`（GUI実装）/`7`（将来拡張）を再導入して連番を復元した。
- Section 3.2 を最新実装へ同期:
  - binary: `train.top_k` 指定時の `precision_at_k` 返却を明記。
  - multiclass: `ovr_roc_auc_macro` / `average_precision_macro` は計算可能時のみ返却し、欠落クラス時は省略する契約を明記。
- Section 4 は要約を維持し、詳細は Section 5〜7 を正とする導線を固定した。

**決定事項**
- Decision: confirmed（確定）
  - 内容: `DESIGN_BLUEPRINT.md` は「連番整合 + 現行実装契約の正確性」を優先し、要約と詳細の責務を分離する。
  - 理由: 設計書の可読性と参照信頼性を維持し、運用時の誤読を防ぐため。
  - 影響範囲: `DESIGN_BLUEPRINT.md` Section 3〜9

**検証結果**
- ドキュメント更新のみ（コード/テスト変更なし）。
