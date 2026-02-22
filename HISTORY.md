# HISTORY.md
（AIエージェント作業履歴・意思決定ログ / 圧縮版）

- 詳細原本: `HISTORY_ARCHIVE.md`
- 本書の目的: 日々の参照に必要な「要点のみ」を維持する
- 圧縮方針: 旧ログ全文は archive へ退避し、HISTORY は要約を保持する

## ルール
- 1エントリ = 1作業/PR（必要に応じて同日作業を統合）
- Decision は `provisional` / `confirmed` のみを使用
- 詳細手順・長いコマンド列・重複説明は archive 側へ寄せる

## 現在の到達点（2026-02-22 時点）
- Core API（`fit/predict/evaluate/tune/simulate/export/estimate_dr`）は regression/binary/multiclass/frontier を一貫サポート
- Causal は DR / DR-DiD（panel/repeated_cs, binary outcome 含む）を実装
- GUI は adapter 原則を維持しつつ、運用基盤（job queue / cancel-retry 枠 / template / studio）まで実装
- Notebook は quick reference 01〜13 を canonical 化し、timeseries partial OOF 契約を適用
- Phase35 テストギャップ（G1/G2/G3）を解消済み

## 主要 Decision（継続有効）
- `Decision: confirmed`
  - Core は GUI/CLI/API 非依存、RunConfig 共通入口、Artifact 受け渡しを維持する
- `Decision: confirmed`
  - Stable API（`veldra.api.*`）は互換優先、破壊的変更は migration 前提でのみ許容
- `Decision: confirmed`
  - Notebook の canonical は `notebooks/quick_reference/` を正とする
- `Decision: confirmed`
  - timeseries split は partial OOF を許容、非-timeseries は OOF 欠損をエラー維持
- `Decision: confirmed`
  - GUI は adapter 層に限定し、Core ロジックを持たない

---

## Log（圧縮）

### 2026-02-09〜2026-02-11（Phase4〜22 基盤整備）
**要約**
- examples 導線、binary/multiclass/frontier、simulate/export、runconfig validation、artifact roundtrip を段階実装
- evaluate(config, data) 導線と Notebook 契約の初期整備を実施
- config migrate MVP（API/CLI）を導入

**Decision**
- `Decision: confirmed`
  - 内容: examples は再現可能な最小ワークフローを優先
  - 理由: 導入コストと検証コストを同時に下げるため

### 2026-02-12（Phase23: DR-DiD Binary + 最小診断）
**要約**
- `causal.method=dr_did` で `task.type=binary` を正式対応
- `panel` / `repeated_cross_section` の2設計で実行契約を固定
- `overlap_metric`, `smd_max_unweighted`, `smd_max_weighted` の診断返却を標準化

**Decision**
- `Decision: confirmed`
  - 内容: binary DR-DiD の推定量は Risk Difference（ATT）解釈を採用
  - 理由: 2値アウトカムでの一貫した因果解釈を確保するため

### 2026-02-12（Phase24: Causal Tune Balance-Priority）
**要約**
- causal tuning 既定を SE 偏重から balance-priority へ移行
- `tuning.causal_balance_threshold` を導入

**Decision**
- `Decision: confirmed`
  - 内容: causal tuning の品質判定はバランス制約を優先する
  - 理由: 推定の安定性と運用妥当性を確保するため

### 2026-02-12〜2026-02-16（Phase25〜25.9: GUI運用強化 + LightGBM強化）
**要約**
- GUI: 非同期ジョブ（SQLite永続）、best-effort cancel、config migrate 統合を実装
- テスト基盤を DRY 化し、GUI callback テスト方針を整理
- LightGBM 機能強化（param 拡張、top_k、feature weights など）を実装
- 不足テスト補完（Phase25.9）を実施し、必要最小本体修正まで同フェーズで収束

**Decision**
- `Decision: confirmed`
  - 内容: GUI 運用機能は adapter 層で閉じ、Core 契約は維持
  - 理由: Stable API と運用拡張の両立のため

### 2026-02-16〜2026-02-18（Phase26〜26.7: UX/UI・Notebook・Core refactor）
**要約**
- Phase26 本線（UI再構成、Help/Guided、Run/Results 導線）を段階完了
- Notebook 命名整理・契約テスト強化・A/B適用・gui_e2e 安定化を実施
- Phase26.7 で core リファクタ（CV共通化、causal learner 抽象化）を完了

**Decision**
- `Decision: confirmed`
  - 内容: notebook テストは責務ベース命名へ統一し、phase依存命名を廃止
  - 理由: 再編時の保守性と検索性を確保するため

### 2026-02-18（Phase27: Priority Queue + Worker Pool）
**要約**
- GUI ジョブキューへ優先度制御とマルチworker運用を導入

### 2026-02-18（Phase28: Realtime Progress + Streaming Logs）
**要約**
- polling + SQLite 永続で進捗/ログ可視化を安定化（WebSocket/SSE は非採用）

### 2026-02-18（Phase29: Cancel & Retry Recovery）
**要約**
- 協調キャンセル導線と自動リトライ枠（既定手動中心）を整備

**Decision**
- `Decision: confirmed`
  - 内容: retry policy は GUI adapter 側に限定
  - 理由: Core RunConfig 契約を肥大化させないため

### 2026-02-18（Phase30: Config Template Library）
**要約**
- template/preset/localstore/wizard 導線を GUI adapter へ統合

### 2026-02-18（Phase31: Advanced Visualization & Compare）
**要約**
- fold metrics、因果診断、artifact比較、HTML/PDF 出力を導入

### 2026-02-18（Phase32: Performance & Scalability）
**要約**
- サーバー側ページング、AG Grid、DBアーカイブで運用性能を改善

### 2026-02-18（Phase33: GUI Memory Re-optimization & Marker Hardening）
**要約**
- lazy import 契約を固定化し、`gui/gui_e2e/notebook_e2e` 分離を厳密化

### 2026-02-18〜2026-02-19（Phase34: Studio UX）
**要約**
- `/studio` 導入、train/inference/model hub 導線、Guided mode 整理を段階完了

**Decision**
- `Decision: confirmed`
  - 内容: `/studio` 追加で既存 Guided URL は維持（破壊的移行なし）
  - 理由: 既存運用との互換維持のため

### 2026-02-20〜2026-02-22（Phase35: Notebook Mainline 完了）
**要約**
- quick_reference 01〜13 を段階実装し、最終的に `quick_reference/` 本線化
- diagnostics 拡張（metrics/plots）を反映
- timeseries split 契約を partial OOF 許容へ更新
- legacy notebook は archive へ退避し、canonical 導線を一本化
- 英語標準化、重複導線整理、命名整理（phase-prefix 廃止）を完了

**Decision**
- `Decision: confirmed`
  - 内容: canonical quick reference は `notebooks/quick_reference/` のみ
  - 理由: 重複運用コストを下げるため

### 2026-02-22（Phase35 test gap closure: G1/G2/G3）
**要約**
- G1: `evaluate()` の新指標返却を実装 + 統合テスト追加
  - regression: `huber`
  - binary: `top5_pct_positive`
  - multiclass: `balanced_accuracy`, `brier_macro`, `ovr_roc_auc_macro`, `average_precision_macro`
- G2: `veldra.diagnostics` package-level import テスト追加
- G3: `training_history` OOF coverage roundtrip テスト追加
- examples の evaluate 契約を「完全一致」から「必須キー包含」に更新

**Decision**
- `Decision: confirmed`
  - 内容: multiclass は欠落クラス時でも evaluate 継続し、計算不能指標のみ省略
  - 理由: 現場データのクラス偏りで過剰失敗しないようにするため

### 2026-02-22（設計書メンテ）
**要約**
- `DESIGN_BLUEPRINT.md` を再編（Phase23〜35要約、重複削減、番号整合、現状能力同期）
- Section 3 の実装能力契約を最新に更新

**Decision**
- `Decision: confirmed`
  - 内容: 設計書は「要約セクション」と「詳細セクション」の責務分離を維持
  - 理由: 可読性と運用時の整合性維持コストを下げるため

---

## Archive参照
- フル履歴（圧縮前全文）: `HISTORY_ARCHIVE.md`
- 圧縮スナップショット: `Snapshot 2026-02-22 ... (source: HISTORY.md before 1/5 compression)`
