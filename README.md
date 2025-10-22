# session-lab — 実験全体像 / 仕様 / 契約（README 完全版 v1）

> 目的：公開データセットの統計とプロジェクト方針（Δt選定・契約CSV）に基づき、**シミュレーション生成ログ**で LSTM による異常検知を実証し、Simulink で LSTM vs PID を時間領域指標で比較・可視化する。収集サーバ（Node.js）は将来用として切り離し、本レポは **Python中心で完結**させる。

---
## 0. 全体像（Pipeline / Dataflow）

```
[Public DS] ──▶ (A) ds-contract
                 ├─ validate/map → contract.csv
                 ├─ sessionize(ΔT_session_max=Otsu/Elbow) → sessioned.csv + meta_session.json
                 └─ deltify(Δt→log→median/MAD z→clip) → deltified.csv + meta_dt.json

(A)の統計 → (B) scenario-design (設計) ──▶ scenario_spec.json
scenario_spec.json ──▶ (C) log-generator (生成) ──▶ normal.csv, anom.csv, run_meta.json, audit.jsonl

normal.csv / anom.csv ──▶ (D) deep-learning (LSTM) ──▶ runs/exp*/best.ckpt, scored.csv, metrics.json

scored.csv ──▶ (E) matlab-bridge ──▶ matlab/ref.mat ──▶ Simulink (LSTM vs PID 比較)
```

- **A. ds-contract**: 公開データを研究スキーマへ正規化・セッション分割・Δtロバスト化。
- **B. scenario-design**: 正常/異常シナリオの「設計図」（確率・制約・注入方針）を JSON へ定義。
- **C. log-generator**: 設計図に従い、契約CSV形式の**擬似ログ**を決定論的に生成。
- **D. deep-learning**: LSTM で学習・推論・スコアリング（GPU_MODEで RTX6000/4060/CPU 切替）。
- **E. matlab-bridge**: MATLAB/Simulink 取り込み用 `.mat` を出力、時間領域指標で LSTM vs PID を比較。

> 非目標：実サーバ収集、Web UI、Burp自動化、運用監視は含めない（別レポ）。

---
## 1. リポ構成（Monorepo）

```
log/
  packages/
    ds_contract/         # A: 契約CSV・セッション化・Δtロバスト化
    scenario_design/     # B: 正常/異常シナリオ設計（JSON化）
    log_generator/       # C: 設計に基づくCSV生成（normal/anom）
    models_lstm/         # D: LSTM 学習・推論・スコアリング
    matlab_bridge/       # E: .mat エクスポート（From Workspace 用）
  methods_5parts/
    A_data_contract_dt.md
    B_scenario_design.md
    C_log_generator.md
    D_lstm_learning.md
    E_matlab_simulink.md
  matlab/
    export_timeseries.m  # 任意：可視化・指標算出補助
  scripts/
    quickstart.sh        # 例：A→B→C→D→E を一括実行
  .env.example
  README.md (本ファイル)
  AGENTS.md              # codex用規約
```

> 将来 3〜5 レポへ分割しても、**I/F は CSV/JSON 契約**で疎結合化済み。

---
## 2. 契約（I/F）

### 2.1 契約CSV（列順固定）
1. `timestamp_utc` (ISO8601 or UNIX epoch[ms])  
2. `uid` (JWT 等→ハッシュ/サロゲート)  
3. `session_id` (Cookie 等→サロゲート/推定)  
4. `method` (GET/POST/PUT/DELETE…)  
5. `path` (正規化URLパス)  
6. `referer`  
7. `user_agent`  
8. `ip` (将来互換のプレースホルダ；空文字可)  
9. `op_category` (READ/UPDATE/AUTH 等の抽象カテゴリ)

**必須**: `timestamp_utc, uid|jwt, session_id, method, path, referer, user_agent, op_category`。全行 UTC 正規化。欠損行は除外（**非0**終了）。

### 2.2 セッション分割 & Δt
- `(uid, session_id)` が無い/壊れている場合、**ログギャップ**でセッション推定。  
- `ΔT_session_max` は **log 空間 Otsu**（二峰性）で自動選定、単峰時は **肘法**へフォールバック。  
- `Δt(i)=t(i)-t(i-1)`（セッション内昇順）。先頭は `0` または `MASK`（本プロジェクトは `0` で統一）。  
- **ロバスト化**: `log(Δt+ε)`→ median/MAD z → クリップ（±5）。単位はユーザ/（必要なら user×session）。


### 2.3 再現性・メタ
- 全生成器 `--seed` 必須。`run_meta.json`（seed, algo_version, 入力 SHA256）を保存。
- 閾値・統計・ヒストグラム・採択根拠は `meta_*.json` に保存（論文再現可）。

### 2.4 GPU モード
- `.env` の `GPU_MODE` で切替：`rtx6000 | rtx4060 | cpu`。  
- CUDA デバイスを列挙し、**名前一致**で優先（見つからなければ `cuda:0`、無ければ CPU）。

---
## 3. Quickstart（最小例）

```bash
# 依存ライブラリを editable インストール
python -m pip install -e packages/ds_contract -e packages/scenario_design   -e packages/log_generator -e packages/models_lstm -e packages/matlab_bridge

# A: 契約化→セッション化→Δt
ds-contract validate data/public.csv --map map.yaml --out contract.csv
ds-contract sessionize contract.csv --out sessioned.csv --meta meta_session.json
ds-contract deltify sessioned.csv --out deltified.csv --meta meta_dt.json

# B: シナリオ設計（統計を反映）
scenario-design fit deltified.csv --out stats.pkl
scenario-design plan --stats stats.pkl --seed 42 --out scenario_spec.json   --anom time(mode=propagate,p=0.02) --anom order(p=0.01) --anom unauth(p=0.005)

# C: ログ生成（契約CSV形式）
log-generator run --spec scenario_spec.json --seed 42   --normal normal.csv --anom anom.csv --audit audit.jsonl --meta run_meta.json

# D: 学習・推論
models-lstm train --normal normal.csv --val normal.csv --out runs/exp1 --seed 42
models-lstm score --model runs/exp1/best.ckpt --in anom.csv --out scored.csv

# E: MATLAB 連携
matlab-bridge export --in scored.csv --out matlab/ref.mat
```

---
## 4. Done の定義（受け入れ基準）
- **決定論性**: 同一 `--seed` で出力がバイト一致（`sha256sum` で確認）。
- **契約順守**: 列順固定/UTC/必須列チェック、違反は**非0**終了。
- **メタ保存**: 閾値・統計・ヒスト・入力ハッシュ・seed を JSON で保存。
- **テスト**: `pytest -q` 緑（Otsu/肘法、Δtロバスト z、propagate/local、LSTM export など）。
- **Simulink**: `ref.mat` を From Workspace で読み込み、LSTM vs PID の IAE/ISE/ITAE 等が算出可能。

---
## 5. ライセンス / セキュリティ
- JWT 生値は保管しない。`uid` はハッシュ/サロゲート化し、再識別防止。
- 外部共有時は匿名化/脱機微を徹底。
