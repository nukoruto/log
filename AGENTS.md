# AGENTS.md — session-lab 開発エージェント向け 制約・手順・I/F仕様（単一ファイル版）

> 目的：本ドキュメントは、コード自動生成エンジン（例: Codex, GPT 等）および開発者が**厳密に従う**ための制約・手順・I/F 仕様を、今回の **5パート構成**（Δt抽出 / シナリオ設計 / ログ生成 / 深層学習 / MATLAB 連携）に合わせて、1ファイルに統合したものです。  
> ここに明記した**入出力契約**と**検証方法**により、並行開発を可能にし、リポジトリの肥大化・混線を防ぎます。

---
## 0. 作業対象：ファイル / フォルダ概要（Monorepo 構成）

```
log/
  packages/
    ds_contract/       # A: 公開DS→契約CSV→セッション分割→Δtロバスト化（CLI: ds-contract）
    scenario_design/   # B: 正常/異常シナリオ設計（CLI: scenario-design）
    log_generator/     # C: 設計に基づく擬似ログ生成（CLI: log-generator）
    models_lstm/       # D: LSTM 学習・推論・スコア（CLI: models-lstm）
    matlab_bridge/     # E: Simulink 連携 .mat 出力（CLI: matlab-bridge）
  methods_5parts/      # 実験手法・契約・方針（md）
  matlab/              # 可視化・指標算出の .m / モデル（任意）
  scripts/             # A→B→C→D→E を繋ぐユーティリティ
  .env.example         # GPU_MODE=rtx6000|rtx4060|cpu 等
  README.md            # 全体 README
  AGENTS.md            # （本ファイル）
```

- **Python 3.11+ / PyTorch 2.x / MATLAB R2025b+**。ネットワークアクセスは既定で禁止（ローカル入出力のみ）。
- タイムゾーンは **UTC** 固定。ログ列は**契約 CSV（9列）**を最小真実源とする。

---
## 1. 5パートの **入出力** と **契約**（並行開発のためのI/F）

### A. Δt抽出（`packages/ds_contract`）
- **入力**：公開データセットの CSV（列名は `map.yaml` で契約列にマッピング）。  
- **出力**：
  - `contract.csv`（契約9列・列順固定・UTC）
  - `sessioned.csv` ＋ `meta_session.json`（`ΔT_session_max`=log空間Otsu/単峰時は肘法）
  - `deltified.csv` ＋ `meta_dt.json`（`Δt`→`log`→`median/MAD z`→`clip ±5`）
- **必須列（契約9列）**：`timestamp_utc, uid, session_id, method, path, referer, user_agent, ip, op_category`  
- **制約**：欠損/非UTC/逆順は**非0終了**。seed により決定論。  
- **CLI 例**：
  ```bash
  ds-contract validate input.csv --map map.yaml --out contract.csv
  ds-contract sessionize contract.csv --out sessioned.csv --meta meta_session.json
  ds-contract deltify sessioned.csv --out deltified.csv --meta meta_dt.json
  ```

### B. シナリオ設計（`packages/scenario_design`）
- **入力**：`deltified.csv`  
- **出力**：
  - `stats.pkl`（op遷移・カテゴリ別Δt分布の統計）
  - `scenario_spec.json`（正常・異常の設計図：長さ/ユーザ/初期分布/遷移/Δt分布/異常注入方針）
- **異常設計**：`time(mode=propagate|local, p, scale)`, `order(p)`, `unauth(p)`, `token_replay(p)` 等を JSON 化。  
- **制約**：JSON Schema に準拠、seed で決定論。  
- **CLI 例**：
  ```bash
  scenario-design fit deltified.csv --out stats.pkl
  scenario-design plan --stats stats.pkl --seed 42 --out scenario_spec.json     --anom time(mode=propagate,p=0.02) --anom order(p=0.01) --anom unauth(p=0.005)
  ```

### C. ログ生成（`packages/log_generator`）
- **入力**：`scenario_spec.json`（＋任意 `catalog.yaml`：op→{method,path,referer,ua} 分布）  
- **出力**：
  - `normal.csv`（契約CSV、異常注入なし）
  - `anom.csv`（契約CSV、注入あり）
  - `run_meta.json`（seed, algo_version, spec_sha256）
  - `audit.jsonl`（注入理由と位置の逐語記録）
- **制約**：契約9列・UTC・seed 決定論。`audit.jsonl` 不整合は**非0終了**。  
- **CLI 例**：
  ```bash
  log-generator run --spec scenario_spec.json --seed 42     --normal normal.csv --anom anom.csv --audit audit.jsonl --meta run_meta.json
  ```

### D. 深層学習（`packages/models_lstm`）
- **入力**：`normal.csv`（学習/検証）、`anom.csv`（評価）  
- **出力**：
  - `runs/exp*/best.ckpt`, `metrics.json`
  - `scored.csv`（`anom_score`, `flag_cls`, `flag_dt` 等）
- **モデル**：入力 `[embed(op_category), z_clipped(Δt), …]`、損失 `CE + λ·Huber/MAE`。  
- **GPU**：`.env` の `GPU_MODE=rtx6000|rtx4060|cpu` で自動選択（名前一致→`cuda:0`→CPU）。  
- **CLI 例**：
  ```bash
  models-lstm train --normal normal.csv --val normal.csv --out runs/exp1 --seed 42
  models-lstm score --model runs/exp1/best.ckpt --in anom.csv --out scored.csv
  ```

### E. MATLAB 連携（`packages/matlab_bridge`）
- **入力**：`scored.csv`  
- **出力**：`matlab/ref.mat`（From Workspace で読める `struct/timetable`）。  
  変数：`ref, y_lstm, y_pid, t`。指標（IAE/ISE/ITAE, Tr, Ts, OS, ESS）算出は MATLAB 側スクリプトで実施。  
- **CLI 例**：
  ```bash
  matlab-bridge export --in scored.csv --out matlab/ref.mat
  ```

---
## 2. 貢献内容とスタイルガイドライン（開発者・エージェント共通）

### 2.1 コーディング規約
- **PEP8**、**型注釈**必須、**docstring（Google形式）**。  
- フォーマッタ **black**、リンタ **ruff**、型チェック **mypy** を通すこと。  
- 例外は `ValueError/RuntimeError/IOError` を中心に**原因と対処**を明示。  
- ロギングは **JSON構造**で stdout（`INFO` 以上）。

### 2.2 乱数・再現性
- Python/NumPy/PyTorch の seed を統一。`--seed` 必須。  
- 決定論モード（cuDNN 等）を可能な限り有効化。実行メタ（seed, algo_version, 入力SHA256）を JSON 保存。

### 2.3 コミット / PR / 変更粒度
- **1 PR = 1 機能単位**（A〜Eのどれかのサブ機能）。  
- コミットメッセージ：`feat(A): …` / `fix(D): …` / `docs(*)` / `refactor(C)`。  
- PR は **差分最小**、**自己完結**（README 追補含む）。

### 2.4 自動生成エンジン（Codex/GPT）専用 ルール
- **出力は**「unified diff」**または**「新規ファイルの完全本文」**のみ**。前置きは**3行以内の要約**。  
- 既存編集時は `diff --unified` でファイルパスと `@@` ハンクを必ず含む。  
- **テストを先に**用意し、その後に実装を提示（TDD 指向）。

---
## 3. 旧 logserver → 新 5パート への移行マッピング

| 旧ディレクトリ/機能 | 新パート | 新ディレクトリ | 備考 |
|---|---|---|---|
| `contract/`（CSV契約・辞書） | A | `packages/ds_contract` | 契約CSV/セッション分割/Δtロバスト化を統合 |
| `collector/`（シミュレーション/収集） | B/C | `packages/scenario_design`, `packages/log_generator` | 設計と生成を分離 |
| `trainer/`（前処理・学習・評価） | D | `packages/models_lstm` | LSTM 学習/推論/スコアを集約 |
| `simulink/`（.slx/.m） | E | `packages/matlab_bridge`, `matlab/` | .mat ブリッジをPython側で提供 |
| `artifacts/`, `outputs/` | 共通 | ルート直下（Git 管理外） | `artifacts/`=生成CSV、`outputs/`=学習成果 |

> **方針**：**Node.js 依存は撤去**（将来用に別レポで保全）。本レポは**Python中心**で完結。

---
## 4. 変更の検証方法（lint / test / E2E）

### 4.1 Lint / Format / Type
```bash
ruff check .
black --check .
mypy packages
```

### 4.2 単体テスト（pytest）
- 目標カバレッジ **80%+**。  
- 代表テスト：
  - A: Otsu/肘法のしきい値推定、Δtロバスト z、一連 CLI の終了コード 0
  - B: `scenario_spec.json` の Schema 検証、seed 再現
  - C: `audit.jsonl` 整合、normal/anom の契約準拠
  - D: `devices`（GPU_MODE 選択）モック、`export-mat` の最小構造検証、学習の smoke test
  - E: `.mat` 読み込み可能性（scipy.io.loadmat で構造チェック）

### 4.3 E2E（Quickstart 再現）
```bash
python -m pip install -e packages/ds_contract -e packages/scenario_design -e packages/log_generator -e packages/models_lstm -e packages/matlab_bridge

ds-contract validate data/public.csv --map map.yaml --out contract.csv
ds-contract sessionize contract.csv --out sessioned.csv --meta meta_session.json
ds-contract deltify sessioned.csv --out deltified.csv --meta meta_dt.json

scenario-design fit deltified.csv --out stats.pkl
scenario-design plan --stats stats.pkl --seed 42 --out scenario_spec.json   --anom time(mode=propagate,p=0.02) --anom order(p=0.01) --anom unauth(p=0.005)

log-generator run --spec scenario_spec.json --seed 42   --normal normal.csv --anom anom.csv --audit audit.jsonl --meta run_meta.json

models-lstm train --normal normal.csv --val normal.csv --out runs/exp1 --seed 42
models-lstm score --model runs/exp1/best.ckpt --in anom.csv --out scored.csv

matlab-bridge export --in scored.csv --out matlab/ref.mat
```

### 4.4 CI チェック（推奨 Make タスク）
```make
lint: ; ruff check . && black --check . && mypy packages
test: ; pytest -q
e2e:  ; bash scripts/quickstart.sh
all: lint test e2e
```

---
## 5. 追加ポリシー（セキュリティ / PII / GPU / ログ）

- **PII/セキュリティ**：JWT 生値は保存不可。`uid` は HMAC-SHA256 等で擬似化（復号不可）。  
- **GPU 切替**：`.env` の `GPU_MODE=rtx6000|rtx4060|cpu`。名前一致→`cuda:0`→CPU。  
- **成果物**：各ステージは `meta_*.json` / `run_meta.json` に seed、ハッシュ、閾値、統計を保存。  
- **ログ**：すべての CLI は JSON 構造ログを stdout に出力（開始/完了/入力ハッシュ/種など）。

---
## 6. エージェント向け 実装プロンプト（テンプレ）

> **以下をそのまま投げる**（必要箇所を `${...}` で埋める）。

```
# OBJECTIVE
${短い一文（例）: Create "ds_contract" CLI to normalize CSV, sessionize with Otsu/Elbow, and compute robust Δt features.}

# FILES (変更範囲)
${新規/修正ファイル一覧。tests を含むこと。}

# REQUIREMENTS
- Follow AGENTS.md: A/B/C/D/E interface contracts strictly.
- Contract CSV columns (fixed order): timestamp_utc, uid, session_id, method, path, referer, user_agent, ip, op_category.
- UTC only; non-UTC / missing required columns -> non-zero exit.
- Deterministic with --seed. Save meta JSON with thresholds/stats/input SHA256.
${機能固有の要求（Otsu/Elbow, audit.jsonl, GPU_MODE 等）}

# CLI
${サブコマンド仕様と例}

# TESTS
${pytest シナリオ：何をどう検証するか}

# DONE
- Ruff/Black/Mypy pass. Pytest green (80%+). CLI --help works.
- Only unified diff or full new files. Minimal preamble (<=3 lines).
```

---
## 7. 失敗条件とエラー設計（全パート共通）

- **契約違反**（列欠損/非UTC/スキーマ不一致）：**非0終了** + 明確なエラー文（修正手順を含む）。  
- **seed 未指定**（指定が必須の CLI）：エラーで中断。  
- **JSON Schema 不一致**（`scenario_spec.json`, `run_meta.json`, `meta_*.json` など）：**非0終了**。  
- **監査欠落**（C の `audit.jsonl` に理由なし行）：**非0終了**。

---
## 8. 参考：各パートの責務境界と並行開発のガイド

- **A→B** は `deltified.csv` のみを介す（B は A の内部実装に依存しない）。  
- **B→C** は `scenario_spec.json`（スキーマ固定）を介す。  
- **C→D** は **契約CSV**（normal/anom）を介す。D は C の監査方式に依存しない。  
- **D→E** は `scored.csv` と最小 `.mat` 仕様を介す。E は学習モデル内部に依存しない。

> これにより、**A〜E は完全に並行開発可能**です。各パートは**docxフォルダ下にある契約ファイルだけ**を参照してください。

---
## 9. 用語集（抜粋）
- **契約CSV**：9 列（timestamp_utc, uid, session_id, method, path, referer, user_agent, ip, op_category）。唯一の最低限真実源。
- **シード固定**：乱数の再現性を保証。すべての生成物には seed と入力ハッシュをメタ保存。
- **propagate/local**：時間異常の注入モード。前者は以降全体シフト、後者は局所のみ変更。
- **Simulink 指標**：IAE/ISE/ITAE、Tr、Ts、OS、ESS。

---

この AGENTS.md に**反している変更**はレビューで差し戻します。疑義がある場合は、**契約（入出力）を優先**し、`TODO:` として最小限の仮定を明記して先に進めてください。
