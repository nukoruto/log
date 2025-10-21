# C. ログ自動生成（旧：手法2の生成手順を独立化）

`scenario_spec.json` に基づき、契約CSV形式の **normal/anom** を決定論的に生成する。

## C.1 操作列と時間の生成
乱数 $U_k\sim \mathrm{Uniform}(0,1)$（seed 固定）。
$$
op_{k+1}\sim \mathrm{Categorical}(A_{op_k,\cdot}),\quad
\Delta t_k = F^{-1}_{\Delta t\mid op_k}(U_k),\quad
t_k=t_{k-1}+\Delta t_k. \tag{C.1}
$$
開始時刻 $t_0$ は UTC で固定（`--t0` 指定が無ければ `1970-01-01T00:00:00Z`）。

## C.2 時間異常の注入
- **local**：$\Delta t_k'=\alpha\Delta t_k$（または $\Delta t_k'=\Delta t_k+\delta$），以降は元スケジュール。
- **propagate**：$k$ 以降を積分し直す：
  $$
  t_i' = t_i + \sum_{j=k}^i(\Delta t_j'-\Delta t_j),\quad i\ge k. \tag{C.2}
  $$

## C.3 仕様違反の注入
- 順序逸脱 / 未認証更新 / トークン流用を、確率 $p$ で注入。**各注入は `audit.jsonl` に逐語記録**（index, 種別, パラメータ, 根拠）。

## C.4 出力とCLI
```bash
log-generator run --spec scenario_spec.json --seed 42 \
  --normal normal.csv --anom anom.csv --audit audit.jsonl --meta run_meta.json
```
- `normal.csv`／`anom.csv` は**契約CSV9列・UTC**。  
- `run_meta.json`：`seed`, `algo_version`, `spec_sha256`。

## 検証と失敗条件
- 契約違反（列欠損/非UTC/順序違反の正常混入）→ **非0終了**。  
- 同一 `--seed` で**バイト一致**。`audit.jsonl` に注入行の根拠がなければ失敗。

### 参考文献（C）
[1] RFC 9110; [2] RFC 7519; [3] RFC 6750; [4] RFC 6265;  
[10] Siffer et al., SPOT; [13] NIST Combinatorial Testing; [14] NIST ACTS.
