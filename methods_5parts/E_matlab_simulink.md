# E. MATLAB/Simulink 連携（旧：手法8）

`scored.csv` を `.mat` にエクスポートし、Simulink で **LSTM vs PID** を同一時間軸で比較する。

## E.1 変数とファイル仕様
- 出力：`matlab/ref.mat`（From Workspace が読み込める `struct` or `timetable`）。
- 変数：`ref`（参照系列）, `y_lstm`（LSTM出力）, `y_pid`（PID出力）, `t`（秒）。  
- `t` は単調増加。既定サンプリングは `Δt = median(Δt)` に合わせて補間。

## E.2 指標（離散化）
誤差 $e(t)=r(t)-y(t)$、離散点 $(t_k,e_k)$：
$$
\Delta t_k=t_k-t_{k-1},\ 
\mathrm{IAE}\approx\sum_k |e_k|\,\Delta t_k,\ 
\mathrm{ISE}\approx\sum_k e_k^2\,\Delta t_k,\ 
\mathrm{ITAE}\approx\sum_k t_k\,|e_k|\,\Delta t_k. \tag{E.1}
$$
過渡応答：
$$
T_r=\min\{t:y(t)\ge0.9r_{\mathrm{ss}}\}-\min\{t:y(t)\ge0.1r_{\mathrm{ss}}\},\quad
T_s=\min\{t:|y(t)-r_{\mathrm{ss}}|\le\epsilon|r_{\mathrm{ss}}|\},\ \epsilon=0.02. \tag{E.2}
$$
オーバーシュート：
$$
\mathrm{OS}=\frac{\max_t y(t)-r_{\mathrm{ss}}}{r_{\mathrm{ss}}}\times 100\%. \tag{E.3}
$$

## E.3 CLI / MATLAB側スケルトン
```bash
matlab-bridge export --in scored.csv --out matlab/ref.mat
```
**MATLAB**（例）:
```matlab
S = load('ref.mat');
t = S.t; ref = S.ref; y1 = S.y_lstm; y2 = S.y_pid;
e = ref - y1;
dt = [0; diff(t)];
IAE  = sum(abs(e).*dt);
ISE  = sum((e.^2).*dt);
ITAE = sum(t.*abs(e).*dt);
```

## E.4 検証と失敗条件
- `.mat` の読み込み失敗、`t` 非増加、変数欠落 → **非0終了**。  
- 指標算出で NaN が出た場合は失敗。

### 参考文献（E）
[1] MathWorks: From Workspace; [2] Zero-Order Hold; [3] PID Controller; [4] 制御工学標準教科書（IAE/ISE/ITAEの定義）。
