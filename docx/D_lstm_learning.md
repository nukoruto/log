# D. 学習・推論・統計的スコアリング（旧：手法6＋5＋7）

A/B/C で得た normal/anom を用いて LSTM を学習し、異常スコアを算出する。

## D.1 特徴量とモデル
- 入力：$\mathbf{x}_i=[\mathrm{Embed}(op_i);\ z_{\mathrm{clip}}(i)]\in\mathbb{R}^{d+1}$。
- LSTM：
  $$
  (\mathbf{h}_i,\mathbf{c}_i)=\mathrm{LSTM}(\mathbf{x}_i,\mathbf{h}_{i-1},\mathbf{c}_{i-1}). \tag{D.1}
  $$
- 出力：
  $$
  \mathbf{p}_{i+1}=\mathrm{softmax}(W_c\mathbf{h}_i+\mathbf{b}_c),\qquad
  \hat{z}_{i+1}=w_t^\top \mathbf{h}_i+b_t. \tag{D.2}
  $$

## D.2 損失（分類＋回帰）
$$
\mathcal{L}=\mathrm{CE}(y_{i+1},\mathbf{p}_{i+1})+\lambda\,\mathrm{Huber}_\delta(\hat{z}_{i+1}-z_{i+1}), \tag{D.3}
$$
$$
\mathrm{Huber}_\delta(r)=\begin{cases}\frac{1}{2}r^2&(|r|\le\delta)\\ \delta(|r|-\frac{1}{2}\delta)&(|r|>\delta)\end{cases}. \tag{D.4}
$$
（任意）RMTPP の連続時間尤度を併用可 [4]。

## D.3 異常スコアと閾値
$$
s_{\mathrm{cls},i}=1-\mathbf{p}_i[op_i],\quad s_{\mathrm{time},i}=|\hat{z}_i-z_i|,\quad
S_i=w_{\mathrm{cls}}s_{\mathrm{cls},i}+w_{\mathrm{time}}s_{\mathrm{time},i}. \tag{D.5}
$$
分位点方式：カテゴリ $c$ ごとに
$$
\tau_{\mathrm{hi}}(c)=Q_{1-\alpha}(s_{\mathrm{time}}\mid c),\ \ \tau_{\mathrm{lo}}(c)=Q_{\alpha}(s_{\mathrm{time}}\mid c). \tag{D.6}
$$
（任意）EVT(POT) により上側尾を GPD 近似：
$$
\tau=u+\frac{\beta}{\xi}\Big[\Big(\frac{p_{\mathrm{ref}}}{p_{\mathrm{target}}}\Big)^{\xi}-1\Big]. \tag{D.7}
$$

## D.4 実装規定（再現性・GPU・分割）
- 乱数：`--seed` 必須（Python/NumPy/PyTorchに適用）。cuDNN 決定論フラグを有効化。AMP 既定ON。  
- デバイス：`.env: GPU_MODE=rtx6000|rtx4060|cpu`。名前一致→`cuda:0`→CPU。  
- データ分割：**user/session グループ**でリーク防止（GroupKFold）。  
- 既定 HP：embedding dim=64, LSTM hidden=128, 層=1, dropout=0.1, ϵ=1e-8, Adam(lr=1e-3, β=(0.9,0.999)), batch=256, epoch上限=50, 早期停止 patience=5。

## D.5 CLI / 出力
```bash
models-lstm train --normal normal.csv --val normal.csv --out runs/exp1 --seed 42
models-lstm score --model runs/exp1/best.ckpt --in anom.csv --out scored.csv
```
- `scored.csv` 列：`timestamp_utc, uid, session_id, op_category, z, z_hat, s_cls, s_time, S, flag_cls, flag_dt`。  
- `metrics.json`：F1, PR-AUC, ROC-AUC, 検知遅延、しきい値。

## D.6 評価
- 主指標：Average Precision（PR-AUC）。副指標：ROC-AUC, F1@opt, 検知遅延。  
- 検証：rolling origin の時系列CV（グループ制約）。

### 参考文献（D）
[1] NIST/ITL (Percentile); [2] Kingma & Ba, Adam; [3] Paszke et al., PyTorch;  
[4] Du et al., RMTPP; [5] Neil et al., Phased LSTM; [6] Micikevicius et al., Mixed Precision;  
[8] Wilson et al., Good enough practices; [9] PyTorch deterministic algos doc.
