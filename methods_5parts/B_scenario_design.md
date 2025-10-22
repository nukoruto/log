# B. 正常／異常シナリオ設計（旧：手法2）

本章は、Aで得た統計から**正常/異常シナリオの設計図**を JSON で定義する。式と実装規定を併記する。

## B.1 正常シナリオ：操作列（Markov）
カテゴリ集合 $\mathcal{C}$。初期分布 $\pi$，遷移行列 $A$：
$$
\pi_j=\Pr(op_1=j),\quad A_{ij}=\Pr(op_{k+1}=j\mid op_k=i),\quad \mathbf{op}\sim \mathrm{MC}(\pi,A).\tag{B.1}
$$
推定は相対頻度＋ラプラス平滑化（$\alpha=1/|\mathcal{C}|$）。未知語は**禁止**（契約違反）。[1]–[4]

## B.2 正常シナリオ：時間間隔
カテゴリ $c$ ごとに $x=\log(\Delta t+\varepsilon)$ のロバスト推定 $(\mu_c,\sigma_c)$ から、
$$
\log(\Delta t\mid c)\sim \mathcal{N}(\mu_c,\sigma_c^2),\quad \Delta t\sim \mathrm{Lognormal}(\mu_c,\sigma_c^2). \tag{B.2}
$$
（任意）日周性は非斉次ポアソン過程で $ \lambda_c(t)=\lambda_{0,c}\,g_h(t) $ として thinning [7],[9]。

## B.3 異常シナリオの設計
- **順序逸脱**：制約集合 $\mathcal{R}$ に対し $\exists k:(op_k,op_{k+1})\notin \mathcal{R}$。
- **未認証アクセス**：述語 $\mathrm{AuthOK}(i)$ に対し $\neg\mathrm{AuthOK}(i)\wedge op_i\in\{\mathrm{UPDATE},\mathrm{DELETE}\}$。
- **トークン流用**：$(\mathrm{uid},\mathrm{session\_id})$ の結合制約を破る。
- **時間異常**：確率 $p$，スケール $\alpha$（乗算）または $\delta$（加算）。[10],[15]

## B.4 仕様（`scenario_spec.json`）最小スキーマ
```json
{
  "length": 512,
  "users": 10,
  "pi": {"AUTH":0.3,"READ":0.6,"UPDATE":0.1},
  "A": {"AUTH":{"READ":0.9,"AUTH":0.1}, "...": {}},
  "dt": {"lognorm": {"mu": {"READ": 0.1, "UPDATE": 0.5}, "sigma": {"READ": 0.6, "UPDATE": 0.8}}},
  "anoms": [
    {"type":"time","mode":"propagate","p":0.02,"scale":3.0},
    {"type":"order","p":0.01},
    {"type":"unauth","p":0.005},
    {"type":"token_replay","p":0.005}
  ],
  "seed": 42
}
```

## 実装規定とCLI
- `scenario-design fit deltified.csv --out stats.pkl`（$ \pi,A,(\mu_c,\sigma_c) $ 推定）。
- `scenario-design plan --stats stats.pkl --seed 42 --out scenario_spec.json ...`  
- JSON Schema を固定し、フィールド欠落は**非0終了**。

## 検証
- 同一 `--seed` で `scenario_spec.json` が**バイト一致**。  
- 制約 $\mathcal{R}$ 違反が正常系に含まれていないことをテスト。

### 参考文献（B）
[1] RFC 9110; [2] RFC 7519; [3] RFC 6750; [4] RFC 6265;  
[7] Reinhart, Hawkes review; [8] Bacry & Muzy; [9] Gu & Zhu;  
[10] Siffer et al., SPOT; [11] Karsai et al., bursty; [12] Vázquez et al., heavy tails; [15] NIST ITL (MAD).
