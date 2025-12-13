# A. データ契約・セッション分割・Δt抽出（旧：手法1＋3＋4）

本章は、公開データから**契約CSV**を作り、セッション分割と `Δt` のロバスト特徴量を生成するまでを**完全実装可能**な粒度で規定する。IEEE形式の引用番号を用いる。

## A.1 データ契約と収集環境（旧・手法1）

- 入力：任意の公開CSV（列名は `map.yaml` で契約列にマップ）。
- 出力：`contract.csv`（**列順固定・UTC**）。
- 契約CSV（9列、順序固定）  
- 契約CSV（6列、順序固定）  
  1) `timestamp_utc`, 2) `uid`, 3) `session_id`, 4) `method`, 5) `path`, 6) `op_category`。  
  必須：1,2,4,5,6。`session_id` は任意。

- ベースID（`uid`）の優先順位と生成式
  $$
  \mathrm{base}_i=
  \begin{cases}
  \mathrm{sid}_i, & \mathrm{sid}_i\neq\varnothing\\
  H_k(\mathrm{uid}^{\text{raw}}_i), & \mathrm{uid}^{\text{raw}}_i\neq\varnothing\\
  H_k(\mathrm{sub}_i), & \mathrm{sub}_i\neq\varnothing\\
  H_k(\mathrm{dc}_i), & \mathrm{dc}_i\neq\varnothing\\
  H_k(\mathrm{ip}_i\parallel \mathrm{ua}_i), & \text{otherwise}
  \end{cases} \tag{A.1}
  $$
  ここで $H_k(x)=\mathrm{hex}(\mathrm{HMAC}_{\mathrm{SHA256}}(k, x))$。$k$ は秘密鍵。

- 時刻正規化：ローカル時刻 $t_{\mathrm{local}}$ とオフセット $\mathrm{offset}$ から、
  $$
  t_{\mathrm{UTC}}=t_{\mathrm{local}}-\mathrm{offset} \tag{A.2}[5]
  $$
  NTPv4 でクロック同期（$P_{95}(|\theta|)\le 50\,\mathrm{ms}$ を推奨）[11]。

- 仕様準拠：HTTP意味論/クッキー/Bearer/JWT/CSV は RFC に準拠 [1]–[6]。

- 実行環境：Docker で依存固定、GPU は環境変数で選択（`.env: GPU_MODE=rtx6000|rtx4060|cpu`）[12],[13]。FAIR/Datasheets を添付 [14],[15]。

## A.2 セッション分割：$\Delta T_{\mathrm{session\_max}}$ の自動選定（旧・手法3）

- 入力：`contract.csv`（UTC整列）。(uid, session_id, timestamp) で昇順ソート。`session_id` 欠落や破損時は**時間ギャップ**で推定する。
- `Δt` の定義：セッション内の時刻列 $\{t(i)\}$ から $\Delta t(i) = t(i) - t(i-1)$。
- セッション分割（同一ベースID内）：
  $$
  s_i=
  \begin{cases}
  0,& i=1\\
  s_{i-1}+1,& \mathrm{base}_i\neq \mathrm{base}_{i-1}\ \lor\ \Delta t_i>\theta\\
  s_{i-1},& \text{otherwise}
  \end{cases} \tag{A.2}
  $$
  最終的なセッションキー：$\mathrm{session\_key}_i := H_k(\mathrm{base}_i \parallel s_i)$（または $\mathrm{sid}_i$ をそのまま利用）。

- 対数領域：$x=\log(\Delta t+\varepsilon)$、$\varepsilon=10^{-3}$ 秒固定。

- **Otsu法**（二峰性が明確な場合）[5]：ヒストグラムのクラス間分散
  $$
  \sigma_B^2(\tau)=\omega_0(\tau)\,\omega_1(\tau)\bigl(\mu_0(\tau)-\mu_1(\tau)\bigr)^2
  \tag{A.4}
  $$
  を**最大化**する $\tau^{*}$ を求め、
$$
\Delta T_{\mathrm{session\_max}}^{\mathrm{Otsu}} = \exp(\tau^{*}) \tag{A.5}
$$

- **肘法（Kneedle）** フォールバック [7],[8]：ギャップ閾値 $\tau$ に対するセッション数 $N_{\mathrm{sess}}(\tau)$ を正規化し、弦からの距離
  $$
  d(\tau)=\frac{|(\tilde{y}(\tau)-y_0)-m(\tilde{x}(\tau)-x_0)|}{\sqrt{1+m^2}} \tag{A.6}
  $$
  を**最大**とする $\tau^\dagger$ を選び、$\Delta T_{\mathrm{session\_max}}=\tau^\dagger$。  
  人のWeb行動は二峰性を示し分離点は概ね1時間付近で観測される [6]。

### 実装規定（決定論のための固定値）
- ビニング：Freedman–Diaconis 幅 $h=2\cdot \mathrm{IQR}\cdot n^{-1/3}$、ビン数は **[32, 256]** にクリップ。
- 二峰性判定：クラス内分散比 < **0.9** で Otsu、それ以外は肘法。
- 分位点：`numpy.quantile(..., method="linear")` に固定。
- グルーピング：基本は **user 単位**。サンプル <5 の user は全体へバックオフ（台帳に記録）。

## A.3 Δt のロバスト正規化（旧・手法4）
- ロバスト Z：
  $$
  z(i)=\frac{x(i)-\tilde{x}}{1.4826\cdot \mathrm{MAD}},\quad
  z_{\mathrm{clip}}(i)=\min\{\max(z(i),-5),\,5\} \tag{A.7}
  $$
  $\tilde{x}$ は中央値，$\mathrm{MAD}=\mathrm{median}(|x-\tilde{x}|)$。

- 補助特徴：近傍分位とバースト度
  $$
  m_q(i)=\mathrm{Quantile}_q\big(\{\Delta t(i-j)\}_{j=1}^{W}\big),\quad
  \mathrm{burst}(i)=\frac{\Delta t(i-1)+\varepsilon}{\Delta t(i)+\varepsilon}. \tag{A.8}
  $$

### CLI / 出力物
```bash
ds-contract validate input.csv --map map.yaml --out contract.csv
ds-contract sessionize contract.csv --out sessioned.csv --meta meta_session.json
ds-contract deltify sessioned.csv --out deltified.csv --meta meta_dt.json
```
- `meta_session.json`: 採用手法（Otsu/Elbow）, 閾値, ヒスト設定, 入力 SHA256, seed。  
- `meta_dt.json`: median/MAD/clip、グループ単位、入力 SHA256, seed。

### エラーと検証
- UTCでない/必須列欠損/逆順 → **非0終了**（修正ガイダンスを表示）。
- 同一 `--seed` で出力が**バイト一致**（決定論）。

### 参考文献（A）
[1] RFC 9110（HTTP Semantics）; [2] RFC 6265（Cookie）; [3] RFC 6750（Bearer）; [4] RFC 7519（JWT）;  
[5] RFC 3339（Timestamps）; [6] Halfaker et al., Session Identification (arXiv:1411.2878);  
[7] Herdiana et al., Precise Elbow Method (arXiv:2502.00851); [8] Fok & Ye, Knee Detection (arXiv:2409.15608);  
[9] NIST SP 800-122; [10] NISTIR 8053; [11] RFC 5905（NTPv4）;  
[12] Nüst et al., Ten simple rules for Dockerfiles; [13] Wilson et al., Good enough practices;  
[14] Wilkinson et al., FAIR; [15] Gebru et al., Datasheets.
