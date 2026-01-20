# ログ異常検知ベンチマーク 実行ガイド

本リポジトリは、DeepLogおよびLogAnomalyモデルを用いたログ異常検知の統合ベンチマーク環境です。
制御工学指標（IAE, ISE, ITAE）および従来の分類指標（Precision, Recall, F1）を用いてモデルを評価・比較します。

## 1. 環境構築

### 必須要件
- Python 3.8+
- PyTorch
- pandas, numpy, scikit-learn, matplotlib

### インストール
本リポジトリをクローンし、必要なライブラリをインストールしてください。
(requirements.txt は別途用意予定)

## 2. データ準備 (HDFS)

### データの配置
`data/HDFS/HDFS_2k.log` (またはフルサイズの `HDFS.log`) を配置します。

### パース (Drain)
ログデータを構造化データ（CSV）に変換します。
```bash
python pipeline/1_parse.py
```
出力: `data/HDFS/parsed/HDFS_2k.log_structured.csv`

## 3. DeepLog モデルの実行

DeepLog用データへの変換、学習、評価を行います。

```bash
# 1. データ変換 (セッション系列化)
# data/HDFS/deeplog_input/ に学習・テスト用データ(hdfs_train, hdfs_test_normal等)を作成します
python pipeline/conversion.py

# 2. モデル学習
# HDFS_2kのような小規模データの場合は、window_size=1 等に調整されています
python pipeline/2_train_deeplog.py

# 3. 評価
# 正常・異常データに対するスコアリングを行い、IAE/ISEおよびF1スコアを算出します
python pipeline/3_evaluate_deeplog.py
```

## 4. LogAnomaly モデルの実行

LogAnomaly用のデータセットアップ（意味ベクトル生成）と学習、評価を行います。

```bash
# 1. データセットアップ
# DeepLog用に作成されたデータをLogAnomaly形式（意味ベクトル付与）に変換します
# HDFS_2kのような小規模データの場合、学習用にデータを自動的に増幅(Augmentation)します
python pipeline/4_setup_loganomaly_data.py

# 2. モデル学習
python pipeline/2_train_loganomaly.py

# 3. 評価
python pipeline/3_evaluate_loganomaly.py
```

## 5. 結果の比較

両モデルの評価結果を集計し、比較表を出力します。

```bash
python pipeline/5_compare_models.py
```

**出力例** (コンソールおよび `evaluation/results/benchmark_comparison.csv`):
```text
=== Benchmark Results ===
     Model       IAE       ISE      ITAE  Precision  Recall   F1
   DeepLog  0.000000  0.000000  0.000000        1.0     0.5  0.66
LogAnomaly  3.873657  3.298336  9.595449        0.0     0.0  0.00
```
*注: 上記はサンプルデータによるダミースコアの場合があります。*

## 評価指標について

*   **制御工学指標 (Anomaly Score Waveform Evaluation)**
    *   **IAE (Integrated Absolute Error)**: 異常スコアの総和。モデルがどれだけ「正常(=0)」から逸脱したかを示します。
    *   **ISE (Integrated Squared Error)**: 異常スコアの二乗和。大きな誤差をより重く評価します。
*   **分類指標 (Binary Classification)**
    *   **Precision/Recall/F1**: 異常検知の正確さ。Top-K予測などで次イベントを予測できなかった場合を「異常」と判定し、正解ラベルと比較します。

以上
