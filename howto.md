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

## 2. デモモード (Demo Mode)

本パイプラインは、開発や動作確認用の「デモモード」と、本格的な学習を行う「フルモード」の2つのモードをサポートしています。
すべてのパイプラインスクリプトにて `--mode demo` 引数を指定可能です。指定しない場合はデフォルトで `full` モードになります。

| 機能 | フルモード (デフォルト) | デモモード (`--mode demo`) |
| :--- | :--- | :--- |
| **入力ログ** | `HDFS.log` (約1.5GB) | `HDFS_2k.log` (2000行) |
| **Window Size** | 10 | 1 |
| **セッションフィルタ** | 長さ >= 11 (学習用) | 長さ >= 2 (テスト用) |
| **DeepLogデータ出力先** | `data/HDFS/deeplog_input` | `data/HDFS/deeplog_input_2k` |
| **LogAnomalyデータ出力先** | `models/LogDeep/data/hdfs` | `models/LogDeep/data/hdfs_2k` |
| **DeepLog Epoch数** | 300 | 5 |
| **LogAnomaly Epoch数** | 50 | 5 |

**デモモードの実行例:**
```bash
python pipeline/1_parse.py --mode demo
python pipeline/conversion.py --mode demo
python pipeline/4_setup_loganomaly_data.py --mode demo
python pipeline/2_train_deeplog.py --mode demo
python pipeline/3_evaluate_deeplog.py --mode demo
```

## 3. データ準備 (HDFS - Full Mode)

### データの配置
`data/HDFS/HDFS_2k.log` (またはフルサイズの `HDFS.log`) を配置します。

### パース (Drain)
ログデータを構造化データ（CSV）に変換します。
```bash
python pipeline/1_parse.py
```
出力: `data/HDFS/parsed/HDFS.log_structured.csv`

## 4. DeepLog モデルの実行

DeepLog用データへの変換、学習、評価を行います。

```bash
# 1. データ変換 (セッション系列化)
# data/HDFS/deeplog_input/ に学習・テスト用データ(hdfs_train, hdfs_test_normal等)を作成します
python pipeline/conversion.py

# 2. モデル学習
# HDFSフルデータの場合は、window_size=10 が一般的です (スクリプト内で設定済)
python pipeline/2_train_deeplog.py

# 3. 評価
# 正常・異常データに対するスコアリングを行い、IAE/ISEおよびF1スコアを算出します。
# デフォルトのg=9ではRecallが低くなる傾向があるため、より高いF1スコアを得るには g=3 などを推奨します。
python pipeline/3_evaluate_deeplog.py --g 3
```

## 5. LogAnomaly モデルの実行

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

## 6. 結果の比較

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
    *   **IAE (Integrated Absolute Error)**: 異常スコアの積分値 ($\sum |e(t)| \cdot \Delta t$)。$\Delta t$ (イベント間時間) を考慮し、異常の持続性と頻度を評価します。
    *   **ISE (Integrated Squared Error)**: 異常スコアの二乗積分 ($\sum e(t)^2 \cdot \Delta t$)。大きな誤差をより重く評価します。
*   **分類指標 (Binary Classification)**
    *   **Precision/Recall/F1**: 異常検知の正確さ。Top-K予測などで次イベントを予測できなかった場合を「異常」と判定し、正解ラベルと比較します。

以上
