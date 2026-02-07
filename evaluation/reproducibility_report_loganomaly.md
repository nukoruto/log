# Reproducibility Report: LogAnomaly

This document records the metadata required for reproducing the experimental results of LogAnomaly on the HDFS dataset.

## 1. Reproducibility Metadata

### (A) Resource Information (Resource)

*   **Dataset Source**: HDFS Dataset (LogHub)
    *   **Original File**: `HDFS.log` (Full Mode)
    *   **Subset**: `HDFS_2k.log` (Demo Mode)
*   **Data Splits**:
    *   **Train**: 80% of Normal sessions (Randomly sampled)
    *   **Test (Normal)**: Remaining 20% of Normal sessions
    *   **Test (Abnormal)**: 100% of Abnormal sessions
    *   **Splitting Script**: `pipeline/conversion.py`
    *   **Generator**: `pipeline/4_setup_loganomaly_data.py` (Augmentation applied if samples < 100)
*   **Codebase**: 
    *   **Root**: `d:\kosen\sotuken\log`
    *   **Models**: `models/LogDeep`

### (B) Methodological Information (Methodological)

#### Model Architectures

| Parameter | LogAnomaly |
| :--- | :--- |
| **Model Type** | LSTM (Semantic + Quant + Seq) |
| **Input Size** | 1 (Seq) + 300 (Sem) + 1 (Quan) |
| **Hidden Size** | 64 |
| **Num Layers** | 2 |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | 1024 |
| **Epochs (Full)** | 50 |

#### Preprocessing & Features

*   **Log Parser**: Drain
    *   **Sim Thresh (st)**: 0.5
    *   **Depth**: 4
    *   **Regex**: `blk_-?\d+`, IP addresses
*   **Semantic Vectors**:
    *   **Method**: Word2Vec + TF-IDF + Retrofitting
    *   **Dimension**: 300
    *   **Window**: 5
    *   **Min Count**: 1
    *   **Retrofitting**: Alpha=0.1, Epochs=10 (using `synonyms_antonyms.json`)
*   **Frameworks**:
    *   **Python**: 3.x
    *   **PyTorch**: (User Environment)
    *   **Pandas**, **Gensim**, **Scikit-learn**

### (C) Randomness Information (Randomness)

*   **Seed Control**: `seed_everything` used in `logdeep/tools/utils.py`.
    *   **Seed Value**: `1234` (Python, NumPy, PyTorch CPU/GPU)
*   **Word2Vec Seed**: `42`
*   **Data Shuffle**: `sample(frac=1, random_state=42)` in `pipeline/conversion.py`.

### (D) Statistical Information (Statistical)

*   **Evaluation Metric**: F1 Score (Precision, Recall)
    *   **Definition**: Based on Top-K prediction matching next event.
    *   **Criteria**: If Ground Truth is in Top-g predictions -> Correct (Normal). Else -> Anomaly.
*   **Latest Results**:
    *   **g=1**: Precision 0.23, Recall 1.00, F1 0.38
    *   **g=4 (Optimal)**: Precision 0.96, Recall 0.91, **F1 0.93**
    *   **g=9**: Precision 1.00, Recall 0.34, F1 0.51
*   **Execution**: Single run reported.

## 2. Log Anomaly Detection Specifics

### Log Format Schema
*   **Format**: `<Date> <Time> <Pid> <Level> <Component>: <Content>`
*   **Parsed Fields**: `EventId`, `EventTemplate`
*   **Session ID**: `BlockId` (extracted from Content via Regex `blk_-?\d+`)

### Log Preprocessing Pipeline
1.  **Parsing**: Raw Log -> Drain -> Structured CSV (`EventId` assigned).
2.  **Mapping**: `EventId` (String) -> `IntId` (1-based Integer).
3.  **Vectorization**: `EventTemplate` -> Tokenize -> Word2Vec -> Semantic Vector.
4.  **Sessionizaton**: Group by `BlockId`. Filtering `len >= window_size`.
5.  **Indexing**: `IntId` - 1 = `Model Input ID` (0-based).

### Window Settings
*   **Strategy**: Sliding Window
*   **Window Size**: 10 (Full Mode)
*   **Step Size**: 1 (Next event prediction)

### Anomaly Labeling Criteria
*   **Unit**: Session (BlockId)
*   **Label Source**: `anomaly_label.csv` (Ground Truth) provided by LogHub.
*   **Policy**: If a session ID exists in label file as 'Anomaly', it is Abnormal. Otherwise Normal.
*   **Prediction Policy**:
    *   For each window in a session, predict next event.
    *   If prediction fails (Target not in Top-g), count as anomaly.
    *   (Note: Current evaluation script aggregates per session: if any window in a session is predicted as anomaly, the entire session is flagged.)
