import os
import sys
from pathlib import Path

def train_deeplog():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
    args = parser.parse_args()

    # Path to DeepLog training script
    script_path = "models/DeepLog/LogKeyModel_train.py"
    
    args = parser.parse_args()

    # Path to DeepLog training script
    script_path = "models/DeepLog/LogKeyModel_train.py"
    
    # Arguments
    if args.mode == 'full':
        data_path = "data/HDFS/deeplog_input/hdfs_train"
        mapping_path = "data/HDFS/deeplog_input/event_mapping.csv"
        window_size = 10
        num_epochs = 300
    else:
        data_path = "data/HDFS/deeplog_input_2k/hdfs_train"
        mapping_path = "data/HDFS/deeplog_input_2k/event_mapping.csv"
        window_size = 1
        num_epochs = 5
        
    # Detect num_classes from event_mapping.csv if exists
    num_classes = 28 # Default
    if Path(mapping_path).exists():
        import pandas as pd
        df = pd.read_csv(mapping_path)
        # Assuming IntId 1..N. Max ID is N.
        # But wait, IDs are 1-based. If Max ID is 48, we need 49 classes if we use 0-based indexing?
        # DeepLog code uses `output = model(seq)` and `loss = criterion(output, label)`.
        # Label is 0..(N-1) usually.
        # `generate` converts `n-1`. So if file has 1..48, inputs become 0..47.
        # Max index is 47. So num_classes must be at least 48.
        # So we should use max(IntId).
        num_classes = df['IntId'].max()
        print(f"Detected max label: {num_classes-1}, setting num_classes to {num_classes}")

    # Command
    cmd = f"python {script_path} -data {data_path} -num_epochs {num_epochs} -window_size {window_size} -num_classes {num_classes}"
    
    print(f"Running: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    train_deeplog()
