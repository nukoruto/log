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
        window_size = 10
        num_epochs = 300
    else:
        data_path = "data/HDFS/deeplog_input_2k/hdfs_train"
        window_size = 1
        num_epochs = 5

    # Command
    cmd = f"python {script_path} -data {data_path} -num_epochs {num_epochs} -window_size {window_size}"
    
    print(f"Running: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    train_deeplog()
