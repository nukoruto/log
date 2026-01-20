import os
import sys
from pathlib import Path

def train_deeplog():
    # Path to DeepLog training script
    script_path = "models/DeepLog/LogKeyModel_train.py"
    
    # Arguments
    data_path = "data/HDFS/deeplog_input/hdfs_train"
    num_epochs = 5 # Reduced for quick verification (was 300)
    
    # Command
    cmd = f"python {script_path} -data {data_path} -num_epochs {num_epochs} -window_size 1"
    
    print(f"Running: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    train_deeplog()
