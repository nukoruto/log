import os
import sys
from pathlib import Path

def train_loganomaly():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
    args = parser.parse_args()

    # LogDeep demo script location
    project_root = Path(__file__).resolve().parent.parent
    demo_dir = project_root / 'models' / 'LogDeep' / 'demo'
    
    if args.mode == 'full':
        window_size = 10
        epochs = 50 # Default for full
        data_dir = '../data/'
        save_dir = '../result/loganomaly/'
    else:
        window_size = 1
        epochs = 5 # Default for demo
        data_dir = '../data/hdfs_2k/'
        save_dir = '../result/loganomaly_2k/'

    cmd = f"python loganomaly.py train --window_size {window_size} --max_epoch {epochs} --data_dir {data_dir} --save_dir {save_dir}"
    
    print(f"Running LogAnomaly from {demo_dir} (Mode: {args.mode})")
    print(f"Command: {cmd}")
    
    os.chdir(demo_dir)
    os.system(cmd)

if __name__ == "__main__":
    train_loganomaly()
