import os
import sys
from pathlib import Path

def train_loganomaly():
    # LogDeep demo script location
    # d:\kosen\sotuken\log\models\LogDeep\demo\loganomaly.py
    # This script assumes it's run from 'demo' folder usually, or we adjust paths.
    # It sets options['data_dir'] = '../data/'
    # So if we run it from 'models/LogDeep/demo', it looks for '../data/' which is 'models/LogDeep/data'.
    # This matches our setup.
    
    project_root = Path(__file__).resolve().parent.parent
    demo_dir = project_root / 'models' / 'LogDeep' / 'demo'
    
    # We need to modify loganomaly.py to:
    # 1. Use small epochs (for verify)
    # 2. Use cpu (maybe)
    # 3. Use 'hdfs' dataset (it hardcodes specific paths?)
    # Let's check loganomaly.py again.
    # It has `options['data_dir'] = '../data/'`
    # Training uses `Trainer`. 
    # Logic in `logdeep/dataset/sample.py` likely looks for 'train.csv' in `options['data_dir'] + 'hdfs/train.csv'`?
    # No, usually LogDeep config specifies subfolder.
    # IN loganomaly.py:
    # options['sample'] = "sliding_window"
    # It doesn't explicitly set dataset name 'hdfs', but maybe it's default?
    # Wait, `loganomaly.py` doesn't seem to specify `dataset_name`.
    # Let's inspect `Trainer`.
    
    # But for now, let's just try running it and see if it picks up our `data/hdfs/train.csv`.
    # Actually, `loganomaly.py` sets `options['save_dir'] = "../result/loganomaly/"`.
    
    cmd = f"python loganomaly.py train"
    
    print(f"Running LogAnomaly from {demo_dir}")
    os.chdir(demo_dir)
    os.system(cmd)

if __name__ == "__main__":
    train_loganomaly()
