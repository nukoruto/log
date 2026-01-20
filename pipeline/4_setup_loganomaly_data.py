import json
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

def setup_loganomaly(mode='full'):
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    
    if mode == 'full':
        source_dir = project_root / 'data' / 'HDFS' / 'deeplog_input'
        target_dir = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs'
    else:
        source_dir = project_root / 'data' / 'HDFS' / 'deeplog_input_2k'
        target_dir = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs_2k' / 'hdfs'
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Dummy Semantic Vectors
    print("Generating dummy semantic vectors...")
    mapping_path = source_dir / 'event_mapping.csv'
    if not mapping_path.exists():
        print(f"Error: {mapping_path} not found. Run conversion.py first.")
        return

    mapping_df = pd.read_csv(mapping_path)
    # IntId is 1-based in my conversion for DeepLog. 
    # LogDeep usually expects 0-based or 1-based?
    # DeepLog subtracts 1. LogAnomaly might do the same or just use as index.
    # Safe bet: generate for range [0, MaxID].
    
    max_id = mapping_df['IntId'].max()
    vec_dim = 300
    
    # Create simple random vectors
    # Key is string representation of integer ID
    # DeepLog and LogDeep usually assume events are 0..N-1 or 1..N
    # If our data has 1..N, we generate for 1..N.
    # Note: LogAnomaly might try to look up '0' if it subtracts 1 from 1. 
    # Let's generate for 0 to MaxID just to be safe.
    
    vec_dict = {}
    for i in range(max_id + 2):
        vec_dict[str(i)] = np.random.rand(vec_dim).tolist()
        
    vec_path = target_dir / 'event2semantic_vec.json'
    with open(vec_path, 'w') as f:
        json.dump(vec_dict, f)
    print(f"Saved {vec_path}")

    # 2. Copy Data Files
    # We copy our generated session files to LogDeep's data folder
    files_to_copy = ['hdfs_train', 'hdfs_test_normal']
    # Create dummy abnormal if missing
    if not (source_dir / 'hdfs_test_abnormal').exists():
        shutil.copy(source_dir / 'hdfs_test_normal', source_dir / 'hdfs_test_abnormal')
        files_to_copy.append('hdfs_test_abnormal')
    else:
        files_to_copy.append('hdfs_test_abnormal')
        
    for fname in files_to_copy:
        src = source_dir / fname
        dst = target_dir / fname
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"Warning: {src} missing")
            
    # 3. Generate train.csv / test.csv using logic similar to gen_train_data.py
    # But dealing with the fact that our data might be space separated integers
    
    # Create dummy data with length >= 2 to ensure LogAnomaly runs (window_size=1)
    # We need valid EventIds.
    # Our generated vectors cover up to max_id.
    # Let's just use "1 1", "1 2" etc.
    
    # LogDeep's sample.py subtracts 1 from input.
    # It treats result 0 (Input 1) as Padding.
    # It treats result > 0 (Input > 1) as valid, and looks up (result-1).
    # i.e. Input 2 -> 1 -> look up "0".
    # So we need inputs >= 2.
    
    # Use actual data from DeepLog input
    print("Using actual data from valid HDFS_2k conversion...")
    
    def read_sequences(path):
        if not path.exists():
            return []
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    # Read sequences and times and labels
    real_train_seqs = read_sequences(source_dir / 'hdfs_train')
    real_test_normal_seqs = read_sequences(source_dir / 'hdfs_test_normal')
    real_test_abnormal_seqs = read_sequences(source_dir / 'hdfs_test_abnormal')
    
    real_train_times = read_sequences(source_dir / 'hdfs_train_time') 
    real_test_normal_times = read_sequences(source_dir / 'hdfs_test_normal_time')
    real_test_abnormal_times = read_sequences(source_dir / 'hdfs_test_abnormal_time')
    
    real_train_labels = read_sequences(source_dir / 'hdfs_train_label') 
    real_test_normal_labels = read_sequences(source_dir / 'hdfs_test_normal_label')
    real_test_abnormal_labels = read_sequences(source_dir / 'hdfs_test_abnormal_label')
    
    # Helper to sync filter
    def filter_triple(seqs, times, labels):
        valid_s = []
        valid_t = []
        valid_l = []
        # If times/labels missing, fill dummy
        if not times: times = ["" for _ in seqs]
        if not labels: labels = ["" for _ in seqs]
        
        for s, t, l in zip(seqs, times, labels):
            if len(s.split()) >= 2:
                valid_s.append(s)
                valid_t.append(t)
                valid_l.append(l)
        return valid_s, valid_t, valid_l

    valid_train, _, _ = filter_triple(real_train_seqs, real_train_times, real_train_labels) 
    valid_test_normal, valid_test_normal_t, valid_test_normal_l = filter_triple(real_test_normal_seqs, real_test_normal_times, real_test_normal_labels)
    valid_test_abnormal, valid_test_abnormal_t, valid_test_abnormal_l = filter_triple(real_test_abnormal_seqs, real_test_abnormal_times, real_test_abnormal_labels)
    
    print(f"Original Train: {len(real_train_seqs)}, Valid Length (>=2): {len(valid_train)}")

    # Augmentation logic (only for TRAIN)
    if len(valid_train) > 0 and len(valid_train) < 100:
        print(f"Duplicating valid sessions (count={len(valid_train)}) to ensure robust training loop...")
        while len(valid_train) < 100:
            valid_train.extend(valid_train)
        print(f"Augmented Train: {len(valid_train)}")

    if len(valid_train) == 0:
        pass 
        
    # Write raw text files
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Write Events
    with open(target_dir / 'hdfs_train', 'w') as f:
        f.write('\n'.join(valid_train))
    with open(target_dir / 'hdfs_test_normal', 'w') as f:
        f.write('\n'.join(valid_test_normal))
    with open(target_dir / 'hdfs_test_abnormal', 'w') as f:
        f.write('\n'.join(valid_test_abnormal))
        
    # Write Times
    with open(target_dir / 'hdfs_test_normal_time', 'w') as f:
        f.write('\n'.join(valid_test_normal_t))
    with open(target_dir / 'hdfs_test_abnormal_time', 'w') as f:
        f.write('\n'.join(valid_test_abnormal_t))
        
    # Write Labels
    with open(target_dir / 'hdfs_test_normal_label', 'w') as f:
        f.write('\n'.join(valid_test_normal_l))
    with open(target_dir / 'hdfs_test_abnormal_label', 'w') as f:
        f.write('\n'.join(valid_test_abnormal_l))
        
    print(f"Generated raw files, times, and labels in {target_dir}")
    
    # Generate CSVs for 'session_window' or reference
    train_df = pd.DataFrame({
        "Sequence": valid_train,
        "label": [0] * len(valid_train)
    })
    
    test_df = pd.DataFrame({
        "Sequence": valid_test_normal + valid_test_abnormal, 
        "label": [0] * len(valid_test_normal) + [1] * len(valid_test_abnormal)
    })
    
    train_df.to_csv(target_dir / 'train.csv', index=False)
    train_df.to_csv(target_dir / 'valid.csv', index=False)
    test_df.to_csv(target_dir / 'test.csv', index=False)
    print(f"Generated train.csv with {len(train_df)} sequences.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
    args = parser.parse_args()
    
    setup_loganomaly(mode=args.mode)
