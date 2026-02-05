import json
import sys
from pathlib import Path
import torch
import numpy as np

def debug_check():
    # パス設定
    project_root = Path(__file__).resolve().parent.parent
    hdfs_dir = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs'
    
    vec_path = hdfs_dir / 'event2semantic_vec.json'
    train_data_path = hdfs_dir / 'hdfs_train'
    
    print(f"--- Debugging Alignment ---")
    print(f"Data Dir: {hdfs_dir}")
    
    # 1. ベクトルファイルの確認
    if not vec_path.exists():
        print(f"ERROR: Vector file missing at {vec_path}")
        return
        
    with open(vec_path, 'r') as f:
        event2vec = json.load(f)
    
    print(f"Vector keys found: {list(event2vec.keys())[:5]} ...")
    print(f"Total vectors: {len(event2vec)}")
    
    # "0"番（元ID:1）のベクトルの中身チェック
    if "0" in event2vec:
        vec0 = np.array(event2vec["0"])
        norm0 = np.linalg.norm(vec0)
        print(f"Vector '0' (Original ID 1): Norm = {norm0:.4f}")
        if norm0 == 0:
            print("WARNING: Vector '0' is all ZEROS! (Model learns nothing for this event)")
    else:
        print("ERROR: Key '0' not found in vectors! (Did you run 4_setup...py?)")

    # 2. トレーニングデータの確認
    if not train_data_path.exists():
        print(f"ERROR: Train data missing at {train_data_path}")
        return
        
    with open(train_data_path, 'r') as f:
        first_line = f.readline().strip()
    
    print(f"First line of hdfs_train (Raw): {first_line[:50]}...")
    raw_ids = list(map(int, first_line.split()))
    print(f"First 5 Raw IDs: {raw_ids[:5]}")
    
    # 3. シミュレーション（Evaluateと同じ処理）
    # Evaluateスクリプトは「n - 1」をしている
    shifted_ids = [n - 1 for n in raw_ids[:5]]
    print(f"Shifted IDs (Input to Model): {shifted_ids}")
    
    # 4. マッピング確認
    print("\n--- Mapping Check ---")
    for raw, shifted in zip(raw_ids[:5], shifted_ids):
        key = str(shifted)
        status = "OK" if key in event2vec else "MISSING"
        vec_norm = 0
        if status == "OK":
            vec_norm = np.linalg.norm(event2vec[key])
            if vec_norm == 0: status = "ZERO_VECTOR"
            
        print(f"Raw ID: {raw} -> Shifted: {shifted} -> Key: '{key}' -> {status} (Norm: {vec_norm:.2f})")

    # 5. モデルの期待値確認
    print("\n--- Hypothesis ---")
    if all(str(s) in event2vec for s in shifted_ids) and all(np.linalg.norm(event2vec[str(s)]) > 0 for s in shifted_ids):
        print("Data alignment seems CORRECT.")
        print("If accuracy is still bad, the issue might be:")
        print("1. 'sample.py' in LogDeep is NOT subtracting 1 (mismatching Evaluate).")
        print("2. The Model converged to a local minimum (predicting only class 0?).")
    else:
        print("Data alignment is BROKEN. See above for details.")

if __name__ == "__main__":
    debug_check()