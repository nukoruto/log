import sys
from pathlib import Path

def compare_data():
    project_root = Path(__file__).resolve().parent.parent
    
    # 1. 学習データ (Training Data)
    train_path = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs' / 'hdfs_train'
    
    # 2. テストデータ (Test Data)
    # 3_evaluate_loganomaly.py が読み込んでいる場所
    test_dir = project_root / 'data' / 'HDFS' / 'deeplog_input'
    # ファイル名が変わっている可能性があるので検索します
    test_files = list(test_dir.glob('*normal*')) 
    if not test_files:
        test_files = list(test_dir.glob('*'))
    test_path = test_files[0] if test_files else None

    print(f"--- Data Mismatch Check ---")
    
    # 学習データの確認
    if train_path.exists():
        with open(train_path, 'r') as f:
            train_seq = f.readline().strip().split()[:20]
        print(f"[Train Data] {train_path.name}:")
        print(f"Sample: {train_seq}")
    else:
        print(f"[Train Data] NOT FOUND at {train_path}")

    # テストデータの確認
    if test_path and test_path.exists():
        with open(test_path, 'r') as f:
            # deeplog_input形式は通常 "1 2 3..." とスペース区切り
            test_seq = f.readline().strip().split()[:20]
        print(f"\n[Test Data]  {test_path.name}:")
        print(f"Sample: {test_seq}")
    else:
        print(f"\n[Test Data] NOT FOUND at {test_dir}")

    print("\n--- Judgment ---")
    if train_path.exists() and test_path:
        # 先頭の数字を比較
        if train_seq[:5] == test_seq[:5]:
            print("MATCH! The sequences look similar. Problem is elsewhere.")
        else:
            print("MISMATCH DETECTED! (Fatal Error)")
            print("The ID '1' in training is NOT the same as ID '1' in testing.")
            print("Solution: You must regenerate Test Data using the SAME mapping as Training Data.")

if __name__ == "__main__":
    compare_data()