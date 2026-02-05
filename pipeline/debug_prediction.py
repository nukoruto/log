import torch
import json
import sys
from pathlib import Path
from collections import Counter
import pandas as pd

# LogDeepのモジュールを読み込むためのパス設定
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'models' / 'LogDeep'))

from logdeep.models.lstm import loganomaly

def debug_prediction():
    print("--- Model Prediction Debug ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 設定
    num_classes = 48  # detected value
    input_size = 1
    hidden_size = 64
    num_layers = 2
    window_size = 10
    
    # 2. モデル読み込み
    model_dir = project_root / 'models' / 'LogDeep' / 'result' / 'loganomaly'
    model_path = model_dir / 'loganomaly_last.pth'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = loganomaly(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_keys=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # 3. ベクトル読み込み
    vec_path = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs' / 'event2semantic_vec.json'
    print(f"Loading vectors from {vec_path}...")
    with open(vec_path, 'r') as f:
        event2vec = json.load(f)
        
    embedding_matrix = torch.zeros((num_classes, 300)).to(device)
    for i in range(num_classes):
        key = str(i)
        if key in event2vec:
            embedding_matrix[i] = torch.tensor(event2vec[key]).to(device)

    # 4. テストデータを1つだけ手動で作って予測させる
    # hdfs_test_normal の最初の1行目を想定（または典型的な正常パターン）
    # 例: 0番(元ID1)が続くパターン
    dummy_seq = [0, 1, 0, 0, 2, 3, 4, 2, 3, 2] # 適当なシーケンス（0始まり）
    ground_truth = 3 # 次に来るべき値
    
    print(f"\nTest Input Sequence (0-based): {dummy_seq}")
    print(f"Ground Truth Next ID: {ground_truth}")
    
    # 入力作成
    seq0 = dummy_seq
    seq1 = [0] * num_classes
    log_counter = Counter(seq0)
    for key in log_counter:
        if key < num_classes:
            seq1[key] = log_counter[key]

    seq0_idx = torch.tensor(seq0, dtype=torch.long).to(device)
    seq0_t = embedding_matrix[seq0_idx].view(-1, window_size, 300)
    seq1_t = torch.tensor(seq1, dtype=torch.float).view(-1, num_classes, input_size).to(device)
    
    # 5. 予測実行
    with torch.no_grad():
        output = model(features=[seq0_t, seq1_t], device=device)
        probs = torch.softmax(output, dim=1)
        
    # 6. 結果表示
    print("\n--- Prediction Results ---")
    top_k = 10
    vals, inds = torch.topk(probs, top_k)
    
    vals = vals[0].cpu().numpy()
    inds = inds[0].cpu().numpy()
    
    print(f"Top-{top_k} Predictions:")
    for i in range(top_k):
        pred_id = inds[i]
        prob = vals[i]
        mark = "<-- CORRECT" if pred_id == ground_truth else ""
        print(f"Rank {i+1}: ID {pred_id} (Prob: {prob:.4f}) {mark}")
        
    # 統計
    all_probs = probs[0].cpu().numpy()
    print(f"\nProbability Stats: Max={all_probs.max():.4f}, Min={all_probs.min():.4f}, Mean={all_probs.mean():.4f}")
    
    if inds[0] == 0 and vals[0] > 0.9:
        print("\nDIAGNOSIS: Model is collapsing to predict '0' for everything.")
    elif all_probs.max() < 0.1:
        print("\nDIAGNOSIS: Model is confused (flat distribution).")
    else:
        print("\nDIAGNOSIS: Model seems to have learned 'something'. Check if Ground Truth is in Top-9.")

if __name__ == "__main__":
    debug_prediction()