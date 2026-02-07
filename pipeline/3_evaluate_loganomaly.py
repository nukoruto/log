import torch
import torch.nn as nn
import time
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

# Add paths for LogDeep modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'models' / 'LogDeep'))

from logdeep.models.lstm import loganomaly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- INLINED METRICS FUNCTION ---
def _plot_waveform(t, scores, labels, metrics, title_prefix, output_dir):
    try:
        plt.figure(figsize=(15, 6))
        plt.plot(t, scores, label='Error Signal e(t) = |r(t)-y(t)|', color='blue', linewidth=0.8, alpha=0.7)
        anomaly_indices = np.where(labels == 1)[0]
        if len(anomaly_indices) > 0:
            valid_indices = anomaly_indices[anomaly_indices < len(scores)]
            if len(valid_indices) > 0:
                plt.scatter(valid_indices, scores[valid_indices], color='red', label='Ground Truth Anomaly', s=10, zorder=5)
        plt.title(f"{title_prefix} - Control Metrics\nIAE: {metrics['IAE']:.2f} | ISE: {metrics['ISE']:.2f}")
        plt.xlabel("Event Index (Time)")
        plt.ylabel("Error Signal")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        filename = output_dir / f"{title_prefix}_waveform.png"
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Warning: Plotting failed: {e}")

def calculate_control_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    experiment_name: str = "experiment",
    output_dir: Path = None,
    **kwargs
) -> Dict[str, float]:
    y_scores = np.array(y_scores)
    # Define Error Signal e_t
    e_t = y_scores 

    if 'dt' in kwargs and kwargs['dt'] is not None:
        dt = np.array(kwargs['dt'])
        if len(dt) != len(y_scores):
             pass
        t = np.insert(np.cumsum(dt), 0, 0)[:-1]
        iae = np.sum(np.abs(e_t) * dt)
        ise = np.sum((e_t ** 2) * dt)
        itae = np.sum(t * np.abs(e_t) * dt)
    else:
        t = np.arange(len(y_scores))
        iae = np.sum(np.abs(e_t))
        ise = np.sum(e_t ** 2)
        itae = np.sum(t * np.abs(e_t))
        
    n = len(y_scores)
    metrics = {
        "IAE": iae, "ISE": ise, "ITAE": itae,
        "Mean_IAE": iae / n if n else 0,
        "Mean_ISE": ise / n if n else 0,
        "Mean_ITAE": itae / n if n else 0
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_waveform(t, y_scores, y_true, metrics, experiment_name, output_dir)
        import json
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
    return metrics
# -------------------------------------------------------------

def generate(name, window_size=1, data_dir=None):
    if data_dir is None:
        data_base = project_root / 'data' / 'HDFS' / 'deeplog_input'
    else:
        data_base = Path(data_dir)

    # Modified to return list of (seq, time_seq, label_seq) tuples
    data_path = data_base / name
    time_path = project_root / 'data' / 'HDFS' / 'deeplog_input' / (name + "_time")
    label_path = project_root / 'data' / 'HDFS' / 'deeplog_input' / (name + "_label")
    
    hdfs_seqs = []
    hdfs_times = []
    hdfs_labels = []
    
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return []

    with open(data_path, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            if len(ln) < window_size: continue
            hdfs_seqs.append(ln)

    if time_path.exists():
        with open(time_path, 'r') as f:
            for ln in f.readlines():
                ln = list(map(float, ln.strip().split()))
                hdfs_times.append(ln)
    else:
        hdfs_times = [[0.0]*len(s) for s in hdfs_seqs]
        
    if label_path.exists():
        with open(label_path, 'r') as f:
            for ln in f.readlines():
                ln = list(map(int, ln.strip().split()))
                hdfs_labels.append(ln)
    else:
        hdfs_labels = [[0]*len(s) for s in hdfs_seqs]

    return list(zip(hdfs_seqs, hdfs_times, hdfs_labels))

def evaluate_loganomaly(window_size=10, mode='full', num_classes=28, num_candidates=9):
    input_size = 1
    hidden_size = 64
    num_layers = 2
    # num_classes = 28  # Removed to use argument
    # window_size arg used
    
    if mode == 'full':
        model_dir = project_root / 'models' / 'LogDeep' / 'result' / 'loganomaly'
        data_dir = project_root / 'data' / 'HDFS' / 'deeplog_input'
        output_dir = Path("evaluation/results/LogAnomaly")
    else:
        model_dir = project_root / 'models' / 'LogDeep' / 'result' / 'loganomaly_2k'
        data_dir = project_root / 'data' / 'HDFS' / 'deeplog_input_2k'
        output_dir = Path("evaluation/results/LogAnomaly_2k")
    
    model_path = model_dir / 'loganomaly_last.pth'
    if not model_path.exists():
        model_path = model_dir / 'loganomaly_bestloss.pth'
        if not model_path.exists():
            print(f"Model definition not found at {model_path}")
            return
            
    model = loganomaly(input_size=input_size, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       num_keys=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Load semantic vectors
    semantic_vec_path = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs' / 'event2semantic_vec.json'
    if mode != 'full':
         # Try 2k path if exists, otherwise fallback
         p = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs_2k' / 'hdfs' / 'event2semantic_vec.json'
         if p.exists():
             semantic_vec_path = p
    
    if not semantic_vec_path.exists():
        print(f"Error: Semantic vector file not found at {semantic_vec_path}")
        return

    import json
    with open(semantic_vec_path, 'r') as f:
        event2vec = json.load(f)
    
    # Create embedding matrix (num_classes x 300)
    embedding_matrix = torch.zeros((num_classes, 300)).to(device)
    for i in range(num_classes):
        # seq line is generated with n-1, so intId = i (0-based)
        # keys in json are now 0-based ('0'..'47')
        key = str(i)
        if key in event2vec:
            embedding_matrix[i] = torch.tensor(event2vec[key]).to(device)
        else:
            # If key not found (e.g. 0 padding or new event), leaving it as zeros is fine
            # But "0" in json is all zeros, maybe use that for i=-1?
            pass
    
    print(f"Generating test data from {data_dir}...")
    test_normal = generate('hdfs_test_normal', window_size, data_dir)
    test_abnormal = generate('hdfs_test_abnormal', window_size, data_dir)
    
    # Storage for results: each g from 1 to 9
    # Key: g value, Value: {'y_true': [], 'y_pred': []}
    results = {g: {'y_true': [], 'y_pred': []} for g in range(1, 10)}
    
    scores = [] # Keeping original score tracking for IAE/ISE (based on num_candidates arg)
    dt_list = []
    gt_labels_list = []
    
    # Pre-compute embedding matrix for faster lookup
    
    print(f"Evaluating Normal Data ({len(test_normal)} sessions)...")
    with torch.no_grad():
        for line, times, labels in test_normal:
            # Initialize session failure flags for each g
            session_failed = {g: False for g in range(1, 10)}
            
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                target_idx = line[i + window_size]
                r_t = labels[i + window_size]
                
                t_target = times[i + window_size]
                t_last = times[i + window_size - 1]
                dt = max(0, t_target - t_last)
                
                seq1 = [0] * num_classes
                log_counter = Counter(seq0)
                for key in log_counter:
                    if key < num_classes:
                        seq1[key] = log_counter[key]

                seq0_idx = torch.tensor(seq0, dtype=torch.long).to(device)
                seq0_t = embedding_matrix[seq0_idx].view(-1, window_size, 300)
                seq1_t = torch.tensor(seq1, dtype=torch.float).view(-1, num_classes, input_size).to(device)
                
                output = model(features=[seq0_t, seq1_t], device=device)
                
                # Check rank of target
                # descending=True means first element is highest prob
                sorted_indices = torch.argsort(output, 1, descending=True)[0]
                
                # Find rank of target_idx (0-based)
                # If target_idx is not in sorted_indices (shouldn't happen if coverage is full), we assume worst
                if target_idx < num_classes:
                     try:
                         rank = (sorted_indices == target_idx).nonzero().item()
                     except ValueError:
                         rank = num_classes # Not found
                else:
                     rank = num_classes # Target out of bounds
                
                # Update session failure for each g
                # If rank < g, then it is in Top-g (Success)
                # If rank >= g, then it is NOT in Top-g (Failure -> Anomaly)
                for g in range(1, 10):
                    if rank >= g:
                        session_failed[g] = True

                # Calculate scalar score for default calculation (using num_candidates input)
                probs = torch.softmax(output, dim=1)
                if 0 <= target_idx < num_classes:
                    prob_gt = probs[0, target_idx].item()
                    y_t = 1.0 - prob_gt
                else:
                    y_t = 1.0
                e_t = abs(r_t - y_t)
                scores.append(e_t)
                dt_list.append(dt)
                gt_labels_list.append(r_t)
            
            # End of session
            for g in range(1, 10):
                results[g]['y_true'].append(0) # Normal
                results[g]['y_pred'].append(1 if session_failed[g] else 0)

    print(f"Evaluating Abnormal Data ({len(test_abnormal)} sessions)...")
    with torch.no_grad():
        for line, times, labels in test_abnormal:
            session_failed = {g: False for g in range(1, 10)}
            
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                target_idx = line[i + window_size]
                r_t = labels[i + window_size]
                
                t_target = times[i + window_size]
                dt = max(0, t_target - times[i + window_size - 1])
                
                seq1 = [0] * num_classes
                log_counter = Counter(seq0)
                for key in log_counter:
                    if key < num_classes:
                        seq1[key] = log_counter[key]

                seq0_idx = torch.tensor(seq0, dtype=torch.long).to(device)
                seq0_t = embedding_matrix[seq0_idx].view(-1, window_size, 300)
                seq1_t = torch.tensor(seq1, dtype=torch.float).view(-1, num_classes, input_size).to(device)
                
                output = model(features=[seq0_t, seq1_t], device=device)
                
                sorted_indices = torch.argsort(output, 1, descending=True)[0]
                
                if target_idx < num_classes:
                     try:
                         rank = (sorted_indices == target_idx).nonzero().item()
                     except ValueError:
                         rank = num_classes
                else:
                     rank = num_classes
                
                for g in range(1, 10):
                    if rank >= g:
                        session_failed[g] = True
                        
                # Default score calc
                probs = torch.softmax(output, dim=1)
                if 0 <= target_idx < num_classes:
                    prob_gt = probs[0, target_idx].item()
                    y_t = 1.0 - prob_gt
                else:
                    y_t = 1.0
                e_t = abs(r_t - y_t)
                scores.append(e_t)
                dt_list.append(dt)
                gt_labels_list.append(r_t)
            
            for g in range(1, 10):
                results[g]['y_true'].append(1) # Abnormal
                results[g]['y_pred'].append(1 if session_failed[g] else 0)
    
    # Calculate Control Metrics (Probabilistic, independent of g)
    y_r_t = np.array(gt_labels_list)
    y_e_t = np.array(scores)
    
    control_metrics = calculate_control_metrics(
        y_r_t, 
        y_e_t, 
        experiment_name=f"LogAnomaly_HDFS_{mode}_Test", 
        output_dir=output_dir,
        dt=dt_list
    ) 

    # Calculate Metrics for all g
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    all_metrics = {}
    print("\n--- Evaluation Results for g=1 to 9 ---")
    print(f"{'g':<3} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'IAE':<12} | {'ISE':<12} | {'ITAE':<12}")
    print("-" * 85)
    
    for g in range(1, 10):
        y_true = results[g]['y_true']
        y_pred = results[g]['y_pred']
        
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Add control metrics (Constant for all g)
        all_metrics[str(g)] = {
            'Precision': float(p),
            'Recall': float(r),
            'F1': float(f1),
            'IAE': control_metrics['IAE'],
            'ISE': control_metrics['ISE'],
            'ITAE': control_metrics['ITAE']
        }
        print(f"{g:<3} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {control_metrics['IAE']:<12.2e} | {control_metrics['ISE']:<12.2e} | {control_metrics['ITAE']:<12.2e}")
        
    # Save combined metrics
    with open(output_dir / 'metrics_multi_g.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    # Also save the standard single file for the 'num_candidates' requested (for compatibility)
    # Add F1 info for the requested g to the standard metrics.json
    req_g = str(num_candidates)
    if req_g in all_metrics:
        control_metrics['Precision'] = all_metrics[req_g]['Precision']
        control_metrics['Recall'] = all_metrics[req_g]['Recall']
        control_metrics['F1'] = all_metrics[req_g]['F1']
        
    print(f"\nEvaluation Results (LogAnomaly) for requested g={num_candidates}:")
    print(control_metrics)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(control_metrics, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
    parser.add_argument('--g', type=int, default=3, help='Top-k candidates (g value)')
    args = parser.parse_args()
    
    mapping_path = None
    if args.mode == 'full':
        window_size = 10
        mapping_path = project_root / "data/HDFS/deeplog_input/event_mapping.csv"
    else:
        window_size = 1
        mapping_path = project_root / "data/HDFS/deeplog_input_2k/event_mapping.csv"
        
    num_classes = 28
    if mapping_path and mapping_path.exists():
        import pandas as pd
        df = pd.read_csv(mapping_path)
        num_classes = df['IntId'].max()
        print(f"Detected num_classes: {num_classes}")
    
    evaluate_loganomaly(window_size=window_size, mode=args.mode, num_classes=num_classes, num_candidates=args.g)
