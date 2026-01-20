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

def evaluate_loganomaly(window_size=10, mode='full'):
    input_size = 1
    hidden_size = 64
    num_layers = 2
    num_classes = 28
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
    
    print(f"Generating test data from {data_dir}...")
    test_normal = generate('hdfs_test_normal', window_size, data_dir)
    test_abnormal = generate('hdfs_test_abnormal', window_size, data_dir)
    
    scores = []
    dt_list = []
    gt_labels_list = []
    
    y_true_binary = []
    y_pred_binary = []
    num_candidates = 1
    
    print(f"Evaluating Normal Data ({len(test_normal)} sessions)...")
    with torch.no_grad():
        for line, times, labels in test_normal:
            session_failed = False
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                target_idx = line[i + window_size]
                r_t = labels[i + window_size]
                
                t_target = times[i + window_size]
                t_last = times[i + window_size - 1]
                dt = max(0, t_target - t_last)
                
                seq1 = [0] * 28
                log_counter = Counter(seq0)
                for key in log_counter:
                    if key < 28:
                        seq1[key] = log_counter[key]

                seq0_t = torch.tensor(seq0, dtype=torch.float).view(-1, window_size, input_size).to(device)
                seq1_t = torch.tensor(seq1, dtype=torch.float).view(-1, num_classes, input_size).to(device)
                
                output = model(features=[seq0_t, seq1_t], device=device)
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
                
                predicted_topk = torch.argsort(output, 1)[0][-num_candidates:]
                if target_idx not in predicted_topk:
                    session_failed = True
            
            y_true_binary.append(0)
            y_pred_binary.append(1 if session_failed else 0)

    print(f"Evaluating Abnormal Data ({len(test_abnormal)} sessions)...")
    with torch.no_grad():
        for line, times, labels in test_abnormal:
            session_failed = False
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                target_idx = line[i + window_size]
                r_t = labels[i + window_size]
                
                t_target = times[i + window_size]
                dt = max(0, t_target - times[i + window_size - 1])
                
                seq1 = [0] * 28
                log_counter = Counter(seq0)
                for key in log_counter:
                    if key < 28:
                        seq1[key] = log_counter[key]

                seq0_t = torch.tensor(seq0, dtype=torch.float).view(-1, window_size, input_size).to(device)
                seq1_t = torch.tensor(seq1, dtype=torch.float).view(-1, num_classes, input_size).to(device)
                
                output = model(features=[seq0_t, seq1_t], device=device)
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
                
                predicted_topk = torch.argsort(output, 1)[0][-num_candidates:]
                if target_idx not in predicted_topk:
                    session_failed = True
            
            y_true_binary.append(1)
            y_pred_binary.append(1 if session_failed else 0)
    
    y_r_t = np.array(gt_labels_list)
    y_e_t = np.array(scores)
    
    metrics = calculate_control_metrics(
        y_r_t, 
        y_e_t, 
        experiment_name=f"LogAnomaly_HDFS_{mode}_Test", 
        output_dir=output_dir,
        dt=dt_list
    )
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    r = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    metrics['Precision'] = float(p)
    metrics['Recall'] = float(r)
    metrics['F1'] = float(f1)
    
    print("Evaluation Results (LogAnomaly):")
    print(metrics)
    
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
    args = parser.parse_args()
    
    window_size = 10 if args.mode == 'full' else 1
    
    # Inject window_size into evaluate function via a wrapper or just modify evaluate_loganomaly
    # Actually evaluate_loganomaly hardcodes window_size. Let's pass it.
    # We need to change evaluate_loganomaly signature.
    evaluate_loganomaly(window_size=window_size, mode=args.mode)
