import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict

# Import generic metrics (Still attempting, but defining inline for safety)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- INLINED METRICS FUNCTION ---
def _plot_waveform(t, scores, labels, metrics, title_prefix, output_dir):
    try:
        plt.figure(figsize=(15, 6))
        plt.plot(t, scores, label='Error Signal e(t) = |r(t)-y(t)|', color='blue', linewidth=0.8, alpha=0.7)
        # Highlight anomalies (Reference = 1)
        anomaly_indices = np.where(labels == 1)[0]
        if len(anomaly_indices) > 0:
            # Check range
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
    e_t = y_scores # In this new definition, y_scores passed IS e(t)

    # Time steps
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

def generate_data(name, window_size, data_dir=Path("data/HDFS/deeplog_input")):
    hdfs = []
    hdfs_times = []
    hdfs_labels = []
    
    base_path = Path(data_dir)
    data_path = base_path / name
    time_path = base_path / (name + "_time")
    label_path = base_path / (name + "_label") # New
    
    # Load Events
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return []

    with open(data_path, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            if len(ln) < window_size:
                ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.append(tuple(ln))

    # Load Times
    if time_path.exists():
        with open(time_path, 'r') as f:
            for ln in f.readlines():
                ln = list(map(float, ln.strip().split()))
                if len(ln) < window_size:
                     ln = ln + [0.0] * (window_size + 1 - len(ln))
                hdfs_times.append(tuple(ln))
    else:
        hdfs_times = [tuple([0.0]*len(x)) for x in hdfs]

    # Load Labels
    if label_path.exists():
        with open(label_path, 'r') as f:
            for ln in f.readlines():
                ln = list(map(int, ln.strip().split()))
                if len(ln) < window_size:
                     ln = ln + [0] * (window_size + 1 - len(ln))
                hdfs_labels.append(tuple(ln))
    else:
        # Fallback (Assume 0 for normal, but warning needed)
        print(f"Warning: Label file not found at {label_path}")
        hdfs_labels = [tuple([0]*len(x)) for x in hdfs]

    # Remove debug slicing for full evaluation
    # if len(hdfs) > 100: 
    #     hdfs = hdfs[:100]
    #     hdfs_times = hdfs_times[:100]
    #     hdfs_labels = hdfs_labels[:100]

    return list(zip(hdfs, hdfs_times, hdfs_labels))

    # Multi-G Evaluation
    metrics_multi_g = {}
    
    # Pre-calculate IAE/ISE (Probabilistic, independent of g)
    y_r_t = np.array(gt_labels_list)
    y_e_t = np.array(scores)
    control_metrics = calculate_control_metrics(
        y_r_t, 
        y_e_t, 
        experiment_name="DeepLog_HDFS_Test", 
        output_dir=result_dir,
        dt=dt_list
    )
    
    # Calculate F1 for each g
    print("\n--- Evaluation Results for g=1 to 9 ---")
    print(f"{'g':<3} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 43)
    
    for g in range(1, 10):
        # Re-evaluate binary predictions for this g
        # We need to check if target is in Top-g.
        # Since we didn't store all outputs to save memory, we might need to re-run or 
        # distinct strategy: Store the RANK of the target in the probability list.
        # But we didn't store ranks. 
        # WAIT: Running model inference 9 times is wasteful. 
        # Optimization: We already have 'scores' (prob of target). 
        # But 'scores' is (1 - P_target). We don't know the P_others.
        # Top-g depends on whether other classes have higher prob.
        # We can't know if target is in Top-g just from P_target.
        pass
    
    # Correction: I need to modify the INFERENCE loop to calculate ranks once or store sufficient info.
    # Storing (N, num_classes) probs is too big (111k * 28 floats).
    # Storing just the RANK of the target is sufficient!
    # Rank 0 = Top-1. Rank k = Top-(k+1).
    # If Rank < g, then Correct.
    
    # I will modify the loop above to store 'target_ranks'.
    pass

def evaluate(model_path, num_classes=28, window_size=10, data_dir=None, result_dir=None, topk=9):
    if data_dir is None: data_dir = Path("data/HDFS/deeplog_input")
    if result_dir is None: result_dir = Path("evaluation/results/DeepLog")
    result_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    
    input_size = 1
    hidden_size = 64
    num_layers = 2
    
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test data
    normal_data = generate_data('hdfs_test_normal', window_size, data_dir)
    abnormal_data = generate_data('hdfs_test_abnormal', window_size, data_dir)
    
    if not normal_data and not abnormal_data:
        print("No test data found.")
        return

    scores = []
    dt_list = [] 
    gt_labels_list = []
    
    # Store ranks of the correct target for each window
    # If target is normal (in vocab), store its rank (0-based).
    # If target is abnormal/unknown, rank is inf? -> effectively num_classes
    target_ranks = [] 
    
    # Session tracking: We need to know which windows belong to which session to aggregate per session
    # Structure: session_ranks = [ [rank_w1, rank_w2], [rank_w1], ... ]
    # Also separate Normal vs Abnormal sessions
    
    normal_session_ranks = []
    abnormal_session_ranks = []
    
    print(f"Evaluating Normal Data ({len(normal_data)} sessions)...")
    with torch.no_grad():
        for line, times, labels in normal_data:
            s_ranks = []
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                target_idx = line[i + window_size]
                r_t = labels[i + window_size]
                
                t_target = times[i + window_size]
                t_last = times[i + window_size - 1]
                dt = max(0, t_target - t_last)
                
                if target_idx == -1: continue
                    
                seq_tensor = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                output = model(seq_tensor) 
                probs = torch.softmax(output, dim=1)
                
                # IAE Score
                if 0 <= target_idx < num_classes:
                    prob_gt = probs[0, target_idx].item()
                    y_t = 1.0 - prob_gt
                    
                    # Calculate Rank
                    # argsort descends? No, ascends. 
                    # We want descending order of prob.
                    # rank = count(p > p_gt)
                    # Let's use argsort
                    sorted_idxs = torch.argsort(output, 1, descending=True)[0]
                    # Find location of target_idx
                    rank = (sorted_idxs == target_idx).nonzero(as_tuple=True)[0].item()
                    s_ranks.append(rank)
                    
                else:
                    y_t = 1.0
                    s_ranks.append(num_classes + 1) # Effectively infinity
                
                e_t = abs(r_t - y_t)
                scores.append(e_t)
                dt_list.append(dt)
                gt_labels_list.append(r_t)
            
            normal_session_ranks.append(s_ranks)

    print(f"Evaluating Abnormal Data ({len(abnormal_data)} sessions)...")
    with torch.no_grad():
        for line, times, labels in abnormal_data:
            s_ranks = []
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                target_idx = line[i + window_size]
                r_t = labels[i + window_size]
                
                t_target = times[i + window_size]
                t_last = times[i + window_size - 1]
                dt = max(0, t_target - t_last)
                
                if target_idx == -1: continue

                seq_tensor = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                output = model(seq_tensor) 
                probs = torch.softmax(output, dim=1)
                
                if 0 <= target_idx < num_classes:
                    prob_gt = probs[0, target_idx].item()
                    y_t = 1.0 - prob_gt
                    
                    sorted_idxs = torch.argsort(output, 1, descending=True)[0]
                    rank = (sorted_idxs == target_idx).nonzero(as_tuple=True)[0].item()
                    s_ranks.append(rank)
                else:
                    y_t = 1.0
                    s_ranks.append(num_classes + 1)
                    
                e_t = abs(r_t - y_t)
                scores.append(e_t)
                dt_list.append(dt)
                gt_labels_list.append(r_t)
                
            abnormal_session_ranks.append(s_ranks)

    # --- Multi-G Analysis ---
    metrics_multi_g = {}
    
    # Calculate IAE (Constant)
    y_r_t = np.array(gt_labels_list)
    y_e_t = np.array(scores)
    
    # Base control metrics
    control_metrics = calculate_control_metrics(
        y_r_t, y_e_t, experiment_name="DeepLog_HDFS_Test", output_dir=result_dir, dt=dt_list
    )
    
    print("\n--- Evaluation Results for g=1 to 9 ---")
    print(f"{'g':<3} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'IAE':<12} | {'ISE':<12} | {'ITAE':<12}")
    print("-" * 85)

    from sklearn.metrics import precision_score, recall_score, f1_score

    for g in range(1, 10):
        # Determine Anomaly for each session based on ranks
        # If ANY rank in session >= g (0-based rank implies rank < g is Top-g), then Anomaly.
        # Wait, Top-k inclusive. 
        # g=1 (Top-1): rank 0 is correct. rank >= 1 is anomaly.
        # g=9 (Top-9): rank 0..8 is correct. rank >= 9 is anomaly.
        # So Condition: Anomaly if rank >= g
        
        y_true = []
        y_pred = []
        
        # Normal Sessions (Should be 0, Predicted 1 if rank >= g)
        for ranks in normal_session_ranks:
            y_true.append(0)
            # Check if any event was mispredicted
            is_anomaly = any(r >= g for r in ranks)
            y_pred.append(1 if is_anomaly else 0)
            
        # Abnormal Sessions (Should be 1, Predicted 1 if rank >= g)
        for ranks in abnormal_session_ranks:
            y_true.append(1)
            is_anomaly = any(r >= g for r in ranks)
            y_pred.append(1 if is_anomaly else 0)
            
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"{g:<3} | {p:.4f}     | {r:.4f}     | {f1:.4f}")
        
        # Add control metrics (Constant for all g)
        metrics_multi_g[str(g)] = {
            "Precision": float(p),
            "Recall": float(r),
            "F1": float(f1),
            "IAE": control_metrics['IAE'], # Constant
            "ISE": control_metrics['ISE'],
            "ITAE": control_metrics['ITAE']
        }
        print(f"{g:<3} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {control_metrics['IAE']:<12.2e} | {control_metrics['ISE']:<12.2e} | {control_metrics['ITAE']:<12.2e}")

    # Save Multi-G Results
    with open(result_dir / 'metrics_multi_g.json', 'w') as f:
        json.dump(metrics_multi_g, f, indent=4)
        
    print(f"\nSaved results to {result_dir / 'metrics_multi_g.json'}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
    parser.add_argument('--g', type=int, default=9, help='Top-K candidates (g) value for evaluation')
    args = parser.parse_args()
    
    if args.mode == 'full':
        window_size = 10
        data_dir = Path("data/HDFS/deeplog_input")
        result_dir = Path("evaluation/results/DeepLog")
        mapping_path = data_dir / "event_mapping.csv"
    else:
        window_size = 1
        data_dir = Path("data/HDFS/deeplog_input_2k")
        result_dir = Path("evaluation/results/DeepLog_2k")
        mapping_path = data_dir / "event_mapping.csv"
        
    num_classes = 28
    if mapping_path.exists():
        import pandas as pd
        df = pd.read_csv(mapping_path)
        num_classes = df['IntId'].max()
        print(f"Detected num_classes: {num_classes}")
    
    model_dir = Path("model")
    models = list(model_dir.glob("*.pt"))
    if not models:
        print("No models found!")
        sys.exit(1)
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    
    print(f"Evaluating with window_size={window_size}, num_classes={num_classes} (Mode: {args.mode})")
    print(f"Data Source: {data_dir}")
    print(f"Results: {result_dir}")
    evaluate(latest_model, window_size=window_size, num_classes=num_classes, data_dir=data_dir, result_dir=result_dir, topk=args.g)
