import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    # Just take first 100 sessions for speed verification
    if len(hdfs) > 100: 
        hdfs = hdfs[:100]
        hdfs_times = hdfs_times[:100]
        hdfs_labels = hdfs_labels[:100]

    return list(zip(hdfs, hdfs_times, hdfs_labels))

def evaluate(model_path, num_classes=28, window_size=10, data_dir=None, result_dir=None):
    if data_dir is None: data_dir = Path("data/HDFS/deeplog_input")
    if result_dir is None: result_dir = Path("evaluation/results/DeepLog")
    
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

    scores = [] # This will store e(t)
    dt_list = [] 
    gt_labels_list = [] # List of r(t) for plotting
    
    y_true_binary = [] # Session binary labels for F1
    y_pred_binary = [] 
    
    num_candidates = 1 
    
    print(f"Evaluating Normal Data ({len(normal_data)} sessions)...")
    with torch.no_grad():
        for line, times, labels in normal_data:
            session_failed = False
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                target_idx = line[i + window_size]
                
                # Ground Truth for this event
                r_t = labels[i + window_size] # 0 or 1
                
                # Time Delta
                t_target = times[i + window_size]
                t_last = times[i + window_size - 1]
                dt = max(0, t_target - t_last)
                
                if target_idx == -1: continue
                    
                seq_tensor = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                output = model(seq_tensor) 
                probs = torch.softmax(output, dim=1)
                
                # Model Anomaly Score y(t) = 1 - P(observed_event)
                # If observed event is valid (in vocab):
                if 0 <= target_idx < num_classes:
                    prob_gt = probs[0, target_idx].item()
                    y_t = 1.0 - prob_gt
                else:
                    # Unknown event -> Score 1.0 (Anomaly)
                    y_t = 1.0
                
                # Control Error e(t) = |r(t) - y(t)|
                e_t = abs(r_t - y_t)
                
                scores.append(e_t)
                dt_list.append(dt)
                gt_labels_list.append(r_t)
                
                # F1 Logic (Top K)
                predicted_topk = torch.argsort(output, 1)[0][-num_candidates:]
                if target_idx not in predicted_topk:
                    # Model flagged anomaly
                    session_failed = True
            
            y_true_binary.append(0)
            y_pred_binary.append(1 if session_failed else 0)

    print(f"Evaluating Abnormal Data ({len(abnormal_data)} sessions)...")
    with torch.no_grad():
        for line, times, labels in abnormal_data:
            session_failed = False
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

    # Calculate P/R/F1
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    r = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Calculate IAE/ISE using e(t) stored in 'scores'
    y_r_t = np.array(gt_labels_list) # Ground truth signal for plot
    y_e_t = np.array(scores)        # Error signal
    
    metrics = calculate_control_metrics(
        y_r_t, 
        y_e_t, 
        experiment_name="DeepLog_HDFS_Test", 
        output_dir=result_dir,
        dt=dt_list
    )
    
    metrics['Precision'] = float(p)
    metrics['Recall'] = float(r)
    metrics['F1'] = float(f1)
    
    print("Evaluation Results:")
    print(metrics)
    
    import json
    with open(result_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
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
    evaluate(latest_model, window_size=window_size, num_classes=num_classes, data_dir=data_dir, result_dir=result_dir)
