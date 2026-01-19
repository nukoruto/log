"""
Control Engineering Metrics for Log Anomaly Detection Evaluation.
Calculates IAE (Integrated Absolute Error), ISE (Integrated Squared Error),
and ITAE (Integrated Time-weighted Absolute Error) for anomaly score sequences.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_control_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    experiment_name: str = "experiment",
    output_dir: Path = None
) -> Dict[str, float]:
    """
    Calculate Control Engineering Metrics (IAE, ISE, ITAE) for a sequence of anomaly scores.
    
    Args:
        y_true (np.ndarray): Ground truth labels (0=Normal, 1=Anomaly). 
                             Note: These metrics typically treat 'anomaly' as deviation from 0.
                             However, in this context, we are evaluating the *anomaly score signal* produced by the model.
                             Ideally, for normal data, score should be 0. For anomalies, score > 0.
        y_scores (np.ndarray): Predicted anomaly scores (probabilities or reconstruction errors).
        experiment_name (str): Name for identifying plots/results.
        output_dir (Path): Directory to save waveform plots.
        
    Returns:
        Dict[str, float]: Dictionary containing IAE, ISE, ITAE values.
    """
    
    # Ensure numpy arrays
    y_scores = np.array(y_scores)
    
    # Time steps (assuming uniform sampling)
    t = np.arange(len(y_scores))
    
    # Error signal e(t). 
    # In control theory, e(t) = Setpoint - ProcessValue.
    # Here, 'Setpoint' is Normal (0). ProcessValue is the Anomaly Score.
    # So e(t) = 0 - Score(t) = -Score(t).
    # Since we take Absolute/Squared values, we can just use Score(t) directly as the magnitude of deviation.
    # We assume the model attempts to minimize this score for normal data.
    e_t = y_scores
    
    # IAE: Integrated Absolute Error = sum(|e(t)|) * dt
    # We assume dt=1 (event step)
    iae = np.sum(np.abs(e_t))
    
    # ISE: Integrated Squared Error = sum(e(t)^2) * dt
    ise = np.sum(e_t ** 2)
    
    # ITAE: Integrated Time-weighted Absolute Error = sum(t * |e(t)|) * dt
    itae = np.sum(t * np.abs(e_t))
    
    # Normalization (Optional but helpful for comparing different sequence lengths)
    # If we want "Mean" metrics, we divide by N. 
    # Standard IAE/ISE are integrals (sums), so they grow with time.
    # For fair comparison across different test set sizes, we often state "Average IAE per event".
    # Let's provide both or standard sum. 
    # The user's previous code used Mean. Let's provide Mean versions as primary `iae`, `ise`.
    n = len(y_scores)
    mean_iae = iae / n if n > 0 else 0.0
    mean_ise = ise / n if n > 0 else 0.0
    mean_itae = itae / n if n > 0 else 0.0 # Note: Mean ITAE is less standard but calculable.
    
    metrics = {
        "IAE": iae,
        "ISE": ise,
        "ITAE": itae,
        "Mean_IAE": mean_iae,
        "Mean_ISE": mean_ise,
        "Mean_ITAE": mean_itae
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_waveform(t, y_scores, y_true, metrics, experiment_name, output_dir)
        
    return metrics

def _plot_waveform(
    t: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    metrics: Dict[str, float],
    title_prefix: str,
    output_dir: Path
):
    plt.figure(figsize=(15, 6))
    
    # Plot Anomaly Score (Error Signal)
    plt.plot(t, scores, label='Anomaly Score (Error)', color='blue', linewidth=0.8, alpha=0.7)
    
    # Highlight actual anomalies
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, scores[anomaly_indices], color='red', label='Ground Truth Anomaly', s=10, zorder=5)
        
    plt.title(f"{title_prefix} - Control Metrics Waveform\nIAE: {metrics['IAE']:.2f} | ISE: {metrics['ISE']:.2f}")
    plt.xlabel("Event Index (Time)")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filename = output_dir / f"{title_prefix}_waveform.png"
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Test stub
    dummy_scores = np.random.rand(100) * 0.1
    dummy_scores[50:60] += 0.8 # Simulated anomaly
    dummy_labels = np.zeros(100)
    dummy_labels[50:60] = 1
    
    print(calculate_control_metrics(dummy_labels, dummy_scores, "Test", Path(".")))
