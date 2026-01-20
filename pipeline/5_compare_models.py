import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def compare_models():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / 'evaluation' / 'results'
    
    models = ['DeepLog', 'LogAnomaly']
    metrics_data = []
    
    for model in models:
        json_path = results_dir / model / 'metrics.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                data['Model'] = model
                metrics_data.append(data)
        else:
            print(f"Warning: Results for {model} not found at {json_path}")
            
    if not metrics_data:
        print("No metrics found.")
        return

    df = pd.DataFrame(metrics_data)
    # Reorder columns
    cols = ['Model', 'IAE', 'ISE', 'ITAE', 'Precision', 'Recall', 'F1']
    # Filter if columns exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    print("\n=== Benchmark Results ===")
    print(df.to_string(index=False))
    print("=========================")
    
    # Save comparison to CSV
    output_path = results_dir / 'benchmark_comparison.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")

if __name__ == "__main__":
    compare_models()
