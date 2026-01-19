#!/usr/bin/env python3
"""
Interactive Pipeline Script
Automates the workflow from AIT Log preprocessing to MATLAB .mat export.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, env=None):
    """Run a shell command and check for errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def prompt_path(prompt_text, check_exists=True):
    """Interactively prompt for a file path."""
    while True:
        path_str = input(prompt_text).strip().strip('"').strip("'")
        if not path_str:
            print("Path cannot be empty.")
            continue
        path = Path(path_str)
        if check_exists and not path.exists():
            print(f"File not found: {path}")
            continue
        return path

def setup_python_path():
    """Ensure package sources are in PYTHONPATH for the subprocesses."""
    # Assuming this script is in scripts/ and packages are in packages/
    root_dir = Path(__file__).parent.parent
    packages_dir = root_dir / "packages"
    
    # Add all package src dirs
    src_paths = []
    for pkg in ["ds_contract", "models_lstm", "scenario_design", "log_generator"]:
        src = packages_dir / pkg / "src"
        if src.exists():
            src_paths.append(str(src))
            
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(src_paths + [current_pythonpath])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env



def main():
    print("=== AIT Log Pipeline Automation ===")
    
    # 1. Gather Inputs
    log_file = prompt_path("Enter path to access.log: ")
    label_file = prompt_path("Enter path to labels.log (or .csv): ")
    
    out_dir_str = input("Enter output directory [default: data/pipeline_out]: ").strip()
    if not out_dir_str:
        out_dir_str = "data/pipeline_out"
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seed = input("Enter Random Seed [default: 42]: ").strip()
    if not seed:
        seed = "42"
    
    env = setup_python_path()
    
    # 2. Process AIT (Preprocessing)
    print("\n--- Step 1: Preprocessing (ds-contract process-ait) ---")
    processed_prefix = out_dir / "processed"
    cmd_process = [
        sys.executable, "-m", "ds_contract.cli",
        "--seed", seed,
        "process-ait",
        str(log_file),
        str(label_file),
        "--out", str(processed_prefix)
    ]
    run_command(cmd_process, env)
    
    train_csv = Path(f"{processed_prefix}_train_normal.csv")
    test_csv = Path(f"{processed_prefix}_test_dataset.csv")
    
    if not train_csv.exists() or not test_csv.exists():
        print("Error: Preprocessing failed to generate expected files.")
        sys.exit(1)

    # 3. Train Model
    print("\n--- Step 2: Training LSTM Model (models-lstm train) ---")
    print("\n--- Training Configuration ---")
    epochs = input("Enter Training Epochs [default: 5]: ").strip() or "5"
    batch_size = input("Enter Batch Size [default: 32]: ").strip() or "32"
    hidden_dim = input("Enter Hidden Dimension [default: 64]: ").strip() or "64"

    run_dir = out_dir / "models"
    cmd_train = [
        sys.executable, "-m", "models_lstm.cli",
        "train",
        "--normal", str(train_csv),
        "--val", str(train_csv), # Using training data for val as simple default
        "--out", str(run_dir),
        "--seed", seed,
        "--epochs", epochs,
        "--batch-size", batch_size,
        "--hidden-dim", hidden_dim,
    ]
    # Check if user wants to customize epochs? For "quickstart" simplicity, fixed is fine.
    run_command(cmd_train, env)
    
    model_ckpt = run_dir / "best.ckpt"
    if not model_ckpt.exists():
        print("Error: Training failed to generate checkpoint.")
        sys.exit(1)

    # 4. Score Data
    print("\n--- Step 3: Scoring (models-lstm score) ---")
    scored_csv = out_dir / "scored.csv"
    cmd_score = [
        sys.executable, "-m", "models_lstm.cli",
        "score",
        "--model", str(model_ckpt),
        "--in", str(test_csv),
        "--out", str(scored_csv),
        "--seed", seed
    ]
    run_command(cmd_score, env)

    # Remove Step 4: MATLAB Export
    print("\n=== Pipeline Complete ===")
    print(f"Scored CSV: {scored_csv}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline aborted by user.")
