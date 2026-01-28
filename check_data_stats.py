
import matplotlib.pyplot as plt
import sys

def check_stats(filepath):
    total_lines = 0
    total_events = 0
    max_len = 0
    lens = []
    
    with open(filepath, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            n = len(tokens)
            total_lines += 1
            total_events += n
            if n > max_len:
                max_len = n
            lens.append(n)
            
    print(f"File: {filepath}")
    print(f"Total lines: {total_lines}")
    print(f"Total events: {total_events}")
    print(f"Max length: {max_len}")
    print(f"Avg length: {total_events / total_lines if total_lines else 0}")
    
    window_size = 10
    estimated_samples = sum(max(0, l - window_size) for l in lens)
    print(f"Estimated samples (window=10): {estimated_samples}")
    
    # Estimate memory for list of tuples
    # Each tuple: 48 bytes base + 8 bytes * 10 items = 128 bytes?
    # Plus reference in 'inputs' list: 8 bytes.
    # Total per sample ~ 140 bytes.
    # Plus integer objects (small ints are shared, so 0).
    print(f"Estimated Python list memory: {estimated_samples * 150 / 1024 / 1024:.2f} MB")
    
    # Estimate FloatTensor memory
    # samples * 10 * 4 bytes
    print(f"Estimated Tensor memory: {estimated_samples * 10 * 4 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    check_stats("data/HDFS/deeplog_input/hdfs_train")
