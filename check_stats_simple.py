
import sys

def check_stats(filepath):
    total_events = 0
    total_windows = 0
    window_size = 10
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            tokens = line.strip().split()
            n = len(tokens)
            total_events += n
            if n > window_size:
                total_windows += n - window_size
            
            if i % 100000 == 0:
                print(f"Processed {i} lines...")

    print(f"Total events: {total_events}")
    print(f"Total windows: {total_windows}")
    
    # 1 window = list of 10 ints.
    # Python list overhead: ~64 bytes + 8*10 = 144 bytes per window.
    # 10M windows => 1.4 GB.
    # But torch.tensor(list_of_lists) creates a temporary copy.
    
    # If total_windows is huge, say 100M, then 14GB.

if __name__ == "__main__":
    check_stats("data/HDFS/deeplog_input/hdfs_train")
