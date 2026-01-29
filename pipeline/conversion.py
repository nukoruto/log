import pandas as pd
import os
from pathlib import Path

def convert_to_deeplog_format(structured_csv, output_dir, log_file_path=None, min_len=11):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(structured_csv)
    
    # 1. Map EventId to Integers
    # DeepLog expects 1-based integers (it subtracts 1 in the code)
    unique_events = df['EventId'].unique()
    event_to_int = {event: i+1 for i, event in enumerate(unique_events)}
    
    # Save mapping for later
    mapping_df = pd.DataFrame(list(event_to_int.items()), columns=['EventId', 'IntId'])
    mapping_path = output_dir / 'event_mapping.csv'
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved event mapping to {mapping_path}")
    print(f"Total unique events: {len(unique_events)}")

    # 2. Group by BlockId (Session)
    # HDFS logs have 'BlockId' in the Content or we extracted it via Regex in Drain.
    # Drain's 'extracted_parameter_list' might act weird, but usually LogHub HDFS has Label file.
    # Wait, HDFS_2k.log might not have BlockId extracted as a column unless we defined it in the regex.
    # My parse code used regex `blk_-?\d+`. Drain usually puts extracted parameters in a list.
    # But usually for HDFS, the raw content contains 'blk_...'.
    # Let's check if 'BlockId' is a column. If not, we extract from 'Content'.
    
    # BlockId extraction from original content
    # Since structured CSV 'Content' is the template, we can't extract specific BlockId from it.
    # We must read the original log file to get BlockIds.
    # Assuming line correspondence 1-to-1.
    
    # log_file_path passed as argument
    if log_file_path and log_file_path.exists():
        with open(log_file_path, 'r', encoding='utf-8') as f:
            raw_logs = f.readlines()
        
        # Extract BlockId from each line
        import re
        block_pattern = re.compile(r'(blk_-?\d+)')
        block_ids = []
        for line in raw_logs:
            match = block_pattern.search(line)
            if match:
                block_ids.append(match.group(1))
            else:
                block_ids.append('Unknown')
        
        # Assign to DF (Assuming LineId matches index+1 or order)
        # Drain usually preserves order.
        if len(block_ids) == len(df):
            df['BlockId'] = block_ids
        else:
             print(f"Warning: Line count mismatch. Log: {len(block_ids)}, CSV: {len(df)}")
             # Fallback: try ParameterList if available, or just fail for now
             # Actually Drain output csv might skipped lines? No usually not.
             pass
    else:
        print(f"Error: Original log file not found at {log_file_path}")
        return

    # Filter out rows with 'Unknown' BlockId
    df = df[df['BlockId'] != 'Unknown']
    
    # Pre-calculate Timestamps
    # HDFS Format: Date (081109) Time (203615) -> 2008-11-09 20:36:15
    # NOTE: Year is ambiguous in HDFS sample usually, but let's assume 2008 or just relative.
    def parse_datetime(row):
        # Ensure encoded as string with leading zeros
        d = str(row['Date']).zfill(6)
        t = str(row['Time']).zfill(6)
        # d="081109", t="203615"
        # 20+d[0:2] ...
        try:
             return pd.to_datetime(f"20{d[0:2]}-{d[2:4]}-{d[4:6]} {t[0:2]}:{t[2:4]}:{t[4:6]}")
        except:
             return pd.Timestamp.now() # Fallback
        
    df['Timestamp'] = df.apply(parse_datetime, axis=1)

    # Group Events and Timestamps
    grouped = df.groupby('BlockId')
    sessions = grouped['EventId'].apply(list)
    timestamps = grouped['Timestamp'].apply(list)
    
    # 3. Convert strings to ints
    # Filter sessions with length < min_len
    # Filter both sessions and timestamps
    valid_indices = [i for i, s in enumerate(sessions) if len(s) >= min_len]
    sessions = sessions.iloc[valid_indices]
    timestamps = timestamps.iloc[valid_indices]
    
    session_ints = sessions.apply(lambda event_list: [str(event_to_int[e]) for e in event_list])
    
    # Calculate Relative Timestamps (floats, seconds from start)
    def get_relative_times(ts_list):
        if not ts_list: return []
        start_t = ts_list[0]
        # Return list of seconds
        return [str((t - start_t).total_seconds()) for t in ts_list]
        
    session_times = timestamps.apply(get_relative_times)
    
    
    # 3.5 Merge with Anomaly Labels
    label_csv_path = Path('data/HDFS/anomaly_label.csv')
    if label_csv_path.exists():
        print(f"Loading labels from {label_csv_path}")
        label_df = pd.read_csv(label_csv_path)
        # Ensure BlockId is string
        label_df['BlockId'] = label_df['BlockId'].astype(str)
        
        # Merge labels into our session data
        # 'sessions' is a Series indexed by BlockId. 
        # We can create a DataFrame from it to merge.
        session_df = pd.DataFrame({'EventInts': sessions, 'Times': session_times})
        session_df.index.name = 'BlockId'
        session_df.reset_index(inplace=True)
        session_df['BlockId'] = session_df['BlockId'].astype(str)
        
        merged_df = pd.merge(session_df, label_df, on='BlockId', how='left')
        merged_df['Label'] = merged_df['Label'].fillna('Normal') # Default to Normal
        
    else:
        print(f"Warning: {label_csv_path} not found! Treating all as Normal and skipping real anomaly split.")
        # Fallback to original logic manually or just error? 
        return

    # 4. Split Train/Test based on Labels
    # Train: Normal sessions
    # Test Normal: Normal sessions (subset)
    # Test Abnormal: Anomaly sessions
    
    normal_df = merged_df[merged_df['Label'] == 'Normal']
    abnormal_df = merged_df[merged_df['Label'] == 'Anomaly']
    
    print(f"Normal sessions: {len(normal_df)}, Abnormal sessions: {len(abnormal_df)}")
    
    # Shuffle Normal
    normal_df = normal_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split Normal into Train (80%) and Test (20%)
    train_size = int(len(normal_df) * 0.8)
    train_df = normal_df.iloc[:train_size]
    test_normal_df = normal_df.iloc[train_size:]
    
    test_abnormal_df = abnormal_df
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Train Events
    with open(output_dir / 'hdfs_train', 'w') as f:
        for seq in train_df['EventInts']:
            f.write(' '.join(seq) + '\n')
            
    # Save Train Times (Optional)
    with open(output_dir / 'hdfs_train_time', 'w') as f:
        for seq in train_df['Times']:
            f.write(' '.join(seq) + '\n')
            
    # Save Train Labels (All 0)
    with open(output_dir / 'hdfs_train_label', 'w') as f:
        for seq in train_df['EventInts']:
            f.write(' '.join(['0'] * len(seq)) + '\n')
            
    # Save Test Normal Events
    with open(output_dir / 'hdfs_test_normal', 'w') as f:
        for seq in test_normal_df['EventInts']:
            f.write(' '.join(seq) + '\n')

    # Save Test Normal Times & Labels
    with open(output_dir / 'hdfs_test_normal_time', 'w') as f_time, \
         open(output_dir / 'hdfs_test_normal_label', 'w') as f_label:
         for seq, times in zip(test_normal_df['EventInts'], test_normal_df['Times']):
            f_time.write(' '.join(times) + '\n')
            f_label.write(' '.join(['0'] * len(seq)) + '\n')
            
    # Save Test Abnormal Events
    with open(output_dir / 'hdfs_test_abnormal', 'w') as f:
        for seq in test_abnormal_df['EventInts']:
            f.write(' '.join(seq) + '\n')
            
    # Save Test Abnormal Times & Labels
    with open(output_dir / 'hdfs_test_abnormal_time', 'w') as f_time, \
         open(output_dir / 'hdfs_test_abnormal_label', 'w') as f_label:
         
         for seq, times in zip(test_abnormal_df['EventInts'], test_abnormal_df['Times']):
            f_time.write(' '.join(times) + '\n')
            # Label all events as 1 for anomalous sessions
            f_label.write(' '.join(['1'] * len(seq)) + '\n')
            
    print(f"Saved sessions to {output_dir / 'hdfs_train'}")
    print(f"Saved timestamps to {output_dir / 'hdfs_test_normal_time'}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
    args = parser.parse_args()

    # Logic switch
    if args.mode == 'full':
        structured_csv = 'data/HDFS/parsed/HDFS.log_structured.csv'
        log_file_path = Path('data/HDFS/HDFS.log')
        min_len = 11 # Window 10
        output_dir = 'data/HDFS/deeplog_input'
    else:
        structured_csv = 'data/HDFS/parsed/HDFS_2k.log_structured.csv'
        log_file_path = Path('data/HDFS/HDFS_2k.log')
        min_len = 2 # Window 1
        output_dir = 'data/HDFS/deeplog_input_2k'

    print(f"Converting HDFS data to DeepLog format (Mode: {args.mode})...")
    # Need to pass these config values to the function or change the function signature
    # Ideally change function signature, but to keep it simple, I will modify the function 
    # to accept an optional 'config' dict or arguments.
    # Let's update `convert_to_deeplog_format` signature first.
    convert_to_deeplog_format(structured_csv, output_dir, log_file_path=log_file_path, min_len=min_len)
    print("Conversion completed.")
