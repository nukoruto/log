import pandas as pd
import os
from pathlib import Path

def convert_to_deeplog_format(structured_csv, output_dir):
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
    
    log_file_path = Path('data/HDFS/HDFS.log')
    if log_file_path.exists():
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
    # Filter sessions with length < 11 (Window 10 + 1 for prediction)
    # Filter both sessions and timestamps
    valid_indices = [i for i, s in enumerate(sessions) if len(s) >= 11]
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
    
    # 4. Split Train/Test (Simple split for demo)
    # We use all for train for now as verification.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Train Events
    with open(output_dir / 'hdfs_train', 'w') as f:
        for seq in session_ints:
            f.write(' '.join(seq) + '\n')

    # Save Train Times (Optional, but good for completeness)
    # DeepLog doesn't use time for training in this impl, but let's save.
    # We won't generate hdfs_train_time unless converting training code, but might be useful later.
    
    import random
    # Create a dummy test set (copy of train)
    with open(output_dir / 'hdfs_test_normal', 'w') as f:
         for seq in session_ints:
            f.write(' '.join(seq) + '\n')
            
    # Save Normal Times & Labels
    with open(output_dir / 'hdfs_test_normal_time', 'w') as f_time, \
         open(output_dir / 'hdfs_test_normal_label', 'w') as f_label:
         for seq in session_times:
            f_time.write(' '.join(seq) + '\n')
            # All 0 for normal
            f_label.write(' '.join(['0'] * len(seq)) + '\n')
            
    # Create synthetic abnormal data (randomly changes 1 event per session)
    with open(output_dir / 'hdfs_test_abnormal', 'w') as f_evt, \
         open(output_dir / 'hdfs_test_abnormal_time', 'w') as f_time, \
         open(output_dir / 'hdfs_test_abnormal_label', 'w') as f_label:
         
         all_event_ids = list(event_to_int.values())
         # session_ints and session_times are Series indexed by BlockId, so they align.
         # Iterate over values
         for seq, times in zip(session_ints, session_times):
            # Copy sequence
            mod_seq = list(seq)
            mod_times = list(times)
            mod_labels = ['0'] * len(mod_seq) # Default all 0
            
            if len(mod_seq) > 0:
                # Pick a random position
                idx = random.randint(0, len(mod_seq) - 1)
                # Pick a random NEW event
                current_val = mod_seq[idx]
                new_val = current_val
                # Ensure we change it
                if len(all_event_ids) > 1:
                    while new_val == current_val:
                        new_val = str(random.choice(all_event_ids))
                else:
                    new_val = "999" # Force unknown if only 1 event type exists
                
                mod_seq[idx] = new_val
                # Label this timestamp as 1 (Anomaly)
                mod_labels[idx] = '1'
                
            f_evt.write(' '.join(mod_seq) + '\n')
            f_time.write(' '.join(mod_times) + '\n')
            f_label.write(' '.join(mod_labels) + '\n')
            
    print(f"Saved sessions to {output_dir / 'hdfs_train'}")
    print(f"Saved timestamps to {output_dir / 'hdfs_test_normal_time'}")

if __name__ == '__main__':
    structured_csv = 'data/HDFS/parsed/HDFS.log_structured.csv'
    output_dir = 'data/HDFS/deeplog_input'
    
    print("Converting HDFS data to DeepLog format...")
    convert_to_deeplog_format(structured_csv, output_dir)
    print("Conversion completed.")
