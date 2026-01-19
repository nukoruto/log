import pandas as pd
import os
from pathlib import Path

def convert_to_deeplog_format(structured_csv, output_dir):
    df = pd.read_csv(structured_csv)
    
    # 1. Map EventId to Integers
    # DeepLog expects 1-based integers (it subtracts 1 in the code)
    unique_events = df['EventId'].unique()
    event_to_int = {event: i+1 for i, event in enumerate(unique_events)}
    
    # Save mapping for later
    mapping_df = pd.DataFrame(list(event_to_int.items()), columns=['EventId', 'IntId'])
    mapping_path = Path(output_dir) / 'event_mapping.csv'
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
    
    if 'BlockId' not in df.columns:
        # Try to extract from Content or ParameterList
        # The regex in parse.py was `blk_-?\d+`.
        # Drain puts regex matches in 'ParameterList'.
        # But robust way is to re-extract.
        df['BlockId'] = df['Content'].str.extract(r'(blk_-?\d+)')
    
    # Filter out rows with no BlockId (if any)
    df = df.dropna(subset=['BlockId'])
    
    # Group
    sessions = df.groupby('BlockId')['EventId'].apply(list)
    
    # 3. Convert strings to ints
    session_ints = sessions.apply(lambda event_list: [str(event_to_int[e]) for e in event_list])
    
    # 4. Split Train/Test (Simple split for demo)
    # We use all for train for now as verification.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'hdfs_train', 'w') as f:
        for seq in session_ints:
            f.write(' '.join(seq) + '\n')
            
    # Create a dummy test set (copy of train) for now
    with open(output_dir / 'hdfs_test_normal', 'w') as f:
         for seq in session_ints:
            f.write(' '.join(seq) + '\n')
            
    print(f"Saved sessions to {output_dir / 'hdfs_train'}")

if __name__ == '__main__':
    structured_csv = 'data/HDFS/parsed/HDFS_2k.log_structured.csv'
    output_dir = 'data/HDFS/deeplog_input'
    
    print("Converting HDFS data to DeepLog format...")
    convert_to_deeplog_format(structured_csv, output_dir)
    print("Conversion completed.")
