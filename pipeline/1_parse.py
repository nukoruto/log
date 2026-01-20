import sys
import pandas as pd
import os
from pathlib import Path

# Add project root to path to allow importing logparser
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'parsers' / 'logparser'))

from logparser.Drain import LogParser

def parse_hdfs(input_dir, output_dir, log_file):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # HDFS/Drain settings
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    # HDFS log content usually looks like: 081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
    # The LogHub format might be slightly different depending on the specific file version, 
    # but standards usually use:
    # <Date> <Time> <Pid> <Level> <Component>: <Content>
    # or just: <Content> if it's raw message.
    # Looking at HDFS_2k.log from LogHub, it usually has headers.
    
    regex = [
        r'blk_-?\d+', # block_id
        r'(\d+\.){3}\d+(:\d+)?', # IP
    ]
    st = 0.5  # Similarity threshold
    depth = 4  # Depth of all leaf nodes: [1, 2, 4, 3, 5]
    
    parser = LogParser(log_format, indir=str(input_dir), outdir=str(output_dir),  depth=depth, st=st, rex=regex)
    parser.parse(log_file)

if __name__ == '__main__':
    input_dir = 'data/HDFS'
    output_dir = 'data/HDFS/parsed'
    log_file = 'HDFS.log'
    
    print(f"Parsing {log_file}...")
    try:
        parse_hdfs(input_dir, output_dir, log_file)
        print("Parsing completed.")
    except Exception as e:
        print(f"Error: {e}")
        # Fallback for path issues
        print("Checking paths...")
        print(f"Input: {Path(input_dir).absolute()}")
