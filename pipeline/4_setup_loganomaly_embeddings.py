
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from gensim.models import Word2Vec
import argparse

def tokenize_template(template):
    # Replace weird characters with spaces
    # Keep <*>, but maybe treat it as a word "<*>" or "blk" etc.
    # LogAnomaly paper: "we first tokenize the log templates into words"
    # It doesn't specify deep cleaning.
    # Let's replace non-alphanumeric (except <*>) with space.
    # Actually, simplistic approach: split by non-alphanumeric.
    
    # 1. Handle common wildcards like <*> or <blk>
    # We treat them as specific tokens if we want, or just remove them.
    # Let's clean them to simple words.
    text = str(template)
    text = re.sub(r'<\*>', ' parameter ', text) 
    
    # 2. Split by non-alphanumeric
    tokens = re.split(r'[^a-zA-Z0-9]+', text)
    
    # 3. Filter empty and lowercase
    tokens = [t.lower() for t in tokens if len(t) > 0]
    
    return tokens

def generate_embeddings(mode='full'):
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    if mode == 'full':
        structured_csv_path = project_root / 'data' / 'HDFS' / 'parsed' / 'HDFS.log_structured.csv'
        mapping_path = project_root / 'data' / 'HDFS' / 'deeplog_input' / 'event_mapping.csv'
        output_dir = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs'
    else:
        structured_csv_path = project_root / 'data' / 'HDFS' / 'parsed' / 'HDFS_2k.log_structured.csv'
        mapping_path = project_root / 'data' / 'HDFS' / 'deeplog_input_2k' / 'event_mapping.csv'
        output_dir = project_root / 'models' / 'LogDeep' / 'data' / 'hdfs_2k' / 'hdfs'

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Semantic Vectors for mode={mode}")
    print(f"Reading {structured_csv_path}...")
    
    # Efficiently read only needed columns
    try:
        df_log = pd.read_csv(structured_csv_path, usecols=['EventId', 'EventTemplate'])
    except ValueError:
         print("Error: Columns EventId or EventTemplate not found in CSV.")
         return

    # Get unique templates
    df_templates = df_log.drop_duplicates(subset=['EventId']).copy()
    print(f"Found {len(df_templates)} unique templates.")
    
    # Read Mapping
    if not mapping_path.exists():
        print(f"Error: {mapping_path} not found. Run conversion first.")
        return
        
    df_mapping = pd.read_csv(mapping_path)
    # Mapping: EventId, IntId
    
    # Merge
    merged = pd.merge(df_mapping, df_templates, on='EventId', how='left')
    
    # Check for missing templates
    if merged['EventTemplate'].isnull().any():
        print("Warning: Some EventIds have no corresponding template in the structured file.")
        merged['EventTemplate'] = merged['EventTemplate'].fillna("unknown")
        
    # Prepare Corpus
    sentences = []
    int_id_to_tokens = {}
    
    for _, row in merged.iterrows():
        int_id = str(row['IntId']) # Use string key for JSON
        template = row['EventTemplate']
        tokens = tokenize_template(template)
        sentences.append(tokens)
        int_id_to_tokens[int_id] = tokens
        
    # Train Word2Vec
    print("Training Word2Vec...")
    vector_dim = 300
    model = Word2Vec(sentences=sentences, vector_size=vector_dim, window=5, min_count=1, workers=1, seed=42)
    # min_count=1 ensures even rare words in templates have vectors
    
    # Generate Event Embeddings (Average of word vectors)
    event2vec = {}
    
    # Add an entry for padding or special tokens? LogAnomaly uses IntId from 1?
    # Usually we generate for all IntIds present.
    # We should also cover '0' if it's used as padding or unknown. 
    # Let's generate a zero vector for '0' or random? "EventId 0" is usually reserved/padding.
    event2vec['0'] = [0.0] * vector_dim 
    
    for int_id, tokens in int_id_to_tokens.items():
        if not tokens:
            vec = np.zeros(vector_dim)
        else:
            vectors = []
            for t in tokens:
                if t in model.wv:
                    vectors.append(model.wv[t])
                else:
                    # Should not happen with min_count=1 and training on same corpus
                    pass
            if vectors:
                vec = np.mean(vectors, axis=0)
            else:
                vec = np.zeros(vector_dim)
        
        event2vec[int_id] = vec.tolist()
        
    # Save
    output_path = output_dir / 'event2semantic_vec.json'
    with open(output_path, 'w') as f:
        json.dump(event2vec, f)
        
    print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'demo'], default='full')
    args = parser.parse_args()
    
    generate_embeddings(args.mode)
