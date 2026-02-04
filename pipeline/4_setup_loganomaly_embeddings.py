
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
        int_id = str(int(row['IntId']) - 1) # Shift to 0-based index
        template = row['EventTemplate']
        tokens = tokenize_template(template)
        sentences.append(tokens)
        int_id_to_tokens[int_id] = tokens
        
    # Train Word2Vec
    print("Training Word2Vec...")
    vector_dim = 300
    model = Word2Vec(sentences=sentences, vector_size=vector_dim, window=5, min_count=1, workers=1, seed=42)
    # min_count=1 ensures even rare words in templates have vectors
    
    # --- 1. Retrofitting (Domain Knowledge Injection) ---
    dict_path = project_root / 'data' / 'dictionaries' / 'synonyms_antonyms.json'
    if dict_path.exists():
        print(f"Loading domain dictionary from {dict_path}...")
        with open(dict_path, 'r') as f:
            domain_dict = json.load(f)
            
        synonyms = domain_dict.get('synonyms', [])
        antonyms = domain_dict.get('antonyms', [])
        
        # Helper to get vector safely
        def get_vec(word):
            if word in model.wv:
                return model.wv[word]
            return None
            
        # Helper to normalize
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0: return v
            return v / norm

        print("Applying Retrofitting...")
        alpha = 0.1 # Learning rate for retrofitting
        epochs = 10
        
        for epoch in range(epochs):
            # Antonyms: Push apart if similar
            for pair in antonyms:
                if len(pair) >= 2:
                    w1, w2 = pair[0], pair[1]
                    v1, v2 = get_vec(w1), get_vec(w2)
                    if v1 is not None and v2 is not None:
                        # Cosine similarity
                        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                        if sim > 0.2:
                            # Push apart
                            new_v1 = v1 - alpha * v2
                            new_v2 = v2 - alpha * v1
                            model.wv[w1] = normalize(new_v1)
                            model.wv[w2] = normalize(new_v2)
                            
            # Synonyms: Pull closer
            for group in synonyms:
                # Pull each word towards the centroid of the group (if they exist)
                valid_words = [w for w in group if get_vec(w) is not None]
                if len(valid_words) < 2: continue
                
                # Simple approach: Pairwise pull
                # Better: Pull to mean of others? Let's do simple pairwise for compatibility/ease.
                # Actually, let's pull towards the group mean.
                group_vecs = [model.wv[w] for w in valid_words]
                centroid = np.mean(group_vecs, axis=0)
                
                for w in valid_words:
                    new_v = model.wv[w] + alpha * (centroid - model.wv[w])
                    model.wv[w] = normalize(new_v)
        
        print("Retrofitting completed.")
    else:
        print(f"Warning: Domain dictionary not found at {dict_path}")

    # --- 2. TF-IDF Calculation ---
    print("Calculating TF-IDF...")
    from collections import Counter
    import math
    
    # Calculate IDF
    # DF: Number of documents (templates) containing word w
    df_counts = Counter()
    total_docs = len(sentences)
    
    for tokens in sentences:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df_counts[t] += 1
            
    idf = {}
    for t, count in df_counts.items():
        idf[t] = math.log(total_docs / (count + 1)) # Add 1 smooth
        
    # Generate Event Embeddings (Weighted Average)
    event2vec = {}
    
    # Add an entry for padding (0) -> Actually '0' is now a class ID.
    # event2vec['0'] = [0.0] * vector_dim 
    
    for int_id, tokens in int_id_to_tokens.items():
        # int_id is already 0-based
        int_id_0 = int_id
        
        if not tokens:
            vec = np.zeros(vector_dim)
        else:
            vectors = []
            weights = []
            for t in tokens:
                if t in model.wv:
                    v = model.wv[t]
                    # TF: term frequency in this template
                    # tokens is the list of words in *this* template
                    tf = tokens.count(t) 
                    w = tf * idf.get(t, 0)
                    
                    vectors.append(v)
                    weights.append(w)
                else:
                    pass
            
            if vectors:
                if sum(weights) > 0:
                    # Weighted average
                    vec = np.average(vectors, axis=0, weights=weights)
                else:
                    # Fallback to simple mean if all weights are 0 (shouldn't happen with IDF)
                    vec = np.mean(vectors, axis=0)
            else:
                vec = np.zeros(vector_dim)
        
        event2vec[int_id_0] = vec.tolist()
        
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
