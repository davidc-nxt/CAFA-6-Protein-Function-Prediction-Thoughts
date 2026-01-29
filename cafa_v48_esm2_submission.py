#!/usr/bin/env python3
"""
Generate V48 Submission: Hybrid SBERT + ESM-2.
Results:
- Local Val F-max: 0.259 (vs Homology 0.225)
- Strategy: Use SBERT predictions where available (High Precision), fill gaps with ESM-2 (High Coverage).
"""

import os
import pickle
import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
from tqdm import tqdm
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Files
SBERT_PREDS = "submission_v39_text_only.tsv"
TRAIN_EMB = "train_esm2_embeddings.pkl"
TEST_EMB = "test_esm2_embeddings.pkl"
TRAIN_TERMS = "Train/train_terms.tsv"
OUTPUT_FILE = "submission_v48_hybrid_esm2.tsv"

def load_terms():
    logger.info("Loading training terms...")
    terms = defaultdict(set)
    counts = Counter()
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2:
                pid, term = row[0], row[1]
                terms[pid].add(term)
                counts[term] += 1
    return terms, counts

def load_sbert_preds():
    logger.info(f"Loading SBERT preds from {SBERT_PREDS}...")
    preds = defaultdict(dict)
    count = 0
    with open(SBERT_PREDS, 'r') as f:
        # header?
        # Check first line
        pos = f.tell()
        line = f.readline()
        if "term" in line.lower():
            pass # Skip header
        else:
            f.seek(pos)
            
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                pid, term, score = row[0], row[1], float(row[2])
                preds[pid][term] = score
                count += 1
    logger.info(f"Loaded {len(preds)} proteins with SBERT predictions.")
    return preds

def train_esm2_knn(emb_file):
    logger.info("Loading Train ESM-2 Embeddings...")
    with open(emb_file, "rb") as f:
        embeddings = pickle.load(f)
    
    pids = list(embeddings.keys())
    matrix = np.array([embeddings[p] for p in pids])
    
    logger.info(f"Training k-NN on {len(pids)} vectors...")
    knn = NearestNeighbors(n_neighbors=15, metric='cosine', n_jobs=-1)
    knn.fit(matrix)
    return knn, pids

def predict_esm2(knn, train_pids, test_emb_file, terms, counts, sbert_pids):
    logger.info("Loading Test ESM-2 Embeddings...")
    with open(test_emb_file, "rb") as f:
        test_embeddings = pickle.load(f)
        
    preds = defaultdict(dict)
    
    # Only predict for proteins NOT in SBERT (or predict all and blend? Validated strategy was waterfall)
    # Validation strategy: If SBERT exists, use it. Else ESM-2.
    
    needed_pids = [p for p in test_embeddings if p not in sbert_pids]
    logger.info(f"Predicting ESM-2 for {len(needed_pids)} proteins (filling gaps)...")
    
    if not needed_pids:
        return preds
        
    needed_matrix = np.array([test_embeddings[p] for p in needed_pids])
    
    # Process in batches to save RAM
    BATCH_SIZE = 1000
    
    for i in tqdm(range(0, len(needed_matrix), BATCH_SIZE)):
        batch_X = needed_matrix[i : i+BATCH_SIZE]
        batch_pids = needed_pids[i : i+BATCH_SIZE]
        
        dists, idxs = knn.kneighbors(batch_X)
        sims = 1 - dists
        
        for j, pid in enumerate(batch_pids):
            neighbors = idxs[j]
            similarities = sims[j]
            
            # Weighted vote
            scores = defaultdict(float)
            total_weight = 0
            
            for k, n_idx in enumerate(neighbors):
                n_pid = train_pids[n_idx]
                weight = similarities[k]
                total_weight += weight
                
                for t in terms[n_pid]:
                    # Ignore very rare terms? (Optional, validation used Min 10 count)
                    if counts[t] >= 5: 
                         scores[t] += weight
            
            # Normalize
            if scores:
                max_score = max(scores.values())
                # Or divide by total weight? 
                # Validation used max_score normalize to 0.1-1.0
                
                for t, s in scores.items():
                    norm = s / max_score
                    # Keep top
                    if norm >= 0.01:
                       preds[pid][t] = norm
                       
    return preds

def main():
    # 1. Load Data
    terms, counts = load_terms()
    
    # 2. Load SBERT Predictions
    sbert_preds = load_sbert_preds()
    sbert_pids = set(sbert_preds.keys())
    
    # 3. Train ESM-2 k-NN
    knn, train_pids = train_esm2_knn(TRAIN_EMB)
    
    # 4. Predict ESM-2 for Gaps
    esm2_preds = predict_esm2(knn, train_pids, TEST_EMB, terms, counts, sbert_pids)
    
    # 5. Merge and Write
    logger.info("Merging and writing...")
    all_pids = sbert_pids.union(esm2_preds.keys())
    
    with open(OUTPUT_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        # Kaggle format: No header? Or header? CAFA usually no header or ignored.
        # Let's check verified submission. v39 had no header in check?
        # But verify logic usually strips header.
        
        count = 0
        for pid in all_pids:
            # Source?
            if pid in sbert_preds:
                p_data = sbert_preds[pid]
            else:
                p_data = esm2_preds.get(pid, {})
                
            for t, s in p_data.items():
                writer.writerow([pid, t, f"{s:.3f}"])
                count += 1
                
    logger.info(f"Done! Written {count} predictions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
