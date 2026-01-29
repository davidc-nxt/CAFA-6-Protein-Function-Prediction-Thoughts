#!/usr/bin/env python3
"""
Phase 33: Learning-to-Rank (LTR) - Inference / Submission.
Applies v55_ltr_xgboost.model to Test Data.
"""

import os
import csv
import pickle
import subprocess
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import logging
import gc

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
TEST_FASTA = "Test/testsuperset.fasta"
DIAMOND_DB = "diamond_train_full.dmnd"
SBERT_EMB = "train_text_embeddings.pkl" # For k-NN reference
ESM2_TRAIN_EMB = "train_esm2_embeddings.pkl"
SBERT_TEST_PREDS = "submission_v39_text_only.tsv" # Use V39 file for fast SBERT features? 
# Better to compute fresh k-NN or reuse? 
# Reusing V39 TSV scores as "SBERT Feature" is faster and consistent.
ESM2_TEST_EMB = "test_esm2_embeddings.pkl"
MODEL_FILE = "v56_ltr_xgboost_tuned.model"
OUTPUT_FILE = "submission_v56_ltr_tuned.tsv"
TRAIN_TERMS = "Train/train_terms.tsv"

def load_term_priors():
    logger.info("Loading Term Priors...")
    counts = Counter()
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            counts[row[1]] += 1
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

def run_diamond_test():
    out_file = "diamond_test_ltr.tsv"
    if os.path.exists(out_file):
        logger.info(f"Using existing {out_file}...")
    elif not os.path.exists(DIAMOND_DB):
         logger.error(f"Diamond DB {DIAMOND_DB} not found! Cannot generate features.")
         raise FileNotFoundError(DIAMOND_DB)
    else:
        logger.info(f"Running Diamond on Test Set...")
        cmd = [
            "diamond", "blastp",
            "-d", DIAMOND_DB,
            "-q", TEST_FASTA,
            "-o", out_file,
            "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore",
            "--max-target-seqs", "50",
            "--sensitive",
            "-b", "4.0" # Block size for speed
        ]
        subprocess.run(cmd, check=True)
    return out_file

def load_diamond_scores(diamond_file):
    logger.info("Loading Diamond Scores...")
    # Map Train PID -> Terms
    train_terms = defaultdict(set)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            train_terms[row[0]].add(row[1])
            
    # Process file line by line to save RAM? 
    # Or load into robust structure. 220k proteins.
    # Scores: {pid: {term: score}}
    scores = defaultdict(dict)
    
    with open(diamond_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            q = row[0]
            s = row[1]
            score = float(row[3])
            
            if s in train_terms:
                for t in train_terms[s]:
                    if score > scores[q].get(t, 0):
                        scores[q][t] = score
                        
    # Normalize
    for pid in scores:
        m = max(scores[pid].values())
        if m > 0:
            for t in scores[pid]:
                scores[pid][t] /= m
    return scores

def load_sbert_scores():
    # Load from V39 submission file directly
    logger.info("Loading SBERT Scores from V39 file...")
    scores = defaultdict(dict)
    # Check archive path if moved?
    # User moved it to archive/submissions/submission_v39_text_only.tsv
    path = "archive/submissions/submission_v39_text_only.tsv"
    if not os.path.exists(path):
         path = "submission_v39_text_only.tsv" # Fallback
         
    if not os.path.exists(path):
        logger.warning(f"SBERT file {path} not found. Skipping SBERT features.")
        return scores

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                scores[row[0]][row[1]] = float(row[2])
    return scores

def run_esm2_test_knn():
    logger.info("Running ESM-2 k-NN for Test...")
    # Train Matrix
    with open(ESM2_TRAIN_EMB, "rb") as f:
        train_data = pickle.load(f)
    train_pids = list(train_data.keys())
    train_X = [train_data[p] for p in train_pids]
    del train_data
    
    knn = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1)
    knn.fit(train_X)
    del train_X
    gc.collect()
    
    # Test Matrix
    with open(ESM2_TEST_EMB, "rb") as f:
        test_data = pickle.load(f)
    test_pids = list(test_data.keys())
    # Process in batches
    
    train_terms = defaultdict(set)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            train_terms[row[0]].add(row[1])
            
    scores = defaultdict(dict)
    
    BATCH = 1000
    for i in tqdm(range(0, len(test_pids), BATCH)):
        batch_pids = test_pids[i:i+BATCH]
        batch_X = [test_data[p] for p in batch_pids]
        
        dists, idxs = knn.kneighbors(batch_X)
        sims = 1 - dists
        
        for j, pid in enumerate(batch_pids):
            votes = defaultdict(float)
            for k, n_idx in enumerate(idxs[j]):
                n_pid = train_pids[n_idx]
                sim = sims[j][k]
                for t in train_terms[n_pid]:
                    votes[t] += sim
            
            if votes:
                m = max(votes.values())
                for t, v in votes.items():
                    norm = v / m
                    if norm > 0.1: # Threshold to reduce size
                        scores[pid][t] = norm
                        
    return scores

def main():
    # 1. Load Model
    logger.info("Loading XGBoost Model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    # 2. Features
    term_priors = load_term_priors()
    
    diamond_file = run_diamond_test()
    diamond_scores = load_diamond_scores(diamond_file)
    
    sbert_scores = load_sbert_scores()
    
    # Heavy Step: ESM2
    # Check if we already have a cache? No.
    esm_scores = run_esm2_test_knn()
    
    # 3. Predict Loop
    logger.info("Generating Predictions...")
    
    # Get all test PIDs
    test_pids = set(diamond_scores.keys()) | set(sbert_scores.keys()) | set(esm_scores.keys())
    
    with open(OUTPUT_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        
        for pid in tqdm(test_pids):
            # Candidate Terms
            candidates = set()
            candidates.update(diamond_scores.get(pid, {}).keys())
            candidates.update(sbert_scores.get(pid, {}).keys())
            candidates.update(esm_scores.get(pid, {}).keys())
            
            if not candidates:
                continue
                
            # Batch Feature Construction
            rows = []
            terms = []
            for t in candidates:
                rows.append([
                    diamond_scores.get(pid, {}).get(t, 0.0),
                    sbert_scores.get(pid, {}).get(t, 0.0),
                    esm_scores.get(pid, {}).get(t, 0.0),
                    term_priors.get(t, 0.0)
                ])
                terms.append(t)
                
            X = np.array(rows)
            # Fast predict
            # Check feature names match training?
            # XGBoost might warn if feature names missing.
            # Convert to DataFrame with names to be safe.
            X_df = pd.DataFrame(X, columns=['f_diamond', 'f_sbert', 'f_esm2', 'f_prior'])
            
            preds = model.predict_proba(X_df)[:, 1]
            
            # Write
            for t, p in zip(terms, preds):
                # Filter low score?
                if p >= 0.01:
                    writer.writerow([pid, t, f"{p:.3f}"])
                    
    logger.info("Done.")

if __name__ == "__main__":
    main()
