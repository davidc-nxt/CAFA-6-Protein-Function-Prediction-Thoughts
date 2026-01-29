#!/usr/bin/env python3
"""
Phase 33: Learning-to-Rank (LTR) - Feature Preparation.
Generates Training Data for XGBoost.
Structure:
- For each Validation Protein:
  - Get Top-50 predictions from Diamond.
  - Get Top-50 from SBERT.
  - Get Top-50 from ESM-2.
  - Union these terms -> Candidate Set.
  - For each Candidate Term:
    - Feature 1: Diamond Score (0 if missing)
    - Feature 2: SBERT Score (0 if missing)
    - Feature 3: ESM-2 Score (0 if missing)
    - Feature 4: Term Frequency (Prior)
    - Label: 1 if in Ground Truth, 0 else.
"""

import os
import csv
import pickle
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
VAL_FASTA = "val_split.fasta"
TRAIN_TERMS = "Train/train_terms.tsv"
DIAMOND_DB = "diamond_train_split.dmnd" # Using split DB to avoid leakage? Or full? 
# Note: For LTR training on Val, we must use Train Split DB (not full).
# Checking list_dir: diamond_train_split.dmnd exists.

SBERT_EMB = "train_text_embeddings.pkl"
ESM2_EMB = "train_esm2_embeddings.pkl"
OUTPUT_FILE = "ltr_val_dataset.pkl"

def load_ground_truth(val_pids):
    logger.info("Loading Ground Truth...")
    gt = defaultdict(set)
    val_set = set(val_pids)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            pid, term = row[0], row[1]
            if pid in val_set:
                gt[pid].add(term)
    return gt

def get_term_counts():
    logger.info("Loading Term Counts...")
    counts = Counter()
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            counts[row[1]] += 1
    total = sum(counts.values())
    freqs = {k: v/total for k, v in counts.items()}
    return freqs

def run_diamond(val_fasta, db_path):
    out_file = "diamond_val_ltr.tsv"
    logger.info(f"Running Diamond BLASTp: {val_fasta} vs {db_path}...")
    
    cmd = [
        "diamond", "blastp",
        "-d", db_path,
        "-q", val_fasta,
        "-o", out_file,
        "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore",
        "--max-target-seqs", "50",
        "--sensitive"
    ]
    subprocess.run(cmd, check=True)
    
    # Process results into term scores
    logger.info("Processing Diamond hits...")
    
    # We need a map of Train PID -> Terms
    train_terms = defaultdict(set)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            train_terms[row[0]].add(row[1])
            
    scores = defaultdict(dict)
    with open(out_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            q_pid = row[0]
            s_pid = row[1]
            bitscore = float(row[3])
            
            # Transfer
            if s_pid in train_terms:
                for t in train_terms[s_pid]:
                    # Simple MaxPool of bitscore
                    if bitscore > scores[q_pid].get(t, 0):
                        scores[q_pid][t] = bitscore
                        
    # Normalize Diamond scores (0-1 range approx, or just leave as raw feature)
    # LTR handles raw, but 0-1 is nicer. Let's normalize by max bitscore in dataset?
    # Or just per protein? Let's do per protein max-norm.
    for pid in scores:
        max_s = max(scores[pid].values())
        if max_s > 0:
            for t in scores[pid]:
                scores[pid][t] /= max_s
                
    return scores

def run_knn(emb_file, name, train_pids_set):
    logger.info(f"Running k-NN for {name}...")
    with open(emb_file, "rb") as f:
        data = pickle.load(f)
        
    # Split into Train and Val matrices
    # We need to know which PIDs are Train vs Val.
    # We can infer from TRAIN_TERMS vs VAL_FASTA or assume split files logic.
    # Let's rely on the input argument `train_pids_set`.
    
    train_x = []
    train_y_pids = []
    val_x = []
    val_y_pids = []
    
    for pid, vec in data.items():
        if pid in train_pids_set:
            train_x.append(vec)
            train_y_pids.append(pid)
        else:
            val_x.append(vec)
            val_y_pids.append(pid)
            
    if not train_x or not val_x:
        logger.warning(f"Missing data for {name}. Train:{len(train_x)}, Val:{len(val_x)}")
        return defaultdict(dict)
        
    logger.info(f"  Fitting {len(train_x)} training vectors...")
    knn = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1)
    knn.fit(train_x)
    
    logger.info(f"  Querying {len(val_x)} validation vectors...")
    dists, idxs = knn.kneighbors(val_x)
    
    # Load Term Map
    train_terms = defaultdict(set)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if row[0] in train_pids_set:
                train_terms[row[0]].add(row[1])
                
    scores = defaultdict(dict)
    sims = 1 - dists
    
    for i, q_pid in enumerate(val_y_pids):
        # Accumulate votes
        raw_scores = defaultdict(float)
        for j, n_idx in enumerate(idxs[i]):
            s_pid = train_y_pids[n_idx]
            sim = sims[i][j]
            
            for t in train_terms[s_pid]:
                raw_scores[t] += sim
                
        # Normalize
        if raw_scores:
            max_s = max(raw_scores.values())
            for t, s in raw_scores.items():
                scores[q_pid][t] = s / max_s
                
    return scores

def main():
    # 1. Identify Split
    # Load Val PIDs from FASTA
    val_pids = []
    with open(VAL_FASTA, 'r') as f:
        for line in f:
            if line.startswith(">"):
                val_pids.append(line[1:].strip().split()[0])
    val_pids_set = set(val_pids)
    logger.info(f"Validation Set: {len(val_pids)} proteins")
    
    # Infer Train PIDs (All - Val)
    all_pids = set()
    with open(TRAIN_TERMS, 'r') as f:
        next(f)
        for line in f:
            all_pids.add(line.split('\t')[0])
            
    train_pids_set = all_pids - val_pids_set
    logger.info(f"Training Set: {len(train_pids_set)} proteins")

    # 2. Score Generators
    diamond_scores = run_diamond(VAL_FASTA, DIAMOND_DB)
    sbert_scores = run_knn(SBERT_EMB, "SBERT", train_pids_set)
    esm2_scores = run_knn(ESM2_EMB, "ESM-2", train_pids_set)
    
    # 3. Ground Truth & Priors
    ground_truth = load_ground_truth(val_pids)
    term_freqs = get_term_counts()
    
    # 4. Build Dataset
    logger.info("Building Feature Matrix...")
    data = []
    
    for pid in tqdm(val_pids):
        # Union of candidate terms (Top 50 from each model implicit in the scores dicts)
        candidates = set()
        candidates.update(diamond_scores.get(pid, {}).keys())
        candidates.update(sbert_scores.get(pid, {}).keys())
        candidates.update(esm2_scores.get(pid, {}).keys())
        
        # Also add Ground Truth terms (positives must be in dataset!)
        # Wait, if models miss them, we train on them? 
        # Yes, otherwise we only learn to rank retrieved items.
        # But usually LTR is re-ranking retrieved lists. If it's not retrieved, we can't rank it.
        # Let's stick to candidates retrieved by at least one model.
        # If a true positive is not retrieved by ANY model, we can't fix it in Step 2.
        
        candidates.update(ground_truth.get(pid, [])) # OPTIONAL: Include all GT to teach valid terms?
        # Let's stick to UNION of candidates for efficiency.
        
        gt_terms = ground_truth.get(pid, set())
        
        for term in candidates:
            # Features
            f_diamond = diamond_scores.get(pid, {}).get(term, 0.0)
            f_sbert = sbert_scores.get(pid, {}).get(term, 0.0)
            f_esm2 = esm2_scores.get(pid, {}).get(term, 0.0)
            f_prior = term_freqs.get(term, 0.0)
            
            # Label
            label = 1 if term in gt_terms else 0
            
            row = {
                'pid': pid,
                'term': term,
                'f_diamond': f_diamond,
                'f_sbert': f_sbert,
                'f_esm2': f_esm2,
                'f_prior': f_prior,
                'label': label
            }
            data.append(row)
            
    # Savc
    df = pd.DataFrame(data)
    logger.info(f"Dataset Stats: {len(df)} rows, {df['label'].sum()} positives.")
    logger.info(f"Saving to {OUTPUT_FILE}...")
    df.to_pickle(OUTPUT_FILE)
    logger.info("Done.")

if __name__ == "__main__":
    main()
