#!/usr/bin/env python3
"""
V57 LTR Prediction: Apply 5-feature model to test set.
Features: Diamond, SBERT (from V39), ESM2, PubMedBERT, GO Prior
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import logging
import gc

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
TEST_FASTA = "Test/testsuperset.fasta"
TRAIN_TERMS = "Train/train_terms.tsv"
DIAMOND_DB = "diamond_train_full.dmnd"
ESM2_TRAIN_EMB = "train_esm2_embeddings.pkl"
ESM2_TEST_EMB = "test_esm2_embeddings.pkl"
PUBMED_EMB = "pubmed_embeddings.pkl"
V39_SBERT_FILE = "archive/submissions/submission_v39_text_only.tsv"
MODEL_FILE = "v57_ltr_xgboost.model"
OUTPUT_FILE = "submission_v57_ltr_pubmed.tsv"

def load_term_priors():
    """Load GO term prior probabilities."""
    logger.info("Loading term priors...")
    counts = defaultdict(int)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            counts[row[1]] += 1
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

def load_train_terms():
    """Load training protein -> GO terms mapping."""
    terms = defaultdict(set)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            terms[row[0]].add(row[1])
    return terms

def run_diamond_test():
    """Run Diamond BLASTp on test set."""
    out_file = "diamond_test_ltr.tsv"
    if os.path.exists(out_file):
        logger.info(f"Using existing {out_file}...")
    elif not os.path.exists(DIAMOND_DB):
        logger.error(f"Diamond DB {DIAMOND_DB} not found!")
        return None
    else:
        logger.info("Running Diamond on Test Set...")
        import subprocess
        cmd = [
            "diamond", "blastp",
            "-d", DIAMOND_DB,
            "-q", TEST_FASTA,
            "-o", out_file,
            "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore",
            "--max-target-seqs", "50",
            "--sensitive",
            "-b", "4.0"
        ]
        subprocess.run(cmd, check=True)
    return out_file

def load_diamond_scores(diamond_file, train_terms):
    """Load Diamond BLASTp scores."""
    logger.info("Loading Diamond scores...")
    if diamond_file is None or not os.path.exists(diamond_file):
        logger.warning("Diamond file not found, returning empty")
        return {}
    
    scores = defaultdict(dict)
    with open(diamond_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            q, s = row[0], row[1]
            bitscore = float(row[3])
            if s in train_terms:
                for t in train_terms[s]:
                    if bitscore > scores[q].get(t, 0):
                        scores[q][t] = bitscore
    
    # Normalize
    for pid in scores:
        m = max(scores[pid].values()) if scores[pid] else 1
        for t in scores[pid]:
            scores[pid][t] /= m
    return scores

def load_sbert_scores():
    """Load SBERT scores from V39 submission file."""
    logger.info("Loading SBERT Scores from V39 file...")
    scores = defaultdict(dict)
    path = V39_SBERT_FILE
    if not os.path.exists(path):
        path = "submission_v39_text_only.tsv"
    if not os.path.exists(path):
        logger.warning(f"SBERT file {path} not found.")
        return scores
    
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                scores[row[0]][row[1]] = float(row[2])
    return scores

def run_esm2_test_knn(train_terms):
    """Run ESM-2 k-NN for test set."""
    logger.info("Running ESM-2 k-NN for Test...")
    
    with open(ESM2_TRAIN_EMB, "rb") as f:
        train_data = pickle.load(f)
    train_pids = list(train_data.keys())
    train_X = [train_data[p] for p in train_pids]
    del train_data
    
    knn = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1)
    knn.fit(train_X)
    del train_X
    gc.collect()
    
    with open(ESM2_TEST_EMB, "rb") as f:
        test_data = pickle.load(f)
    test_pids = list(test_data.keys())
    
    scores = defaultdict(dict)
    BATCH = 1000
    
    for i in tqdm(range(0, len(test_pids), BATCH), desc="ESM2 k-NN"):
        batch_pids = test_pids[i:i+BATCH]
        batch_X = [test_data[p] for p in batch_pids]
        
        dists, idxs = knn.kneighbors(batch_X)
        sims = 1 - dists
        
        for j, pid in enumerate(batch_pids):
            votes = defaultdict(float)
            for k, n_idx in enumerate(idxs[j]):
                n_pid = train_pids[n_idx]
                sim = sims[j][k]
                for t in train_terms.get(n_pid, set()):
                    votes[t] += sim
            
            if votes:
                m = max(votes.values())
                for t, v in votes.items():
                    norm = v / m
                    if norm > 0.1:
                        scores[pid][t] = norm
    return scores

def run_pubmed_test_knn(test_pids, train_terms):
    """Run PubMed k-NN for test set."""
    logger.info("Running PubMed k-NN for Test...")
    
    if not os.path.exists(PUBMED_EMB):
        logger.warning(f"{PUBMED_EMB} not found")
        return {p: {} for p in test_pids}
    
    with open(PUBMED_EMB, 'rb') as f:
        pubmed_data = pickle.load(f)
    
    # Split into train (has terms) and test (no terms)
    train_pids_with_emb = [p for p in pubmed_data if p in train_terms]
    
    if not train_pids_with_emb:
        logger.warning("No training proteins in PubMed embeddings")
        return {p: {} for p in test_pids}
    
    train_X = np.array([pubmed_data[p] for p in train_pids_with_emb])
    knn = NearestNeighbors(n_neighbors=min(50, len(train_pids_with_emb)), metric='cosine', n_jobs=-1)
    knn.fit(train_X)
    
    scores = {}
    for pid in tqdm(test_pids, desc="PubMed k-NN"):
        if pid not in pubmed_data:
            scores[pid] = {}
            continue
        
        emb = pubmed_data[pid].reshape(1, -1)
        dists, idxs = knn.kneighbors(emb)
        sims = 1 - dists[0]
        
        votes = defaultdict(float)
        for s, idx in zip(sims, idxs[0]):
            n_pid = train_pids_with_emb[idx]
            for t in train_terms.get(n_pid, set()):
                votes[t] += s
        
        if votes:
            m = max(votes.values())
            scores[pid] = {t: v/m for t, v in votes.items() if v/m > 0.1}
        else:
            scores[pid] = {}
    return scores

def main():
    # Load model
    logger.info("Loading V57 model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    priors = load_term_priors()
    train_terms = load_train_terms()
    
    # Load features
    diamond_file = run_diamond_test()
    diamond_scores = load_diamond_scores(diamond_file, train_terms)
    sbert_scores = load_sbert_scores()
    esm_scores = run_esm2_test_knn(train_terms)
    
    # Get test PIDs
    test_pids = set(diamond_scores.keys()) | set(sbert_scores.keys()) | set(esm_scores.keys())
    logger.info(f"Test proteins with scores: {len(test_pids)}")
    
    # PubMed k-NN
    pubmed_scores = run_pubmed_test_knn(list(test_pids), train_terms)
    
    # Predict
    logger.info("Generating Predictions...")
    with open(OUTPUT_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        
        for pid in tqdm(test_pids):
            # Candidate terms
            candidates = set()
            candidates.update(diamond_scores.get(pid, {}).keys())
            candidates.update(sbert_scores.get(pid, {}).keys())
            candidates.update(esm_scores.get(pid, {}).keys())
            candidates.update(pubmed_scores.get(pid, {}).keys())
            
            if not candidates:
                continue
            
            # Build features
            rows = []
            terms = []
            for t in candidates:
                rows.append([
                    diamond_scores.get(pid, {}).get(t, 0.0),
                    sbert_scores.get(pid, {}).get(t, 0.0),
                    esm_scores.get(pid, {}).get(t, 0.0),
                    pubmed_scores.get(pid, {}).get(t, 0.0),
                    priors.get(t, 0.0)
                ])
                terms.append(t)
            
            X = pd.DataFrame(rows, columns=['f_diamond', 'f_sbert', 'f_esm2', 'f_pubmed', 'f_prior'])
            preds = model.predict_proba(X)[:, 1]
            
            for t, p in zip(terms, preds):
                if p >= 0.01:
                    writer.writerow([pid, t, f"{p:.3f}"])
    
    logger.info(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
