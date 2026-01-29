#!/usr/bin/env python3
"""
V58: Generate PubMed-only predictions using k-NN on PubMedBERT embeddings as a pure signal.
Output: submission_v58_pubmed.tsv
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd
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
PUBMED_EMB = "pubmed_embeddings.pkl"
OUTPUT_FILE = "submission_v58_pubmed.tsv"

def load_train_terms():
    """Load training protein -> GO terms mapping."""
    logger.info("Loading train terms...")
    terms = defaultdict(set)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            terms[row[0]].add(row[1])
    return terms

def run_pubmed_knn(train_terms):
    """Run PubMed k-NN for test set."""
    logger.info("Running PubMed k-NN...")
    
    if not os.path.exists(PUBMED_EMB):
        logger.error(f"{PUBMED_EMB} not found")
        raise FileNotFoundError(PUBMED_EMB)
    
    with open(PUBMED_EMB, 'rb') as f:
        pubmed_data = pickle.load(f)
    
    # Identify Test Proteins (those in fasta)
    # Actually, we can just predict for ALL proteins in pubmed_data that represent test set
    # But better to filter by test superset to match submission format?
    # Or just predict for everything in embed file that isn't in train?
    
    # Let's load test PIDs
    test_pids = []
    logger.info("Loading test proteins from fasta...")
    with open(TEST_FASTA, 'r') as f:
         for line in f:
             if line.startswith(">"):
                 test_pids.append(line[1:].strip().split()[0])
    
    # Split embeddings into Train (has terms) and Test (we need preds)
    train_pids_with_emb = [p for p in pubmed_data if p in train_terms]
    test_pids_with_emb = [p for p in test_pids if p in pubmed_data]
    
    logger.info(f"PubMed Coverage: Train={len(train_pids_with_emb)}, Test={len(test_pids_with_emb)}")
    
    if not train_pids_with_emb:
        logger.error("No training proteins have PubMed embeddings!")
        return
    
    # k-NN Setup
    train_X = np.array([pubmed_data[p] for p in train_pids_with_emb])
    knn = NearestNeighbors(n_neighbors=min(50, len(train_pids_with_emb)), metric='cosine', n_jobs=-1)
    knn.fit(train_X)
    
    logger.info("Generating predictions...")
    with open(OUTPUT_FILE, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Batch predict for memory safety
        BATCH = 1000
        for i in tqdm(range(0, len(test_pids_with_emb), BATCH)):
            batch_pids = test_pids_with_emb[i:i+BATCH]
            batch_X = np.array([pubmed_data[p] for p in batch_pids])
            
            dists, idxs = knn.kneighbors(batch_X)
            sims = 1 - dists
            
            for j, pid in enumerate(batch_pids):
                votes = defaultdict(float)
                for k, idx in enumerate(idxs[j]):
                    n_pid = train_pids_with_emb[idx]
                    sim = sims[j][k]
                    for t in train_terms[n_pid]:
                        votes[t] += sim
                
                if votes:
                    m = max(votes.values())
                    for t, v in votes.items():
                        score = v / m
                        if score >= 0.01:
                            writer.writerow([pid, t, f"{score:.3f}"])

    logger.info(f"Done. Saved to {OUTPUT_FILE}")

def main():
    train_terms = load_train_terms()
    run_pubmed_knn(train_terms)

if __name__ == "__main__":
    main()
