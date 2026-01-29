#!/usr/bin/env python3
"""
V57 LTR: Add PubMedBERT literature signal as 5th feature.
Trains on: Diamond, SBERT, ESM2, PubMedBERT, GO Prior
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import logging
import gc

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
TRAIN_TERMS = "Train/train_terms.tsv"
TRAIN_SEQS = "Train/train_sequences.fasta"
SBERT_EMB = "train_text_embeddings.pkl"
ESM2_EMB = "train_esm2_embeddings.pkl"
PUBMED_EMB = "pubmed_embeddings.pkl"
DIAMOND_DB = "diamond_train_split.dmnd"
VAL_FASTA = "val_split.fasta"
VAL_TERMS = "val_terms.tsv"
OUTPUT_DATASET = "ltr_v57_dataset.pkl"
OUTPUT_MODEL = "v57_ltr_xgboost.model"

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
    logger.info("Loading train terms...")
    terms = defaultdict(set)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            terms[row[0]].add(row[1])
    return terms

def load_val_terms():
    """Load validation protein -> GO terms mapping."""
    if not os.path.exists(VAL_TERMS):
        logger.warning(f"{VAL_TERMS} not found, will generate from split")
        return {}
    terms = defaultdict(set)
    with open(VAL_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            terms[row[0]].add(row[1])
    return terms

def run_embedding_knn(val_pids, train_emb_file, train_terms, k=50, name="EMB"):
    """Run k-NN on embeddings to get GO scores."""
    logger.info(f"Running {name} k-NN...")
    
    with open(train_emb_file, 'rb') as f:
        train_data = pickle.load(f)
    
    train_pids = list(train_data.keys())
    train_X = np.array([train_data[p] for p in train_pids])
    
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_X)
    
    scores = {}
    for pid in tqdm(val_pids, desc=f"{name} k-NN"):
        if pid not in train_data:
            scores[pid] = {}
            continue
            
        emb = train_data[pid].reshape(1, -1)
        dists, idxs = knn.kneighbors(emb)
        sims = 1 - dists[0]
        
        votes = defaultdict(float)
        for s, idx in zip(sims, idxs[0]):
            n_pid = train_pids[idx]
            if n_pid in train_terms:
                for t in train_terms[n_pid]:
                    votes[t] += s
        
        if votes:
            m = max(votes.values())
            scores[pid] = {t: v/m for t, v in votes.items()}
        else:
            scores[pid] = {}
            
    return scores

def build_feature_matrix(val_pids, val_terms, diamond_scores, sbert_scores, esm2_scores, pubmed_scores, priors):
    """Build the LTR feature matrix."""
    logger.info("Building feature matrix...")
    
    rows = []
    for pid in tqdm(val_pids, desc="Building features"):
        # Get all candidate terms
        candidates = set()
        candidates.update(diamond_scores.get(pid, {}).keys())
        candidates.update(sbert_scores.get(pid, {}).keys())
        candidates.update(esm2_scores.get(pid, {}).keys())
        candidates.update(pubmed_scores.get(pid, {}).keys())
        
        if not candidates:
            continue
            
        true_terms = val_terms.get(pid, set())
        
        for term in candidates:
            rows.append({
                'protein': pid,
                'term': term,
                'f_diamond': diamond_scores.get(pid, {}).get(term, 0.0),
                'f_sbert': sbert_scores.get(pid, {}).get(term, 0.0),
                'f_esm2': esm2_scores.get(pid, {}).get(term, 0.0),
                'f_pubmed': pubmed_scores.get(pid, {}).get(term, 0.0),
                'f_prior': priors.get(term, 0.0),
                'label': 1 if term in true_terms else 0
            })
    
    df = pd.DataFrame(rows)
    logger.info(f"Dataset: {len(df)} rows, {df['label'].sum()} positives")
    return df

def train_ltr(df):
    """Train XGBoost LTR model."""
    logger.info("Training LTR model...")
    
    features = ['f_diamond', 'f_sbert', 'f_esm2', 'f_pubmed', 'f_prior']
    X = df[features]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use tuned parameters from V56
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    logger.info(f"Test AUC: {auc:.4f}")
    
    # Feature importance
    logger.info("Feature Importance:")
    for f, i in zip(features, model.feature_importances_):
        logger.info(f"  {f}: {i:.4f}")
    
    # Save
    model.save_model(OUTPUT_MODEL)
    logger.info(f"Saved to {OUTPUT_MODEL}")
    
    return model, auc

def main():
    # Check if PubMed embeddings exist
    if not os.path.exists(PUBMED_EMB):
        logger.error(f"{PUBMED_EMB} not found. Run cafa_v57_pubmedbert_embed.py first.")
        return
    
    # Load data
    priors = load_term_priors()
    train_terms = load_train_terms()
    val_terms_dict = load_val_terms()
    
    # Get validation PIDs
    val_pids = list(val_terms_dict.keys())
    if not val_pids:
        logger.warning("No validation split found, using subset of training")
        all_pids = list(train_terms.keys())
        val_pids = all_pids[:len(all_pids)//5]  # 20%
        val_terms_dict = {p: train_terms[p] for p in val_pids}
    
    logger.info(f"Validation proteins: {len(val_pids)}")
    
    # Get k-NN scores for each embedding type
    sbert_scores = run_embedding_knn(val_pids, SBERT_EMB, train_terms, name="SBERT")
    esm2_scores = run_embedding_knn(val_pids, ESM2_EMB, train_terms, name="ESM2")
    pubmed_scores = run_embedding_knn(val_pids, PUBMED_EMB, train_terms, name="PubMed")
    
    # Diamond scores (simplified - use existing if available)
    diamond_scores = {}
    if os.path.exists("diamond_val_scores.pkl"):
        with open("diamond_val_scores.pkl", 'rb') as f:
            diamond_scores = pickle.load(f)
    else:
        logger.info("Diamond scores not pre-computed, using zeros")
    
    # Build dataset
    df = build_feature_matrix(val_pids, val_terms_dict, diamond_scores, sbert_scores, esm2_scores, pubmed_scores, priors)
    
    # Save dataset
    df.to_pickle(OUTPUT_DATASET)
    logger.info(f"Saved dataset to {OUTPUT_DATASET}")
    
    # Train
    model, auc = train_ltr(df)
    
    logger.info("Done! Ready for V57 predictions.")

if __name__ == "__main__":
    main()
