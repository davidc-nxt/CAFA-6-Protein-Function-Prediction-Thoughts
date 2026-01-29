#!/usr/bin/env python3
"""
Local Validation Framework.
Splits training data, trains models on subset, evaluates F-max on holdout.
Essential for debugging V47 failure.
"""

import csv
import random
import numpy as np
from collections import defaultdict, Counter
from cafa_go import load_go_basic_obo
from cafa_metrics import (
    calculate_fmax,
    calculate_mean_fmax_over_aspects,
    calculate_mean_weighted_fmax_over_aspects,
    compute_information_content_by_aspect,
)
from tqdm import tqdm
import os
import subprocess
import pickle
from sklearn.neighbors import NearestNeighbors

# Files
TRAIN_TERMS = "Train/train_terms.tsv"
TRAIN_SEQS = "Train/train_sequences.fasta"

# Validation Settings
VAL_RATIO = 0.2
SEED = 42

def load_data():
    """Load all training data."""
    print("Loading training data...")
    # CAFA-6 training terms include an aspect column:
    # - F: Molecular Function
    # - P: Biological Process
    # - C: Cellular Component
    terms_all = defaultdict(set)
    targets_by_aspect = {
        "F": defaultdict(set),
        "P": defaultdict(set),
        "C": defaultdict(set),
    }
    term_to_aspect = {}
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 3:
                pid, term, aspect = row[0], row[1], row[2]
                terms_all[pid].add(term)
                if aspect in targets_by_aspect:
                    targets_by_aspect[aspect][pid].add(term)
                    # GO terms are single-namespace; use training as a practical map.
                    term_to_aspect.setdefault(term, aspect)
                
    seqs = {}
    with open(TRAIN_SEQS, 'r') as f:
        pid = None
        for line in f:
            if line.startswith(">"):
                # Clean header: >sp|A0A0C5B5G6|... -> A0A0C5B5G6
                # Or >P12345 ... -> P12345 (if tr without sp|)
                # Robust parsing: try splitting by | first
                parts = line[1:].strip().split('|')
                if len(parts) >= 2:
                    pid = parts[1]
                else:
                    # Fallback for simple headers like >P12345
                    pid = parts[0].split()[0]
            else:
                seqs[pid] = line.strip()
                
    proteins = list(terms_all.keys())
    print(f"Loaded {len(proteins)} proteins with annotations")
    return proteins, terms_all, targets_by_aspect, term_to_aspect, seqs

def create_split(proteins):
    """Create Train/Val split."""
    random.seed(SEED)
    random.shuffle(proteins)
    split_idx = int(len(proteins) * (1 - VAL_RATIO))
    train_pids = set(proteins[:split_idx])
    val_pids = set(proteins[split_idx:])
    print(f"Split: {len(train_pids)} Train, {len(val_pids)} Val")
    return train_pids, val_pids

def _slice_targets_by_aspect(targets_by_aspect, pids):
    """Return targets_by_aspect limited to the provided protein IDs."""
    sliced = {}
    for a, d in targets_by_aspect.items():
        sliced[a] = {pid: d[pid] for pid in pids if pid in d}
    return sliced


def mock_baseline_accuracy(val_pids, terms_all, targets_by_aspect, term_to_aspect):
    """
    Calculate max theoretical performance (score = 1.0 for true terms).
    Just to verify the metric function.
    """
    preds = {}
    for pid in val_pids:
        preds[pid] = {t: 1.0 for t in terms_all[pid]}
    
    truth_by_aspect = _slice_targets_by_aspect(targets_by_aspect, val_pids)
    score = calculate_mean_fmax_over_aspects(preds, truth_by_aspect, term_to_aspect=term_to_aspect)
    print(f"Theoretical Max Mean F-max (F/P/C): {score:.3f}")

def main():
    proteins, terms_all, targets_by_aspect, term_to_aspect, seqs = load_data()
    train_pids, val_pids = create_split(proteins)

    # Load GO graph for "full" evaluation (propagation + IC weights)
    go = load_go_basic_obo("go-basic.obo", include_part_of=True, include_alt_ids=True)
    # Build IC on training split only (to avoid peeking at validation labels)
    train_targets_by_aspect = _slice_targets_by_aspect(targets_by_aspect, train_pids)
    ic_by_aspect = compute_information_content_by_aspect(
        train_targets_by_aspect,
        parents=go.parents,
        term_to_aspect=go.term_to_aspect,
    )
    
    # 1. Sanity Check Metric
    mock_baseline_accuracy(val_pids, terms_all, targets_by_aspect, term_to_aspect)
    
    # 2. Run Diamond Validation
    run_diamond_validation(train_pids, val_pids, seqs, seqs, terms_all, targets_by_aspect, term_to_aspect)

    # 3. Validation: Text Mining (SBERT)
    run_embedding_validation(
        "SBERT (Text)",
        "train_text_embeddings.pkl",
        train_pids,
        val_pids,
        terms_all,
        targets_by_aspect,
        term_to_aspect,
        go=go,
        ic_by_aspect=ic_by_aspect,
    )
    
    # 4. Validation: Protein Language (ESM-2)
    run_embedding_validation(
        "ESM-2 (Seq)",
        "train_esm2_embeddings.pkl",
        train_pids,
        val_pids,
        terms_all,
        targets_by_aspect,
        term_to_aspect,
        go=go,
        ic_by_aspect=ic_by_aspect,
    )

    # 5. Validation: Hybrid (SBERT + ESM-2)
    run_hybrid_validation(train_pids, val_pids, terms_all, targets_by_aspect, term_to_aspect, go=go, ic_by_aspect=ic_by_aspect)
    

def run_diamond_validation(train_pids, val_pids, val_seqs, train_seqs, terms_all, targets_by_aspect, term_to_aspect):
    """
    Run Diamond BLASTp validation:
    1. Make Diamond DB from Train sequences
    2. Query Val sequences against Train DB
    3. Transfer annotations (simple 1-NN or weighted)
    """
    print(f"\nEvaluating Diamond Homology Model...")
    
    # 1. Write temp FASTA files
    print("  Writing FASTA files...")
    with open("val_subset.fasta", "w") as f:
        for pid in val_pids:
            if pid in val_seqs:
                f.write(f">{pid}\n{val_seqs[pid]}\n")
                
    with open("train_subset.fasta", "w") as f:
        for pid in train_pids:
            if pid in train_seqs:
                f.write(f">{pid}\n{train_seqs[pid]}\n")
                
    # 2. Make/Run Diamond
    print("  Running Diamond...")
    cmd_db = "diamond makedb --in train_subset.fasta -d train_subset"
    subprocess.run(cmd_db, shell=True, check=True, stdout=subprocess.DEVNULL)
    
    cmd_run = "diamond blastp -q val_subset.fasta -d train_subset -o val_diamond.tsv --sensitive -k 5"
    subprocess.run(cmd_run, shell=True, check=True, stdout=subprocess.DEVNULL)
    
    # 3. Transfer Annotations
    print("  Transferring annotations...")
    preds = defaultdict(dict)
    
    with open("val_diamond.tsv", "r") as f:
        reader = csv.reader(f, delimiter='\t')
        current_q = None
        hits = []  # [(train_pid, bitscore)]

        def flush(q_pid, hits):
            if not q_pid or not hits:
                return
            denom = sum(bs for _, bs in hits) + 1e-12
            term_scores = defaultdict(float)
            for t_pid, bs in hits:
                hit_terms = terms_all.get(t_pid)
                if not hit_terms:
                    continue
                w = bs / denom
                for t in hit_terms:
                    term_scores[t] += w
            for t, s in term_scores.items():
                if s > preds[q_pid].get(t, 0.0):
                    preds[q_pid][t] = float(min(1.0, max(0.0, s)))

        for row in reader:
            # Default DIAMOND outfmt (blast tab) has bitscore at column 12 (0-based 11).
            if len(row) < 12:
                continue
            q_pid, t_pid = row[0], row[1]
            try:
                bitscore = float(row[11])
            except ValueError:
                continue

            if current_q is None:
                current_q = q_pid
            if q_pid != current_q:
                flush(current_q, hits)
                current_q = q_pid
                hits = []
            hits.append((t_pid, bitscore))

        flush(current_q, hits)
                    
    # 4. Calculate F-max
    truth_by_aspect = _slice_targets_by_aspect(targets_by_aspect, val_pids)
    mean_fmax = calculate_mean_fmax_over_aspects(preds, truth_by_aspect, term_to_aspect=term_to_aspect)
    print(f"  Diamond Homology Mean F-max (F/P/C): {mean_fmax:.3f}")
    
    # Cleanup
    os.remove("val_subset.fasta")
    os.remove("train_subset.fasta")
    os.remove("val_diamond.tsv")
    os.remove("train_subset.dmnd")


def run_embedding_validation(name, emb_file, train_pids, val_pids, terms_all, targets_by_aspect, term_to_aspect, *, go=None, ic_by_aspect=None):
    """
    Run Validation for any embedding model (SBERT, ESM-2, etc).
    """
    print(f"\nEvaluating {name} Model...")
    if not os.path.exists(emb_file):
        print(f"  Error: {emb_file} not found! (Skipping)")
        return

    with open(emb_file, "rb") as f:
        embeddings = pickle.load(f)
        
    print(f"  Loaded {len(embeddings)} embeddings.")
    
    # 1. Prepare Training Data
    train_x = []
    train_y_pids = []
    
    for pid in train_pids:
        if pid in embeddings:
            train_x.append(embeddings[pid])
            train_y_pids.append(pid)
            
    if not train_x:
        print("  No training embeddings found!")
        return
        
    print(f"  Training on {len(train_x)} proteins (Coverage: {len(train_x)/len(train_pids):.1%})...")
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(train_x)
    
    # 2. Predict on Validation
    val_x = []
    val_target_pids = []
    
    for pid in val_pids:
        if pid in embeddings:
            val_x.append(embeddings[pid])
            val_target_pids.append(pid)
            
    if not val_x:
        print("  No validation embeddings found!")
        return
        
    print(f"  Predicting for {len(val_x)} validation proteins (Coverage: {len(val_x)/len(val_pids):.1%})...")
    
    distances, indices = knn.kneighbors(val_x)
    
    # Transfer logic
    preds_subset = defaultdict(dict) # Only for those with text
    preds_all = defaultdict(dict)    # Zeros for missing
    
    go_counts = Counter()
    for pid in train_pids:
        for t in terms_all[pid]:
            go_counts[t] += 1
            
    for i, pid in enumerate(val_target_pids):
        neighbor_idx = indices[i]
        neighbor_dist = distances[i]
        sims = 1 - neighbor_dist
        
        go_scores = defaultdict(float)
        for j, idx in enumerate(neighbor_idx):
            n_pid = train_y_pids[idx]
            sim = sims[j]
            for t in terms_all[n_pid]:
                if go_counts[t] >= 10: # Min count check
                    go_scores[t] += sim
                    
        # Max Normalize
        if go_scores:
            max_s = max(go_scores.values())
            for t, s in go_scores.items():
                norm = s / max_s
                if norm >= 0.1:
                    preds_subset[pid][t] = norm
                    preds_all[pid][t] = norm
                    
    # 3. Calculate F-max
    
    # Subset Mean F-max (Quality of signal when present)
    truth_subset_by_aspect = _slice_targets_by_aspect(targets_by_aspect, val_target_pids)
    mean_fmax_sub = calculate_mean_fmax_over_aspects(
        preds_subset,
        truth_subset_by_aspect,
        term_to_aspect=term_to_aspect,
    )
    print(f"  {name} Mean F-max (Subset, {len(val_target_pids)} proteins): {mean_fmax_sub:.3f}")

    if go is not None and ic_by_aspect is not None:
        mean_wfmax_sub = calculate_mean_weighted_fmax_over_aspects(
            preds_subset,
            truth_subset_by_aspect,
            ic_by_aspect=ic_by_aspect,
            parents=go.parents,
            term_to_aspect=go.term_to_aspect,
        )
        print(f"  {name} Mean weighted F-max (Subset, full eval): {mean_wfmax_sub:.3f}")
    
    # Overall Mean F-max (Impact on whole dataset)
    truth_all_by_aspect = _slice_targets_by_aspect(targets_by_aspect, val_pids)
    mean_fmax_all = calculate_mean_fmax_over_aspects(
        preds_all,
        truth_all_by_aspect,
        term_to_aspect=term_to_aspect,
    )
    print(f"  {name} Mean F-max (Overall, {len(val_pids)} proteins): {mean_fmax_all:.3f}")

    if go is not None and ic_by_aspect is not None:
        mean_wfmax_all = calculate_mean_weighted_fmax_over_aspects(
            preds_all,
            truth_all_by_aspect,
            ic_by_aspect=ic_by_aspect,
            parents=go.parents,
            term_to_aspect=go.term_to_aspect,
        )
        print(f"  {name} Mean weighted F-max (Overall, full eval): {mean_wfmax_all:.3f}")


def run_hybrid_validation(train_pids, val_pids, terms_all, targets_by_aspect, term_to_aspect, *, go=None, ic_by_aspect=None):
    """Hybrid SBERT + ESM-2 Validation."""
    print(f"\nEvaluating Hybrid (SBERT + ESM-2) Model...")
    with open("train_text_embeddings.pkl", "rb") as f: emb_sbert = pickle.load(f)
    with open("train_esm2_embeddings.pkl", "rb") as f: emb_esm2 = pickle.load(f)

    # Train k-NNs
    sbert_x = [emb_sbert[p] for p in train_pids if p in emb_sbert]
    knn_sbert = NearestNeighbors(n_neighbors=10, metric='cosine').fit(sbert_x)
    sbert_pids = [p for p in train_pids if p in emb_sbert]
    
    esm2_x = [emb_esm2[p] for p in train_pids if p in emb_esm2]
    knn_esm2 = NearestNeighbors(n_neighbors=10, metric='cosine').fit(esm2_x)
    esm2_pids = [p for p in train_pids if p in emb_esm2]
    
    preds = defaultdict(dict)
    go_counts = Counter()
    for pid in train_pids:
        for t in terms_all[pid]: go_counts[t] += 1

    print(f"  Predicting for {len(val_pids)} validation proteins...")
    for pid in val_pids:
        if pid in emb_sbert:
            dists, idxs = knn_sbert.kneighbors([emb_sbert[pid]])
            ref_pids = sbert_pids
        elif pid in emb_esm2:
            dists, idxs = knn_esm2.kneighbors([emb_esm2[pid]])
            ref_pids = esm2_pids
        else:
            continue
            
        sims = 1 - dists[0]
        go_scores = defaultdict(float)
        for i, idx in enumerate(idxs[0]):
            n_pid = ref_pids[idx]
            sim = sims[i]
            for t in terms_all[n_pid]:
                if go_counts[t] >= 10: go_scores[t] += sim
                    
        if go_scores:
            max_s = max(go_scores.values())
            for t, s in go_scores.items():
                if s/max_s >= 0.1: preds[pid][t] = s/max_s

    truth_by_aspect = _slice_targets_by_aspect(targets_by_aspect, val_pids)
    mean_fmax = calculate_mean_fmax_over_aspects(preds, truth_by_aspect, term_to_aspect=term_to_aspect)
    print(f"  Hybrid Mean F-max (F/P/C): {mean_fmax:.3f}")

    if go is not None and ic_by_aspect is not None:
        mean_wfmax = calculate_mean_weighted_fmax_over_aspects(
            preds,
            truth_by_aspect,
            ic_by_aspect=ic_by_aspect,
            parents=go.parents,
            term_to_aspect=go.term_to_aspect,
        )
        print(f"  Hybrid Mean weighted F-max (full eval): {mean_wfmax:.3f}")
if __name__ == "__main__":
    main()
