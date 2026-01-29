#!/usr/bin/env python3
"""
Phase 40: Label Propagation on Protein Graph.
Algorithm: Y_new = alpha * S * Y_old + (1 - alpha) * Y_init

Inputs:
- graph_adj.npz
- graph_node_map.pkl
- Train/train_terms.tsv

Output:
- submission_v60_graph.tsv
"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

GRAPH_ADJ = "graph_adj.npz"
NODE_MAP_FILE = "graph_node_map.pkl"
TRAIN_TERMS = "Train/train_terms.tsv"
OUTPUT_FILE = "submission_v60_graph.tsv"

ALPHA = 0.8  # Propagation strength (0.1 - 0.9). Higher = more diffusion.
MAX_ITER = 10
TOL = 1e-4
CHUNK_SIZE = 500 # Propagate 500 terms at a time to save RAM

def load_graph():
    logger.info("Loading graph...")
    adj = sp.load_npz(GRAPH_ADJ)
    with open(NODE_MAP_FILE, "rb") as f:
        maps = pickle.load(f)
    return adj, maps["str2int"], maps["int2str"]

def load_train_labels(node_map):
    logger.info("Loading training labels...")
    # Map Term -> ID -> 1.0
    # But we need Matrix Y: (N_nodes, N_terms)
    
    # 1. Collect all terms and train proteins
    df = pd.read_csv(TRAIN_TERMS, sep='\t')
    # df columns: EntryID, term, aspect
    
    # Filter only nodes in our graph
    valid_mask = df['EntryID'].isin(node_map)
    df = df[valid_mask]
    
    # Unique terms
    unique_terms = sorted(df['term'].unique())
    term_map = {t: i for i, t in enumerate(unique_terms)}
    rev_term_map = {i: t for i, t in enumerate(unique_terms)}
    
    logger.info(f"Training Data: {len(df)} annotations, {len(unique_terms)} unique terms.")
    
    # Build Sparse Y_init
    # Rows: Nodes, Cols: Terms
    N = len(node_map)
    M = len(unique_terms)
    
    row_idx = [node_map[uid] for uid in df['EntryID']]
    col_idx = [term_map[t] for t in df['term']]
    data = np.ones(len(row_idx), dtype=np.float32)
    
    Y_init = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, M))
    
    return Y_init, term_map, rev_term_map

def propagate_chunk(adj, Y_init_chunk, alpha=ALPHA, max_iter=MAX_ITER, tol=TOL):
    """
    Run Label Propagation on a subset of columns (terms).
    Y has shape (N_nodes, Chunk_Size)
    """
    Y = Y_init_chunk.toarray().astype(np.float32) 
    # Dense is faster for matrix mul if chunk is small.
    # N=140k. Chunk=1000. Matrix 140MB. Fine.
    
    Y_init_dense = Y.copy()
    
    for i in range(max_iter):
        Y_old = Y.copy()
        # Y_new = alpha * A * Y + (1-alpha) * Y_init
        Y = alpha * (adj.dot(Y)) + (1 - alpha) * Y_init_dense
        
        diff = np.linalg.norm(Y - Y_old)
        logger.info(f"    Iter {i+1}/{max_iter} diff={diff:.6f}") # Verbose?
        if diff < tol:
            break
            
    return Y

def main():
    if not os.path.exists(GRAPH_ADJ):
        logger.error(f"{GRAPH_ADJ} not found. Run builder.")
        return

    adj, str2int, int2str = load_graph()
    num_nodes = adj.shape[0]
    
    Y_init_all, term2int, int2term = load_train_labels(str2int)
    num_terms = Y_init_all.shape[1]
    
    # Result container for Test Nodes
    # We only care about Test Nodes.
    # Identify Test Node Indices
    test_node_indices = []
    test_node_ids = []
    
    # Assume anything not in Train (and has no label) is Test? 
    # Or rely on ID patterns? 
    # Better: If it's in the graph but has 0 in Y_init (check row sum), it *might* be test.
    # Or strictly reading "Test/test_proteins.fasta" if available?
    # Let's rely on `test_diamond.tsv` queries being the test set.
    # Simplified: Just output scores for ALL nodes, filter later? No, huge.
    
    # Let's filter output for rows that match our Test IDs (from previous submissions or known list)
    # Getting test IDs from test_diamond query column seems safest if available.
    # For now, we will store ALL non-zero predictions for top terms. (Sparse).
    
    logger.info("Starting batch propagation...")
    
    final_preds = [] # List of (ProteinID, Term, Score)
    
    # Process terms in chunks
    for start in tqdm(range(0, num_terms, CHUNK_SIZE)):
        end = min(start + CHUNK_SIZE, num_terms)
        
        logger.info(f"Processing chunk {start}-{end}...")
        
        # Slice columns
        Y_chunk = Y_init_all[:, start:end]
        
        # Propagate
        Y_pred = propagate_chunk(adj, Y_chunk)
        
        # Extract predictions
        # Threshold > 0.01 to save space
        # Y_pred is (N_nodes, Chunk_Size)
        
        # Optimization: Only look at Test Nodes?
        # Since we don't have explicit Test mask here easily, let's just 
        # extract everything > 0.01 for now, but that's huge (140k * 30k * density).
        # We really need a target list.
        
        # Hack: Check if node name is in previous submission or typical Test ID format?
        # Let's assume user wants predictions for ALL nodes that had 0 labels initially?
        # Or Just dump everything and filter by ID later.
        
        rows, cols = np.where(Y_pred > 0.01)
        vals = Y_pred[rows, cols]
        
        current_chunk_preds = []
        for r, c, v in zip(rows, cols, vals):
            node_id = int2str[r]
            term = int2term[start + c]
            current_chunk_preds.append((node_id, term, v))
            
        final_preds.extend(current_chunk_preds)
        
        gc.collect()

    logger.info(f"Saving {len(final_preds)} predictions...")
    
    df_out = pd.DataFrame(final_preds, columns=['id', 'term', 'score'])
    df_out.to_csv(OUTPUT_FILE, sep='\t', index=False, header=False, float_format='%.3f')
    logger.info("Done.")

if __name__ == "__main__":
    main()
