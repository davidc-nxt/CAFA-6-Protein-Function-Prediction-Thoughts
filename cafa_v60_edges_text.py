#!/usr/bin/env python3
"""
Phase 40: Text Edge Extractor (PubMed).
Builds a k-NN graph from PubMedBERT embeddings.

Inputs:
- pubmed_embeddings.pkl (Dict: ProteinID -> Embedding Vector)

Output:
- graph_edges_text.pkl: List of (u, v, weight)
"""

import os
import pickle
import numpy as np
import logging
from scipy.spatial.distance import cdist
# Try importing faiss, fallback to sklearn/scipy
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

EMBED_FILE = "pubmed_embeddings.pkl"
OUTPUT_FILE = "graph_edges_text.pkl"
K_NEIGHBORS = 15  # Connect to top-15 text similar proteins
BATCH_SIZE = 5000

def load_embeddings():
    if not os.path.exists(EMBED_FILE):
        raise FileNotFoundError(f"{EMBED_FILE} not found!")
    
    logger.info(f"Loading embeddings from {EMBED_FILE}...")
    with open(EMBED_FILE, "rb") as f:
        data = pickle.load(f)
        
    ids = list(data.keys())
    # Stack vectors
    matrix = np.stack([data[uid] for uid in ids])
    return ids, matrix

def build_edges_faiss(ids, matrix, k=K_NEIGHBORS):
    logger.info(f"Building k-NN graph with FAISS (N={len(ids)}, Dim={matrix.shape[1]})...")
    d = matrix.shape[1]
    index = faiss.IndexFlatIP(d) # Inner Product (Cosine if normalized)
    
    # Normalize for Cosine Similarity
    faiss.normalize_L2(matrix)
    index.add(matrix)
    
    # Search
    # Query: All proteins against All proteins
    D, I = index.search(matrix, k + 1) # k+1 because self is included
    
    edges = []
    for i in range(len(ids)):
        u = ids[i]
        # Skip self (usually index 0 in results, but check)
        # Distances D[i], Indices I[i]
        for rank, j in enumerate(I[i]):
            if j == i or j == -1: continue # Skip self
            v = ids[j]
            w = float(D[i][rank])
            if w > 0.5: # Hard threshold for relevance
                edges.append((u, v, w))
                
    return edges

def build_edges_slow(ids, matrix, k=K_NEIGHBORS):
    logger.warning("FAISS not found. Using naive block processing. This might be slow.")
    # Normalize
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / (norm + 1e-9)
    
    edges = []
    n = len(ids)
    
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = matrix[start:end]
        
        # Dot product: (Batch, D) x (N, D).T -> (Batch, N)
        sims = np.dot(batch, matrix.T)
        
        # Top-k
        # Argpartition is faster than sort
        # We want top k+1
        top_k_idx = np.argpartition(sims, -(k+1), axis=1)[:, -(k+1):]
        
        for local_i in range(len(batch)):
            global_i = start + local_i
            u = ids[global_i]
            
            # Get exact scores for top k
            indices = top_k_idx[local_i]
            scores = sims[local_i, indices]
            
            for j, w in zip(indices, scores):
                if j == global_i: continue
                if w > 0.5:
                    edges.append((u, ids[j], float(w)))
                    
        if start % 10000 == 0:
            logger.info(f"Processed {start}/{n}...")
            
    return edges

def main():
    try:
        ids, matrix = load_embeddings()
        
        if HAS_FAISS:
            edges = build_edges_faiss(ids, matrix)
        else:
            edges = build_edges_slow(ids, matrix)
            
        logger.info(f"Extracted {len(edges)} text edges.")
        
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(edges, f)
            
        logger.info(f"Saved to {OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"Failed: {e}")

if __name__ == "__main__":
    main()
