#!/usr/bin/env python3
"""
Phase 40: Graph Builder.
Assembles edges from Homology, Structure, and Text into a Sparse Adjacency Matrix.

Inputs:
- graph_edges_homology.pkl
- graph_edges_structure.pkl
- graph_edges_text.pkl

Outputs:
- graph_adj.npz (SciPy CSR Matrix)
- graph_node_map.pkl (Dict: ProteinID -> Index)
- graph_node_map_rev.pkl (Dict: Index -> ProteinID)
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

EDGE_FILES = [
    ("graph_edges_homology.pkl", 1.0),  # Weight 1.0
    ("graph_edges_structure.pkl", 1.0), # Weight 1.0
    ("graph_edges_text.pkl", 0.8),      # Weight 0.8
    ("graph_edges_ppi.pkl", 1.5)        # Weight 1.5 (Strong Interaction Signal)
]

OUTPUT_ADJ = "graph_adj.npz"
OUTPUT_MAP = "graph_node_map.pkl"

def load_edges(filename):
    if not os.path.exists(filename):
        logger.warning(f"{filename} missing. Skipping.")
        return []
    with open(filename, "rb") as f:
        return pickle.load(f)

def main():
    all_edges = []
    
    # 1. Load All Edges
    for fname, global_w in EDGE_FILES:
        edges = load_edges(fname)
        if not edges: continue
        
        # Apply global weight
        # e is (u, v, w)
        weighted_edges = [(u, v, w * global_w) for u, v, w in edges]
        all_edges.extend(weighted_edges)
        logger.info(f"Added {len(weighted_edges)} edges from {fname}")

    if not all_edges:
        logger.error("No edges found! Cannot build graph.")
        return

    # 2. Build Node Map (Protein ID -> Int)
    unique_nodes = set()
    for u, v, w in all_edges:
        unique_nodes.add(u)
        unique_nodes.add(v)
    
    # Add all Train/Test nodes just in case they are isolated
    # (Optional, but good for indexing)
    
    sorted_nodes = sorted(list(unique_nodes))
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    rev_map = {i: node for i, node in enumerate(sorted_nodes)}
    
    logger.info(f"Total Unique Nodes: {len(sorted_nodes)}")
    
    with open(OUTPUT_MAP, "wb") as f:
        pickle.dump({"str2int": node_map, "int2str": rev_map}, f)

    # 3. Build Sparse Matrix
    rows = []
    cols = []
    data = []
    
    for u, v, w in all_edges:
        if u not in node_map or v not in node_map: continue
        i = node_map[u]
        j = node_map[v]
        
        # Symmetrize?
        # Usually dependencies are directed (Diamond Query->Target).
        # But similarity is semantic: If A is like B, likely B is like A.
        # Let's add both directions.
        
        rows.append(i)
        cols.append(j)
        data.append(w)
        
        # Reverse edge
        rows.append(j)
        cols.append(i)
        data.append(w)
        
    # Coo Matrix
    N = len(sorted_nodes)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    
    # Convert to CSR for arithmetic
    adj_csr = adj.tocsr()
    
    # Normalize? 
    # Row normalize: D^-1 A
    # Or D^-1/2 A D^-1/2
    # Simple Row Normalize usually works for Label Prop.
    # But wait, summing duplicates? (A->B from diamond AND text).
    # coo_matrix sums duplicates by default when converting to csr/csc. Perfect.
    
    logger.info("Normalizing matrix...")
    # Row sums
    row_sums = np.array(adj_csr.sum(axis=1)).flatten()
    # Avoid div by zero
    row_sums[row_sums == 0] = 1.0
    diag_inv = sp.diags(1.0 / row_sums)
    
    norm_adj = diag_inv.dot(adj_csr)
    
    logger.info(f"Graph Shape: {norm_adj.shape}")
    sp.save_npz(OUTPUT_ADJ, norm_adj)
    logger.info(f"Saved to {OUTPUT_ADJ}")

if __name__ == "__main__":
    main()
