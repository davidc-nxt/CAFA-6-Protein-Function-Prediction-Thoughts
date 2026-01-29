#!/usr/bin/env python3
"""
Phase 40: Homology Edge Extractor.
Parses Diamond alignment (BLASTP) results to build edges for the Protein Graph.

Inputs:
- test_diamond.tsv (Test -> Train)
- train_matches.tsv (Train -> Train)

Output:
- graph_edges_homology.pkl: List of (source_id, target_id, weight)
  Weight = Normalized Bitscore (0.0 - 1.0)
"""

import os
import pickle
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_EDGES_FILE = "test_diamond.tsv"
TRAIN_EDGES_FILE = "train_matches.tsv"
OUTPUT_FILE = "graph_edges_homology.pkl"

MIN_BITSCORE = 20.0  # Filter weak edges

def load_diamond_tsv(filename, edge_type="Test->Train"):
    if not os.path.exists(filename):
        logger.warning(f"{filename} not found!")
        return []
    
    logger.info(f"Loading {edge_type} from {filename}...")
    # Cols: qseqid, sseqid, pident, bitscore, length
    df = pd.read_csv(filename, sep='\t', names=['u', 'v', 'pident', 'bitscore', 'length'])
    
    # Filter self-loops for Train->Train
    if edge_type == "Train->Train":
        df = df[df['u'] != df['v']]
        
    # Filter weak edges
    df = df[df['bitscore'] >= MIN_BITSCORE]
    
    # Normalize weights (simple scaling for now, refined in Graph Builder)
    # Bitscores typically 20 - 1000+
    # Log transform might be better: log(1 + bitscore)
    df['weight'] = np.log1p(df['bitscore'])
    
    # Return as list of tuples
    edges = list(zip(df['u'], df['v'], df['weight']))
    logger.info(f"Loaded {len(edges)} edges from {filename}")
    return edges

def main():
    all_edges = []
    
    # 1. Test -> Train
    all_edges.extend(load_diamond_tsv(TEST_EDGES_FILE, "Test->Train"))
    
    # 2. Train -> Train
    all_edges.extend(load_diamond_tsv(TRAIN_EDGES_FILE, "Train->Train"))
    
    logger.info(f"Total Homology Edges: {len(all_edges)}")
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_edges, f)
    
    logger.info(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
