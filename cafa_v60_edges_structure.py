#!/usr/bin/env python3
"""
Phase 40: Structure Edge Extractor (FoldSeek).
Parses FoldSeek output (Test->Train and Train->Train) to build edges.

Inputs:
- foldseek_results.m8 (Computed via foldseek_bin)

Output:
- graph_edges_structure.pkl: List of (u, v, weight)
"""

import os
import pickle
import subprocess
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
FOLDSEEK_BIN = "./foldseek_bin"
TRAIN_PDB = "structures/train" # Folder with PDBs
TEST_PDB = "structures/test" # Folder with PDBs
OUTPUT_FILE = "graph_edges_structure.pkl"
TEMP_RES = "foldseek_edges.m8"

# Check if we have pre-computed foldseek results
PRECOMPUTED_RES = "test_foldseek.m8" # Example

def run_foldseek():
    # Placeholder: Structural mining is heavy (needs GBs of PDBs).
    # If we don't have the PDBs downloading/aligning might take days.
    # Check if we have results from previous runs?
    if os.path.exists(PRECOMPUTED_RES):
        logging.info(f"Using precomputed {PRECOMPUTED_RES}")
        return PRECOMPUTED_RES
    
    # If not, check if we can run it.
    if not os.path.exists(TRAIN_PDB) or not os.path.exists(TEST_PDB):
        logging.warning("Structure folders not found. Skipping Structure Edges.")
        return None
        
    cmd = [
        FOLDSEEK_BIN, "easy-search", 
        TEST_PDB, TRAIN_PDB, TEMP_RES, "tmp_fs",
        "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    ]
    logging.info(f"Running FoldSeek: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return TEMP_RES

def load_edges(filename):
    if not filename or not os.path.exists(filename):
        return []
    
    logger.info(f"Loading structure edges from {filename}...")
    # Formats: query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits
    df = pd.read_csv(filename, sep='\t', names=['u', 'v', 'fident', 'alnlen', 'm', 'g', 'qs', 'qe', 'ts', 'te', 'evalue', 'bits'])
    
    # Filter
    df = df[df['evalue'] < 1e-3]
    
    # Normalize weight (FoldSeek bits can be high)
    df['weight'] = np.log1p(df['bits'])
    
    edges = list(zip(df['u'], df['v'], df['weight']))
    logger.info(f"Loaded {len(edges)} structure edges.")
    return edges

def main():
    res_file = run_foldseek()
    edges = load_edges(res_file)
    
    if not edges:
        logger.warning("No structure edges found.")
        # Create empty file
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump([], f)
        return

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(edges, f)
    logger.info(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
