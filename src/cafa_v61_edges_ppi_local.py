#!/usr/bin/env python3
"""
Parse protein.physical.links.v12.0.txt.gz locally to extract PPI edges.
Uses global_map.pkl (UniProt -> STRING) to filter for our proteins.
Output: graph_edges_ppi.pkl
"""
import gzip
import pickle
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_SCORE = 400  # STRING combined score threshold

def main():
    # 1. Load global map (UniProt -> STRING)
    logger.info("Loading global_map.pkl...")
    with open("global_map.pkl", "rb") as f:
        uniprot_to_string = pickle.load(f)
    
    # Create reverse map (STRING -> UniProt)
    string_to_uniprot = {v: k for k, v in uniprot_to_string.items()}
    logger.info(f"Loaded {len(string_to_uniprot)} STRING IDs in reverse map")
    
    # 2. Parse physical links file
    edges = []
    links_file = "protein.physical.links.v12.0.txt.gz"
    
    logger.info(f"Parsing {links_file}...")
    
    with gzip.open(links_file, "rt") as f:
        # Skip header
        next(f)
        
        for line in tqdm(f):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
                
            protein1, protein2, score_str = parts[0], parts[1], parts[2]
            score = int(score_str)
            
            # Filter by score
            if score < MIN_SCORE:
                continue
            
            # Check if both proteins are in our map
            if protein1 in string_to_uniprot and protein2 in string_to_uniprot:
                u1 = string_to_uniprot[protein1]
                u2 = string_to_uniprot[protein2]
                
                # Normalize score to 0-1 range (STRING scores are 0-1000)
                weight = score / 1000.0
                
                edges.append((u1, u2, weight))
    
    logger.info(f"Extracted {len(edges)} PPI edges (score >= {MIN_SCORE})")
    
    # 3. Save edges
    with open("graph_edges_ppi.pkl", "wb") as f:
        pickle.dump(edges, f)
    logger.info("Saved graph_edges_ppi.pkl")

if __name__ == "__main__":
    main()
