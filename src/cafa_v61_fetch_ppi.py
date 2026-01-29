#!/usr/bin/env python3
"""
Phase 41: Fetch PPI Data from STRING DB.
Uncovers "Orthogonal" edges: Physical/Functional interactions.

Inputs:
- Test/test_sequences.fasta (or similar) to get IDs.

Output:
- graph_edges_ppi.pkl: List of (u, v, weight)
  Weight = combined_score / 1000.0 (0.0 - 1.0)
"""

import os
import time
import requests
import pickle
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
OUTPUT_FILE = "graph_edges_ppi.pkl"
STRING_API_URL = "https://string-db.org/api"
MIN_SCORE = 400 # Medium confidence

# We need a list of UniProt IDs ("Q9VH19", etc.)
# Where to get them?
# Let's try reading the FASTA headers.
TEST_FASTA = "Test/testsuperset.fasta" 

def get_ids_from_fasta(fasta_path):
    ids = []
    if not os.path.exists(fasta_path):
        logger.error(f"{fasta_path} not found.")
        return []
    
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                # >ID Description
                pid = line[1:].split()[0].strip()
                ids.append(pid)
    return ids

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504, 524],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def map_uniprot_to_string(uniprot_ids):
    """
    Map UniProt IDs to STRING 'stringId' using STRING API.
    Batch processing.
    """
    logger.info(f"Mapping {len(uniprot_ids)} IDs to STRING...")
    mapped_ids = {} # uniprot -> string_id
    
    # Chunk size for API - REDUCED to 50 for stability
    CHUNK = 50 
    
    url = f"{STRING_API_URL}/tsv/get_string_ids"
    session = create_session()
    
    for i in tqdm(range(0, len(uniprot_ids), CHUNK)):
        chunk = uniprot_ids[i:i+CHUNK]
        
        try:
            resp = session.post(url, data={
                "identifiers": "\n".join(chunk),
                "limit": 1,
                "echo_query": 1
            }, timeout=30)
            
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                for line in lines[1:]: # Skip header
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        query = parts[0]
                        string_id = parts[1]
                        mapped_ids[query] = string_id
            else:
                logger.warning(f"Batch {i} failed: {resp.status_code}")
                
        except Exception as e:
            logger.error(f"Error mapping batch {i}: {e}")
            # Skip bad batch
            
        time.sleep(0.1) 
        
    logger.info(f"Mapped {len(mapped_ids)}/{len(uniprot_ids)} IDs.")
    return mapped_ids

def fetch_interactions(string_ids):
    logger.info(f"Fetching interactions for {len(string_ids)} proteins...")
    edges = []
    url = f"{STRING_API_URL}/tsv/network"
    
    # Chunk size 50
    CHUNK = 50
    id_list = list(string_ids.values())
    rev_map = {v: k for k, v in string_ids.items()}
    session = create_session()
    
    for i in tqdm(range(0, len(id_list), CHUNK)):
        chunk = id_list[i:i+CHUNK]
        
        try:
            resp = session.post(url, data={
                "identifiers": "\n".join(chunk),
                "required_score": MIN_SCORE
            }, timeout=30)
            
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                if not lines: continue
                
                header = lines[0].split("\t")
                try:
                    score_idx = header.index("score")
                    id_a_idx = header.index("stringId_A")
                    id_b_idx = header.index("stringId_B")
                except ValueError:
                    continue
                
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) > score_idx:
                        s_a = parts[id_a_idx]
                        s_b = parts[id_b_idx]
                        score = float(parts[score_idx])
                        
                        if s_a in rev_map and s_b in rev_map:
                            u_a = rev_map[s_a]
                            u_b = rev_map[s_b]
                            w = score
                            if w > 1.0: w /= 1000.0
                            edges.append((u_a, u_b, w))
            else:
                 pass
                 
        except Exception as e:
            logger.error(f"Error fetching network batch {i}: {e}")
            
        time.sleep(0.1)
        
    return edges

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="Shard index")
    parser.add_argument("--total-shards", type=int, default=1, help="Total shards")
    args = parser.parse_args()

    if not os.path.exists(TEST_FASTA):
        logger.error(f"Cannot find Test FASTA: {TEST_FASTA}")
        return

    all_ids = get_ids_from_fasta(TEST_FASTA)
    
    # Sharding logic
    n = len(all_ids)
    shard_size = int(np.ceil(n / args.total_shards))
    start = args.shard * shard_size
    end = min(start + shard_size, n)
    ids = all_ids[start:end]
    
    logger.info(f"Shard {args.shard}: Processing {len(ids)} IDs ({start}-{end})")
    
    # Map
    string_map = map_uniprot_to_string(ids)
    
    # Fetch
    edges = []
    if string_map:
        edges = fetch_interactions(string_map)
        
    # Save
    out_name = f"{OUTPUT_FILE.replace('.pkl', '')}_{args.shard}.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(edges, f)
    logger.info(f"Saved {len(edges)} edges to {out_name}")

if __name__ == "__main__":
    main()
