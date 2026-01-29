import argparse
import os
import pickle
import logging
import time
import requests
import glob
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

STRING_API_URL = "https://string-db.org/api"
MIN_SCORE = 400

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

def get_ids_from_fasta(fasta_paths):
    ids = []
    for path in fasta_paths:
        if not os.path.exists(path):
            continue
        logger.info(f"Reading {path}...")
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    pid = line[1:].split()[0].strip()
                    ids.append(pid)
    return ids

def run_mapping(args):
    """
    Map UniProt -> STRING.
    """
    # Load ALL IDs (Train + Test)
    # Why? We need to map Train too, to find Test-Train edges?
    # Yes. But we only query interactions for TEST proteins (to save time) 
    # OR we need global network?
    # If we only query Test proteins, we receive (Test, Partner).
    # If Partner is Train, we keep it.
    # So we need to map ALL IDs to know "Who is Partner?".
    
    inputs = args.inputs.split(",")
    all_ids = get_ids_from_fasta(inputs)
    logger.info(f"Total IDs: {len(all_ids)}")
    
    # Shard
    n = len(all_ids)
    shard_size = (n + args.total_shards - 1) // args.total_shards
    start = args.shard * shard_size
    end = min(start + shard_size, n)
    ids = all_ids[start:end]
    
    logger.info(f"Shard {args.shard}: Mapping {len(ids)} IDs ({start}-{end})")
    
    session = create_session()
    url = f"{STRING_API_URL}/tsv/get_string_ids"
    mapped_ids = {}
    CHUNK = 200
    
    for i in tqdm(range(0, len(ids), CHUNK)):
        chunk = ids[i:i+CHUNK]
        try:
            resp = session.post(url, data={
                "identifiers": "\n".join(chunk),
                "limit": 1,
                "echo_query": 1
            }, timeout=30)
            
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        query = parts[0]
                        string_id = parts[1]
                        mapped_ids[query] = string_id
        except Exception as e:
            logger.warning(f"Batch {i} error: {e}")
            
        # Intermediate save every 50 batches (10k IDs)
        if (i // CHUNK) % 50 == 0:
             out_file = f"map_shard_{args.shard}_partial.pkl"
             with open(out_file, "wb") as f:
                 pickle.dump(mapped_ids, f)

    out_file = f"map_shard_{args.shard}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(mapped_ids, f)
    logger.info(f"Saved {len(mapped_ids)} mappings to {out_file}")

def run_fetching(args):
    """
    Fetch interactions using Global Map.
    """
    # 1. Load Global Map
    # Assume 'global_map.pkl' exists (merged from map_shard_*.pkl)
    if not os.path.exists("global_map.pkl"):
        # Try to merge on the fly?
        shards = glob.glob("map_shard_*.pkl")
        global_map = {}
        for s in shards:
            with open(s, "rb") as f:
                global_map.update(pickle.load(f))
        # Save for future
        with open("global_map.pkl", "wb") as f:
            pickle.dump(global_map, f)
    else:
        with open("global_map.pkl", "rb") as f:
            global_map = pickle.load(f)
            
    # Reverse Map: string -> uniprot
    rev_map = {v: k for k, v in global_map.items()}
    logger.info(f"Global Map Size: {len(global_map)}")
    
    # 2. Identify Target IDs to Query
    # We want edges for ALL proteins in the graph (Train+Test).
    # Querying 140k proteins via API is heavy.
    # Maybe prioritize Test proteins?
    # If we query Test, we get Test-Test and Test-Train. 
    # Do we need Train-Train? Yes, for propagation.
    # But Train-Train edges might be dense.
    # Let's try to query ALL mapped IDs.
    
    all_string_ids = list(global_map.values())
    
    # Shard the QUERY list
    n = len(all_string_ids)
    shard_size = (n + args.total_shards - 1) // args.total_shards
    start = args.shard * shard_size
    end = min(start + shard_size, n)
    target_ids = all_string_ids[start:end]
    
    logger.info(f"Shard {args.shard}: Fetching for {len(target_ids)} proteins")
    
    session = create_session()
    url = f"{STRING_API_URL}/tsv/network"
    edges = []
    CHUNK = 50
    
    for i in tqdm(range(0, len(target_ids), CHUNK)):
        chunk = target_ids[i:i+CHUNK]
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
                        
                        # Resolve s_b using Global Map
                        # s_a is definitely in our list (we queried it)
                        # s_b might be anything. Check against rev_map.
                        
                        if s_a in rev_map and s_b in rev_map:
                            u_a = rev_map[s_a]
                            u_b = rev_map[s_b]
                            w = score
                            if w > 1.0: w /= 1000.0
                            edges.append((u_a, u_b, w))
        except Exception:
            pass
            
    out_file = f"edge_shard_{args.shard}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(edges, f)
    logger.info(f"Saved {len(edges)} edges to {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["map", "fetch"])
    parser.add_argument("--inputs", help="Comma-sep fasta files")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    args = parser.parse_args()
    
    if args.mode == "map":
        run_mapping(args)
    elif args.mode == "fetch":
        run_fetching(args)

if __name__ == "__main__":
    main()
