import gzip
import os
import pickle
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ids_from_fasta(fasta_paths):
    ids = set()
    for path in fasta_paths:
        if not os.path.exists(path):
            continue
        logger.info(f"Reading {path}...")
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    # Extract ID from header
                    # CAFA Test: ">A0A0C5B5G6 9606" -> "A0A0C5B5G6"
                    # Train: ">sp|A0A0C5B5G6|MOTSC_HUMAN" -> "A0A0C5B5G6"
                    
                    raw_id = line[1:].split()[0].strip()
                    if "|" in raw_id:
                        parts = raw_id.split("|")
                        if len(parts) >= 2:
                            ids.add(parts[1])
                        else:
                            ids.add(raw_id)
                    else:
                        ids.add(raw_id)
    return ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", help="Comma-sep fasta files")
    parser.add_argument("--alias-file", default="protein.aliases.v12.0.txt.gz")
    args = parser.parse_args()
    
    # 1. Load target IDs
    inputs = args.inputs.split(",")
    target_ids = get_ids_from_fasta(inputs)
    logger.info(f"Target UniProt IDs: {len(target_ids)}")
    
    # 2. Parse Aliases
    # File format: species_id \t string_protein_id \t alias \t source
    # We want: alias (UniProt) -> string_protein_id
    # Filter by source? "UniProt_AC", "UniProt_ID"?
    # Let's keep any alias that matches our target_ids.
    
    mapping = {}
    found_count = 0
    
    logger.info(f"Parsing {args.alias_file}...")
    
    try:
        with gzip.open(args.alias_file, "rt") as f:
            for line in tqdm(f):
                if line.startswith("#"): continue
                
                parts = line.strip().split("\t")
                if len(parts) < 3: continue
                
                # File format: species_id (0) \t alias (1) \t source (2)
                # Example: 9606.ENSP00000269305 \t P04637 \t UniProt_AC
                
                string_id = parts[0]
                alias = parts[1]
                
                # Check if alias is in our target set
                if alias in target_ids:
                    mapping[alias] = string_id
                    found_count += 1
                    
                    # Optimization: Remove from target set?
                    # No, one ID might map to multiple (rare) or we want to double check coverage.
                    # Actually, if we found it, we are good.
                    
    except Exception as e:
        logger.error(f"Error reading alias file: {e}")
        return

    logger.info(f"Mapped {len(mapping)} / {len(target_ids)} unique UniProt IDs.")
    
    with open("global_map.pkl", "wb") as f:
        pickle.dump(mapping, f)
    logger.info("Saved global_map.pkl")

if __name__ == "__main__":
    main()
