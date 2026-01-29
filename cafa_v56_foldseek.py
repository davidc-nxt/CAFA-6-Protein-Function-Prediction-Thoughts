#!/usr/bin/env python3
"""
FoldSeek Pipeline for CAFA-6.
1. Build database from training structures.
2. Query validation/test against training.
3. Transfer GO terms by structural similarity.
"""

import os
import csv
import subprocess
from collections import defaultdict
from tqdm import tqdm
import logging

# Config
FOLDSEEK = "./foldseek_bin"
TRAIN_STRUCTS = "structures/train"
TEST_STRUCTS = "structures/test"  
DB_PATH = "foldseek_train_db"
TRAIN_TERMS = "Train/train_terms.tsv"
OUTPUT_VAL = "foldseek_val_preds.tsv"
OUTPUT_TEST = "foldseek_test_preds.tsv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def build_database():
    """Build FoldSeek database from training structures."""
    logger.info("Building FoldSeek database...")
    
    # Check structure count
    structs = [f for f in os.listdir(TRAIN_STRUCTS) if f.endswith('.cif')]
    logger.info(f"Found {len(structs)} training structures")
    
    # Build DB
    cmd = [
        FOLDSEEK, "createdb",
        TRAIN_STRUCTS,
        DB_PATH,
        "--threads", "8"
    ]
    subprocess.run(cmd, check=True)
    
    # Create index for faster search
    logger.info("Creating index...")
    cmd = [
        FOLDSEEK, "createindex",
        DB_PATH,
        "tmp_foldseek",
        "--threads", "8"
    ]
    subprocess.run(cmd, check=True)
    
    logger.info("Database built successfully")

def query_structures(query_dir, output_file, top_k=50):
    """Query structures against training database."""
    logger.info(f"Querying {query_dir} against training DB...")
    
    result_file = "foldseek_results.tsv"
    
    # Create query DB
    query_db = "foldseek_query_db"
    cmd = [
        FOLDSEEK, "createdb",
        query_dir,
        query_db,
        "--threads", "8"
    ]
    subprocess.run(cmd, check=True)
    
    # Search
    cmd = [
        FOLDSEEK, "search",
        query_db,
        DB_PATH,
        result_file,
        "tmp_foldseek",
        "--threads", "8",
        "-e", "10",  # E-value cutoff
        "-s", "7.5",  # Sensitivity
        "--max-seqs", str(top_k)
    ]
    subprocess.run(cmd, check=True)
    
    # Convert to TSV
    cmd = [
        FOLDSEEK, "convertalis",
        query_db,
        DB_PATH,
        result_file,
        result_file + ".txt",
        "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    ]
    subprocess.run(cmd, check=True)
    
    # Process results into GO term predictions
    logger.info("Processing results into GO predictions...")
    
    # Load training GO terms
    train_terms = defaultdict(set)
    with open(TRAIN_TERMS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            train_terms[row[0]].add(row[1])
    
    # Process hits
    scores = defaultdict(dict)
    
    with open(result_file + ".txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 12:
                query = parts[0].replace('.cif', '')
                target = parts[1].replace('.cif', '')
                bitscore = float(parts[11])
                
                # Transfer GO terms
                if target in train_terms:
                    for term in train_terms[target]:
                        if bitscore > scores[query].get(term, 0):
                            scores[query][term] = bitscore
    
    # Normalize and write
    logger.info(f"Writing predictions to {output_file}...")
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for pid in scores:
            if scores[pid]:
                max_score = max(scores[pid].values())
                for term, score in scores[pid].items():
                    norm = score / max_score
                    if norm >= 0.1:
                        writer.writerow([pid, term, f"{norm:.3f}"])
    
    logger.info("Done")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-db', action='store_true', help='Build training database')
    parser.add_argument('--query-val', action='store_true', help='Query validation structures')
    parser.add_argument('--query-test', action='store_true', help='Query test structures')
    args = parser.parse_args()
    
    if args.build_db:
        build_database()
        
    if args.query_val:
        val_structs = "structures/val"  # Need to create this from val_split
        if os.path.exists(val_structs):
            query_structures(val_structs, OUTPUT_VAL)
        else:
            logger.warning(f"Validation structures not found at {val_structs}")
            
    if args.query_test:
        query_structures(TEST_STRUCTS, OUTPUT_TEST)
        
    if not (args.build_db or args.query_val or args.query_test):
        print("Usage: python cafa_v56_foldseek.py --build-db [--query-val] [--query-test]")

if __name__ == "__main__":
    main()
