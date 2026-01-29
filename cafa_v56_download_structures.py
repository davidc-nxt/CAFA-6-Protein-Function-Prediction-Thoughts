#!/usr/bin/env python3
"""
Download AlphaFold structures for CAFA-6 proteins.
Uses AlphaFold Database REST API.
Saves as CIF files (smaller than PDB).
"""

import os
import asyncio
import aiohttp
import csv
from tqdm.asyncio import tqdm
import logging

# Config
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v6.cif"
CONCURRENT = 50
TRAIN_SEQS = "Train/train_sequences.fasta"
TEST_SEQS = "Test/testsuperset.fasta"
TRAIN_OUT = "structures/train"
TEST_OUT = "structures/test"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def download_structure(session, pid, out_dir, semaphore):
    """Download AlphaFold structure for a UniProt ID."""
    async with semaphore:
        url = ALPHAFOLD_URL.format(pid)
        out_path = os.path.join(out_dir, f"{pid}.cif")
        
        if os.path.exists(out_path):
            return True
            
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    return True
                elif resp.status == 404:
                    # No AlphaFold structure available
                    return False
                else:
                    return False
        except Exception as e:
            return False

async def download_batch(pids, out_dir):
    """Download structures for a list of protein IDs."""
    os.makedirs(out_dir, exist_ok=True)
    
    semaphore = asyncio.Semaphore(CONCURRENT)
    connector = aiohttp.TCPConnector(limit=CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [download_structure(session, pid, out_dir, semaphore) for pid in pids]
        results = await tqdm.gather(*tasks, desc="Downloading")
        
    success = sum(results)
    logger.info(f"Downloaded {success}/{len(pids)} structures")
    return success

def load_pids(fasta_path):
    """Extract protein IDs from FASTA file."""
    pids = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                # Parse header: >sp|P12345|NAME or >P12345
                header = line[1:].strip()
                if '|' in header:
                    pid = header.split('|')[1]
                else:
                    pid = header.split()[0]
                pids.append(pid)
    return pids

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Download training structures')
    parser.add_argument('--test', action='store_true', help='Download test structures')
    args = parser.parse_args()
    
    if args.train:
        logger.info("Loading Training PIDs...")
        pids = load_pids(TRAIN_SEQS)
        logger.info(f"Found {len(pids)} training proteins")
        asyncio.run(download_batch(pids, TRAIN_OUT))
        
    if args.test:
        logger.info("Loading Test PIDs...")
        pids = load_pids(TEST_SEQS)
        logger.info(f"Found {len(pids)} test proteins")
        asyncio.run(download_batch(pids, TEST_OUT))
        
    if not args.train and not args.test:
        print("Usage: python cafa_v56_download_structures.py --train [--test]")

if __name__ == "__main__":
    main()
