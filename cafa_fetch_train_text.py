#!/usr/bin/env python3
"""
Fetch UniProt text for TRAINING data.
Necessary for Local Validation of Text Mining.
"""

import os
import pickle
import asyncio
import aiohttp
import logging
from tqdm.asyncio import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_text_mining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Files
TRAIN_SEQS = "Train/train_sequences.fasta"
TEXT_CACHE = "train_text_cache.pkl"
CHECKPOINT_FILE = "train_text_checkpoint.txt"

# Settings
CONCURRENT_REQUESTS = 500
BATCH_SIZE = 1000

def load_train_ids():
    """Load protein IDs from Training FASTA."""
    ids = []
    with open(TRAIN_SEQS, "r") as f:
        for line in f:
            if line.startswith(">"):
                # Clean header: >sp|A0A0C5B5G6|... -> A0A0C5B5G6
                parts = line[1:].strip().split('|')
                if len(parts) >= 2:
                    pid = parts[1]
                else:
                    pid = parts[0].split()[0]
                ids.append(pid)
    return ids

async def fetch_text_async(session, accession, semaphore):
    """Fetch text from UniProt asynchronously."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    async with semaphore:
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    text_parts = []
                    
                    # Recommended name
                    rec_name = data.get("proteinDescription", {}).get("recommendedName", {})
                    if rec_name:
                        full_name = rec_name.get("fullName", {}).get("value", "")
                        if full_name: text_parts.append(full_name)
                    
                    # Function & Subcellular Location
                    for comment in data.get("comments", []):
                        if comment.get("commentType") == "FUNCTION":
                            for t in comment.get("texts", []):
                                if t.get("value"): text_parts.append(t.get("value"))
                        if comment.get("commentType") == "SUBCELLULAR LOCATION":
                            for loc in comment.get("subcellularLocations", []):
                                val = loc.get("location", {}).get("value", "")
                                if val: text_parts.append(f"Located in {val}")
                        
                    full_text = " ".join(text_parts)
                    return accession, full_text if full_text else None
                    
                else:
                    return accession, None
        except Exception as e:
            return accession, None
        return accession, None

async def main_loop():
    logger.info("=== Training Data Text Fetcher ===")
    
    # 1. Load IDs
    train_ids = load_train_ids()
    logger.info(f"Total training proteins: {len(train_ids)}")
    
    # 2. Load Cache
    if os.path.exists(TEXT_CACHE):
        with open(TEXT_CACHE, "rb") as f:
            text_cache = pickle.load(f)
        logger.info(f"Loaded cache: {len(text_cache)} entries")
    else:
        text_cache = {}
        
    # 3. Filter
    remaining_ids = [pid for pid in train_ids if pid not in text_cache]
    logger.info(f"Remaining to fetch: {len(remaining_ids)}")
    
    if not remaining_ids:
        logger.info("Nothing to do!")
        return

    # 4. Async Fetch
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(remaining_ids), BATCH_SIZE):
            batch = remaining_ids[i : i + BATCH_SIZE]
            tasks = [fetch_text_async(session, pid, semaphore) for pid in batch]
            
            results = await tqdm.gather(*tasks, desc=f"Batch {i//BATCH_SIZE}")
            
            for pid, text in results:
                if text:
                    text_cache[pid] = text
            
            # Save
            with open(TEXT_CACHE, "wb") as f:
                pickle.dump(text_cache, f)
            logger.info(f"Saved checkpoint: {len(text_cache)} entries")
            
    logger.info("Done fetching training text!")

if __name__ == "__main__":
    asyncio.run(main_loop())
