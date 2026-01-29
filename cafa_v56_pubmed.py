#!/usr/bin/env python3
"""
PubMed Mining for CAFA-6.
1. Fetch PubMed IDs linked to proteins from UniProt.
2. Fetch abstracts from NCBI Entrez.
3. Embed with PubMedBERT.
"""

import os
import asyncio
import aiohttp
import pickle
import csv
import time
import xml.etree.ElementTree as ET
from tqdm.asyncio import tqdm
import logging

# Config
CONCURRENT = 20  # Conservative for NCBI rate limits
TRAIN_SEQS = "Train/train_sequences.fasta"
TEST_SEQS = "Test/testsuperset.fasta"
PUBMED_CACHE = "pubmed_cache.pkl"
ABSTRACTS_CACHE = "abstracts_cache.pkl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_pubmed_ids(session, uniprot_id, semaphore):
    """Fetch PubMed IDs linked to a UniProt protein."""
    async with semaphore:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pubmed_ids = []
                    
                    # Extract from references
                    refs = data.get('references', [])
                    for ref in refs:
                        citation = ref.get('citation', {})
                        db_refs = citation.get('citationCrossReferences', [])
                        for db_ref in db_refs:
                            if db_ref.get('database') == 'PubMed':
                                pubmed_ids.append(db_ref.get('id'))
                    
                    return uniprot_id, pubmed_ids
                else:
                    return uniprot_id, []
        except Exception as e:
            return uniprot_id, []

async def fetch_abstract(session, pmid, semaphore):
    """Fetch abstract from NCBI Entrez."""
    async with semaphore:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    # Parse XML
                    root = ET.fromstring(content)
                    abstract_elem = root.find('.//AbstractText')
                    if abstract_elem is not None and abstract_elem.text:
                        return pmid, abstract_elem.text
                return pmid, None
        except Exception as e:
            return pmid, None

def load_pids(fasta_path):
    """Extract protein IDs from FASTA file."""
    pids = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                if '|' in header:
                    pid = header.split('|')[1]
                else:
                    pid = header.split()[0]
                pids.append(pid)
    return pids

async def fetch_pubmed_batch(pids):
    """Fetch PubMed IDs for a batch of proteins."""
    semaphore = asyncio.Semaphore(CONCURRENT)
    connector = aiohttp.TCPConnector(limit=CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=30)
    
    # Load existing cache
    cache = {}
    if os.path.exists(PUBMED_CACHE):
        with open(PUBMED_CACHE, 'rb') as f:
            cache = pickle.load(f)
        logger.info(f"Loaded {len(cache)} from cache")
    
    # Filter already cached
    to_fetch = [p for p in pids if p not in cache]
    logger.info(f"Fetching PubMed IDs for {len(to_fetch)} proteins...")
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_pubmed_ids(session, pid, semaphore) for pid in to_fetch]
        results = await tqdm.gather(*tasks, desc="Fetching PubMed IDs")
        
    # Update cache
    for pid, pmids in results:
        cache[pid] = pmids
        
    # Save
    with open(PUBMED_CACHE, 'wb') as f:
        pickle.dump(cache, f)
        
    # Stats
    with_papers = sum(1 for p in cache.values() if p)
    total_papers = sum(len(v) for v in cache.values())
    logger.info(f"Proteins with papers: {with_papers}/{len(cache)} ({100*with_papers/len(cache):.1f}%)")
    logger.info(f"Total unique papers: {len(set(pmid for pmids in cache.values() for pmid in pmids))}")
    
    return cache

async def fetch_abstracts_batch(pubmed_ids):
    """Fetch abstracts for a list of PubMed IDs."""
    semaphore = asyncio.Semaphore(CONCURRENT)
    connector = aiohttp.TCPConnector(limit=CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=30)
    
    # Load existing cache
    cache = {}
    if os.path.exists(ABSTRACTS_CACHE):
        with open(ABSTRACTS_CACHE, 'rb') as f:
            cache = pickle.load(f)
        logger.info(f"Loaded {len(cache)} abstracts from cache")
    
    # Filter
    to_fetch = [p for p in pubmed_ids if p not in cache]
    logger.info(f"Fetching {len(to_fetch)} abstracts...")
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_abstract(session, pmid, semaphore) for pmid in to_fetch]
        
        # Batch with delays for NCBI rate limiting
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await tqdm.gather(*batch, desc=f"Abstracts {i}-{i+batch_size}")
            for pmid, abstract in results:
                cache[pmid] = abstract
            
            # Rate limit pause
            if i + batch_size < len(tasks):
                await asyncio.sleep(1)
                
    # Save
    with open(ABSTRACTS_CACHE, 'wb') as f:
        pickle.dump(cache, f)
        
    with_text = sum(1 for v in cache.values() if v)
    logger.info(f"Abstracts with text: {with_text}/{len(cache)}")
    
    return cache

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-pmids', action='store_true', help='Fetch PubMed IDs from UniProt')
    parser.add_argument('--fetch-abstracts', action='store_true', help='Fetch abstracts from NCBI')
    parser.add_argument('--train', action='store_true', help='Process training proteins')
    parser.add_argument('--test', action='store_true', help='Process test proteins')
    args = parser.parse_args()
    
    # For abstract fetching, we don't need pids - just use the cache
    if args.fetch_abstracts:
        if os.path.exists(PUBMED_CACHE):
            with open(PUBMED_CACHE, 'rb') as f:
                pubmed_cache = pickle.load(f)
            all_pmids = list(set(pmid for pmids in pubmed_cache.values() for pmid in pmids))
            logger.info(f"Found {len(all_pmids)} unique PubMed IDs to fetch abstracts for")
            asyncio.run(fetch_abstracts_batch(all_pmids))
        else:
            logger.error("Run --fetch-pmids first")
        return
    
    # For PubMed ID fetching, we need pids
    pids = []
    if args.train:
        pids.extend(load_pids(TRAIN_SEQS))
    if args.test:
        pids.extend(load_pids(TEST_SEQS))
        
    if not pids:
        print("Usage: python cafa_v56_pubmed.py --fetch-pmids --train [--test]")
        print("       python cafa_v56_pubmed.py --fetch-abstracts")
        return
        
    if args.fetch_pmids:
        pubmed_cache = asyncio.run(fetch_pubmed_batch(pids))

if __name__ == "__main__":
    main()
