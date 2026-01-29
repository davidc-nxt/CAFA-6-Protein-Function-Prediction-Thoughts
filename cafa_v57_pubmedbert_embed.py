#!/usr/bin/env python3
"""
Generate PubMedBERT embeddings from literature abstracts.
Creates embeddings for proteins based on their linked paper abstracts.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import logging

# Config
PUBMED_CACHE = "pubmed_cache.pkl"
ABSTRACTS_CACHE = "abstracts_cache.pkl"
OUTPUT_FILE = "pubmed_embeddings.pkl"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load PubMed IDs and abstracts."""
    logger.info("Loading PubMed cache...")
    with open(PUBMED_CACHE, 'rb') as f:
        pubmed_cache = pickle.load(f)
        
    logger.info("Loading abstracts cache...")
    with open(ABSTRACTS_CACHE, 'rb') as f:
        abstracts_cache = pickle.load(f)
        
    return pubmed_cache, abstracts_cache

def generate_embeddings():
    """Generate PubMedBERT embeddings for proteins."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    # Load model
    logger.info(f"Loading PubMedBERT model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Load data
    pubmed_cache, abstracts_cache = load_data()
    
    # Build protein -> abstracts mapping
    logger.info("Building protein -> abstract mapping...")
    protein_texts = {}
    for pid, pmids in pubmed_cache.items():
        texts = []
        for pmid in pmids:
            if pmid in abstracts_cache and abstracts_cache[pmid]:
                texts.append(abstracts_cache[pmid])
        if texts:
            # Combine all abstracts for a protein (truncate if too long)
            combined = " ".join(texts)[:4000]  # Max ~4k chars
            protein_texts[pid] = combined
            
    logger.info(f"Proteins with abstract text: {len(protein_texts)}")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = {}
    batch_size = 16
    
    pids = list(protein_texts.keys())
    for i in tqdm(range(0, len(pids), batch_size)):
        batch_pids = pids[i:i+batch_size]
        batch_texts = [protein_texts[p] for p in batch_pids]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        # Store
        for pid, emb in zip(batch_pids, batch_embeddings):
            embeddings[pid] = emb
            
    logger.info(f"Generated embeddings for {len(embeddings)} proteins")
    
    # Save
    logger.info(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
        
    logger.info("Done!")
    return embeddings

if __name__ == "__main__":
    generate_embeddings()
