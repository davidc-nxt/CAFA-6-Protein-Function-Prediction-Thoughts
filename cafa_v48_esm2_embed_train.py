#!/usr/bin/env python3
"""
Generate ESM-2 embeddings for TRAINING data.
Model: facebook/esm2_t33_650M_UR50D
"""

import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('esm2_train_embed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config
# MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # Fast debug
MODEL_NAME = "facebook/esm2_t33_650M_UR50D" # Production
BATCH_SIZE = 16 # Adjust based on VRAM (650M fits ~16-32 on 16GB RAM)
TRAIN_SEQS = "Train/train_sequences.fasta"
EMBEDDINGS_FILE = "train_esm2_embeddings.pkl"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_train_seqs():
    logger.info("Loading sequences...")
    seqs = {}
    with open(TRAIN_SEQS, 'r') as f:
        pid = None
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split('|')
                if len(parts) >= 2:
                    pid = parts[1]
                else:
                    pid = parts[0].split()[0]
            else:
                if pid:
                    seqs[pid] = line.strip()
    logger.info(f"Loaded {len(seqs)} sequences.")
    return seqs

def main():
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load Model
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    
    # Data
    seqs_dict = load_train_seqs()
    pids = list(seqs_dict.keys())
    sequences = [seqs_dict[pid] for pid in pids]
    
    embeddings = {}
    tokens_processed = 0
    start_time = time.time()
    
    # Resume?
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                embeddings = pickle.load(f)
            logger.info(f"Resumed from checkpoint: {len(embeddings)} computed.")
            
            # Filter done
            needed_pids = [p for p in pids if p not in embeddings]
            needed_seqs = [seqs_dict[p] for p in needed_pids]
            pids = needed_pids
            sequences = needed_seqs
            logger.info(f"Remaining to compute: {len(pids)}")
        except:
            logger.warning("Corrupt checkpoint, starting fresh.")

    if not pids:
        logger.info("All done!")
        return

    # Loop batches
    for i in tqdm(range(0, len(pids), BATCH_SIZE), desc="Embedding"):
        batch_pids = pids[i : i + BATCH_SIZE]
        batch_seqs = sequences[i : i + BATCH_SIZE]
        
        try:
            # Tokenize
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean Pooling (excluding padding)
            # attention_mask: [batch, seq_len]
            # last_hidden_state: [batch, seq_len, dim]
            
            mask = inputs['attention_mask'].unsqueeze(-1)
            last_hidden = outputs.last_hidden_state
            
            # Sum over seq dimension
            sum_embeddings = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            
            mean_embeddings = sum_embeddings / sum_mask
            
            # To CPU numpy
            batch_embs = mean_embeddings.cpu().numpy()
            
            for pid, emb in zip(batch_pids, batch_embs):
                embeddings[pid] = emb
                
            # Periodical Save (every 10 batches to avoid IO bottleneck)
            if i % (BATCH_SIZE * 10) == 0:
                 with open(EMBEDDINGS_FILE, "wb") as f:
                    pickle.dump(embeddings, f)
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("OOM! Reduce Batch Size.")
                return
            else:
                logger.error(f"Error in batch {i}: {e}")
                
    # Final save
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)
        
    logger.info(f"Finished. Total embeddings: {len(embeddings)}")

if __name__ == "__main__":
    main()
