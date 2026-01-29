#!/usr/bin/env python3
"""
V58 Blend: Boost V49 (Champion Anchor) with pure PubMed signal.
V49 Score: 0.352 (High quality blend)
PubMed: High precision literature signal (~22% importance in V57)

Logic:
- Load V49
- Load PubMed (V58 pure)
- Boost: Final = V49 + 0.2 * PubMed
- Clip at 1.0
"""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

V31_FILE = "submission-v31-0365.tsv"
PUBMED_FILE = "submission_v58_pubmed.tsv"
OUTPUT_FILE = "submission_v58_boosted_v31.tsv"

def main():
    logger.info("Loading V31 (Champion Anchor 0.365)...")
    if not os.path.exists(V31_FILE):
        logger.error(f"{V31_FILE} not found!")
        return

    # Load V31
    # Use pandas for speed? But files are large (1-2GB).
    # 6M rows. Pandas might use ~5GB RAM. M1 Mac has Unified Memory, likely fine.
    df_main = pd.read_csv(V31_FILE, sep='\t', header=None, names=['id', 'term', 'score'])
    logger.info(f"V31 loaded: {len(df_main)} rows")
    
    logger.info("Loading PubMed predictions...")
    if not os.path.exists(PUBMED_FILE):
        logger.error(f"{PUBMED_FILE} not found!")
        return
        
    df_pub = pd.read_csv(PUBMED_FILE, sep='\t', header=None, names=['id', 'term', 'pub_score'])
    logger.info(f"PubMed loaded: {len(df_pub)} rows")
    
    # Merge
    logger.info("Merging...")
    # Outer join to keep both? 
    # V49 should be dense. PubMed is sparse.
    # Left join V49 <- PubMed? No, we might want to Add interactions.
    # But usually baseline (V49) covers mostly everything.
    # Let's do Left Merge on V49, and if PubMed has *new* predictions, append them?
    # PubMed k-NN should be restricted to valid terms.
    
    df_merged = pd.merge(df_main, df_pub, on=['id', 'term'], how='outer')
    
    # Fill NA
    df_merged['score'] = df_merged['score'].fillna(0.0)
    df_merged['pub_score'] = df_merged['pub_score'].fillna(0.0)
    
    # Boost Logic
    # V58 = V49 + 0.2 * PubMed
    logger.info("Applying Boost...")
    df_merged['final_score'] = df_merged['score'] + (0.2 * df_merged['pub_score'])
    
    # Clip
    df_merged['final_score'] = df_merged['final_score'].clip(upper=1.0)
    
    # Threshold filter
    df_merged = df_merged[df_merged['final_score'] >= 0.01]
    
    # Sort (optional, but good for inspection)
    # Not strictly needed for submission but good for consistency
    # df_merged.sort_values(['id', 'final_score'], ascending=[True, False], inplace=True)
    
    # Output
    logger.info(f"Saving {len(df_merged)} rows to {OUTPUT_FILE}...")
    df_merged[['id', 'term', 'final_score']].to_csv(
        OUTPUT_FILE, sep='\t', header=False, index=False, float_format='%.3f'
    )
    logger.info("Done!")

if __name__ == "__main__":
    main()
