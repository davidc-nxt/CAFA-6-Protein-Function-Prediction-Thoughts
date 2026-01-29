#!/usr/bin/env python3
"""
Phase 40: V60 Integration.
Blends V31 Champion (Anchor) with V60 Graph Signal.
"""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

ANCHOR_FILE = "submission-v31-0365.tsv"
GRAPH_FILE = "submission_v60_graph.tsv"
OUTPUT_FILE = "submission_v60_graph_boosted.tsv"

WEIGHT_GRAPH = 0.3 # Graph is strong context signal

def main():
    logger.info("Loading Anchors...")
    # Load V31
    df_anchor = pd.read_csv(ANCHOR_FILE, sep='\t', header=None, names=['id', 'term', 'score'])
    logger.info(f"Anchor V31: {len(df_anchor)} rows")

    # Load Graph
    logger.info("Loading Graph Predictions...")
    if not os.path.exists(GRAPH_FILE):
        logger.error("Graph predictions missing!")
        return

    df_graph = pd.read_csv(GRAPH_FILE, sep='\t', header=None, names=['id', 'term', 'g_score'])
    logger.info(f"Graph V60: {len(df_graph)} rows")

    # Merge
    logger.info("Merging...")
    df_merged = pd.merge(df_anchor, df_graph, on=['id', 'term'], how='outer')
    df_merged['score'] = df_merged['score'].fillna(0.0)
    df_merged['g_score'] = df_merged['g_score'].fillna(0.0)

    # Blend
    # Linear combination
    df_merged['final_score'] = df_merged['score'] + (WEIGHT_GRAPH * df_merged['g_score'])
    df_merged['final_score'] = df_merged['final_score'].clip(upper=1.0)

    # Filter
    df_merged = df_merged[df_merged['final_score'] >= 0.01]

    # Save
    logger.info(f"Saving {len(df_merged)} rows to {OUTPUT_FILE}...")
    df_merged[['id', 'term', 'final_score']].to_csv(
        OUTPUT_FILE, sep='\t', header=False, index=False, float_format='%.3f'
    )
    logger.info("Done.")

if __name__ == "__main__":
    main()
