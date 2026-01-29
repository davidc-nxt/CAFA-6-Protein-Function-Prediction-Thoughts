import pickle
import glob
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_FILE = "graph_edges_ppi.pkl"

def main():
    shards = glob.glob("graph_edges_ppi_*.pkl")
    # Exclude the output file itself if it exists
    shards = [s for s in shards if s != OUTPUT_FILE]
    
    logger.info(f"Found {len(shards)} shards.")
    
    all_edges = []
    for s in shards:
        try:
            with open(s, "rb") as f:
                edges = pickle.load(f)
                all_edges.extend(edges)
        except Exception as e:
            logger.error(f"Error loading {s}: {e}")
            
    logger.info(f"Total PPI edges: {len(all_edges)}")
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_edges, f)
    logger.info(f"Saved merged edges to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
