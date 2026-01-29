---
description: CAFA-6 Protein Function Prediction - Project Skills & Lessons Learned
---

# CAFA-6 Protein Function Prediction Pipeline

## Project Overview
Kaggle competition to predict Gene Ontology (GO) terms for 224,309 test proteins.

**Best Score**: 0.366 (V58) | **Target**: 0.46+

---

## 🎓 Key Lessons Learned

### What Worked ✅
1. **Diamond homology baseline** (V7) achieved 0.360 - strong foundation
2. **DeepGO-SE neural blend** added +0.005 improvement
3. **PubMed text boost** (V58) added +0.001 improvement
4. **Local file parsing** bypasses API rate limits entirely
5. **Checkpointing** for long API fetches (24h runs) is essential
6. **Anchor-based blending** - keep best model as base, blend conservatively

### What Didn't Work ❌
1. **Domain2GO** (V37=0.361) - signal overlapped with Diamond homology
2. **Hierarchy Propagation** (V43=0.350, V45=0.334) - artificial enforcement hurts scores
3. **SBERT Text Reranking** (V47=0.322) - noisy embeddings degraded good predictions
4. **Score Calibration** (V44=0.261) - distorted probability distribution
5. **LTR with data leak** (V55=0.212) - validation proteins leaked into training
6. **Graph Label Prop alone** (V60=0.365) - signals redundant with V31 anchor

### Key Insight 💡
**Orthogonal signals are the key to breakthrough.**
- Homology + Text = redundant (both identify similar proteins)
- **PPI = orthogonal** (physical interaction ≠ sequence similarity)
- **Structure = orthogonal** (3D shape ≠ sequence)

---

## Score History

| Version | Score | Method | Notes |
|---------|-------|--------|-------|
| V7 | 0.360 | Diamond homology | Baseline |
| **V31** | **0.365** | Homology + DeepGO-SE | Anchor |
| V37 | 0.361 | + Domain2GO | Redundant |
| V43 | 0.350 | Hierarchy Propagation | Degraded |
| V47 | 0.322 | Text-First Rerank | Failed |
| V48 | 0.233 | ESM-2 Hybrid (Naive) | Failed |
| V55 | 0.212 | LTR (data leak) | Failed |
| **V58** | **0.366** | PubMed Boosted V31 | **Current Best** |
| V60 | 0.365 | Graph Label Propagation | Neutral |
| V61 | Pending | + PPI Network | Target: 0.46+ |

---

## Current Strategy: Phase 41 - PPI Network Integration

### Rationale
V60 matched V31 (0.365) because Homology + Text signals were **redundant**.
PPI (Protein-Protein Interaction) is **orthogonal**:
- Proteins that interact physically often share CC/BP terms
- No sequence similarity required

### Implementation
1. **Map IDs locally** - Parse `protein.aliases.v12.0.txt.gz` (bypasses API)
2. **Extract edges locally** - Parse `protein.physical.links.v12.0.txt.gz`
3. **Build graph** - Add PPI edges to existing homology+text graph
4. **Label propagation** - Propagate GO terms through PPI network
5. **Blend** - Combine with V31 anchor

---

## Workflows

### `/submit-kaggle`
Compress and submit a prediction file to Kaggle.

### `/create-blend`
Create a weighted ensemble blend of existing predictions.

### `/overnight-fetch`
Run long-running data fetching jobs (InterPro, UniProt text).

---

## Local Validation 🧪

Use `cafa_local_validate.py` before burning Kaggle submissions:
- **Split**: 80/20 train/val
- **Baseline**: Diamond gets F-max ~0.225 on this split
- **Rule**: Only submit models that beat local baseline significantly

---

## Environment
- Python 3.14 (via venv)
- Key packages: sentence-transformers, xgboost, scipy, requests
- External APIs: STRING DB, InterPro, PubMed

---

## Data Files Summary

| File | Size | Purpose |
|------|------|---------|
| `global_map.pkl` | 4.5MB | UniProt→STRING mapping |
| `graph_edges_homology.pkl` | 108MB | Homology edges |
| `graph_edges_text.pkl` | 14MB | Text similarity edges |
| `test_esm2_embeddings.pkl` | 1.1GB | ESM-2 embeddings |
| `pubmed_embeddings.pkl` | 134MB | PubMed embeddings |
