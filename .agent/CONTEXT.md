# CAFA-6 Project Context

## Competition Overview
**Goal**: Predict Gene Ontology (GO) terms for 224,309 test proteins.
**Metric**: F-max score (harmonic mean of precision and recall).
**Best Score**: 0.366 (V58) | **Target**: 0.46+

---

## Pipeline Architecture (V60+)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAFA-6 Prediction Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │   Diamond    │   │   DeepGO-SE  │   │   ESM-2      │        │
│  │  Homology    │   │   Neural Net │   │  Supervised  │        │
│  │   (V7)       │   │              │   │   (V48+)     │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                   │                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │   PubMed     │   │   Text       │   │   PPI        │        │
│  │  Embeddings  │   │  Similarity  │   │  Network     │        │
│  │   (V58)      │   │   (SBERT)    │   │  (Phase 41)  │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                   │                │
│         └────────────┬─────┴───────────────────┘                │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │  Graph-Based  │ ← PPI + Homology + Text edges    │
│              │  Label Prop   │                                  │
│              └───────┬───────┘                                  │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │   Ensemble    │ ← Weighted blend with V31 anchor │
│              │   Blending    │                                  │
│              └───────┬───────┘                                  │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │  submission   │                                  │
│              │    .tsv.gz    │                                  │
│              └───────────────┘                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current Phase: Phase 41 - PPI Network Integration

### Status
- [x] Map IDs (Train+Test) to STRING (Local: 161k proteins, 72% coverage)
- [ ] **BLOCKED**: Fetch Interactions (Waiting for `protein.physical.links.v12.0.txt.gz`)
- [ ] Update Graph with PPI edges
- [ ] Re-run Label Propagation
- [ ] Blend and Submit V61

### Data Files
| File | Size | Description |
|------|------|-------------|
| `global_map.pkl` | 4.5MB | UniProt → STRING ID mapping |
| `graph_edges_homology.pkl` | 108MB | Diamond homology edges |
| `graph_edges_text.pkl` | 14MB | Text similarity edges |
| `protein.aliases.v12.0.txt.gz` | 3.2GB | STRING aliases (downloaded) |

---

## Score History

| Version | Score | Method | Notes |
|---------|-------|--------|-------|
| V7 | 0.360 | Diamond homology | Baseline |
| **V31** | **0.365** | Homology + DeepGO-SE | Anchor |
| V55 | 0.212 | LTR (XGBoost) | Data leak issue |
| **V58** | **0.366** | PubMed Boosted V31 | Current Best |
| V60 | 0.365 | Graph Label Propagation | Neutral (redundant signals) |
| V61 | Pending | PPI + Graph | Target: 0.46+ |

---

## Key Scripts

### Data Acquisition
- `cafa_v61_map_local.py` - Parse STRING aliases locally (bypasses API)
- `cafa_v61_global_ppi.py` - Fetch PPI interactions (API or local)

### Graph Pipeline
- `cafa_v60_edges_homology.py` - Extract homology edges from Diamond
- `cafa_v60_edges_text.py` - Extract text similarity edges from PubMed
- `cafa_v60_build_graph.py` - Build sparse adjacency matrix
- `cafa_v60_label_prop.py` - Label propagation algorithm

### Blending & Submission
- `cafa_v58_blend.py` - Blend graph predictions with V31 anchor
- `/submit-kaggle` workflow - Compress and submit to Kaggle

---

## Key Learnings

### What Worked ✅
1. **Diamond homology** - Strong baseline (0.360)
2. **DeepGO-SE blend** - +0.005 improvement
3. **PubMed text boost** - +0.001 improvement (V58)
4. **Local parsing** - Bypass STRING API rate limits

### What Didn't Work ❌
1. **LTR (V55)** - Data leak in training caused regression (0.212)
2. **Graph Label Prop (V60)** - Neutral (signals were redundant with V31)
3. **API-based PPI fetch** - Rate limits made it impractical at scale

### Key Insight 💡
**Orthogonal signals are required for breakthrough.**
- Homology + Text = redundant (both agree on same proteins)
- PPI = orthogonal (physical interactions ≠ sequence similarity)
- Structure = orthogonal (3D shape ≠ sequence)

---

## Environment
- Python 3.14 (via venv)
- Key packages: sentence-transformers, xgboost, scipy, requests
- External: STRING DB, InterPro, PubMed
