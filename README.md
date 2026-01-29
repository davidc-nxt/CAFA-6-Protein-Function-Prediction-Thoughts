# 🧬 CAFA-6 Protein Function Prediction

> **Kaggle Competition**: Predicting Gene Ontology (GO) terms for 224,309 proteins  
> **Best Score**: 0.366 F-max | **Leaderboard Target**: 0.46+

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Status](https://img.shields.io/badge/Status-Active%20Research-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

This repository documents my journey through the **Critical Assessment of Functional Annotation (CAFA-6)** Kaggle competition. The challenge: predict biological functions for hundreds of thousands of proteins using Gene Ontology terms.

What started as a simple sequence homology approach evolved into a sophisticated multi-signal ensemble spanning **60+ experimental phases**, incorporating:
- Sequence homology (Diamond BLASTP)
- Protein language models (ESM-2, ProtT5)
- Neural function predictors (DeepGO-SE)
- Protein-protein interaction networks (STRING DB)
- Literature mining (PubMed abstracts + SBERT)
- Structural alignment (FoldSeek + AlphaFold)
- Graph-based label propagation

---

## 📊 Score Evolution

| Version | Score | Method | Key Innovation |
|---------|-------|--------|----------------|
| V1 | 0.225 | Diamond baseline | Identity-tiered transfer |
| V7 | 0.360 | Homology + GOA | Power calibration + Noisy-OR |
| V31 | 0.365 | + DeepGO-SE | Neural-homology 60/40 blend |
| **V58** | **0.366** | + PubMed Mining | Literature-boosted ensemble |
| V60 | 0.365 | + Graph Label Prop | Multi-edge heterogeneous graph |
| V61 | 0.365 | + PPI Network | STRING physical interactions |

### The 0.365 Wall 🧱

A persistent plateau emerged around 0.365, where additional signals (text, PPI) provided **redundant information** that was already captured by sequence homology. Breaking this wall requires truly **orthogonal signals** like 3D structure.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAFA-6 Prediction Pipeline                    │
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
│  │   (V58)      │   │   (SBERT)    │   │  (V61)       │        │
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
│              │   Ensemble    │ ← Weighted blend with anchor     │
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

## 🔬 Key Methodologies

### 1. Sequence Homology (Diamond BLASTP)
The backbone of the pipeline. Proteins with similar sequences often share function.

```python
# Noisy-OR aggregation rewards consistency across multiple homologs
P(term) = 1 - ∏(1 - wᵢ)  
# where wᵢ = (pident/100)⁴ × qcov
```

**Key Finding**: Sharpened identity weighting (`pident⁴`) outperforms linear weighting by emphasizing high-confidence matches.

### 2. Protein Language Models (ESM-2)
Captures "semantic" similarity where sequence identity has decayed.

- **Model**: `facebook/esm2_t33_650M_UR50D` (1280-dim embeddings)
- **Classifier**: 3-layer MLP (1280 → 512 → 512 → 3000)
- **Discovery**: Raw ESM scores cap at ~0.807, creating a "Calibration Ceiling" when fused with 1.0 confidence GOA terms

### 3. Neural Function Prediction (DeepGO-SE)
Combines ESM-2 embeddings with neuro-symbolic Gene Ontology logic.

**Breakthrough**: Hierarchical constraint loss enforces biological axioms: `P(Child) ≤ P(Parent)`

### 4. Graph-Based Label Propagation
Propagates functional annotations through a heterogeneous protein graph.

```
Edge Types:
├── Homology edges (Diamond, ~5M edges)
├── Text similarity edges (SBERT cosine, ~14M edges)  
├── PPI edges (STRING physical, ~1.5M edges)
└── Structure edges (FoldSeek TM-score, WIP)
```

### 5. Literature Mining (PubMed + SBERT)
Extracts functional hints from scientific literature associated with proteins.

- 194,758 PubMed papers linked to proteins (99.3% coverage)
- PubMedBERT embeddings for semantic similarity
- +0.001 score improvement (V58)

---

## 💡 Key Lessons Learned

### ✅ What Worked

| Strategy | Impact | Notes |
|----------|--------|-------|
| **Power calibration** | +0.05 | `S = (S/S_max)^0.8 × 0.95` prevents over-confidence |
| **Noisy-OR aggregation** | +0.03 | Better than max() for multi-hit consolidation |
| **Hierarchical constraints** | +0.02 | Biological axioms prevent impossible predictions |
| **Local data processing** | ∞ | Bypassing APIs (STRING, UniProt) saved weeks |
| **Anchor-based blending** | Stability | Always blend with proven baseline, never replace |

### ❌ What Didn't Work

| Strategy | Result | Lesson |
|----------|--------|--------|
| **LTR (XGBoost ranker)** | 0.109 (-0.25) | Data leak in training poisoned the model |
| **Aggressive neural weights** | 0.028 | Balanced class weights destroy precision |
| **GNN global blend** | 0.305 | 200M low-confidence predictions act as noise |
| **Rank-based fusion** | 0.109 | Forcing rank=1.0 for weak signals creates false positives |
| **SOTA thresholds (0.1-0.2)** | 0.268 | "Density trap" — reduced predictions = destroyed recall |

### 🔑 Critical Insights

1. **Orthogonal signals are everything**
   - Homology, text, and PPI all correlate strongly (~80% overlap)
   - Only **structure** provides truly independent information
   - CAFA-5 winners used FoldSeek structural alignment for breakthrough

2. **The Calibration Ceiling**
   - When GOA priors = 1.0 and neural scores ≤ 0.81, conservative `max()` fusion is useless
   - Solution: Weighted-sum of union, or "boost-only" selective injection

3. **Precision > Recall in ensembles**
   - Adding weak signals can destroy strong baselines
   - Always gate new predictions with strict confidence thresholds

4. **Local processing >> API calls**
   - Downloaded STRING (15GB), UniProt (3GB), AlphaFold (300k structures)
   - Saved weeks of rate-limited API calls

---

## 📁 Project Structure

```
cafa-6-protein-function-prediction/
├── Train/                      # Training data (FASTA + GO terms)
├── Test/                       # Test proteins (FASTA)
├── structures/                 # 300k AlphaFold PDB structures
├── esm2_embeddings/           # ESM-2 protein embeddings
├── 
├── # Core Pipeline
├── cafa_v60_build_graph.py    # Graph construction
├── cafa_v60_label_prop.py     # Label propagation algorithm
├── cafa_v60_edges_*.py        # Edge extraction (homology/text/PPI)
├── cafa_v58_blend.py          # Ensemble blending
├── 
├── # Data Acquisition
├── cafa_v56_download_structures.py  # AlphaFold structure download
├── cafa_v61_map_local.py            # Local STRING ID mapping
├── cafa_v56_pubmed.py               # PubMed abstract fetching
├── 
├── # Models & Features
├── cafa_v48_esm2_embed_*.py   # ESM-2 embedding generation
├── cafa_v57_ltr_*.py          # Learning-to-rank experiments
└── cafa_local_validate.py     # Local validation framework
```

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/davidc-nxt/CAFA-6-Protein-Function-Prediction-Thoughts.git
cd CAFA-6-Protein-Function-Prediction-Thoughts

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scipy scikit-learn sentence-transformers xgboost tqdm requests

# Run the graph-based pipeline
python cafa_v60_build_graph.py      # Build protein graph
python cafa_v60_label_prop.py       # Propagate labels
python cafa_v60_blend.py           # Blend with anchor
```

---

## 🔮 Future Directions

### Structure-Aware Breakthrough (Current Priority)
- **FoldSeek alignment** (in progress): 219k × 81k structural comparisons
- Expected: 3D fold similarity for remote homologs undetectable by BLASTP
- Target: Break 0.40 barrier

### Advanced Methods to Explore
1. **GORetriever**: CAFA-5 1st place solution using PubMedBERT for GO reranking
2. **ProTrek**: Tri-modal (Sequence + Structure + Function) protein language model
3. **SaProt**: Structure-aware vocabulary with FoldSeek tokens

---

## 📚 References

- [CAFA-6 Kaggle Competition](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
- [Gene Ontology Consortium](http://geneontology.org/)
- [STRING Database](https://string-db.org/)
- [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)
- [ESM-2 Protein Language Model](https://github.com/facebookresearch/esm)
- [FoldSeek Structure Search](https://github.com/steineggerlab/foldseek)

---

## 📄 License

MIT License - Feel free to use and adapt for your own protein function prediction projects.

---

*Built with persistence through 60+ experimental phases, countless API timeouts, and one very patient FoldSeek run.* 🧬
