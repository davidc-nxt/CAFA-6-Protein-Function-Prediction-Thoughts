# CAFA-6 Protein Function Prediction — Project Status Summary

Last updated: 2026-01-23

## Goal

Improve Kaggle CAFA-6 score (target discussed: **0.45+**), starting from an initial baseline around **0.233** (V48).

## Best achieved (Kaggle, per our notes)

- **Best known Kaggle public score so far**: **0.365** (user-reported historical best).
- **Best file found in this workspace that’s close to that peak**: `submission_v49_blend_v43_anchor.tsv` (noted previously as scoring **~0.352**).
- Current working default submission file in repo root: `submission.tsv` (you can overwrite this when choosing what to submit).

## Key fixes and infrastructure added

### GO term canonicalization (alt_id handling)

Why: some sources can emit GO alt IDs; evaluators may not resolve them, causing silent score loss.

- Added `load_go_alt_id_map()` to `cafa_go.py` to build `alt_id -> primary_id`.
- Updated blending/tuning scripts to canonicalize GO IDs using that mapping:
  - `cafa_v49_blend_submissions.py`
  - `cafa_v49_tune_blend.py`

### Submission validity + robustness fixes

- **Score formatting bug fixed**: some scripts wrote the literal string `".6f"` instead of numeric scores, yielding Kaggle **0.000** submissions. Fixed formatting to `f"{score:.6f}"` in:
  - `cafa_v50_weighted_blend.py`
  - `cafa_v50_ensemble_meta.py`
  - `cafa_v50_gnn_boost.py`
- **Score clamping**: blend outputs are clamped to \([0, 1]\) to avoid invalid values.
- **Coverage-safe submissions**: added `--force-coverage` option in `cafa_v49_blend_submissions.py` to ensure every test protein has at least one prediction row.

### Metric alignment (local vs Kaggle)

- Updated local tuning workflow to optimize **weighted Fmax** (closer to CAFA official scoring), not just unweighted mean Fmax.

## Models / methods implemented or integrated

### Homology transfer (DIAMOND)

- Implemented DIAMOND-based transfer and related tooling:
  - `cafa_v49_run_diamond.py`
  - `cafa_v49_diamond_transfer.py`
  - DIAMOND artifacts in repo: `diamond_train_full.dmnd`, `diamond_test_vs_train_full.tsv`, etc.

### Domain-based signal (Domain2GO)

- Domain2GO present under `Domain2GO/` and used in blends/tuning.

### Protein language model embeddings (ESM2)

- ESM2 embedding generation + hybrid submission pipeline exists:
  - `cafa_v48_esm2_embed_train.py`
  - `cafa_v48_esm2_embed_test.py`
  - `cafa_v48_esm2_submission.py`
  - Output example: `submission_v48_hybrid_esm2.tsv`

### Blending + tuning framework (submission-level ensembling)

- Blend builder: `cafa_v49_blend_submissions.py`
- Blend tuner: `cafa_v49_tune_blend.py`
- Multiple blend config JSONs are present (e.g. `v51_blend_config_v48_anchor.json`, `v53_blend_config_v48_taxon_strict.json`).

### DeepGOPlus-style CNN motif model (new signal)

Implemented a sequence-only motif CNN to add **motif-level** evidence beyond homology + embeddings:

- Script: `cafa_v54_cnn_motif.py`
- Outputs produced:
  - Trained bundle: `v54_cnn_motif.pkl`
  - Validation predictions: `cnn_val_preds.tsv` (**106 MB**, produced)
  - Test submission (currently being generated): `submission_v54_cnn_motif.tsv` (in progress at the time of writing)

## What we achieved in the latest run (V54 CNN motif)

- Successfully **trained** the CNN motif model on the existing train/val split.
- Successfully wrote a **tuner-ready validation TSV**: `cnn_val_preds.tsv`.
- Started generating the **test submission TSV**: `submission_v54_cnn_motif.tsv` (this run can take ~15–25 minutes on CPU).

## Current state (what’s running / next steps)

1. **Finish** generating `submission_v54_cnn_motif.tsv`.
2. **Tune/blend** CNN motif predictions with your strongest existing components (e.g. V48 anchor + DIAMOND + Domain2GO) using `cafa_v49_tune_blend.py`.
3. Produce a new `submission.tsv` with the best local tuned blend for Kaggle submission.

## Key files to look at

- **Best-known anchors**:
  - `submission_v48_hybrid_esm2.tsv`
  - `submission_v49_blend_v43_anchor.tsv`
- **CNN motif outputs**:
  - `v54_cnn_motif.pkl`
  - `cnn_val_preds.tsv`
  - `submission_v54_cnn_motif.tsv` (once finished)
- **Blending/tuning**:
  - `cafa_v49_blend_submissions.py`
  - `cafa_v49_tune_blend.py`

