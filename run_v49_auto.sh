#!/usr/bin/env bash
set -euo pipefail

echo "=== V49 Auto Pipeline ==="

echo "0) Activating venv is recommended:"
echo "   source venv/bin/activate"

echo "1) Train supervised ESM2 model (skip if already trained)..."
if [ ! -f "v49_esm2_supervised.pkl" ]; then
  venv/bin/python cafa_v49_esm2_supervised.py --mode train --model-out v49_esm2_supervised.pkl
else
  echo "   Found v49_esm2_supervised.pkl, skipping training."
fi

echo "2) Predict ESM2 supervised submission..."
venv/bin/python cafa_v49_esm2_supervised.py --mode predict --model-out v49_esm2_supervised.pkl --output submission_v49_esm2_supervised.tsv

echo "3) Generate Domain2GO submission..."
venv/bin/python cafa_v49_domain2go_submission.py --test-fasta Test/testsuperset.fasta --output submission_v49_domain2go.tsv

echo "4) Generate DIAMOND transfer submission (from existing test_diamond.tsv)..."
venv/bin/python cafa_v49_diamond_transfer.py --diamond-tsv test_diamond.tsv --output submission_v49_diamond_transfer.tsv

echo "5) Blend into final submission..."
if [ ! -f "v49_blend_config.json" ]; then
  echo "   No v49_blend_config.json found; copying example."
  cp v49_blend_config.example.json v49_blend_config.json
fi
venv/bin/python cafa_v49_blend_submissions.py --config v49_blend_config.json --output submission_v49_blend.tsv

echo "Done."
echo "Next: submit with:"
echo "  cp submission_v49_blend.tsv submission.tsv"
echo "  venv/bin/kaggle competitions submit -c cafa-6-protein-function-prediction -f submission.tsv -m \"V49 blend\""

