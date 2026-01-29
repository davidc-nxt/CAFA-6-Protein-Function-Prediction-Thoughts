---
description: Submit a prediction file to Kaggle
---

# Submit to Kaggle

## Prerequisites
- Kaggle API token set as environment variable
- Submission file exists (`.tsv` format)

## Steps

1. Copy your submission to `submission.tsv`:
```bash
cp submission_vXX.tsv submission.tsv
```

// turbo
2. Export Kaggle API token:
```bash
export KAGGLE_API_TOKEN=<your_token>
```

// turbo
3. Submit to competition:
```bash
venv/bin/kaggle competitions submit -c cafa-6-protein-function-prediction -f submission.tsv -m "Your description here"
```

// turbo
4. Check submission status:
```bash
venv/bin/kaggle competitions submissions -c cafa-6-protein-function-prediction | head -10
```

## Notes
- Wait 5-10 minutes for scoring to complete
- Daily limit is approximately 5 submissions
- Large files (>1GB) may take 30+ seconds to upload
