---
description: Create a new ensemble blend of existing predictions
---

# Create New Blend

## Prerequisites
- At least 2 submission files to blend
- Python environment activated

## Steps

1. Identify the base submissions to blend:
```bash
ls -la submission*.tsv | head -20
```

2. Create a new blend script based on existing pattern:
```python
#!/usr/bin/env python3
"""
V{next}: Description of blend
"""
import csv
from collections import defaultdict
from tqdm import tqdm

# Paths - adjust these
BASE1 = "submission_v29_neural_blend.tsv"  # Weight: 0.7
BASE2 = "submission_v40_structure_only.tsv"  # Weight: 0.3
OUTPUT = "submission_v{next}_blend.tsv"

# Load and blend...
```

// turbo
3. Run the blend script:
```bash
venv/bin/python cafa_vXX_blend.py
```

4. Check output statistics:
```bash
wc -l submission_vXX.tsv
head -5 submission_vXX.tsv
```

5. Submit to Kaggle (see `/submit-kaggle` workflow)

## Best Weights Discovered
- V7 (Diamond homology): 60-70%
- DeepGO-SE: 30-40%
- Domain2GO: 20-30%
- Structure (V40): 20-30%

## Common Blend Patterns
- **Neural Blend**: V7 + DeepGO-SE → 0.365
- **Domain Blend**: V7 + Domain2GO → diminishing returns
- **Structure Blend**: V29 + V40 Structure → pending
