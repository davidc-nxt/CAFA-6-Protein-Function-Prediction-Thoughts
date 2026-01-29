#!/usr/bin/env python3
"""
Create a reproducible train/val split of training protein IDs.

Useful for:
- local validation aligned to Kaggle metric
- tuning ensemble weights (v49 tuner expects a val-ids file)
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-terms", default="Train/train_terms.tsv")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", default="split_train_ids.txt")
    ap.add_argument("--out-val", default="split_val_ids.txt")
    args = ap.parse_args()

    proteins = set()
    with open(args.train_terms, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                proteins.add(row[0])

    proteins = list(proteins)
    random.Random(args.seed).shuffle(proteins)
    split_idx = int(len(proteins) * (1.0 - args.val_ratio))
    train_ids = proteins[:split_idx]
    val_ids = proteins[split_idx:]

    with open(args.out_train, "w") as f:
        for pid in train_ids:
            f.write(pid + "\n")
    with open(args.out_val, "w") as f:
        for pid in val_ids:
            f.write(pid + "\n")

    print(f"Wrote {len(train_ids):,} train IDs to {args.out_train}")
    print(f"Wrote {len(val_ids):,} val IDs to {args.out_val}")


if __name__ == "__main__":
    main()

