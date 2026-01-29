#!/usr/bin/env python3
"""
Extract a subset of sequences from a FASTA file by protein ID.

Works with both:
- Train FASTA headers like: >sp|A0A0C5B5G6|MOTSC_HUMAN ...
- Test FASTA headers like:  >A0A0C5B5G6 9606
"""

from __future__ import annotations

import argparse


def parse_pid(header_line: str) -> str:
    h = header_line[1:].strip()
    token = h.split()[0]
    parts = token.split("|")
    if len(parts) >= 2 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return token


def load_ids(path: str) -> set[str]:
    ids: set[str] = set()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            tok = s.split()[0]
            if tok.lower() in {"entryid", "protein_id"}:
                continue
            ids.add(tok)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input FASTA")
    ap.add_argument("--ids", required=True, help="File of protein IDs (one per line)")
    ap.add_argument("--output", required=True, help="Output FASTA")
    args = ap.parse_args()

    wanted = load_ids(args.ids)
    kept = 0

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        write_block = False
        for line in fin:
            if line.startswith(">"):
                pid = parse_pid(line)
                write_block = pid in wanted
                if write_block:
                    fout.write(f">{pid}\n")
                    kept += 1
            else:
                if write_block:
                    fout.write(line)

    print(f"Wrote {kept:,} sequences to {args.output}")


if __name__ == "__main__":
    main()

