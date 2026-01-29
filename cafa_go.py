from __future__ import annotations

"""
GO (Gene Ontology) helpers used across CAFA scripts.

This repo historically parsed `go-basic.obo` ad-hoc in multiple scripts.
Centralizing it avoids drift and makes it easier to apply safe hierarchy
consistency logic without exploding the prediction set (a key regression in V43).
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, Mapping, MutableMapping, Optional, Set, Tuple


ASPECT_BY_NAMESPACE = {
    "molecular_function": "F",
    "biological_process": "P",
    "cellular_component": "C",
}


@dataclass(frozen=True)
class GOData:
    term_to_aspect: Dict[str, str]
    parents: Dict[str, Set[str]]


def load_go_alt_id_map(obo_path: str) -> Dict[str, str]:
    """
    Parse `go-basic.obo` and return a mapping:
      alt_id (GO:xxxxxxx) -> primary_id (GO:yyyyyyy)

    Why:
    - Some sources (e.g., InterPro2GO/Domain2GO) can emit GO alt IDs.
    - Kaggle/CAFA evaluators may not resolve alt IDs automatically.
    - Canonicalizing improves interoperability and reduces silent score loss.
    """
    alt_to_primary: Dict[str, str] = {}
    cur_id: Optional[str] = None
    in_term = False

    with open(obo_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "[Term]":
                in_term = True
                cur_id = None
                continue
            if line == "[Typedef]":
                in_term = False
                cur_id = None
                continue
            if not in_term:
                continue
            if line.startswith("id: GO:"):
                cur_id = line.split("id: ", 1)[1].strip()
                continue
            if cur_id and line.startswith("alt_id: GO:"):
                alt = line.split("alt_id:", 1)[1].strip()
                alt_to_primary[alt] = cur_id

    return alt_to_primary


def load_go_basic_obo(
    obo_path: str,
    include_part_of: bool = True,
    include_alt_ids: bool = True,
) -> GOData:
    """
    Parse `go-basic.obo` and return:
    - term_to_aspect: GO:xxxx -> {'F','P','C'}
    - parents: GO:xxxx -> set(parent_terms) for is_a (+ optionally part_of)
    """
    term_to_aspect: Dict[str, str] = {}
    parents: DefaultDict[str, Set[str]] = defaultdict(set)

    cur_id: Optional[str] = None
    cur_ns: Optional[str] = None
    cur_alt: Set[str] = set()
    cur_obsolete = False
    in_term = False

    def commit() -> None:
        nonlocal cur_id, cur_ns, cur_alt, cur_obsolete
        if not cur_id or not cur_ns or cur_obsolete:
            return
        aspect = ASPECT_BY_NAMESPACE.get(cur_ns)
        if not aspect:
            return
        term_to_aspect[cur_id] = aspect
        if include_alt_ids:
            for alt in cur_alt:
                term_to_aspect.setdefault(alt, aspect)
                # Best-effort: share parent edges with alt ids too.
                if cur_id in parents:
                    parents[alt].update(parents[cur_id])

    with open(obo_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line == "[Term]":
                # Commit previous term, then reset.
                commit()
                in_term = True
                cur_id = None
                cur_ns = None
                cur_alt = set()
                cur_obsolete = False
                continue

            if line == "[Typedef]":
                # Commit previous term, then stop parsing term blocks until next [Term].
                commit()
                in_term = False
                cur_id = None
                cur_ns = None
                cur_alt = set()
                cur_obsolete = False
                continue

            if not in_term:
                continue

            if line.startswith("id: GO:"):
                cur_id = line.split("id: ", 1)[1].strip()
                continue

            if line.startswith("namespace:"):
                cur_ns = line.split("namespace:", 1)[1].strip()
                continue

            if include_alt_ids and line.startswith("alt_id: GO:"):
                cur_alt.add(line.split("alt_id:", 1)[1].strip())
                continue

            if line.startswith("is_obsolete:"):
                cur_obsolete = line.split("is_obsolete:", 1)[1].strip().lower() == "true"
                continue

            if cur_id and line.startswith("is_a: GO:"):
                parent = line.split("is_a:", 1)[1].split("!")[0].strip()
                parents[cur_id].add(parent)
                continue

            if include_part_of and cur_id and line.startswith("relationship: part_of GO:"):
                parent = line.split("relationship: part_of", 1)[1].split("!")[0].strip()
                parents[cur_id].add(parent)
                continue

    # Commit last term.
    commit()

    return GOData(term_to_aspect=dict(term_to_aspect), parents=dict(parents))


def enforce_hierarchy_clip_in_place(
    preds: MutableMapping[str, MutableMapping[str, float]],
    parents: Mapping[str, Set[str]],
    max_iters: int = 4,
) -> int:
    """
    "Safe" hierarchy consistency:
    - DO NOT add new ancestor terms.
    - Only ensure that if an ancestor term is already present in predictions,
      its score is at least the max of any descendant already present.

    Returns number of score increases performed.
    """
    fixed = 0
    for _ in range(max_iters):
        changed = 0
        for pid, term_scores in preds.items():
            # Iterate over a snapshot since we mutate.
            for term, score in list(term_scores.items()):
                for parent in parents.get(term, set()):
                    if parent in term_scores and term_scores[parent] < score:
                        term_scores[parent] = float(score)
                        fixed += 1
                        changed += 1
        if changed == 0:
            break
    return fixed


def get_all_ancestors(
    term: str,
    parents: Mapping[str, Set[str]],
    cache: Optional[MutableMapping[str, Set[str]]] = None,
) -> Set[str]:
    """Return the transitive closure of ancestors for `term` (excluding `term`)."""
    if cache is None:
        cache = {}
    if term in cache:
        return cache[term]

    out: Set[str] = set()
    for p in parents.get(term, set()):
        out.add(p)
        out.update(get_all_ancestors(p, parents, cache))
    cache[term] = out
    return out


def propagate_terms(
    terms: Iterable[str],
    parents: Mapping[str, Set[str]],
    *,
    term_to_aspect: Optional[Mapping[str, str]] = None,
    aspect: Optional[str] = None,
    ancestor_cache: Optional[MutableMapping[str, Set[str]]] = None,
) -> Set[str]:
    """
    True Path Rule closure for a set of GO terms:
    - include each term and all its ancestors
    - optionally restrict to a specific aspect (F/P/C) using `term_to_aspect`.
    """
    out: Set[str] = set()
    for t in terms:
        if aspect and term_to_aspect is not None and term_to_aspect.get(t) != aspect:
            continue
        out.add(t)
        for a in get_all_ancestors(t, parents, cache=ancestor_cache):
            if aspect and term_to_aspect is not None and term_to_aspect.get(a) != aspect:
                continue
            out.add(a)
    return out

