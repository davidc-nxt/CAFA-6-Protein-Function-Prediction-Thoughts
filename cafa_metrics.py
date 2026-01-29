import numpy as np
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

PredDict = Dict[str, Dict[str, float]]  # {protein_id: {go_term: score}}
TargetDict = Dict[str, Set[str]]        # {protein_id: set(go_terms)}


def calculate_fmax(
    preds: PredDict,
    targets: TargetDict,
    ontology=None,
    thresholds: Optional[Iterable[float]] = None,
) -> float:
    """
    Protein-centric F-max (CAFA-style):
    - Precision is averaged over proteins with ≥1 predicted term at threshold.
    - Recall is averaged over proteins with ≥1 true term (in the evaluated ontology/aspect).

    Notes:
    - `ontology` is kept for backward compatibility; it is not used here.
    """
    if thresholds is None:
        # CAFA implementations commonly sweep 0.00..1.00 in 0.01 steps.
        thresholds = np.linspace(0.0, 1.0, 101)
    
    fmax = 0.0
    
    # Pre-calculate term information content if needed, but for simple F-max 
    # we just need precision/recall
    
    # preds: dict {protein_id: {term_id: score}}
    # targets: dict {protein_id: set(term_ids)}
    
    # Filter targets to only those in ontology and with annotations
    target_proteins = set(targets.keys())
    pred_proteins = set(preds.keys())
    all_proteins = target_proteins | pred_proteins
    
    for t in thresholds:
        precision_sum = 0.0
        recall_sum = 0.0
        n_preds = 0
        n_targets = 0
        
        for pid in all_proteins:
            # Ground truth
            truth = targets.get(pid, set())
            if not truth:
                continue # Skip proteins with no annotation in this ontology aspect
                
            n_targets += 1
            
            # Predictions at this threshold
            p_terms = {term for term, score in preds.get(pid, {}).items() if score >= t}
            
            # Intersection
            intersect = len(p_terms & truth)
            
            # Precision
            if p_terms:
                precision_sum += intersect / len(p_terms)
                n_preds += 1
            
            # Recall
            recall_sum += intersect / len(truth)
            
        avg_precision = precision_sum / n_preds if n_preds > 0 else 0.0
        avg_recall = recall_sum / n_targets if n_targets > 0 else 0.0
        
        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if f1 > fmax:
                fmax = f1
                
    return fmax


def calculate_mean_fmax_over_aspects(
    preds: PredDict,
    targets_by_aspect: Mapping[str, TargetDict],
    term_to_aspect: Optional[Mapping[str, str]] = None,
    aspects: Tuple[str, ...] = ("F", "P", "C"),
    thresholds: Optional[Iterable[float]] = None,
) -> float:
    """
    Kaggle CAFA-6 metric (per competition overview):
    - Compute F-max separately for each subontology (MF/BP/CC == F/P/C),
    - return the arithmetic mean of the three F-max values.
    """
    fmaxes = []
    for a in aspects:
        aspect_targets = targets_by_aspect.get(a, {})
        if term_to_aspect is None:
            # WARNING: Without a term→aspect map we cannot correctly filter predictions,
            # so fall back to treating all predicted terms as belonging to this aspect.
            aspect_preds = preds
        else:
            aspect_preds: PredDict = {}
            for pid, p_terms in preds.items():
                filtered = {t: s for t, s in p_terms.items() if term_to_aspect.get(t) == a}
                if filtered:
                    aspect_preds[pid] = filtered

        fmaxes.append(calculate_fmax(aspect_preds, aspect_targets, ontology=a, thresholds=thresholds))
    return float(np.mean(fmaxes)) if fmaxes else 0.0


def compute_information_content_by_aspect(
    targets_by_aspect: Mapping[str, TargetDict],
    *,
    parents: Optional[Mapping[str, Set[str]]] = None,
    term_to_aspect: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute information content (IC) per GO term per aspect:
      IC(term) = -log( P(term) )

    Where P(term) is estimated by protein frequency in the provided target set.
    If `parents` is provided, we first propagate annotations to ancestors (CAFA "full").
    """
    from collections import Counter

    # Local import to avoid circular import at module load time.
    if parents is not None:
        from cafa_go import propagate_terms
    else:
        propagate_terms = None  # type: ignore

    ic_by_aspect: Dict[str, Dict[str, float]] = {}
    for a, targets in targets_by_aspect.items():
        # Only consider proteins with at least one annotation in this aspect.
        pids = [pid for pid, ts in targets.items() if ts]
        n = len(pids)
        if n == 0:
            ic_by_aspect[a] = {}
            continue

        counts: Counter = Counter()
        ancestor_cache = {}
        for pid in pids:
            ts = targets[pid]
            if parents is not None and propagate_terms is not None:
                ts = propagate_terms(ts, parents, term_to_aspect=term_to_aspect, aspect=a, ancestor_cache=ancestor_cache)
            for t in ts:
                counts[t] += 1

        ic = {}
        for t, c in counts.items():
            p = max(1e-12, float(c) / float(n))
            ic[t] = float(-np.log(p))
        ic_by_aspect[a] = ic

    return ic_by_aspect


def calculate_weighted_fmax(
    preds: PredDict,
    targets: TargetDict,
    *,
    ic: Mapping[str, float],
    parents: Optional[Mapping[str, Set[str]]] = None,
    term_to_aspect: Optional[Mapping[str, str]] = None,
    aspect: Optional[str] = None,
    thresholds: Optional[Iterable[float]] = None,
) -> float:
    """
    CAFA-style weighted Fmax:
    - For each threshold, compute weighted precision/recall per protein using IC weights:
        wpr = sum_{t in P∩T} IC(t) / sum_{t in P} IC(t)
        wrc = sum_{t in P∩T} IC(t) / sum_{t in T} IC(t)
      then average wpr over proteins with predictions, and wrc over all proteins with truth.
    - If `parents` is provided, propagate both predictions and truth to ancestors (full evaluation).
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    if parents is not None:
        from cafa_go import propagate_terms
    else:
        propagate_terms = None  # type: ignore

    fmax = 0.0

    all_pids = set(targets.keys()) | set(preds.keys())
    ancestor_cache = {}

    # Pre-propagate truth (independent of threshold)
    truth_by_pid: Dict[str, Set[str]] = {}
    for pid in all_pids:
        ts = targets.get(pid, set())
        if not ts:
            continue
        if parents is not None and propagate_terms is not None:
            ts = propagate_terms(ts, parents, term_to_aspect=term_to_aspect, aspect=aspect, ancestor_cache=ancestor_cache)
        truth_by_pid[pid] = ts

    for thr in thresholds:
        wpr_sum = 0.0
        wrc_sum = 0.0
        n_pred = 0
        n_true = 0

        for pid, truth in truth_by_pid.items():
            if not truth:
                continue
            n_true += 1

            # Predicted terms at threshold
            raw_pred = preds.get(pid, {})
            if raw_pred:
                pset = {t for t, s in raw_pred.items() if s >= thr and (aspect is None or term_to_aspect is None or term_to_aspect.get(t) == aspect)}
            else:
                pset = set()

            if parents is not None and propagate_terms is not None and pset:
                pset = propagate_terms(pset, parents, term_to_aspect=term_to_aspect, aspect=aspect, ancestor_cache=ancestor_cache)

            if pset:
                inter = pset & truth
                num = sum(ic.get(t, 0.0) for t in inter)
                den = sum(ic.get(t, 0.0) for t in pset) + 1e-12
                wpr_sum += num / den
                n_pred += 1

            # recall (include proteins even if pset empty)
            inter = pset & truth if pset else set()
            num = sum(ic.get(t, 0.0) for t in inter)
            den = sum(ic.get(t, 0.0) for t in truth) + 1e-12
            wrc_sum += num / den

        wpr = wpr_sum / n_pred if n_pred else 0.0
        wrc = wrc_sum / n_true if n_true else 0.0
        if wpr + wrc > 0:
            f1 = 2 * wpr * wrc / (wpr + wrc)
            if f1 > fmax:
                fmax = f1

    return float(fmax)


def calculate_mean_weighted_fmax_over_aspects(
    preds: PredDict,
    targets_by_aspect: Mapping[str, TargetDict],
    *,
    ic_by_aspect: Mapping[str, Mapping[str, float]],
    parents: Optional[Mapping[str, Set[str]]] = None,
    term_to_aspect: Optional[Mapping[str, str]] = None,
    aspects: Tuple[str, ...] = ("F", "P", "C"),
    thresholds: Optional[Iterable[float]] = None,
) -> float:
    fmaxes = []
    for a in aspects:
        ic = ic_by_aspect.get(a, {})
        fmaxes.append(
            calculate_weighted_fmax(
                preds,
                targets_by_aspect.get(a, {}),
                ic=ic,
                parents=parents,
                term_to_aspect=term_to_aspect,
                aspect=a,
                thresholds=thresholds,
            )
        )
    return float(np.mean(fmaxes)) if fmaxes else 0.0

# Validation script content will go here
