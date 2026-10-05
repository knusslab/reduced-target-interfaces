"""The predictive-gain ladder rule as one pure function.

The rule already existed three times: inline in the 2D Stage B producer, inline in the 3D
Stage B producer, and inline again in the first SELECTOR draft.  Two implementations of one
rule either share a mistake or drift apart, and an auditor that re-implements it cannot
falsify the producer.  This module states it once; `tests/test_bigdata_predictive_gain_selector_v1.py`
measures that it agrees with the 2D and 3D producers on synthetic input.

No new rule and no new threshold: the gain is clipped at zero, ordered stably by descending
gain, the tail is the suffix sum in that order, and the selected rung is the first ladder
entry whose tail falls at or below the declared threshold, with the maximum rung as the
declared fallback.
"""
from __future__ import annotations

import numpy as np


def canonical_predictive_gain_selector(gains, ladder, threshold, *, basis_rank=None,
                                       r_gram_resolved=None):
    """Order, tail, feasible set, selected rung and censor reasons from raw gains."""
    gain = np.clip(np.asarray(gains, dtype=np.float64), 0.0, None)
    if gain.ndim != 1 or gain.size < 1:
        raise ValueError("gains must be a non-empty 1-D array")
    rungs = [int(k) for k in ladder]
    if rungs != sorted(set(rungs)) or not rungs:
        raise ValueError("ladder must be increasing and unique")
    if max(rungs) > gain.size:
        raise ValueError(f"ladder maximum {max(rungs)} exceeds the gain width {gain.size}")
    threshold = float(threshold)

    order = np.argsort(-gain, kind="stable")
    tail = np.concatenate([np.cumsum(gain[order][::-1])[::-1], [0.0]])
    feasible = [k for k in rungs if float(tail[k]) <= threshold]
    selected = int(feasible[0]) if feasible else int(rungs[-1])
    position = rungs.index(selected)

    reasons = []
    if selected == rungs[-1]:
        reasons.append("ladder_cap")
    if basis_rank is not None and selected == int(basis_rank):
        reasons.append("basis_cap")
    if r_gram_resolved is not None and selected == int(r_gram_resolved):
        reasons.append("gram_resolution_cap")

    return {
        "censor_reasons": reasons,
        "feasible_rung_count": len(feasible),
        "feasible_rungs": feasible,
        "ladder": rungs,
        "order": order.astype(np.int64),
        "selected_k": selected,
        "selected_slack": threshold - float(tail[selected]),
        "tail_at_previous_rung": (None if position == 0
                                  else float(tail[rungs[position - 1]])),
        "tail_at_selected": float(tail[selected]),
        "tail_by_rung": {str(k): float(tail[k]) for k in rungs},
        "threshold": threshold,
        "used_max_rung_fallback": not feasible,
    }
