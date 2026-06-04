"""Outcome-contrastive latent reward R_ctr (LA-DAPO, Eq. ctr)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_CTR_CASE_TO_ID = {"none": 0.0, "mixed": 1.0, "neg_only": 2.0, "pos_only": 3.0}


def is_correct_outcome(accuracy: float) -> bool:
    """True when accuracy is 1.0 (judge/rule); 0.0 and -1.0 are wrong."""
    return float(accuracy) >= 1.0


def normalize_trajectory(latents: Any, as_float_latents) -> Optional[np.ndarray]:
    arr = as_float_latents(latents)
    if arr is None:
        return None
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def traj_similarity(tr_a: np.ndarray, tr_b: np.ndarray) -> float:
    steps = min(tr_a.shape[0], tr_b.shape[0])
    if steps <= 0:
        return 0.0
    sims: List[float] = []
    for t in range(steps):
        cos_t = float(np.dot(tr_a[t], tr_b[t]))
        cos_t = max(-1.0, min(1.0, cos_t))
        sims.append((1.0 + cos_t) * 0.5)
    return float(np.mean(sims))


def reward_one(
    tr_now: np.ndarray,
    pos_trs: List[np.ndarray],
    neg_trs: List[np.ndarray],
    *,
    temperature: float,
) -> Tuple[float, str]:
    tau = max(float(temperature), 1e-6)
    if pos_trs and neg_trs:
        sim_pos = max(traj_similarity(tr_now, p) for p in pos_trs)
        num = math.exp(sim_pos / tau)
        denom = num + sum(math.exp(traj_similarity(tr_now, n) / tau) for n in neg_trs)
        return (num / denom if denom > 0.0 else 0.0), "mixed"
    if neg_trs and not pos_trs:
        denom = 1.0 + sum(math.exp(traj_similarity(tr_now, n) / tau) for n in neg_trs)
        return (1.0 / denom if denom > 0.0 else 0.0), "neg_only"
    if pos_trs and not neg_trs:
        sim_pos = max(traj_similarity(tr_now, p) for p in pos_trs)
        return sim_pos / tau, "pos_only"
    return 0.0, "none"


def group_key(item: Dict[str, Any]) -> str:
    uid = item.get("uid")
    if uid is not None and str(uid).strip():
        return f"uid:{uid}"
    problem = str(item.get("problem", ""))
    ground_truth = str(item.get("ground_truth", ""))
    return f"pg:{problem}\x00{ground_truth}"


def latent_rewards_batch(
    items: List[Dict[str, Any]],
    final_accuracies: List[float],
    repetition_flags: List[bool],
    *,
    temperature: float,
    as_float_latents,
) -> Tuple[List[float], List[str], List[float]]:
    n = len(items)
    rewards: List[float] = [0.0] * n
    cases: List[str] = ["none"] * n
    trajectories: List[Optional[np.ndarray]] = [
        normalize_trajectory(items[i].get("latents"), as_float_latents) for i in range(n)
    ]

    groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(n):
        groups[group_key(items[i])].append(i)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        correct_by_idx: List[Tuple[int, np.ndarray]] = []
        wrong_pool: List[np.ndarray] = []
        for idx in indices:
            tr = trajectories[idx]
            if tr is None:
                continue
            acc = final_accuracies[idx]
            if repetition_flags[idx]:
                acc = -1.0
            if is_correct_outcome(acc):
                correct_by_idx.append((idx, tr))
            else:
                wrong_pool.append(tr)

        for idx in indices:
            tr_now = trajectories[idx]
            if tr_now is None:
                continue
            pos_for_idx = [t for j, t in correct_by_idx if j != idx]
            if not pos_for_idx and correct_by_idx:
                pos_for_idx = [t for _, t in correct_by_idx]
            r_latent, case = reward_one(
                tr_now, pos_for_idx, wrong_pool, temperature=temperature
            )
            rewards[idx] = r_latent
            cases[idx] = case

    case_ids = [_CTR_CASE_TO_ID.get(c, 0.0) for c in cases]
    return rewards, cases, case_ids
