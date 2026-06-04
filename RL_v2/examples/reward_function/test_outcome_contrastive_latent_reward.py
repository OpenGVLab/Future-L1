"""Unit tests for outcome-contrastive latent reward R_ctr."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.reward_function import outcome_contrastive_latent_reward as ctr


def _tr(*rows):
    return np.asarray(rows, dtype=np.float32)


def test_traj_similarity_identity():
    tr = _tr([1.0, 0.0], [0.0, 1.0])
    assert ctr.traj_similarity(tr, tr) == 1.0


def test_reward_one_mixed():
    tr_now = _tr([1.0, 0.0], [0.0, 1.0])
    tr_pos = _tr([0.9, 0.1], [0.1, 0.9])
    tr_neg = _tr([-1.0, 0.0], [0.0, -1.0])
    tau = 0.5
    sim_pos = ctr.traj_similarity(tr_now, tr_pos)
    sim_neg = ctr.traj_similarity(tr_now, tr_neg)
    assert sim_pos > sim_neg
    got, case = ctr.reward_one(tr_now, [tr_pos], [tr_neg], temperature=tau)
    assert case == "mixed"
    assert 0.0 < got < 1.0


def test_reward_one_neg_only():
    tr_now = _tr([1.0, 0.0], [0.0, 1.0])
    tr_neg = _tr([-1.0, 0.0], [0.0, -1.0])
    tau = 0.5
    sim_neg = ctr.traj_similarity(tr_now, tr_neg)
    assert sim_neg < 0.5
    got, case = ctr.reward_one(tr_now, [], [tr_neg], temperature=tau)
    assert case == "neg_only"
    assert 0.0 < got < 1.0


def test_reward_one_pos_only():
    tr_now = _tr([1.0, 0.0], [0.0, 1.0])
    tr_pos = _tr([0.9, 0.1], [0.1, 0.9])
    tau = 0.5
    sim_pos = ctr.traj_similarity(tr_now, tr_pos)
    got, case = ctr.reward_one(tr_now, [tr_pos], [], temperature=tau)
    assert case == "pos_only"
    assert got == pytest.approx(sim_pos / tau)


def test_latent_rewards_batch_groups_by_uid():
    items = [
        {"uid": "g1", "latents": _tr([1.0, 0.0])},
        {"uid": "g1", "latents": _tr([0.9, 0.1])},
        {"uid": "g1", "latents": _tr([-1.0, 0.0])},
    ]
    accs = [1.0, 1.0, 0.0]
    reps = [False, False, False]

    def as_float(x):
        return np.asarray(x, dtype=np.float32)

    rewards, cases, _ = ctr.latent_rewards_batch(
        items, accs, reps, temperature=0.5, as_float_latents=as_float
    )
    assert len(rewards) == 3
    assert any(c != "none" for c in cases)


def test_is_correct_outcome_binary():
    assert ctr.is_correct_outcome(1.0)
    assert not ctr.is_correct_outcome(0.0)
    assert not ctr.is_correct_outcome(-1.0)
