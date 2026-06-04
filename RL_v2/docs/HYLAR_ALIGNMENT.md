# HyLar Baseline Alignment — Audit Record

> **Scope.** This document is the formal audit record for aligning the RL_v2
> (FutureL1 × EasyR1) GRPO / DAPO / DePO training pipelines with HyLar/RL.
> It is intended to be citable from the paper appendix and from any reviewer
> rebuttal. The mechanical content here is what proves "RL_v2 GRPO/DAPO/DePO
> baselines differ from HyLar's only in the base model".

## 1. TL;DR

After alignment, the only mechanical divergence between RL_v2's
GRPO / DAPO / DePO baselines and HyLar/RL's GRPO / DAPO / DePO is:

* **Base model.** RL_v2 wraps `FutureL1_Qwen3VL`; HyLar/RL wraps
  `FutureL1_Qwen2_5_VL`. Both expose the same `<|latent_start|>`,
  `<|latent|>`, `<|latent_end|>` special tokens and the same projection head
  conventions.
* **Chat-format prompt.** RL_v2 uses `examples/format_prompt/future_l1_format.jinja`
  matched to Qwen3-VL's chat template; HyLar uses the corresponding Qwen2.5-VL
  template. The semantic format (image/video placeholders + reasoning span +
  `<answer>` tag) is identical.

Every other RL ingredient — rollout state machine, latent recording, vMF
log-prob substitution at latent positions, decoupled PPO surrogate, vMF KL
constraint, advantage formulae (GRPO outcome / DAPO outcome / DePO), KL
controller, clip ratios, and data filtering — is byte-identical in semantics
(modulo refactoring) to HyLar/RL.

## 2. Problem statement (pre-alignment)

### 2.1 The visible symptom

The user's training logs printed, repeatedly, lines such as:

```
[FutureL1 RL patch] code_root=/mnt/shared-storage-user/jiangtianxiang/Zone/VideoL1/RL_v2 backbone=qwen3_vl
[verl.trainer.main] FutureL1 patch skipped: No module named 'src'
[verl.workers.fsdp_workers] FutureL1 patch skipped: No module named 'src'
```

### 2.2 Two compounding bugs

**Bug A — broken `sys.path` for the transformers monkey patches.**
`future_l1_rl_patch.patch_transformers()` imports `src.train.monkey_patch_forward`,
which requires `VideoL1/` (the parent of `RL_v2/`) on `sys.path`. The
launcher shell scripts computed `FUTURE_L1_CODE_ROOT` from
`SCRIPT_DIR/..` which resolves to `VideoL1/RL_v2/` from inside `rjob/`,
not `VideoL1/`. Consequence: `patch_transformers()` raised `ModuleNotFoundError`,
which propagated to `patch_vllm()` never running.

**Bug B — silent fallback to stock vLLM.**
Every importer of `future_l1_rl_patch` (`verl/trainer/main.py`,
`verl/workers/fsdp_workers.py`, `verl/workers/actor/dp_actor.py`) wrapped
the import in `try: ... except Exception as _e: print("FutureL1 patch skipped:")`.
When Bug A fired, the workers continued with **stock vLLM**: no latent
state machine, no `<|latent|>` forcing, no projected-hidden-state feedback,
no `LatentRecorder`.

**Bug C — RL_v2's design split `future_l1_text` (GRPO/DAPO) vs `future_l1_depo`
(DePO). The `future_l1_text` path was intentionally lighter: it kept the
latent-aware rollout but disabled latent recording and the actor-side vMF
log-prob substitution. HyLar does NOT have such a split — HyLar's single
`hylar` strategy always records latents and always replaces log-probs at
latent positions. Therefore GRPO and DAPO baselines run on RL_v2 under
`future_l1_text` were structurally NOT equivalent to HyLar's GRPO/DAPO even
when the patch loaded correctly.

### 2.3 Per-mode impact (pre-fix)

|                          | rollout state machine | latent vector recorded | actor vMF replacement | matches HyLar baseline |
|--------------------------|:--:|:--:|:--:|:--:|
| RL_v2 GRPO/DAPO, **patch loaded**, `future_l1_text`   | ✓ | ✗ | ✗ | **No** (Bug C) |
| RL_v2 GRPO/DAPO, **patch failed**, `future_l1_text`   | ✗ | ✗ | ✗ | **No** (Bug A+B+C) |
| RL_v2 DePO, **patch loaded**, `future_l1_depo`        | ✓ | ✓ | ✓ | **Yes** |
| RL_v2 DePO, **patch failed**, `future_l1_depo`        | ✗ | ✗ | ✗ (no `latents` key) | **No** (Bug A+B) |

The headline GRPO/DAPO baselines were therefore always misaligned with
HyLar (Bug C), and could additionally suffer from a silent rollout
degradation (Bug A+B) depending on whether the launcher path was correct.

## 3. Equivalence definition

We declare an RL_v2 mode **HyLar-equivalent** iff, given the same prompt,
tokenizer, data loader, optimiser settings, RNG seed, and a FutureL1
checkpoint whose state is identical to HyLar's (modulo the base model
substitution), the trajectory of `(loss, grad-norm, advantage, log-prob,
KL, latent_mu, z_rollout)` produced by the actor's `_forward_micro_batch`
and `update_policy` is identical to that produced by the corresponding
HyLar mode under the matching configuration.

Reduced to mechanical claims:

1. **Rollout semantics** — `<|latent|>` is forced for every step inside
   a latent span; `<|latent_end|>` is forced when the budget is hit;
   the previous step's projected hidden state replaces the input embedding
   at the next step. (Implemented by `future_l1_gpu_model_runner.py`.)
2. **Latent recording** — for every latent step in every rollout, the
   pre-projection hidden state `z` is recorded and shipped to the driver
   under `non_tensor_batch["latents"]`. (Implemented by `LatentRecorder`.)
3. **Actor vMF substitution** — at every position whose token id is
   `<|latent|>` (excluding the open/close markers), the policy log-prob
   is replaced by `κ · cos(μ, z)`, where `μ` is the actor's current
   hidden state and `z` is the rolled-out latent vector at the same step.
   (Implemented by `dp_actor._forward_micro_batch`.)
4. **Mode-specific PPO surrogate** — GRPO and DAPO compute a single
   unified policy loss over `response_mask`. DePO splits the surrogate
   into token / latent halves with different clip ratios and optionally
   adds the closed-form vMF KL term `κ · (1 − cos(μ_actor_new, z_rollout))`.
   (Implemented by `dp_actor.update_policy` gated on
   `enable_decoupled_hybrid_ppo` and `enable_latent_vmf_kl`.)
5. **Advantage estimator** — GRPO uses outcome-normalised GRPO advantage;
   DAPO uses the same outcome advantage paired with online accuracy
   filtering and asymmetric clip. (Implemented by
   `verl/trainer/core_algos.py`.)

## 4. Fix summary (this audit)

| Layer | Files | Change |
|---|---|---|
| Launcher (GRPO/DAPO modes) | `train.sh`, `rjob/grpo.sh`, `rjob/cava.sh`, `rjob/grpo_20k_llava5k.sh`, `rjob/grpo_20k_llava5k_seed6k.sh`, `rjob/grpo_train_merge_eval.sh` | `sampling_strategy=future_l1_text` → `sampling_strategy=future_l1_depo` for both `grpo` and `dapo` cases. Inline comment in each case documenting HyLar equivalence. |
| Config default | `examples/config_future_l1.yaml` | `sampling_strategy: future_l1_depo` (was `future_l1_text`). Comment explicitly marks `default` / `future_l1_text` as ablation-only and NOT HyLar-equivalent. |
| Patch loader hardening | `verl/trainer/main.py`, `verl/workers/fsdp_workers.py`, `verl/workers/actor/dp_actor.py` | When `FUTURE_L1_RL_PATCH_REQUIRED=1` (set by every launcher), the `except Exception` arm of the `import future_l1_rl_patch` block re-raises instead of swallowing the failure. Closes Bug B (silent fallback). |
| vLLM runner hardening | `future_l1_rl_patch.py` | Updated `patch_vllm()` docstring + error message. When `FUTURE_L1_REQUIRE_VLLM_RUNNER=1` (set by every launcher), a failure to load `future_l1_gpu_model_runner` raises instead of degrading. |
| Launcher default flags | all 6 .sh listed above | `export FUTURE_L1_RL_PATCH_REQUIRED=${FUTURE_L1_RL_PATCH_REQUIRED:-1}` and `export FUTURE_L1_REQUIRE_VLLM_RUNNER=${FUTURE_L1_REQUIRE_VLLM_RUNNER:-1}`. |
| Documentation | `README.md`, `verl/workers/rollout/config.py` | Mode summary table rewritten with `future_l1_depo` everywhere; docstring of `RolloutConfig.sampling_strategy` annotates ablation-only modes. |

Bug A (the original `FUTURE_L1_CODE_ROOT` mis-computation in `rjob/*.sh`)
was fixed in a prior commit by switching to absolute paths
(`VIDEO_L1_ROOT=/mnt/shared-storage-user/jiangtianxiang/Zone/VideoL1`,
`RL_V2_ROOT=${VIDEO_L1_ROOT}/RL_v2`). This audit assumes that fix is in
place; see `train.sh` / `rjob/*.sh` headers.

## 5. Equivalence proof — file by file

### 5.1 Rollout state machine

* RL_v2: `future_l1_rl/vllm_runner/future_l1_gpu_model_runner.py`
* HyLar: `HyLar/RL/hylar_models/vllm/hylar_gpu_model_runner.py`

Both files override `GPUModelRunner.execute_model` (and the inputs-embeds
preparation block) with the same logic:

1. When `<|latent_start|>` is sampled at step *t*, mark
   `state[req_id].just_saw_start = True`.
2. At step *t + 1*, set `state[req_id].active = True`, capture the
   last-token hidden state `μ`, store it as the *pending embedding*,
   and force-emit `<|latent|>` as the sampled token id (so the seqlen
   accounting matches).
3. On every subsequent decode step *t + k* (*k ≥ 2*) while
   `active=True`: feed `pending` as the next-step input embedding,
   capture the new hidden state as the new `pending`, force-emit
   `<|latent|>`.
4. When the latent budget is hit OR when the policy samples
   `<|latent_end|>`, clear state, force-emit `<|latent_end|>`,
   and return to normal decoding.

The implementation is byte-equivalent up to identifier renames
(`hylar_id` ↔ `latent_id` ↔ `FUTURE_L1_LATENT_ID`, `LATENT_SIZE` ↔
`FUTURE_L1_LATENT_SIZE`, `latent_state` dict layout).

### 5.2 Latent recording

* RL_v2: `future_l1_rl/vllm_runner/latent_recorder.py`
* HyLar: `HyLar/RL/hylar_models/vllm/latent_recorder.py` and `latent_hook.py`

Both spin up a per-process TCP listener that the GPU runner connects to,
receive `(req_id, latents[T, D])` binary frames per step, and assemble
them into a `(batch, rollout_n)` object array keyed by request id.
RL_v2's `to_object_array_auto(bsz, rollout_n, min_req_id)` is the
direct equivalent of HyLar's `to_object_array_auto`.

`vllm_rollout_spmd.generate_sequences` in both repos invokes the
recorder as a context manager around `inference_engine.generate(...)` and
writes the result into `non_tensor_batch["latents"]`. The only diff is
the gate:

* HyLar: `if self.config.sampling_strategy in ["hylar"]` (always on).
* RL_v2: `if sampling_strategy == "future_l1_depo"` (now always on for
  HyLar-aligned baselines after the alignment patch).

### 5.3 Actor vMF substitution

* RL_v2: `verl/workers/actor/dp_actor.py::_forward_micro_batch`
* HyLar: `HyLar/RL/verl/workers/actor/dp_actor.py::_forward_micro_batch`

Both:

1. Find latent segments in `input_ids` between `LATENT_START_ID` and
   `LATENT_END_ID`, returning `(flat_indices, per_batch_segments)`.
2. For each sample, check `latents[i].shape[0]` matches the count of
   indices inside its latent segments; on mismatch, drop the sample's
   latent contribution but keep its token PPO term.
3. Forward the actor with `output_hidden_states=True`.
4. At the latent positions, compute `μ = h[latent_poss]` and replace
   `log_prob[latent_poss]` with `κ · (μ · z).sum(-1)`. (Both
   implementations skip the L2 normalisation step that the docstring
   mentions, treating the model's projection head as producing
   approximately-unit-norm vectors; this is consistent across the two
   stacks.)

The gating differs only in the strategy name (`hylar` ↔ `future_l1_depo`).

### 5.4 PPO surrogate

* RL_v2: `verl/workers/actor/dp_actor.py::update_policy`
* HyLar: `HyLar/RL/verl/workers/actor/dp_actor.py::update_policy`

GRPO / DAPO branch: `if not is_depo: pg_loss = compute_policy_loss(old, new, adv, response_mask, ...)`.

DePO branch: builds `latent_mask` from `input_ids` between the
`LATENT_START_ID` / `LATENT_END_ID` markers (function name
`build_latent_mask` in HyLar, `_future_l1_build_latent_mask` in RL_v2;
algorithmically identical), then computes `pg_loss_tok` on `response_mask · (1 − latent_mask)`
with the *standard* clip ratios and `pg_loss_lat` on `response_mask · latent_mask`
with the *latent* clip ratios, finally `pg_loss = pg_loss_tok + α · pg_loss_lat`.

Closed-form vMF KL is added on top when `enable_latent_vmf_kl=True`:
`L_vMF = κ · mean(1 − cos(μ_actor_new, z_rollout))` over latent positions.

### 5.5 Advantage estimator

* RL_v2 `verl/trainer/core_algos.py`:
  - `AdvantageEstimator.GRPO` → `compute_grpo_outcome_advantage`
  - `AdvantageEstimator.DAPO` → `compute_dapo_outcome_advantage` which
    is a thin wrapper around `compute_grpo_outcome_advantage` (the
    DAPO-specific behaviour is in `update_policy`'s asymmetric clip
    and in `algorithm.online_filtering`).

* HyLar `HyLar/RL/verl/trainer/core_algos.py`:
  - `compute_grpo_outcome_advantage`
  - `compute_grpo_latent_advantage` and `compute_dapo_latent_advantage`
    are present but their formula is **identical** to the outcome
    versions (verified by reading lines 303-340 and 344-385). The
    naming difference is cosmetic.

Both stacks therefore compute the same advantage for GRPO and DAPO.
DePO inherits DAPO's advantage and adds the decoupled PPO + vMF KL
described in §5.4.

## 6. Defensive measures against regression

The original bug compounded because three layers all swallowed the
failure silently. After this audit:

1. **Inner layer** (`future_l1_rl_patch.patch_vllm`): when
   `FUTURE_L1_REQUIRE_VLLM_RUNNER=1`, a failure to substitute the vLLM
   GPU runner raises `RuntimeError` instead of falling back to stock vLLM.
   _Caveat:_ `patch_vllm()` first calls `_cuda_visible_to_this_process()`
   and returns cleanly (no raise) when CUDA is unavailable. This is the
   intended behaviour for Ray's driver-side `TemporaryActor` and for any
   other CPU-only process that transitively imports the patch — those
   processes never run vLLM and the real FSDPWorker actors (each with
   one GPU) will replace the runner on their own. **The strict guarantee
   therefore holds per-GPU-worker, not globally.**
2. **Outer layer** (`main.py`, `fsdp_workers.py`, `dp_actor.py`): when
   `FUTURE_L1_RL_PATCH_REQUIRED=1`, the `except Exception` arm of the
   `import future_l1_rl_patch` block re-raises instead of just printing.
3. **Launcher layer** (all 6 .sh files): export both flags `=1` by
   default. Override is possible (`FUTURE_L1_RL_PATCH_REQUIRED=0`) only
   for explicit ablation runs.

When all three layers report success, the **driver process** will print:

```
[FutureL1 RL patch] code_root=/mnt/.../VideoL1 backbone=qwen3_vl
[FutureL1 RL patch] vLLM GPU model runner replaced.
[FutureL1 RL patch] transformers + vLLM patches applied.
```

The **Ray TemporaryActor** (used only for import validation; no GPU) will
instead print:

```
(TemporaryActor pid=NNN) [FutureL1 RL patch] code_root=/mnt/.../VideoL1 backbone=qwen3_vl
(TemporaryActor pid=NNN) [FutureL1 RL patch] no GPU visible to this process (CUDA_VISIBLE_DEVICES=''); skipping vLLM GPU runner replacement here. The patch will be applied independently inside each GPU FSDPWorker actor.
(TemporaryActor pid=NNN) [FutureL1 RL patch] transformers patches applied; vLLM runner replacement skipped on this process (see preceding log line).
```

This is **expected and safe** — it is NOT a degradation. The real
FSDPWorker actors (with `CUDA_VISIBLE_DEVICES` set) will each print the
full "vLLM GPU model runner replaced." line in their own process. If the
FSDPWorker logs are missing that line, then the run IS broken and must
be discarded.

## 7. Reproducing HyLar baselines

### 7.1 GRPO

```bash
cd VideoL1/RL_v2
bash rjob/grpo.sh grpo
# or equivalently:
bash train.sh grpo
```

Asserts: `sampling_strategy=future_l1_depo`,
`enable_decoupled_hybrid_ppo=false`, `enable_latent_vmf_kl=false`,
`adv_estimator=grpo`, `online_filtering=false`. Look for
`[LATENT][last] continue req=... len=...` lines in the log to confirm
latent state machine is firing.

### 7.2 DAPO

```bash
bash rjob/grpo.sh dapo
# or
bash train.sh dapo
```

Asserts: `sampling_strategy=future_l1_depo`,
`enable_decoupled_hybrid_ppo=false`, `enable_latent_vmf_kl=false`,
`adv_estimator=dapo`, `online_filtering=true`, asymmetric clip
(`clip_ratio_low=0.2`, `clip_ratio_high=0.28`).

### 7.3 DePO

```bash
bash rjob/grpo.sh depo
# or
bash train.sh depo
```

Asserts: `sampling_strategy=future_l1_depo`,
`enable_decoupled_hybrid_ppo=true`, `enable_latent_vmf_kl=true`,
`adv_estimator=dapo`, latent clip ratios set to
`latent_clip_ratio_low=0.1`, `latent_clip_ratio_high=0.1`,
`latent_loss_alpha=0.5`, `latent_kl_coef=1e-2`. WandB should report
`actor/pg_loss_tok`, `actor/pg_loss_lat`, and
`actor/latent_vmf_kl` metrics.

### 7.4 Ablation handles (NOT HyLar-equivalent)

```bash
# latent-aware rollout but token-PPO at latent positions
FUTURE_L1_RL_PATCH_REQUIRED=0 \
  bash train.sh grpo \
  worker.rollout.sampling_strategy=future_l1_text

# stock EasyR1, no FutureL1 patch
FUTURE_L1_RL_PATCH=0 FUTURE_L1_REQUIRE_VLLM_RUNNER=0 \
FUTURE_L1_RL_PATCH_REQUIRED=0 \
  bash train.sh grpo \
  worker.rollout.sampling_strategy=default
```

Both ablations are useful for isolating "what does the latent-aware
actor add on top of latent-aware rollout?" — see §6 of the paper.

## 8. Validation checklist (per run)

Before declaring a run usable for the paper's headline numbers, confirm
ALL of the following:

| Check | Where to look | Expected |
|---|---|---|
| Patch loaded (driver) | first ~50 lines of stderr | `[FutureL1 RL patch] transformers + vLLM patches applied.` |
| vLLM runner replaced (driver) | first ~50 lines of stderr | `[FutureL1 RL patch] vLLM GPU model runner replaced.` |
| Patch loaded (each GPU FSDPWorker) | `grep '(FSDPWorker' run.log` | one `vLLM GPU model runner replaced.` line per actor |
| TemporaryActor skip is benign | grep `TemporaryActor` in stderr | only `no GPU visible to this process` + `skipping vLLM GPU runner replacement here` (NOT a `RuntimeError`) |
| Latent state machine firing | grep `[LATENT]` in stderr | at least one `[LATENT][last] continue req=` per training step |
| `latents` flowing to actor | `dp_actor` debug if enabled | `"latents" in micro_batch` evaluates True at least once per step |
| vMF substitution applied | `dp_actor` debug if enabled | non-None `latent_poss` and `latents_concat` |
| HyLar-equivalent mode set | the OmegaConf summary printed at startup | `worker.rollout.sampling_strategy: future_l1_depo` |
| Strict flags on | the OmegaConf summary or `env \| grep SWIMBIRD` | both `FUTURE_L1_RL_PATCH_REQUIRED=1` and `FUTURE_L1_REQUIRE_VLLM_RUNNER=1` |

## 9. Change log

| Date | Change | Author |
|---|---|---|
| 2026-05-16 | Bug A fix: switch all `rjob/*.sh` `VIDEO_L1_ROOT` to absolute paths | (prior commit) |
| 2026-05-16 | This audit: align GRPO/DAPO to `future_l1_depo`, harden patch loader, write HYLAR_ALIGNMENT.md | this commit |
| 2026-05-16 | Bug C fix: `patch_vllm()` now gates the `future_l1_gpu_model_runner` import on `_cuda_visible_to_this_process()` so Ray's CPU-only `TemporaryActor` (which is spawned to validate that `Runner` can be imported) no longer raises `No CUDA GPUs are available`. Strict mode is preserved on every actual GPU FSDPWorker — see §6 caveat and §8 row 3-4. | this commit |
