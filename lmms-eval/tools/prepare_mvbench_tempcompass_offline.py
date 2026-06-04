#!/usr/bin/env python3
"""One-time export of MVBench / TempCompass HF datasets to disk for offline lmms-eval.

Run on a machine with Hugging Face access. Then copy the output directories plus
video files to your offline cluster and set:

  MVBench:
    export LMMS_EVAL_MVBENCH_DATASET_DIR=/path/to/mvbench_arrow
    export LMMS_EVAL_MVBENCH_VIDEO_ROOT=/path/to/mvbench_video   # unpacked tree

  TempCompass:
    export LMMS_EVAL_TEMPCOMPASS_DATASET_DIR=/path/to/tempcompass_arrow
    export LMMS_EVAL_TEMPCOMPASS_VIDEO_ROOT=/path/to/tempcompass/videos

Evaluate with task groups: mvbench_local, tempcompass_local.
"""

from __future__ import annotations

import argparse

from datasets import load_dataset

MVBENCH_SUBSETS = [
    "action_sequence",
    "moving_count",
    "action_prediction",
    "episodic_reasoning",
    "action_antonym",
    "action_count",
    "scene_transition",
    "object_shuffle",
    "object_existence",
    "fine_grained_pose",
    "unexpected_action",
    "moving_direction",
    "state_change",
    "object_interaction",
    "character_order",
    "action_localization",
    "counterfactual_inference",
    "fine_grained_action",
    "moving_attribute",
    "egocentric_navigation",
]

TEMPCOMPASS_SUBSETS = ["multi-choice", "yes_no", "caption_matching", "captioning"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "benchmark",
        choices=["mvbench", "tempcompass", "all"],
        help="Which benchmark to export",
    )
    p.add_argument("--out-mvbench", default="./offline_mvbench_arrow", help="Output root for MVBench save_to_disk folders")
    p.add_argument(
        "--out-tempcompass",
        default="./offline_tempcompass_arrow",
        help="Output root for TempCompass save_to_disk folders",
    )
    args = p.parse_args()

    if args.benchmark in ("mvbench", "all"):
        for name in MVBENCH_SUBSETS:
            target = f"{args.out_mvbench.rstrip('/')}/{name}"
            print(f"[mvbench] loading {name!r} (revision=video) …")
            ds = load_dataset("OpenGVLab/MVBench", name, revision="video", trust_remote_code=True)
            ds.save_to_disk(target)
            print(f"  saved -> {target}")
        print("\nVideos: unpack the MVBench video release so paths match lmms_eval.tasks.mvbench.utils.DATA_LIST "
              f"under your LMMS_EVAL_MVBENCH_VIDEO_ROOT (default layout: $HF_HOME/mvbench_video).")

    if args.benchmark in ("tempcompass", "all"):
        for name in TEMPCOMPASS_SUBSETS:
            target = f"{args.out_tempcompass.rstrip('/')}/{name}"
            print(f"[tempcompass] loading {name!r} …")
            ds = load_dataset("lmms-lab/TempCompass", name, trust_remote_code=True)
            ds.save_to_disk(target)
            print(f"  saved -> {target}")
        print("\nVideos: place all TempCompass .mp4 files in the directory set by "
              "LMMS_EVAL_TEMPCOMPASS_VIDEO_ROOT (same as …/tempcompass/videos when using HF cache).")


if __name__ == "__main__":
    main()
