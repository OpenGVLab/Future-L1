#!/usr/bin/env python3
"""One-shot decoder for the TwiFF-Bench parquet shards.

Writes a flat on-disk cache so ``post_score.py`` does not have to reload the
~GBs of HF datasets every run:

    <out_dir>/
        meta.jsonl                         # one record per doc_id
        imgs/<doc_id>/q_<i>.jpg            # question_images, in original order
        imgs/<doc_id>/r_<i>.jpg            # reasoning_images, in original order

``doc_id`` is the lmms-eval row index, i.e. the position after concatenating
the parquet shards in the order listed in twiffbench.yaml.

Usage::

    python tools/twiffbench/build_cache.py --out logs_twiffbench_future_l1/_cache

Override the parquet shards with ``--parquet`` if your data lives elsewhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.tasks.twiffbench.utils import _to_pil  # noqa: E402

DEFAULT_PARQUET = [
    "/path/to/your/data/VideoL1-Bench/TwiFF-Bench/test-00000-of-00004.parquet",
    "/path/to/your/data/VideoL1-Bench/TwiFF-Bench/test-00001-of-00004.parquet",
    "/path/to/your/data/VideoL1-Bench/TwiFF-Bench/test-00002-of-00004.parquet",
    "/path/to/your/data/VideoL1-Bench/TwiFF-Bench/test-00003-of-00004.parquet",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", nargs="+", default=DEFAULT_PARQUET)
    p.add_argument("--out", required=True, help="Output cache dir")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality (1-95)")
    p.add_argument("--overwrite", action="store_true", help="Re-decode even if cache exists")
    return p.parse_args()


def _save_images(images: Iterable, out_dir: Path, prefix: str, quality: int) -> List[str]:
    paths: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images or []):
        pil = _to_pil(im)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        rel = f"{prefix}_{i}.jpg"
        pil.save(out_dir / rel, format="JPEG", quality=quality)
        paths.append(rel)
    return paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    meta_path = out_dir / "meta.jsonl"

    if meta_path.exists() and not args.overwrite:
        n_lines = sum(1 for _ in open(meta_path))
        print(
            f"[twiffbench/build_cache] cache already exists at {out_dir} "
            f"({n_lines} records). Use --overwrite to rebuild."
        )
        return

    import datasets

    print(f"[twiffbench/build_cache] loading {len(args.parquet)} parquet shard(s)...")
    ds_list = [datasets.load_dataset("parquet", data_files=p, split="train") for p in args.parquet]
    ds = datasets.concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]
    print(f"[twiffbench/build_cache] {len(ds)} rows total")

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs_root = out_dir / "imgs"
    imgs_root.mkdir(exist_ok=True)

    n_total = len(ds)
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(range(n_total), desc="decode", dynamic_ncols=True)
    except ImportError:
        iterator = range(n_total)

    with open(meta_path, "w") as fmeta:
        for doc_id in iterator:
            row = ds[doc_id]
            doc_dir = imgs_root / str(doc_id)
            q_paths = _save_images(row.get("question_images"), doc_dir, "q", args.quality)
            r_paths = _save_images(row.get("reasoning_images"), doc_dir, "r", args.quality)

            rec = {
                "doc_id": doc_id,
                "video": row.get("video"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "frames": list(row.get("question_images_index") or []),
                "recon_frames": list(row.get("reasoning_images_index") or []),
                "question_images": q_paths,
                "reasoning_images": r_paths,
                "meta_data": row.get("meta_data"),
            }
            fmeta.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[twiffbench/build_cache] done -> {out_dir}")


if __name__ == "__main__":
    main()
