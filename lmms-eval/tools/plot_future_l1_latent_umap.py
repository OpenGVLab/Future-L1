#!/usr/bin/env python3
"""Plot FutureL1 latent embeddings exported during lmms-eval.

Usage:
  python tools/plot_future_l1_latent_umap.py \
    --inputs no_div=/path/to/no_div_exports full=/path/to/full_exports \
    --samples no_div=/path/to/no_div_samples.jsonl full=/path/to/full_samples.jsonl \
    --output latent_umap.pdf \
    --method umap --color-by run

The exporter writes latent_embeddings_rank*_part*.npz files when running eval with:
  FUTURE_L1_EXPORT_LATENTS=1
  FUTURE_L1_LATENT_EXPORT_DIR=/path/to/exports
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def _parse_named_paths(items: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        parsed[name] = path
    return parsed


def _load_samples(path: str) -> list[dict]:
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def _sample_meta(samples: list[dict], idx: int) -> Tuple[str, str, int]:
    if idx < 0 or idx >= len(samples):
        return "unknown", "unknown", -1
    sample = samples[idx]
    future_acc = sample.get("future_acc") if isinstance(sample.get("future_acc"), dict) else {}
    doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
    split = future_acc.get("question_type") or doc.get("question_type") or "unknown"
    pred = str(future_acc.get("pred_answer", "")).strip()
    ans = str(future_acc.get("answer", "")).strip()
    correct = int(bool(pred and ans and pred == ans))
    sample_id = future_acc.get("video_id") or doc.get("id") or str(sample.get("doc_id", idx))
    return str(split), str(sample_id), correct


def _load_exports(name: str, export_path: str, samples: list[dict] | None, max_points: int | None) -> dict:
    files = sorted(glob.glob(os.path.join(export_path, "latent_embeddings_rank*_part*.npz")))
    if not files and export_path.endswith(".npz"):
        files = [export_path]
    if not files:
        raise FileNotFoundError(f"No latent_embeddings_rank*_part*.npz files found in {export_path}")

    embeddings = []
    run = []
    split = []
    sample_id = []
    correct = []
    block_id = []
    token_pos = []
    global_pos = []

    for file in files:
        data = np.load(file, allow_pickle=False)
        emb = data["embeddings"].astype(np.float32)
        original_request_pos = data.get("original_request_pos", data.get("request_pos"))
        n = emb.shape[0]
        if samples is not None and original_request_pos is not None:
            meta = [_sample_meta(samples, int(i)) for i in original_request_pos]
        else:
            meta = [("unknown", "unknown", -1)] * n
        embeddings.append(emb)
        run.extend([name] * n)
        split.extend([m[0] for m in meta])
        sample_id.extend([m[1] for m in meta])
        correct.extend([m[2] for m in meta])
        block_id.extend(data["block_id"].astype(int).tolist())
        token_pos.extend(data["token_pos"].astype(int).tolist())
        global_pos.extend(data["global_pos"].astype(int).tolist())

    out = {
        "embeddings": np.concatenate(embeddings, axis=0),
        "run": np.asarray(run, dtype=object),
        "split": np.asarray(split, dtype=object),
        "sample_id": np.asarray(sample_id, dtype=object),
        "correct": np.asarray(correct, dtype=np.int32),
        "block_id": np.asarray(block_id, dtype=np.int32),
        "token_pos": np.asarray(token_pos, dtype=np.int32),
        "global_pos": np.asarray(global_pos, dtype=np.int32),
    }
    if max_points is not None and out["embeddings"].shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(out["embeddings"].shape[0], size=max_points, replace=False)
        for key in out:
            out[key] = out[key][idx]
    return out


def _embed_2d(x: np.ndarray, method: str) -> np.ndarray:
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise SystemExit("Please install umap-learn or use --method tsne") from exc
        return umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, metric="cosine", init="pca", learning_rate="auto", random_state=42).fit_transform(x)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(x)
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="name=export_dir_or_npz")
    parser.add_argument("--samples", nargs="*", default=[], help="optional name=samples.jsonl for metadata join")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--color-by", choices=["run", "split", "block_id", "token_pos", "correct"], default="run")
    parser.add_argument("--max-points-per-run", type=int, default=5000)
    args = parser.parse_args()

    inputs = _parse_named_paths(args.inputs)
    sample_paths = _parse_named_paths(args.samples)

    chunks = []
    for name, path in inputs.items():
        samples = _load_samples(sample_paths[name]) if name in sample_paths else None
        chunks.append(_load_exports(name, path, samples, args.max_points_per_run))

    keys = chunks[0].keys()
    merged = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}
    coords = _embed_2d(merged["embeddings"], args.method)

    import matplotlib.pyplot as plt

    color_values = merged[args.color_by]
    labels = sorted(set(color_values.tolist()), key=lambda x: str(x))
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(5.2, 4.2))
    for i, label in enumerate(labels):
        mask = color_values == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.45, color=cmap(i % 10), label=str(label), linewidths=0)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"{args.method.upper()}-1")
    plt.ylabel(f"{args.method.upper()}-2")
    plt.legend(markerscale=3, fontsize=8, frameon=False, loc="best")
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)

    csv_path = str(Path(args.output).with_suffix(".csv"))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("x,y,run,split,sample_id,correct,block_id,token_pos,global_pos\n")
        for i in range(coords.shape[0]):
            f.write(
                f"{coords[i,0]},{coords[i,1]},{merged['run'][i]},{merged['split'][i]},"
                f"{merged['sample_id'][i]},{merged['correct'][i]},{merged['block_id'][i]},"
                f"{merged['token_pos'][i]},{merged['global_pos'][i]}\n"
            )
    print(f"Saved plot to {args.output}")
    print(f"Saved coordinates to {csv_path}")


if __name__ == "__main__":
    main()
