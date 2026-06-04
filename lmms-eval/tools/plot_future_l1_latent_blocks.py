#!/usr/bin/env python3
"""Plot generated latent embeddings colored by latent block index.

Reads latent_embeddings_rank*_part*.npz files exported by FUTURE_L1_EXPORT_LATENTS=1.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _parse_named_paths(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        out[name] = path
    return out


def _load_run(name: str, export_path: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(export_path, "latent_embeddings_rank*_part*.npz")))
    if not files and export_path.endswith(".npz"):
        files = [export_path]
    if not files:
        raise FileNotFoundError(f"No latent_embeddings_rank*_part*.npz files found in {export_path}")

    chunks = {"embeddings": [], "request_pos": [], "block_id": [], "token_pos": [], "global_pos": []}
    for file in files:
        data = np.load(file, allow_pickle=False)
        chunks["embeddings"].append(data["embeddings"].astype(np.float32))
        chunks["request_pos"].append(data.get("original_request_pos", data.get("request_pos")).astype(np.int32))
        chunks["block_id"].append(data["block_id"].astype(np.int32))
        chunks["token_pos"].append(data["token_pos"].astype(np.int32))
        chunks["global_pos"].append(data["global_pos"].astype(np.int32))

    out = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    out["run"] = np.asarray([name] * out["embeddings"].shape[0], dtype=object)
    return out


def _aggregate_by_sample_block(data: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    if mode == "token":
        return data
    if mode not in {"sample_block_mean", "sample_block_first"}:
        raise ValueError(f"Unknown block representation: {mode}")

    keys = sorted({(int(req), int(block)) for req, block in zip(data["request_pos"], data["block_id"]) if int(req) >= 0 and int(block) >= 0})
    if not keys:
        return data

    embeddings = []
    request_pos = []
    block_id = []
    token_pos = []
    global_pos = []
    for req, block in keys:
        idx = np.flatnonzero((data["request_pos"] == req) & (data["block_id"] == block))
        if idx.size == 0:
            continue
        if mode == "sample_block_first":
            chosen = idx[np.argmin(data["token_pos"][idx])]
            emb = data["embeddings"][chosen]
            tok = data["token_pos"][chosen]
            glob = data["global_pos"][chosen]
        else:
            emb = data["embeddings"][idx].mean(axis=0)
            tok = -1
            glob = -1
        embeddings.append(emb.astype(np.float32))
        request_pos.append(req)
        block_id.append(block)
        token_pos.append(tok)
        global_pos.append(glob)

    return {
        "run": np.asarray([data["run"][0]] * len(embeddings), dtype=object),
        "embeddings": np.stack(embeddings, axis=0),
        "request_pos": np.asarray(request_pos, dtype=np.int32),
        "block_id": np.asarray(block_id, dtype=np.int32),
        "token_pos": np.asarray(token_pos, dtype=np.int32),
        "global_pos": np.asarray(global_pos, dtype=np.int32),
    }


def _filter_blocks(data: dict[str, np.ndarray], max_block: int, max_points_per_block: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep = []
    blocks = sorted(int(x) for x in set(data["block_id"].tolist()) if int(x) >= 0 and (max_block < 0 or int(x) <= max_block))
    for block in blocks:
        idx = np.flatnonzero(data["block_id"] == block)
        if idx.size == 0:
            continue
        if max_points_per_block > 0 and idx.size > max_points_per_block:
            idx = rng.choice(idx, size=max_points_per_block, replace=False)
        keep.append(idx)
    if not keep:
        raise ValueError("No latent blocks left after filtering")
    idx = np.concatenate(keep)
    rng.shuffle(idx)
    return {key: value[idx] for key, value in data.items()}


def _embed_2d(x: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, metric="cosine", init="pca", learning_rate="auto", random_state=seed).fit_transform(x)
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise SystemExit("Please install umap-learn or use --method tsne") from exc
        return umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(x)
    if method == "pca":
        x = x.astype(np.float32, copy=False)
        x = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        return x @ vt[:2].T
    raise ValueError(f"Unknown method: {method}")


def _write_csv(path: Path, coords: np.ndarray, data: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("x,y,run,block_id,request_pos,token_pos,global_pos\n")
        for i in range(coords.shape[0]):
            f.write(
                f"{coords[i,0]},{coords[i,1]},{data['run'][i]},{int(data['block_id'][i])},"
                f"{int(data['request_pos'][i])},{int(data['token_pos'][i])},{int(data['global_pos'][i])}\n"
            )


def _plot_panel(ax, coords: np.ndarray, data: dict[str, np.ndarray], title: str, point_size: float, alpha: float, max_block: int) -> None:
    import matplotlib.pyplot as plt

    blocks = sorted(int(x) for x in set(data["block_id"].tolist()) if int(x) >= 0 and (max_block < 0 or int(x) <= max_block))
    cmap = plt.get_cmap("tab20")
    for i, block in enumerate(blocks):
        mask = data["block_id"] == block
        if not np.any(mask):
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=point_size, alpha=alpha, color=cmap(i % 20), label=f"B{block+1}", linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="name=latent_export_dir_or_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--block-repr", choices=["token", "sample_block_mean", "sample_block_first"], default="token")
    parser.add_argument("--max-block", type=int, default=5, help="Plot blocks 0..max-block; set -1 to keep all")
    parser.add_argument("--max-points-per-block", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--separate", action="store_true")
    args = parser.parse_args()

    runs = _parse_named_paths(args.runs)
    loaded = []
    for run_idx, (name, path) in enumerate(runs.items()):
        data = _load_run(name, path)
        data = _aggregate_by_sample_block(data, args.block_repr)
        data = _filter_blocks(data, args.max_block, args.max_points_per_block, args.seed + run_idx)
        coords = _embed_2d(data["embeddings"], args.method, args.seed)
        loaded.append((name, data, coords))
        counts = {int(b): int((data["block_id"] == int(b)).sum()) for b in sorted(set(data["block_id"].tolist()))}
        print(f"{name}: {counts}")

    import matplotlib.pyplot as plt

    n = len(loaded)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, data, coords) in zip(axes.ravel(), loaded):
        ax.axis("on")
        _plot_panel(ax, coords, data, name, args.point_size, args.alpha, args.max_block)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), frameon=False, markerscale=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    print(f"Saved combined figure to {output}")

    csv_dir = output.with_suffix("")
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, data, coords in loaded:
        _write_csv(csv_dir / f"{name}_{args.method}_blocks.csv", coords, data)

    if args.separate:
        for name, data, coords in loaded:
            fig_one, ax = plt.subplots(figsize=(5.2, 4.2))
            _plot_panel(ax, coords, data, name, args.point_size, args.alpha, args.max_block)
            ax.legend(markerscale=4, fontsize=8, frameon=False, loc="best")
            fig_one.tight_layout()
            out_one = output.with_name(f"{output.stem}_{name}{output.suffix}")
            fig_one.savefig(out_one, dpi=300)
            print(f"Saved {name} figure to {out_one}")
            plt.close(fig_one)


if __name__ == "__main__":
    main()
