#!/usr/bin/env python3
"""Plot Fig.7-style FutureL1/Mirage embedding visualizations.

This script reads ``mirage_embeddings_rank*_part*.npz`` files exported by
``FUTURE_L1_EXPORT_MIRAGE_EMBEDDINGS=1`` and plots text, vision, and latent
embeddings with t-SNE/UMAP/PCA.

Example:
  python tools/plot_future_l1_mirage_fig7.py \
    --runs \
      sft=logs_futurebench_future_l1_mirage_fig7/sft/mirage_exports \
      dapo=logs_futurebench_future_l1_mirage_fig7/dapo/mirage_exports \
      colvr050=logs_futurebench_future_l1_mirage_fig7/colvr050/mirage_exports \
      colvr020_div010=logs_futurebench_future_l1_mirage_fig7/colvr020_div010/mirage_exports \
    --output figures/future_l1_mirage_fig7_tsne.pdf \
    --method tsne --max-per-type 3000
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

TYPE_ORDER = ["text", "vision", "latent"]
COLORS = {
    "text": "#2F6DB3",
    "vision": "#E5B83F",
    "latent": "#C8463A",
}
BLOCK_COLORS = {
    0: "#D62728",
    1: "#9467BD",
    2: "#2CA02C",
    3: "#000000",
    4: "#7F7F7F",
    5: "#17BECF",
    6: "#FF7F0E",
    7: "#8C564B",
}
BLOCK_GRADIENT_COLORS = {
    0: "#F6B3A6",
    1: "#E87461",
    2: "#C83D36",
    3: "#7F1D1D",
    4: "#4A0F12",
}
LABELS = {
    "text": "Text",
    "vision": "Vision",
    "latent": "Latent",
}


def _parse_named_paths(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        out[name] = path
    return out


def _load_mirage_run(name: str, export_path: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(export_path, "mirage_embeddings_rank*_part*.npz")))
    if not files and export_path.endswith(".npz"):
        files = [export_path]
    if not files:
        raise FileNotFoundError(f"No mirage_embeddings_rank*_part*.npz files found in {export_path}")

    embeddings = []
    type_names_chunks = []
    request_pos = []
    source_pos = []

    for file in files:
        data = np.load(file, allow_pickle=False)
        emb = data["embeddings"].astype(np.float32)
        type_id = data["type_id"].astype(np.int32)
        names = [str(x) for x in data.get("type_names", np.asarray(TYPE_ORDER))]
        type_names = np.asarray([names[int(i)] if int(i) < len(names) else f"type_{int(i)}" for i in type_id], dtype=object)
        embeddings.append(emb)
        type_names_chunks.append(type_names)
        request_pos.append(data.get("original_request_pos", data.get("request_pos", np.full(emb.shape[0], -1))).astype(np.int32))
        source_pos.append(data.get("source_pos", np.full(emb.shape[0], -1)).astype(np.int32))

    return {
        "run": np.asarray([name] * sum(x.shape[0] for x in embeddings), dtype=object),
        "embeddings": np.concatenate(embeddings, axis=0),
        "type": np.concatenate(type_names_chunks, axis=0),
        "request_pos": np.concatenate(request_pos, axis=0),
        "source_pos": np.concatenate(source_pos, axis=0),
    }


def _load_latent_block_run(name: str, export_path: str, max_blocks: int) -> dict[str, np.ndarray]:
    mirage_files = sorted(glob.glob(os.path.join(export_path, "mirage_embeddings_rank*_part*.npz")))
    latent_files = sorted(glob.glob(os.path.join(export_path, "latent_embeddings_rank*_part*.npz")))
    if not mirage_files:
        raise FileNotFoundError(f"No mirage_embeddings_rank*_part*.npz files found in {export_path}")
    if not latent_files:
        raise FileNotFoundError(f"No latent_embeddings_rank*_part*.npz files found in {export_path}")

    embeddings = []
    labels = []
    request_pos = []
    source_pos = []

    for file in mirage_files:
        data = np.load(file, allow_pickle=False)
        emb = data["embeddings"].astype(np.float32)
        type_id = data["type_id"].astype(np.int32)
        names = [str(x) for x in data.get("type_names", np.asarray(TYPE_ORDER))]
        type_names = np.asarray([names[int(i)] if int(i) < len(names) else f"type_{int(i)}" for i in type_id], dtype=object)
        keep = np.flatnonzero((type_names == "text") | (type_names == "vision"))
        if keep.size == 0:
            continue
        embeddings.append(emb[keep])
        labels.append(type_names[keep])
        req = data.get("original_request_pos", data.get("request_pos", np.full(emb.shape[0], -1))).astype(np.int32)
        src = data.get("source_pos", np.full(emb.shape[0], -1)).astype(np.int32)
        request_pos.append(req[keep])
        source_pos.append(src[keep])

    for file in latent_files:
        data = np.load(file, allow_pickle=False)
        emb = data["embeddings"].astype(np.float32)
        block_id = data["block_id"].astype(np.int32)
        keep = np.flatnonzero(block_id >= 0)
        if max_blocks > 0:
            keep = keep[block_id[keep] < max_blocks]
        if keep.size == 0:
            continue
        embeddings.append(emb[keep])
        labels.append(np.asarray([f"latent_block_{int(b)}" for b in block_id[keep]], dtype=object))
        request_pos.append(data.get("original_request_pos", data.get("request_pos", np.full(emb.shape[0], -1))).astype(np.int32)[keep])
        source_pos.append(data.get("global_pos", data.get("token_pos", np.full(emb.shape[0], -1))).astype(np.int32)[keep])

    return {
        "run": np.asarray([name] * sum(x.shape[0] for x in embeddings), dtype=object),
        "embeddings": np.concatenate(embeddings, axis=0),
        "type": np.concatenate(labels, axis=0),
        "request_pos": np.concatenate(request_pos, axis=0),
        "source_pos": np.concatenate(source_pos, axis=0),
    }


def _label_order(data: dict[str, np.ndarray]) -> list[str]:
    present = set(str(x) for x in data["type"].tolist())
    labels = [x for x in TYPE_ORDER if x in present]
    block_labels = sorted(
        [x for x in present if x.startswith("latent_block_")],
        key=lambda x: int(x.rsplit("_", 1)[-1]),
    )
    labels.extend(block_labels)
    labels.extend(sorted(present - set(labels)))
    return labels


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _label_text(label: str, block_label_style: str) -> str:
    if label in LABELS:
        return LABELS[label]
    if label.startswith("latent_block_"):
        block_id = int(label.rsplit('_', 1)[-1]) + 1
        if block_label_style == "ordinal":
            return _ordinal(block_id)
        if block_label_style == "ordinal-span":
            return f"{_ordinal(block_id)} Latent Span"
        return f"Latent block {block_id}"
    return label


def _label_color(label: str, idx: int, block_palette: str):
    if label in COLORS:
        return COLORS[label]
    if label.startswith("latent_block_"):
        block_id = int(label.rsplit("_", 1)[-1])
        if block_palette == "gradient":
            return BLOCK_GRADIENT_COLORS.get(block_id, "#4A0F12")
        return BLOCK_COLORS.get(block_id, "#7F7F7F")
    import matplotlib.pyplot as plt
    return plt.get_cmap("tab20")(idx % 20)


def _aggregate_latents(data: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    if mode == "token":
        return data
    if mode not in {"sample_mean", "sample_first"}:
        raise ValueError(f"Unknown latent aggregation mode: {mode}")

    latent_idx = np.flatnonzero(data["type"] == "latent")
    nonlatent_idx = np.flatnonzero(data["type"] != "latent")
    if latent_idx.size == 0:
        return data

    latent_request = data["request_pos"][latent_idx]
    valid_requests = sorted(int(x) for x in set(latent_request.tolist()) if int(x) >= 0)
    if not valid_requests:
        return data

    new_embeddings = []
    new_request_pos = []
    new_source_pos = []
    for req in valid_requests:
        req_idx = latent_idx[latent_request == req]
        if req_idx.size == 0:
            continue
        if mode == "sample_first":
            chosen = req_idx[np.argmin(data["source_pos"][req_idx])]
            emb = data["embeddings"][chosen]
            src = data["source_pos"][chosen]
        else:
            emb = data["embeddings"][req_idx].mean(axis=0)
            src = -1
        new_embeddings.append(emb.astype(np.float32))
        new_request_pos.append(req)
        new_source_pos.append(src)

    if not new_embeddings:
        return data

    latent_data = {
        "run": np.asarray([data["run"][0]] * len(new_embeddings), dtype=object),
        "embeddings": np.stack(new_embeddings, axis=0),
        "type": np.asarray(["latent"] * len(new_embeddings), dtype=object),
        "request_pos": np.asarray(new_request_pos, dtype=np.int32),
        "source_pos": np.asarray(new_source_pos, dtype=np.int32),
    }
    return {
        key: np.concatenate([data[key][nonlatent_idx], latent_data[key]], axis=0)
        for key in data
    }


def _aggregate_latent_blocks(data: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    if mode == "token":
        return data
    if mode not in {"sample_block_mean", "sample_block_first"}:
        raise ValueError(f"Unknown latent block aggregation mode: {mode}")

    non_block_idx = np.flatnonzero(~np.char.startswith(data["type"].astype(str), "latent_block_"))
    block_idx = np.flatnonzero(np.char.startswith(data["type"].astype(str), "latent_block_"))
    if block_idx.size == 0:
        return data

    keys = sorted(
        {
            (int(data["request_pos"][i]), str(data["type"][i]))
            for i in block_idx
            if int(data["request_pos"][i]) >= 0
        },
        key=lambda x: (x[0], int(x[1].rsplit("_", 1)[-1])),
    )

    new = {key: [] for key in data}
    for req, label in keys:
        idx = np.flatnonzero((data["request_pos"] == req) & (data["type"] == label))
        if idx.size == 0:
            continue
        if mode == "sample_block_first":
            chosen = idx[np.argmin(data["source_pos"][idx])]
            emb = data["embeddings"][chosen]
            src = data["source_pos"][chosen]
        else:
            emb = data["embeddings"][idx].mean(axis=0)
            src = -1
        new["run"].append(data["run"][0])
        new["embeddings"].append(emb.astype(np.float32))
        new["type"].append(label)
        new["request_pos"].append(req)
        new["source_pos"].append(src)

    block_data = {
        "run": np.asarray(new["run"], dtype=object),
        "embeddings": np.stack(new["embeddings"], axis=0),
        "type": np.asarray(new["type"], dtype=object),
        "request_pos": np.asarray(new["request_pos"], dtype=np.int32),
        "source_pos": np.asarray(new["source_pos"], dtype=np.int32),
    }
    return {
        key: np.concatenate([data[key][non_block_idx], block_data[key]], axis=0)
        for key in data
    }


def _sample_by_label(data: dict[str, np.ndarray], max_per_label: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep = []
    for label in _label_order(data):
        idx = np.flatnonzero(data["type"] == label)
        if idx.size == 0:
            continue
        if max_per_label > 0 and idx.size > max_per_label:
            idx = rng.choice(idx, size=max_per_label, replace=False)
        keep.append(idx)
    if not keep:
        raise ValueError("No embeddings found after filtering")
    idx = np.concatenate(keep)
    rng.shuffle(idx)
    return {key: value[idx] for key, value in data.items()}


def _embed_2d(x: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2,
            perplexity=30,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(x)
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
        f.write("x,y,run,type,request_pos,source_pos\n")
        for i in range(coords.shape[0]):
            f.write(
                f"{coords[i,0]},{coords[i,1]},{data['run'][i]},{data['type'][i]},"
                f"{int(data['request_pos'][i])},{int(data['source_pos'][i])}\n"
            )


def _plot_panel(
    ax,
    coords: np.ndarray,
    data: dict[str, np.ndarray],
    title: str,
    point_size: float,
    alpha: float,
    show_title: bool,
    block_palette: str,
    block_label_style: str,
) -> None:
    for i, label in enumerate(_label_order(data)):
        mask = data["type"] == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[_label_color(label, i, block_palette)],
            label=_label_text(label, block_label_style),
            linewidths=0,
            rasterized=True,
        )
    if show_title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="name=mirage_export_dir_or_npz")
    parser.add_argument("--output", required=True, help="Output figure path (.pdf/.png)")
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--max-per-type", type=int, default=3000, help="Sample this many points per type/label per run; <=0 keeps all")
    parser.add_argument(
        "--latent-repr",
        choices=["token", "sample_mean", "sample_first"],
        default="token",
        help="How to represent latent embeddings when --latent-color-by type is used.",
    )
    parser.add_argument("--latent-color-by", choices=["type", "block"], default="type")
    parser.add_argument(
        "--latent-block-repr",
        choices=["token", "sample_block_mean", "sample_block_first"],
        default="token",
        help="How to represent latent embeddings when coloring by block.",
    )
    parser.add_argument("--max-block", type=int, default=5, help="Keep latent block ids < max-block when --latent-color-by block; <=0 keeps all blocks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--separate", action="store_true", help="Also save one figure per run next to --output")
    parser.add_argument("--no-title", action="store_true", help="Do not draw subplot titles")
    parser.add_argument("--font-file", default=None, help="Path to a .ttf font file used by the figure")
    parser.add_argument("--legend-bold", action="store_true", help="Draw legend text in bold")
    parser.add_argument("--legend-font-size", type=float, default=10.0)
    parser.add_argument("--legend-ncol", type=int, default=3)
    parser.add_argument("--legend-bottom", type=float, default=0.04)
    parser.add_argument("--fig-width", type=float, default=5.2)
    parser.add_argument("--fig-height", type=float, default=4.2)
    parser.add_argument("--block-palette", choices=["categorical", "gradient"], default="categorical")
    parser.add_argument("--block-label-style", choices=["latent-block", "ordinal", "ordinal-span"], default="latent-block")
    args = parser.parse_args()

    runs = _parse_named_paths(args.runs)
    loaded = []
    for run_idx, (name, path) in enumerate(runs.items()):
        if args.latent_color_by == "block":
            data = _load_latent_block_run(name, path, args.max_block)
            data = _aggregate_latent_blocks(data, args.latent_block_repr)
        else:
            data = _load_mirage_run(name, path)
            data = _aggregate_latents(data, args.latent_repr)
        data = _sample_by_label(data, args.max_per_type, args.seed + run_idx)
        coords = _embed_2d(data["embeddings"], args.method, args.seed)
        loaded.append((name, data, coords))
        counts = {label: int((data["type"] == label).sum()) for label in _label_order(data)}
        print(f"{name}: sampled {counts}")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
    })

    if args.font_file:
        from matplotlib import font_manager

        font_manager.fontManager.addfont(args.font_file)
        font_name = font_manager.FontProperties(fname=args.font_file).get_name()
        plt.rcParams["font.family"] = font_name

    n = len(loaded)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(args.fig_width * ncols, args.fig_height * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, (name, data, coords) in zip(axes.ravel(), loaded):
        ax.axis("on")
        _plot_panel(
            ax,
            coords,
            data,
            name,
            args.point_size,
            args.alpha,
            not args.no_title,
            args.block_palette,
            args.block_label_style,
        )

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=args.legend_ncol, frameon=False, markerscale=4, fontsize=args.legend_font_size)
    if args.legend_bold:
        for text in legend.get_texts():
            text.set_fontweight("bold")
    fig.tight_layout(rect=(0, args.legend_bottom, 1, 1))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    print(f"Saved combined figure to {output}")

    csv_dir = output.with_suffix("")
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, data, coords in loaded:
        _write_csv(csv_dir / f"{name}_{args.method}.csv", coords, data)

    if args.separate:
        for name, data, coords in loaded:
            fig_one, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
            _plot_panel(
                ax,
                coords,
                data,
                name,
                args.point_size,
                args.alpha,
                not args.no_title,
                args.block_palette,
                args.block_label_style,
            )
            legend_one = ax.legend(markerscale=4, fontsize=args.legend_font_size, frameon=False, loc="best")
            if args.legend_bold:
                for text in legend_one.get_texts():
                    text.set_fontweight("bold")
            fig_one.tight_layout()
            out_one = output.with_name(f"{output.stem}_{name}{output.suffix}")
            fig_one.savefig(out_one, dpi=300)
            print(f"Saved {name} figure to {out_one}")
            plt.close(fig_one)


if __name__ == "__main__":
    main()
