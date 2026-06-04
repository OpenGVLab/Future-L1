#!/usr/bin/env python3
"""Standalone TwiFF-Bench GPT-judge runner.

This script mirrors ``eval/TwiFFBench/gencot_vqa_eval.py`` from the upstream
TwiFF repo, but reads its inputs from lmms-eval's ``--log_samples`` JSONL
output (the per-sample records emitted by ``twiffbench_process_results``).

Use it when you ran inference with ``TWIFF_SKIP_JUDGE=1`` to defer scoring.

Usage:
    python tools/twiffbench/post_score.py \
        --input  logs_twiffbench/.../samples_twiffbench_*.jsonl

    # Optional: override output path (default: same dir as --input,
    # filename ``<input_stem>.scored.jsonl``).

Aggregate metrics are also written next to ``--output`` as ``*.summary.json``
(override with ``--summary PATH``).

Output jsonl omits ``question_images`` / ``reasoning_images`` base64 blobs by
default (add ``--include-image-b64`` if you need them on disk).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.tasks.twiffbench.utils import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    _build_judge_messages,
    _coerce_twiff_judge_score_pair,
    _load_env_file,
    _normalize_api_base,
    _pil_to_base64,
    _sanitize_record_for_judge,
    _to_pil,
    twiffbench_log_judge_mode,
)


# Default parquet shards (same as twiffbench[_future_l1].yaml). Override with --parquet.
DEFAULT_PARQUET = [
    "/path/to/your/data/Future-L1-Bench/TwiFF-Bench/test-00000-of-00004.parquet",
    "/path/to/your/data/Future-L1-Bench/TwiFF-Bench/test-00001-of-00004.parquet",
    "/path/to/your/data/Future-L1-Bench/TwiFF-Bench/test-00002-of-00004.parquet",
    "/path/to/your/data/Future-L1-Bench/TwiFF-Bench/test-00003-of-00004.parquet",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="lmms-eval samples_twiffbench_*.jsonl")
    p.add_argument(
        "--output",
        default=None,
        help="Where to write scored jsonl. Default: same directory as --input, "
        "file ``<input_stem>.scored.jsonl``.",
    )
    p.add_argument("--max_tokens", type=int, default=int(os.environ.get("TWIFF_JUDGE_MAX_TOKENS", 1024)))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TWIFF_JUDGE_TEMPERATURE", 0.0)))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument(
        "--cache",
        default=None,
        help="Pre-decoded cache dir (built by build_cache.py). When provided, "
        "questions / answers / images are looked up here by doc_id instead of "
        "reloading the parquet on every run.",
    )
    p.add_argument(
        "--parquet",
        nargs="+",
        default=None,
        help="Fallback parquet shard(s) to rehydrate from when --cache is not "
        "given. Defaults to the TwiFF-Bench shards declared in the yaml.",
    )
    p.add_argument(
        "--strip-latent",
        dest="strip_latent",
        action="store_true",
        default=True,
        help="Strip latent/template tokens, merge <reason>...</reason> inner text, "
        "remove remaining <reason> markup, normalize to ``<answer>...</answer>`` before judge. "
        "On by default.",
    )
    p.add_argument(
        "--no-strip-latent",
        dest="strip_latent",
        action="store_false",
        help="Disable response sanitization (e.g. to evaluate vanilla Qwen-VL).",
    )
    p.add_argument(
        "--include-image-b64",
        action="store_true",
        default=False,
        help="Write question_images / reasoning_images (base64 JPEG dicts) into "
        "the output jsonl. Default is off so scored.jsonl stays small and readable.",
    )
    p.add_argument(
        "--summary",
        default=None,
        metavar="PATH",
        help="Where to write aggregate scores (JSON). Default: same basename as "
        "--output with .summary.json (e.g. scored.jsonl -> scored.summary.json).",
    )
    p.add_argument(
        "--suffix",
        default="dpsk",
        help="Tag appended to the auto-generated output filename: "
        "``<input_stem>.scored.<suffix>.jsonl``. Default: ``dpsk``. "
        "Pass ``--suffix ''`` to revert to the old ``<stem>.scored.jsonl`` style.",
    )
    return p.parse_args()


def _record_for_jsonl(rec: Dict[str, Any], include_image_b64: bool) -> Dict[str, Any]:
    """Drop embedded JPEG base64 from disk output unless explicitly requested."""
    if include_image_b64:
        return rec
    out = {k: v for k, v in rec.items() if k not in ("question_images", "reasoning_images")}
    nq = len(rec.get("question_images") or [])
    nr = len(rec.get("reasoning_images") or [])
    if nq:
        out["question_images_count"] = nq
    if nr:
        out["reasoning_images_count"] = nr
    return out


def _serialize_pils(images) -> List[Dict[str, str]]:
    out = []
    for im in images or []:
        pil = _to_pil(im)
        out.append({"format": "JPEG", "b64": _pil_to_base64(pil, "JPEG")})
    return out


def _load_parquet_index(paths: List[str]):
    """Load all parquet shards in declared order and return a list of rows
    (HF datasets concatenates shards in the listed order, so the row index
    is the lmms-eval ``doc_id``).
    """
    import datasets  # lazy

    ds_list = [datasets.load_dataset("parquet", data_files=p, split="train") for p in paths]
    ds = datasets.concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]
    print(f"[twiffbench/post_score] loaded {len(ds)} rows from {len(paths)} parquet shard(s)")
    return ds


def _load_disk_cache(cache_dir: Path):
    """Load build_cache.py output: meta.jsonl + imgs/<doc_id>/{q,r}_*.jpg."""
    meta_path = cache_dir / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.jsonl under {cache_dir}; run build_cache.py first.")
    by_doc_id: Dict[int, Dict[str, Any]] = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            by_doc_id[int(obj["doc_id"])] = obj
    print(f"[twiffbench/post_score] cache: {len(by_doc_id)} records from {meta_path}")
    return by_doc_id


def _b64_from_jpg(path: Path) -> Dict[str, str]:
    return {"format": "JPEG", "b64": base64.b64encode(path.read_bytes()).decode("utf-8")}


def _record_from_log_line(
    line: str,
    parquet_ds=None,
    cache_by_doc_id: Dict[int, Dict[str, Any]] | None = None,
    cache_dir: Path | None = None,
) -> Dict[str, Any] | None:
    obj = json.loads(line)

    candidates = []
    for key in ("twiffbench_score", "process_results", "results"):
        if isinstance(obj.get(key), dict):
            candidates.append(obj[key])
    if "model_response" in obj and "question" in obj:
        candidates.append(obj)

    rec = None
    for cand in candidates:
        if "twiffbench_score" in cand and isinstance(cand["twiffbench_score"], dict):
            rec = cand["twiffbench_score"]
            break
        if "model_response" in cand:
            rec = cand
            break
    if rec is None:
        return None

    needs_rehydrate = (
        (not rec.get("question_images") and (rec.get("frames") or []))
        or (not rec.get("reasoning_images") and (rec.get("recon_frames") or []))
    )
    if not needs_rehydrate:
        return rec

    doc_id = obj.get("doc_id")
    if doc_id is None:
        return rec
    rec["_doc_id"] = int(doc_id)

    if cache_by_doc_id is not None and cache_dir is not None:
        meta = cache_by_doc_id.get(int(doc_id))
        if meta is not None:
            doc_imgs = cache_dir / "imgs" / str(doc_id)
            if not rec.get("question_images"):
                rec["question_images"] = [
                    _b64_from_jpg(doc_imgs / p) for p in (meta.get("question_images") or [])
                ]
            if not rec.get("reasoning_images"):
                rec["reasoning_images"] = [
                    _b64_from_jpg(doc_imgs / p) for p in (meta.get("reasoning_images") or [])
                ]
            for k_src, k_dst in (
                ("answer", "answer"),
                ("question", "question"),
                ("frames", "frames"),
                ("recon_frames", "recon_frames"),
            ):
                if not rec.get(k_dst) and meta.get(k_src) is not None:
                    rec[k_dst] = meta[k_src]
            return rec

    if parquet_ds is not None:
        try:
            row = parquet_ds[int(doc_id)]
        except (IndexError, KeyError):
            return rec
        if not rec.get("question_images"):
            rec["question_images"] = _serialize_pils(row.get("question_images"))
        if not rec.get("reasoning_images"):
            rec["reasoning_images"] = _serialize_pils(row.get("reasoning_images"))
        for k_src, k_dst in (
            ("answer", "answer"),
            ("question", "question"),
            ("question_images_index", "frames"),
            ("reasoning_images_index", "recon_frames"),
        ):
            if not rec.get(k_dst) and row.get(k_src) is not None:
                rec[k_dst] = list(row[k_src]) if isinstance(row[k_src], (list, tuple)) else row[k_src]
    return rec


async def _score_one(client, sem, model, args, record):
    async with sem:
        if args.strip_latent:
            _sanitize_record_for_judge(record)
        messages = _build_judge_messages(record)
        for attempt in range(args.max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                verdict = json.loads(content)
                record["judge_reasoning"] = verdict.get("reasoning")
                pair = _coerce_twiff_judge_score_pair(verdict.get("score"))
                if pair is None:
                    raise ValueError(f"invalid judge score: {verdict.get('score')!r}")
                record["judge_score"] = pair
                return record
            except Exception as exc:
                if attempt == args.max_retries - 1:
                    record["judge_error"] = str(exc)
                    return record


async def _amain():
    args = parse_args()
    if not args.output:
        inp = Path(args.input).expanduser().resolve()
        suffix_part = f".{args.suffix}" if args.suffix else ""
        args.output = str(inp.with_name(f"{inp.stem}.scored{suffix_part}.jsonl"))
    print(f"[twiffbench/post_score] output -> {args.output}")
    _load_env_file()
    twiffbench_log_judge_mode()
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    api_base = _normalize_api_base(os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    model = os.environ.get("LOCAL_LLM") or os.environ.get("JUDGE_MODEL") or "gpt-4o"

    from openai import AsyncOpenAI  # lazy
    client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=600)

    cache_dir = Path(args.cache) if args.cache else None
    cache_by_doc_id = _load_disk_cache(cache_dir) if cache_dir else None
    parquet_ds = None
    if cache_by_doc_id is None and args.parquet:
        parquet_ds = _load_parquet_index(args.parquet)

    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    n_rehydrated = 0
    with open(args.input, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = _record_from_log_line(
                line,
                parquet_ds=parquet_ds,
                cache_by_doc_id=cache_by_doc_id,
                cache_dir=cache_dir,
            )
            if rec is None:
                continue
            if "_doc_id" in rec and (rec.get("question_images") or rec.get("reasoning_images")):
                n_rehydrated += 1
            tasks.append(_score_one(client, sem, model, args, rec))
    if cache_by_doc_id is not None or parquet_ds is not None:
        src = "cache" if cache_by_doc_id is not None else "parquet"
        print(f"[twiffbench/post_score] rehydrated images for {n_rehydrated}/{len(tasks)} records from {src}")

    print(f"[twiffbench/post_score] scoring {len(tasks)} records via {model} @ {api_base}")

    results = []
    n_ok_running = 0
    n_err_running = 0
    rs_running: List[int] = []
    ans_running: List[int] = []
    try:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(total=len(tasks), desc="judge", dynamic_ncols=True)
    except ImportError:
        pbar = None

    for fut in asyncio.as_completed(tasks):
        rec = await fut
        results.append(rec)
        pair = _coerce_twiff_judge_score_pair(rec.get("judge_score"))
        if pair is not None:
            n_ok_running += 1
            rs_running.append(pair[0])
            ans_running.append(pair[1])
        elif rec.get("judge_error"):
            n_err_running += 1
            doc = rec.get("_doc_id", rec.get("doc_id", "?"))
            err_msg = f"[twiffbench/post_score] judge_error doc_id={doc}: {rec['judge_error']}"
            if pbar is not None:
                tqdm.write(err_msg)
            else:
                print(err_msg, file=sys.stderr)
        if pbar is not None:
            if n_ok_running:
                pbar.set_postfix(
                    ok=n_ok_running,
                    err=n_err_running,
                    r=f"{sum(rs_running)/n_ok_running:.2f}",
                    a=f"{sum(ans_running)/n_ok_running:.2f}",
                )
            else:
                pbar.set_postfix(ok=n_ok_running, err=n_err_running)
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    rs, ans_s = [], []
    with open(out_path, "w") as fout:
        for rec in results:
            line_obj = _record_for_jsonl(rec, args.include_image_b64)
            fout.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
            pair = _coerce_twiff_judge_score_pair(rec.get("judge_score"))
            if pair is not None:
                n_ok += 1
                rs.append(pair[0])
                ans_s.append(pair[1])

    n_err = sum(1 for r in results if r.get("judge_error"))
    mr = sum(rs) / n_ok if n_ok else None
    ma = sum(ans_s) / n_ok if n_ok else None
    overall = (mr + ma) / 2 if mr is not None and ma is not None else None

    summary_path = Path(args.summary) if args.summary else out_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input).resolve()),
        "output": str(out_path.resolve()),
        "judge_model": model,
        "api_base": api_base,
        "n_records": len(results),
        "n_scored_ok": n_ok,
        "n_scored_err": n_err,
        "reasoning_mean": mr,
        "answer_mean": ma,
        "overall_mean": overall,
    }
    with open(summary_path, "w") as fsum:
        json.dump(summary, fsum, indent=2, ensure_ascii=False)
        fsum.write("\n")
    print(f"[twiffbench/post_score] wrote final scores -> {summary_path}")

    if n_ok:
        print(
            f"[twiffbench/post_score] scored={n_ok}/{len(results)}  "
            f"reasoning={mr:.3f}/5  answer={ma:.3f}/5  overall={overall:.3f}/5"
        )
    else:
        print(f"[twiffbench/post_score] no valid scores; check API config in .env")


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
