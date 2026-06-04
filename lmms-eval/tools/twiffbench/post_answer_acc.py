#!/usr/bin/env python3
"""Rule-based TwiFF-Bench answer accuracy (no GPT judge).

Compares the inner text of ``<answer>...</answer>`` (GT may use legacy
``<ans>``) in the ground-truth ``answer`` field vs the model response.

Logic mirrors ``lmms_eval.tasks.twiffbench.utils.twiffbench_extract_answer_tag_span``
and ``twiffbench_sanitize_model_response_for_judge`` (inlined here so the
script runs without the full lmms-eval import graph).

Use after inference with ``TWIFF_SKIP_JUDGE=1`` (same jsonl as post_score.py).

Usage:
    python tools/twiffbench/post_answer_acc.py \\
        --input logs_twiffbench/.../samples_twiffbench_*.jsonl

    # Match modes: exact (default) | relaxed (lowercase, strip punctuation)
    python tools/twiffbench/post_answer_acc.py --input ... --match relaxed

Writes ``<input_stem>.answer_acc.jsonl`` and ``<input_stem>.answer_acc.summary.json``.

For GPT judge on the same two spans (0–5, semantic match), use ``post_answer_judge.py``.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ANSWER_SPAN_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_LATENT_TOK_RE = re.compile(r"<\|latent(?:_start|_end)?\|>|<\|latent\|>", re.IGNORECASE)
_JUDGE_TEMPLATE_TOK_RE = re.compile(
    r"<\|im_(?:start|end)\|>|<\|endoftext\|>|<\|redacted_im_end\|>",
    re.IGNORECASE,
)
_REASON_XML_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)


def _legacy_ans_tags_to_answer(text: str) -> str:
    if not text:
        return text
    t = re.sub(r"</ans\s*>", "</answer>", text, flags=re.IGNORECASE)
    return re.sub(r"<ans\s*>", "<answer>", t, flags=re.IGNORECASE)


def extract_answer_tag_span(text: str) -> str:
    """Inner text of first ``<answer>...</answer>`` (``<ans>`` OK). No fallbacks."""
    if not text:
        return ""
    m = _ANSWER_SPAN_RE.search(_legacy_ans_tags_to_answer(text))
    return m.group(1).strip() if m else ""


def _merge_reason_xml_blocks(fragment: str) -> str:
    if not fragment:
        return ""
    parts = [m.group(1).strip() for m in _REASON_XML_RE.finditer(fragment) if m.group(1).strip()]
    rest = _REASON_XML_RE.sub("", fragment).strip()
    merged = "\n\n".join(parts)
    if rest:
        merged = f"{merged}\n\n{rest}".strip() if merged else rest
    return merged.strip()


def sanitize_model_response_for_answer(text: str) -> str:
    """Strip FutureL1 tokens and keep a single ``<answer>`` block for extraction."""
    if not text:
        return text
    cleaned = _LATENT_TOK_RE.sub("", text)
    cleaned = _JUDGE_TEMPLATE_TOK_RE.sub("", cleaned)
    cleaned = _legacy_ans_tags_to_answer(cleaned)
    m = _ANSWER_SPAN_RE.search(cleaned)
    if m:
        head = _merge_reason_xml_blocks(cleaned[: m.start()].rstrip())
        ans = m.group(1).strip()
        return f"{head}\n<answer>{ans}</answer>".strip() if head else f"<answer>{ans}</answer>"
    return _merge_reason_xml_blocks(cleaned)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TwiFF-Bench rule-based answer tag accuracy")
    p.add_argument("--input", required=True, help="lmms-eval samples_twiffbench_*.jsonl")
    p.add_argument(
        "--output",
        default=None,
        help="Per-sample output jsonl (default: <input_stem>.answer_acc.jsonl)",
    )
    p.add_argument(
        "--summary",
        default=None,
        help="Aggregate summary JSON (default: <output_stem>.summary.json)",
    )
    p.add_argument(
        "--match",
        choices=("exact", "relaxed"),
        default="exact",
        help="exact: whitespace-normalized equality; relaxed: also lowercase + drop punctuation",
    )
    p.add_argument(
        "--strip-latent",
        dest="strip_latent",
        action="store_true",
        default=True,
        help="Sanitize FutureL1 latent/template tokens before extracting <answer> (default: on)",
    )
    p.add_argument(
        "--no-strip-latent",
        dest="strip_latent",
        action="store_false",
        help="Read model <answer> from raw response without sanitization",
    )
    p.add_argument(
        "--suffix",
        default="",
        help="Optional tag in output name: <stem>.answer_acc.<suffix>.jsonl",
    )
    return p.parse_args()


def _normalize_exact(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _normalize_relaxed(s: str) -> str:
    s = _normalize_exact(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _answers_match(gt: str, pred: str, mode: str) -> bool:
    if mode == "relaxed":
        return _normalize_relaxed(gt) == _normalize_relaxed(pred)
    return _normalize_exact(gt) == _normalize_exact(pred)


def _record_from_log_line(line: str) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    obj = json.loads(line)
    doc_id = obj.get("doc_id")

    if isinstance(obj.get("twiffbench_score"), dict):
        return obj["twiffbench_score"], doc_id

    doc = obj.get("doc") or {}
    gt = doc.get("answer")
    pred = None
    if obj.get("filtered_resps"):
        pred = obj["filtered_resps"][0]
    elif obj.get("resps"):
        r = obj["resps"][0]
        pred = r[0] if isinstance(r, list) else r

    if gt is None and pred is None:
        return None, doc_id

    return {
        "video": doc.get("video"),
        "question": doc.get("question"),
        "answer": gt,
        "model_response": [pred] if pred is not None else [],
        "meta_data": doc.get("meta_data"),
    }, doc_id


def _model_response_text(rec: Dict[str, Any]) -> str:
    mr = rec.get("model_response")
    if isinstance(mr, list) and mr:
        return str(mr[-1] or "")
    if isinstance(rec.get("raw_response"), str):
        return rec["raw_response"]
    return ""


def score_record(
    rec: Dict[str, Any],
    *,
    match: str,
    strip_latent: bool,
) -> Dict[str, Any]:
    gt_raw = rec.get("answer") or ""
    pred_raw = _model_response_text(rec)
    if strip_latent:
        pred_raw = sanitize_model_response_for_answer(pred_raw)

    gt_span = extract_answer_tag_span(gt_raw)
    pred_span = extract_answer_tag_span(pred_raw)

    has_gt = bool(gt_span)
    has_pred = bool(pred_span)
    correct = has_gt and has_pred and _answers_match(gt_span, pred_span, match)

    out = {
        "video": rec.get("video"),
        "gt_answer_span": gt_span,
        "pred_answer_span": pred_span,
        "has_gt_answer_tag": has_gt,
        "has_pred_answer_tag": has_pred,
        "answer_correct": correct,
        "match_mode": match,
    }
    meta = rec.get("meta_data")
    if isinstance(meta, dict) and meta.get("classification"):
        out["classification"] = meta["classification"]
    return out


def main() -> None:
    args = parse_args()
    inp = Path(args.input).expanduser().resolve()
    suffix_part = f".{args.suffix}" if args.suffix else ""
    out_path = (
        Path(args.output)
        if args.output
        else inp.with_name(f"{inp.stem}.answer_acc{suffix_part}.jsonl")
    )
    summary_path = (
        Path(args.summary)
        if args.summary
        else out_path.with_name(f"{out_path.stem}.summary.json")
    )

    rows: List[Dict[str, Any]] = []
    n_skip = 0

    with open(inp, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec, doc_id = _record_from_log_line(line)
            if rec is None:
                n_skip += 1
                continue
            scored = score_record(rec, match=args.match, strip_latent=args.strip_latent)
            if doc_id is not None:
                scored["doc_id"] = doc_id
            rows.append(scored)

    n = len(rows)
    n_gt = sum(1 for r in rows if r["has_gt_answer_tag"])
    n_pred = sum(1 for r in rows if r["has_pred_answer_tag"])
    n_both = sum(1 for r in rows if r["has_gt_answer_tag"] and r["has_pred_answer_tag"])
    n_correct = sum(1 for r in rows if r["answer_correct"])

    acc_both = n_correct / n_both if n_both else None
    acc_all = n_correct / n if n else None

    per_class: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        if r.get("classification") and r["has_gt_answer_tag"] and r["has_pred_answer_tag"]:
            per_class[r["classification"]].append(r["answer_correct"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for r in rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "input": str(inp),
        "output": str(out_path.resolve()),
        "match_mode": args.match,
        "strip_latent": args.strip_latent,
        "n_records": n,
        "n_skipped_lines": n_skip,
        "n_with_gt_answer_tag": n_gt,
        "n_with_pred_answer_tag": n_pred,
        "n_with_both_tags": n_both,
        "n_correct": n_correct,
        "answer_accuracy_both_tags": acc_both,
        "answer_accuracy_all_samples": acc_all,
        "missing_gt_tag": n - n_gt,
        "missing_pred_tag": n - n_pred,
    }
    if per_class:
        summary["per_classification"] = {
            cls: {"n": len(vals), "accuracy": sum(vals) / len(vals) if vals else None}
            for cls, vals in sorted(per_class.items())
        }

    with open(summary_path, "w") as fsum:
        json.dump(summary, fsum, indent=2, ensure_ascii=False)
        fsum.write("\n")

    print(f"[twiffbench/post_answer_acc] wrote per-sample -> {out_path}")
    print(f"[twiffbench/post_answer_acc] wrote summary -> {summary_path}")
    if acc_both is not None:
        print(
            f"[twiffbench/post_answer_acc] answer_accuracy ({args.match}, both tags): "
            f"{n_correct}/{n_both} = {acc_both * 100:.2f}%"
        )
    if acc_all is not None:
        print(
            f"[twiffbench/post_answer_acc] answer_accuracy ({args.match}, all samples): "
            f"{n_correct}/{n} = {acc_all * 100:.2f}%"
        )
    print(f"[twiffbench/post_answer_acc] missing tags: gt={n - n_gt} pred={n - n_pred}")


if __name__ == "__main__":
    main()
