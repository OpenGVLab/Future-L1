#!/usr/bin/env python3
"""
Clean TwiFF-style RL json/jsonl by dropping rows that would fail materialization.

Default behavior mirrors RL_v2/verl/utils/dataset.py failure conditions:
1) missing `human` turn in `conversations`
2) frame decode failure for rows with non-empty `image` indices

Outputs:
- cleaned dataset (`.json` or `.jsonl`)
- bad-row report (`.jsonl`)
- summary (`.json`)
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _coerce_conversations(value: Any) -> List[dict]:
    """Normalize TwiFF's `conversations` field across shard variants."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        froms = value.get("from") or []
        values = value.get("value") or []
        return [{"from": f, "value": v} for f, v in zip(froms, values)]
    return []


def _get_frame_indices_uni(num_frames: int, vlen: int, start_frames: int = 1, end_frames: int = 1) -> List[int]:
    if vlen <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        return []
    if vlen <= num_frames:
        indices = list(range(vlen))
        if len(indices) < num_frames:
            indices.extend([indices[-1]] * (num_frames - len(indices)))
        return indices
    start = min(start_frames, vlen)
    end = max(vlen - end_frames, start)
    if num_frames == 1:
        return [(start + end - 1) // 2]
    step = (end - 1 - start) / float(num_frames - 1)
    return [int(round(start + i * step)) for i in range(num_frames)]


def _iter_records(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception as exc:  # noqa: BLE001
                    yield i, {
                        "__parse_error__": str(exc),
                        "__raw_line__": raw[:200],
                    }
                    continue
                if not isinstance(obj, dict):
                    yield i, {"__type_error__": f"row is {type(obj).__name__}, expected dict"}
                    continue
                yield i, obj
        return

    # Default: .json with either [records] or {"data":[records]}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for i, rec in enumerate(obj):
            if isinstance(rec, dict):
                yield i, rec
            else:
                yield i, {"__type_error__": f"row is {type(rec).__name__}, expected dict"}
        return
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        for i, rec in enumerate(obj["data"]):
            if isinstance(rec, dict):
                yield i, rec
            else:
                yield i, {"__type_error__": f"row is {type(rec).__name__}, expected dict"}
        return
    raise ValueError("Unsupported JSON format. Expect list[dict] or {'data': list[dict]}.")


def _validate_one(
    idx: int,
    rec: Dict[str, Any],
    check_decode: bool,
    decord_mod: Any,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Return (is_good, bad_info_if_any)."""
    rid = rec.get("id", f"idx_{idx}")
    video_path = str(rec.get("video", "") or "")

    if "__parse_error__" in rec:
        return False, {
            "index": idx,
            "id": rid,
            "video": video_path,
            "reason": f"parse_error: {rec['__parse_error__']}",
        }
    if "__type_error__" in rec:
        return False, {
            "index": idx,
            "id": rid,
            "video": video_path,
            "reason": rec["__type_error__"],
        }

    # RLHFDataset._materialize_twiff: if no human turn => returns None => hard fail.
    conversations = _coerce_conversations(rec.get("conversations"))
    has_human = any(str(turn.get("from", "")) == "human" for turn in conversations)
    if not has_human:
        return False, {
            "index": idx,
            "id": rid,
            "video": video_path,
            "reason": "missing_human_turn",
        }

    # Follow dataset.py behavior:
    # parse image to int-list; on parse failure it falls back to [] (no hard fail).
    try:
        image_indices = [int(x) for x in (rec.get("image") or [])]
    except (TypeError, ValueError):
        image_indices = []

    # Text-only row in this pathway: no decode needed.
    if not image_indices:
        return True, None

    # decode path requires valid video path
    if not video_path:
        return False, {
            "index": idx,
            "id": rid,
            "video": video_path,
            "reason": "missing_video_path",
        }
    if not os.path.isfile(video_path):
        return False, {
            "index": idx,
            "id": rid,
            "video": video_path,
            "reason": "video_not_found",
        }

    # Keep this check aligned with selected = [pool[int(i) - 1] for i in clip_indices].
    for ci in image_indices:
        if ci <= 0 or ci > 8:
            return False, {
                "index": idx,
                "id": rid,
                "video": video_path,
                "reason": f"invalid_image_index:{ci}",
            }

    if not check_decode:
        return True, None

    if decord_mod is None:
        return False, {
            "index": idx,
            "id": rid,
            "video": video_path,
            "reason": "decord_not_installed",
        }

    try:
        vr = decord_mod.VideoReader(video_path, num_threads=1)
        pool = _get_frame_indices_uni(8, len(vr), start_frames=1, end_frames=1)
        selected = [pool[int(i) - 1] for i in image_indices]
        _ = vr.get_batch(selected).asnumpy()
    except Exception as exc:  # noqa: BLE001
        return False, {
            "index": idx,
            "id": rid,
            "video": video_path,
            "reason": f"decode_failed:{type(exc).__name__}",
            "detail": str(exc),
        }

    return True, None


def _write_clean(path: Path, rows: List[Dict[str, Any]], force_jsonl: bool) -> None:
    if force_jsonl or path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
        f.write("\n")


def _write_bad_jsonl(path: Path, bad_rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in bad_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean TwiFF RL json/jsonl by dropping bad rows.")
    parser.add_argument("--input", required=True, help="Input dataset path (.json/.jsonl)")
    parser.add_argument("--output-clean", required=True, help="Output cleaned dataset path")
    parser.add_argument("--output-bad", default=None, help="Output bad rows jsonl (default: <output-clean>.bad.jsonl)")
    parser.add_argument("--output-summary", default=None, help="Output summary json (default: <output-clean>.summary.json)")
    parser.add_argument("--workers", type=int, default=16, help="Thread workers for validation")
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip decord decode checks; only run structural checks.",
    )
    parser.add_argument("--force-jsonl", action="store_true", help="Force cleaned output format as jsonl")
    parser.add_argument("--print-every", type=int, default=500, help="Progress print interval")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_clean = Path(args.output_clean)
    output_bad = Path(args.output_bad) if args.output_bad else output_clean.with_suffix(output_clean.suffix + ".bad.jsonl")
    output_summary = (
        Path(args.output_summary) if args.output_summary else output_clean.with_suffix(output_clean.suffix + ".summary.json")
    )

    check_decode = not args.skip_decode
    decord_mod = None
    if check_decode:
        try:
            import decord  # noqa: PLC0415

            decord_mod = decord
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Decode check requested but `decord` is unavailable. Install decord, "
                "or rerun with --skip-decode."
            ) from exc

    records = list(_iter_records(input_path))
    total = len(records)
    print(f"[clean_twiff_json] loaded {total} rows from {input_path}")
    print(f"[clean_twiff_json] decode_check={'on' if check_decode else 'off'} workers={args.workers}")

    good_rows: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []
    bad_reason_counter: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_to_payload = {
            pool.submit(_validate_one, idx, rec, check_decode, decord_mod): (idx, rec) for idx, rec in records
        }
        done = 0
        for fut in as_completed(future_to_payload):
            idx, rec = future_to_payload[fut]
            is_good, bad_info = fut.result()
            if is_good:
                # only keep original valid dict rows
                if "__parse_error__" not in rec and "__type_error__" not in rec:
                    good_rows.append(rec)
            else:
                assert bad_info is not None
                bad_rows.append(bad_info)
                reason = str(bad_info.get("reason", "unknown"))
                bad_reason_counter[reason] = bad_reason_counter.get(reason, 0) + 1

            done += 1
            if done % args.print_every == 0 or done == total:
                print(f"[clean_twiff_json] checked {done}/{total}, bad={len(bad_rows)}, good={len(good_rows)}")

    # preserve original order in cleaned output
    # (as_completed changes order, so rebuild from accepted index set)
    bad_indices = {int(b["index"]) for b in bad_rows}
    ordered_good = [rec for idx, rec in records if idx not in bad_indices and "__parse_error__" not in rec and "__type_error__" not in rec]

    output_clean.parent.mkdir(parents=True, exist_ok=True)
    output_bad.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    _write_clean(output_clean, ordered_good, force_jsonl=args.force_jsonl)
    _write_bad_jsonl(output_bad, sorted(bad_rows, key=lambda x: int(x["index"])))

    summary = {
        "input": str(input_path),
        "output_clean": str(output_clean),
        "output_bad": str(output_bad),
        "decode_check": check_decode,
        "workers": args.workers,
        "total_rows": total,
        "kept_rows": len(ordered_good),
        "dropped_rows": len(bad_rows),
        "drop_ratio": (len(bad_rows) / total) if total else 0.0,
        "dropped_by_reason": dict(sorted(bad_reason_counter.items(), key=lambda kv: kv[0])),
    }
    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[clean_twiff_json] wrote clean dataset: {output_clean}")
    print(f"[clean_twiff_json] wrote bad report:  {output_bad}")
    print(f"[clean_twiff_json] wrote summary:     {output_summary}")
    print(f"[clean_twiff_json] done: kept={len(ordered_good)} dropped={len(bad_rows)} total={total}")


if __name__ == "__main__":
    main()
