import datetime
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "futurebench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_dir = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


textscore_dict, videoscore_dict = {}, {}


def future_pred_doc_to_visual(doc):
    ###
    # cache_dir = '/path/to/V1-33K/first_part_video/' 
    ###
    raw_video_path = doc["video_path"]

    if os.path.isabs(raw_video_path):
        video_path = raw_video_path
    elif "video_dataset/" in raw_video_path:
        sub_path = raw_video_path.split("video_dataset/", 1)[1]
        video_path = os.path.join(cache_dir, sub_path)
    else:
        # Fallback for datasets that already store a cache_dir-relative path.
        video_path = os.path.join(cache_dir, raw_video_path.lstrip("/"))

    if not os.path.exists(video_path):
        raise Exception(
            f"video path:{video_path} does not exist, please check "
            f"(raw video_path: {raw_video_path}, cache_dir: {cache_dir})"
        )

    return [video_path]




def future_pred_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = lmms_eval_specific_kwargs["pre_prompt"] if "pre_prompt" in lmms_eval_specific_kwargs else "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc['qa']["Question"]
    options = doc['qa']["Options"]
    # 部分 FutureBench 样本的 Question 字段已经把 A)/B)/C)/D) 选项内嵌在文本里。
    # 如果检测到内嵌选项,就不再额外拼一份 "A. ... B. ... C. ... D. ..." 以避免重复。
    has_inline_options = bool(re.search(r"\bA\)\s+", question)) and bool(re.search(r"\bD\)\s+", question))
    if not has_inline_options:
        option = "A. "+options["A"]+"\nB. "+options["B"]+"\nC. "+options["C"]+"\nD. "+options["D"]
        question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt

    return full_prompt


def future_pred_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return doc['qa']['Answer']


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def extract_thinking_answer_regex(s):
    s = s.strip()
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, s, re.DOTALL)

    if match:
        ans = match.group(1).strip()
    else:
        return ""
    
    if len(ans.split()) > 10 and not re.search("[ABCD]", ans):
        return ""

    matches = re.search(r"[ABCD]", ans)
    if matches is None:
        return ""

    return matches[0]


TYPES = ['hop1', 'hop2', 'hop3', 'hop5']

def future_pred_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    data_dict = {"video_id": doc["id"], 'question_type': doc['question_type'], "pred_answer": pred_ans, "answer": doc["qa"]["Answer"].upper()}

    return {f"future_acc": data_dict}


def future_pred_process_reasoning_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_thinking_answer_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    data_dict = {"video_id": doc["id"], 'question_type': doc['question_type'], "pred_answer": pred_ans, "answer": doc["qa"]["Answer"].upper()}
    latent_payload = _future_latent_metric_payload(doc)
    if latent_payload["sample_count"] > 0:
        data_dict["latent_similarity"] = doc.get("_future_l1_latent_similarity_stats", [])

    out = {f"future_acc": data_dict}
    if latent_payload["sample_count"] > 0:
        out.update(
            {
                "latent_adjacent_cos2": latent_payload,
                "latent_adjacent_cos": latent_payload,
                "latent_adjacent_mse": latent_payload,
                "latent_adjacent_pair_count": latent_payload,
                "latent_block_cos2": latent_payload,
                "latent_block_cos": latent_payload,
                "latent_block_mse": latent_payload,
                "latent_block_pair_count": latent_payload,
                "latent_block_count": latent_payload,
                "latent_vector_count": latent_payload,
            }
        )
    return out


def _future_latent_metric_payload(doc):
    stats_list = doc.get("_future_l1_latent_similarity_stats", [])
    payload = {
        "adj_cos2_sum": 0.0,
        "adj_cos_sum": 0.0,
        "adj_mse_sum": 0.0,
        "adj_pair_count": 0,
        "cos2_sum": 0.0,
        "cos_sum": 0.0,
        "mse_sum": 0.0,
        "pair_count": 0,
        "block_count": 0,
        "vector_count": 0,
        "sample_count": 0,
    }
    for stats in stats_list:
        if not isinstance(stats, dict):
            continue
        payload["sample_count"] += 1
        adj_pair_count = int(stats.get("latent_adjacent_pair_count") or 0)
        payload["adj_pair_count"] += adj_pair_count
        payload["block_count"] += int(stats.get("latent_block_count") or 0)
        payload["vector_count"] += int(stats.get("latent_vector_count") or 0)
        adj_cos2 = stats.get("latent_adjacent_cos2_mean")
        adj_cos = stats.get("latent_adjacent_cos_mean")
        if adj_pair_count > 0 and adj_cos2 is not None:
            payload["adj_cos2_sum"] += float(adj_cos2) * adj_pair_count
        if adj_pair_count > 0 and adj_cos is not None:
            payload["adj_cos_sum"] += float(adj_cos) * adj_pair_count
        adj_mse = stats.get("latent_adjacent_mse_mean")
        if adj_pair_count > 0 and adj_mse is not None:
            payload["adj_mse_sum"] += float(adj_mse) * adj_pair_count

        pair_count = int(stats.get("latent_block_pair_count") or 0)
        payload["pair_count"] += pair_count
        cos2 = stats.get("latent_block_cos2_mean")
        cos = stats.get("latent_block_cos_mean")
        if pair_count > 0 and cos2 is not None:
            payload["cos2_sum"] += float(cos2) * pair_count
        if pair_count > 0 and cos is not None:
            payload["cos_sum"] += float(cos) * pair_count
        block_mse = stats.get("latent_block_mse_mean")
        if pair_count > 0 and block_mse is not None:
            payload["mse_sum"] += float(block_mse) * pair_count
    return payload


def _per_hop_acc(results, hop):
    correct = sum(1 for r in results if r["question_type"] == hop and r["pred_answer"] == r["answer"])
    total   = sum(1 for r in results if r["question_type"] == hop)
    acc = 100 * correct / total if total > 0 else 0
    eval_logger.info(f"Evaluation on question types: {hop}: {acc:.1f}%")
    return acc


def future_pred_aggregate_results(results):
    for hop in TYPES:
        _per_hop_acc(results, hop)
    correct = sum(1 for r in results if r["pred_answer"] == r["answer"])
    total   = len(results)
    acc = 100 * correct / total if total > 0 else 0
    eval_logger.info(f"Overall Performance: {acc:.1f}%")
    return acc


def future_pred_aggregate_hop1(results):
    return _per_hop_acc(results, "hop1")


def future_pred_aggregate_hop2(results):
    return _per_hop_acc(results, "hop2")


def future_pred_aggregate_hop3(results):
    return _per_hop_acc(results, "hop3")


def future_pred_aggregate_hop5(results):
    return _per_hop_acc(results, "hop5")


def future_pred_aggregate_latent_adjacent_cos2(results):
    pair_count = sum(int(r.get("adj_pair_count", 0)) for r in results if isinstance(r, dict))
    cos2_sum = sum(float(r.get("adj_cos2_sum", 0.0)) for r in results if isinstance(r, dict))
    value = cos2_sum / pair_count if pair_count > 0 else 0.0
    eval_logger.info(f"FutureL1 latent step-adjacent cos^2 (LASER): {value:.4f} over {pair_count} pairs")
    return value


def future_pred_aggregate_latent_adjacent_cos(results):
    pair_count = sum(int(r.get("adj_pair_count", 0)) for r in results if isinstance(r, dict))
    cos_sum = sum(float(r.get("adj_cos_sum", 0.0)) for r in results if isinstance(r, dict))
    value = cos_sum / pair_count if pair_count > 0 else 0.0
    eval_logger.info(f"FutureL1 latent step-adjacent cosine (LASER): {value:.4f} over {pair_count} pairs")
    return value


def future_pred_aggregate_latent_adjacent_pair_count(results):
    value = sum(int(r.get("adj_pair_count", 0)) for r in results if isinstance(r, dict))
    eval_logger.info(f"FutureL1 latent step-adjacent pair count (LASER): {value}")
    return value


def future_pred_aggregate_latent_adjacent_mse(results):
    pair_count = sum(int(r.get("adj_pair_count", 0)) for r in results if isinstance(r, dict))
    mse_sum = sum(float(r.get("adj_mse_sum", 0.0)) for r in results if isinstance(r, dict))
    value = mse_sum / pair_count if pair_count > 0 else 0.0
    eval_logger.info(f"FutureL1 latent step-adjacent MSE (LASER): {value:.6f} over {pair_count} pairs")
    return value


def future_pred_aggregate_latent_block_cos2(results):
    pair_count = sum(int(r.get("pair_count", 0)) for r in results if isinstance(r, dict))
    cos2_sum = sum(float(r.get("cos2_sum", 0.0)) for r in results if isinstance(r, dict))
    value = cos2_sum / pair_count if pair_count > 0 else 0.0
    eval_logger.info(f"FutureL1 latent block adjacent cos^2: {value:.4f} over {pair_count} pairs")
    return value


def future_pred_aggregate_latent_block_cos(results):
    pair_count = sum(int(r.get("pair_count", 0)) for r in results if isinstance(r, dict))
    cos_sum = sum(float(r.get("cos_sum", 0.0)) for r in results if isinstance(r, dict))
    value = cos_sum / pair_count if pair_count > 0 else 0.0
    eval_logger.info(f"FutureL1 latent block adjacent cosine: {value:.4f} over {pair_count} pairs")
    return value


def future_pred_aggregate_latent_block_pair_count(results):
    value = sum(int(r.get("pair_count", 0)) for r in results if isinstance(r, dict))
    eval_logger.info(f"FutureL1 latent block adjacent pair count: {value}")
    return value


def future_pred_aggregate_latent_block_mse(results):
    pair_count = sum(int(r.get("pair_count", 0)) for r in results if isinstance(r, dict))
    mse_sum = sum(float(r.get("mse_sum", 0.0)) for r in results if isinstance(r, dict))
    value = mse_sum / pair_count if pair_count > 0 else 0.0
    eval_logger.info(f"FutureL1 latent block adjacent MSE: {value:.6f} over {pair_count} pairs")
    return value


def future_pred_aggregate_latent_block_count(results):
    sample_count = sum(int(r.get("sample_count", 0)) for r in results if isinstance(r, dict))
    block_count = sum(int(r.get("block_count", 0)) for r in results if isinstance(r, dict))
    value = block_count / sample_count if sample_count > 0 else 0.0
    eval_logger.info(f"FutureL1 latent blocks per sample: {value:.2f}")
    return value


def future_pred_aggregate_latent_vector_count(results):
    sample_count = sum(int(r.get("sample_count", 0)) for r in results if isinstance(r, dict))
    vector_count = sum(int(r.get("vector_count", 0)) for r in results if isinstance(r, dict))
    value = vector_count / sample_count if sample_count > 0 else 0.0
    eval_logger.info(f"FutureL1 latent vectors per sample: {value:.2f}")
    return value
