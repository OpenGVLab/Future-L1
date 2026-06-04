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
    return {f"future_acc": data_dict}


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
