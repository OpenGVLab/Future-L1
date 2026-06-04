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
from PIL import Image
from loguru import logger as eval_logger


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "seedbench_r1_l2.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_dir = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]



def seedbench_r1_doc_to_visual(doc):
    ###
    #cache_dir = 'path/to/seedbench_r1'
    ###
    video_path = ''
    if len(doc['task_progress_metadata']) > 0:
        video_path = os.path.join(cache_dir, doc['video_basename'])

        if not os.path.exists(video_path):
            raise Exception(f"video path:{video_path} does not exist, please check")
    
    image_path = os.path.join(cache_dir, 'images', doc['video_source'], doc['current_observation_basename'])

    if not os.path.exists(image_path):
        raise Exception(f"video path:{image_path} does not exist, please check")

    return [(video_path, image_path)]


def seedbench_r1_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = lmms_eval_specific_kwargs["pre_prompt"] if "pre_prompt" in lmms_eval_specific_kwargs else "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    options = {"A": doc['choice_a'], "B": doc['choice_b'], "C": doc['choice_c'], "D": doc['choice_d']}
    option = "A. "+options["A"]+"\nB. "+options["B"]+"\nC. "+options["C"]+"\nD. "+options["D"]
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt

    return full_prompt


def seedbench_r1_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return doc['golden_choice_idx']


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


def seedbench_r1_process_results(doc, results):
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

    data_dict = {"video_id": doc["video_id"], "pred_answer": pred_ans, "answer": doc["golden_choice_idx"].upper()}

    return {f"seedbench_r1_acc": data_dict}


def seedbench_r1_process_reasoning_results(doc, results):
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

    data_dict = {"video_id": doc["video_id"], "pred_answer": pred_ans, "answer": doc["golden_choice_idx"].upper()}

    return {f"seedbench_r1_acc": data_dict}


def seedbench_r1_aggregate_results(results):
    category2score = {}

    category2score['overall'] = {"correct": 0, "answered": 0}

    for result in results:
        category2score['overall']["answered"] += 1
        category2score['overall']["correct"] += result["pred_answer"] == result["answer"]

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
