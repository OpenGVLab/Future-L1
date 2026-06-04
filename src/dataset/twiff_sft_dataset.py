import re
import os
import json
import random
import logging
import hashlib
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import decord
from PIL import Image

from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, Features, Value, Sequence
from qwen_vl_utils import process_vision_info

from .data_utils import (
    remove_assistant_images,
    remove_user_images,
    replace_visual_spectial_tokens,
    replace_visual_spectial_tokens_merged,
    collator_replace_latent,
    generate_labels_after_multi_token_start,
    mask_image_output_tokens,
)
from .future_l1_dataset import resolve_system_message, _parse_debug_ce_budget
from src.constants import SYSTEM_MESSAGE


# --------------------------------------------------------------------------- #
# Frame extraction                                                             #
# --------------------------------------------------------------------------- #

def _get_frame_indices_uni(num_frames: int, vlen: int, start_frames: int = 1, end_frames: int = 1) -> List[int]:
    """Uniform 'middle' sampling of num_frames indices from a video of vlen frames."""
    if start_frames + 1 >= vlen:
        start_frames = 0
    if vlen - end_frames - 1 <= start_frames:
        end_idx = vlen - 1
    else:
        end_idx = vlen - end_frames - 1

    acc_samples = min(num_frames - 2, end_idx - start_frames)
    intervals = np.linspace(start=start_frames + 1, stop=end_idx, num=acc_samples + 1).astype(int)
    middle = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(len(intervals) - 1)]
    frame_indices = [start_frames] + middle + [end_idx]

    if len(frame_indices) < num_frames:  # pad with last frame
        padded = [frame_indices[-1]] * num_frames
        padded[: len(frame_indices)] = frame_indices
        frame_indices = padded

    return frame_indices


def read_frames_by_clip(video_path: str, clip_indices: List[int]) -> List[Image.Image]:
    """
    Extract frames from *video_path* at positions given by *clip_indices*.

    *clip_indices* are 1-based indices into a uniformly sampled pool of
    ``8`` frames (same convention as TwiFF's video_think_dataset, which
    hard-codes ``num_frames=8``). Returns PIL Images in the same order
    as *clip_indices*.
    """
    # NOTE: hard-coded to 8 to match TwiFF's video_think_dataset.read_frames_decord_uni.
    # The training JSON only emits clip indices in {1..8}, so this never goes out of bounds.
    # Using `max(clip_indices)` here would shift the underlying pool positions whenever
    # max(clip) < 8 (e.g. for samples that only reference frames 1..4), producing frame
    # locations that diverge from the TwiFF reference implementation.
    num_frames = 8
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    pool = _get_frame_indices_uni(num_frames, vlen, start_frames=1, end_frames=1)
    selected = [pool[i - 1] for i in clip_indices]
    raw = video_reader.get_batch(selected).asnumpy()
    return [Image.fromarray(raw[i]) for i in range(raw.shape[0])]


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

class TwiFFSFTDataset(Dataset):
    """
    Map-style dataset for TwiFF-format JSON files.

    Expected JSON fields per sample:
        conversations  – list of {from, value} dicts
        video          – absolute path to the video file
        image          – list of 1-based frame indices for the question image(s)
        reasoning_image– list of 1-based frame indices for the reasoning image(s)
        answer         – final answer string
        cot            – (optional) raw CoT string
    """

    def __init__(self, data_root: Union[str, List[str]]):
        super().__init__()
        self.raw_dataset = self._load_from_source(data_root)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, i: int):
        return self.raw_dataset[i]

    def _collect_json_files(self, path: Path) -> List[str]:
        if not path.exists():
            logging.warning(f"Path does not exist, skipping: {path}")
            return []
        if path.is_dir():
            found = [str(p) for p in path.glob("*.json") if p.is_file()]
            if not found:
                logging.warning(f"No .json files in directory: {path}")
            return found
        if path.is_file() and path.suffix == ".json":
            return [str(path)]
        logging.warning(f"Not a valid .json file or directory: {path}")
        return []

    def _load_from_source(self, data_root: Union[str, List[str]]):
        if isinstance(data_root, str):
            paths = (
                data_root.split(",")
                if "," in data_root and not os.path.exists(data_root)
                else [data_root]
            )
        elif isinstance(data_root, list):
            paths = data_root
        else:
            raise TypeError(f"Unsupported data_root type: {type(data_root)}")

        all_json_files: List[str] = []
        for p in paths:
            all_json_files.extend(self._collect_json_files(Path(p.strip())))

        if not all_json_files:
            raise ValueError("No valid .json files found in the provided sources.")

        unique_files = sorted(set(all_json_files))
        logging.info(f"TwiFFSFTDataset: loading from {len(unique_files)} JSON file(s).")

        def normalize_messages(messages):
            roles = []
            contents = []
            if isinstance(messages, dict):
                messages = [
                    {"role": r, "content": c}
                    for r, c in zip(messages.get("role", []), messages.get("content", []))
                ]
            for msg in messages or []:
                role = str(msg.get("role", ""))
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(str(part.get("text", "")))
                    content = "".join(text_parts)
                roles.append(role)
                contents.append(str(content))
            return {"role": roles, "content": contents}

        def normalize_conversations(conversations):
            from_values = []
            value_values = []
            if isinstance(conversations, dict):
                conversations = [
                    {"from": f, "value": v}
                    for f, v in zip(conversations.get("from", []), conversations.get("value", []))
                ]
            for turn in conversations or []:
                from_values.append(str(turn.get("from", "")))
                value_values.append(str(turn.get("value", "")))
            return {"from": from_values, "value": value_values}

        def resolve_video_path(video_path: str, base_dir: Path) -> str:
            if video_path and not os.path.isabs(video_path):
                return str(base_dir / video_path)
            return video_path or ""

        def gen():
            for file_path in unique_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    base_dir = Path(file_path).parent
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        if "reasoning_image" in item:
                            video = resolve_video_path(str(item.get("video", "")), base_dir)
                            yield {
                                "source_format": "twiff_frames",
                                "conversations": normalize_conversations(item.get("conversations", [])),
                                "messages": {"role": [], "content": []},
                                "videos": [video] if video else [],
                                "images": [],
                                "video": video,
                                "image": [int(x) for x in item.get("image", [])],
                                "reasoning_image": [int(x) for x in item.get("reasoning_image", [])],
                                "answer": str(item.get("answer", "")),
                                "cot": str(item.get("cot", "")),
                            }
                        elif "messages" in item and "videos" in item:
                            videos = [resolve_video_path(str(v), base_dir) for v in item.get("videos", [])]
                            images = [resolve_video_path(str(v), base_dir) for v in item.get("images", [])]
                            yield {
                                "source_format": "chat_video_distill",
                                "conversations": {"from": [], "value": []},
                                "messages": normalize_messages(item.get("messages", [])),
                                "videos": videos,
                                "images": images,
                                "video": videos[0] if videos else "",
                                "image": [],
                                "reasoning_image": [],
                                "answer": "",
                                "cot": "",
                            }
                        else:
                            logging.warning(
                                f"Sample in {file_path} has neither 'reasoning_image' nor 'messages'/'videos', skipping."
                            )
                except Exception as e:
                    logging.warning(f"Error reading {file_path}: {e}")
                    continue

        features = Features(
            {
                "source_format": Value("string"),
                "conversations": Sequence(
                    feature={"from": Value("string"), "value": Value("string")}
                ),
                "messages": Sequence(
                    feature={"role": Value("string"), "content": Value("string")}
                ),
                "videos": Sequence(Value("string")),
                "images": Sequence(Value("string")),
                "video": Value("string"),
                "image": Sequence(Value("int64")),
                "reasoning_image": Sequence(Value("int64")),
                "answer": Value("string"),
                "cot": Value("string"),
            }
        )

        dataset = HFDataset.from_generator(gen, features=features)
        logging.info(f"TwiFFSFTDataset: loaded {len(dataset)} samples.")
        return dataset


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #

def _shuffle_frames(frames: List[Image.Image], seed: int, key: str) -> List[Image.Image]:
    digest = hashlib.md5(key.encode("utf-8")).digest()
    salt = int.from_bytes(digest[:4], byteorder="little", signed=False)
    rng = random.Random(seed + salt)
    out = list(frames)
    rng.shuffle(out)
    return out


def _iter_role_content(messages):
    if isinstance(messages, dict):
        roles = messages.get("role", [])
        contents = messages.get("content", [])
        return [{"role": r, "content": c} for r, c in zip(roles, contents)]
    return messages or []


def _first_message(messages, role: str) -> Optional[str]:
    for msg in _iter_role_content(messages):
        if msg.get("role") == role:
            return msg.get("content", "")
    return None


def _build_video_content(video_path, max_pixels, min_pixels, fps, nframes, resized_width, resized_height):
    vc = {
        "type": "video",
        "video": video_path,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
    }
    if nframes is not None:
        try:
            total_frames = len(decord.VideoReader(video_path, num_threads=1))
            nframes = min(nframes, total_frames)
        except Exception:
            pass
        vc["nframes"] = nframes
    elif fps is not None:
        vc["fps"] = fps
    if resized_width is not None:
        vc["resized_width"] = resized_width
    if resized_height is not None:
        vc["resized_height"] = resized_height
    return vc


def video_distill_preprocess_function(
    example,
    max_pixels: int = 5120 * 32 * 32,
    min_pixels: int = 128 * 32 * 32,
    system_message: str = SYSTEM_MESSAGE,
    fps: Optional[float] = None,
    nframes: Optional[int] = None,
    resized_width: Optional[int] = None,
    resized_height: Optional[int] = None,
):
    messages = example.get("messages", [])
    system_text = _first_message(messages, "system") or system_message
    user_text = _first_message(messages, "user")
    assistant_text = _first_message(messages, "assistant")
    video_paths = list(example.get("videos", []))
    image_paths = list(example.get("images", []))

    if not user_text or not assistant_text:
        logging.warning("Missing user or assistant turn in chat_video_distill sample, skipping.")
        return None

    vid_idx = 0
    img_idx = 0
    user_content = []

    for part in re.split(r"(<video>|<image>)", user_text):
        part = part.strip()
        if not part:
            continue
        if part == "<video>":
            if vid_idx < len(video_paths):
                user_content.append(_build_video_content(
                    video_paths[vid_idx], max_pixels, min_pixels,
                    fps, nframes, resized_width, resized_height,
                ))
                vid_idx += 1
            else:
                logging.warning("<video> in user text but no video path available.")
        elif part == "<image>":
            if img_idx < len(image_paths):
                try:
                    img = Image.open(image_paths[img_idx]).convert("RGB")
                    user_content.append({
                        "type": "image",
                        "image": img,
                        "max_pixels": max_pixels,
                        "min_pixels": min_pixels,
                    })
                    img_idx += 1
                except Exception as e:
                    logging.warning(f"Failed to load image '{image_paths[img_idx]}': {e}")
                    img_idx += 1
            else:
                logging.warning("<image> in user text but no image path available.")
        else:
            user_content.append({"type": "text", "text": part})

    if not user_content:
        logging.warning("Empty user_content after parsing, skipping.")
        return None

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def twiff_cot_preprocess_function(
    example,
    max_pixels: int = 5120 * 32 * 32,
    min_pixels: int = 128 * 32 * 32,
    latent_max_pixels: int = 64 * 32 * 32,
    system_message: str = SYSTEM_MESSAGE,
    shuffle_latent_images: bool = False,
    shuffle_seed=None,
    response_mode: str = "twiff",
):
    """
    TwiFF variant of cot_preprocess_function.

    Reads the 'video', 'image' (frame indices), and 'reasoning_image' (frame
    indices) fields to extract PIL frames on the fly via decord, then builds
    the same message structure as cot_preprocess_function.

    GPT-turn text segments are wrapped with <reason>...</reason> tags.
    """
    mode = (response_mode or "twiff").strip().lower()
    if mode not in {"twiff", "cot"}:
        raise ValueError(f"Unsupported response_mode: {response_mode}. Expected one of: twiff, cot.")

    video_path = example.get("video", "")
    image_indices = list(example.get("image", []))
    reasoning_indices = list(example.get("reasoning_image", []))

    frame_indices = image_indices + reasoning_indices if mode == "twiff" else image_indices
    frames = []
    if frame_indices:
        try:
            frames = read_frames_by_clip(video_path, frame_indices)
        except Exception as e:
            logging.warning(f"Failed to load frames from '{video_path}': {e}")
            return None
    elif mode == "twiff":
        logging.warning(f"No frame indices for video '{video_path}', skipping twiff sample.")
        return None

    question_frames = frames[: len(image_indices)] if frames else []
    reasoning_frames = frames[len(image_indices) :] if mode == "twiff" and frames else []

    # Normalise conversations
    conversations = example.get("conversations", [])
    if isinstance(conversations, dict):
        try:
            keys = list(conversations.keys())
            length = len(conversations[keys[0]])
            conversations = [{k: conversations[k][i] for k in keys} for i in range(length)]
        except Exception as e:
            logging.error(f"Failed to normalise conversations: {e}")
            return None

    human_turn = next((t for t in conversations if t.get("from") == "human"), None)
    gpt_turn = next((t for t in conversations if t.get("from") == "gpt"), None)

    if not human_turn or not gpt_turn:
        logging.warning("Missing 'human' or 'gpt' turn, skipping.")
        return None

    # ---- User content ----
    user_content = []
    question_text = human_turn.get("value", "")
    q_frame_idx = 0

    for part in re.split(r"(<image>)", question_text):
        part = part.strip()
        if not part:
            continue
        if part == "<image>":
            if q_frame_idx < len(question_frames):
                user_content.append(
                    {
                        "type": "image",
                        "image": question_frames[q_frame_idx],
                        "max_pixels": max_pixels,
                        "min_pixels": min_pixels,
                    }
                )
                q_frame_idx += 1
            else:
                logging.warning("<image> in human turn but no question frame available.")
        else:
            user_content.append({"type": "text", "text": part})

    # ---- Assistant content ----
    assistant_content = []
    reasoning_text = (example.get("cot", "") or "").strip() if mode == "cot" else gpt_turn.get("value", "")
    if mode == "cot" and not reasoning_text:
        reasoning_text = gpt_turn.get("value", "") if gpt_turn else ""
    final_answer = (example.get("answer", "") or "").strip()

    if mode == "cot":
        reasoning_parts = re.split(r"<image>", reasoning_text)
        text_parts = []
        for part in reasoning_parts:
            cleaned = re.sub(r"THOUGHT \d+:\s*", "", part).strip()
            if cleaned:
                text_parts.append(cleaned)
        combined_reason = "\n".join(text_parts).strip()
        if not combined_reason or not final_answer:
            return None
        assistant_content.append({"type": "text", "text": f"<reason>{combined_reason}</reason>\n"})
        assistant_content.append({"type": "text", "text": f"<answer>{final_answer}</answer>"})
    else:
        if shuffle_latent_images and reasoning_frames:
            seed = int(shuffle_seed) if shuffle_seed is not None else 0
            key = gpt_turn.get("value", "") + "|" + video_path + "|" + str(reasoning_indices)
            reasoning_frames = _shuffle_frames(reasoning_frames, seed=seed, key=key)

        r_frame_idx = 0
        img_latent_min = min(min_pixels, latent_max_pixels)

        for part in re.split(r"(<image>)", reasoning_text):
            part = part.strip()
            if not part:
                continue
            if part == "<image>":
                if r_frame_idx < len(reasoning_frames):
                    assistant_content.append(
                        {
                            "type": "image",
                            "image": reasoning_frames[r_frame_idx],
                            "max_pixels": latent_max_pixels,
                            "min_pixels": img_latent_min,
                        }
                    )
                    assistant_content.append({"type": "text", "text": "\n"})
                    r_frame_idx += 1
                else:
                    logging.warning("<image> in gpt turn but no reasoning frame available.")
            else:
                cleaned = re.sub(r"THOUGHT \d+:\s*", "", part).strip()
                if cleaned:
                    assistant_content.append(
                        {"type": "text", "text": f"<reason>{cleaned}</reason>\n"}
                    )

        if final_answer:
            assistant_content.append(
                {"type": "text", "text": f"<answer>{final_answer}</answer>"}
            )

    if not user_content or not assistant_content:
        logging.warning("Empty user_content or assistant_content, skipping.")
        return None

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# --------------------------------------------------------------------------- #
# Collator                                                                     #
# --------------------------------------------------------------------------- #

class TwiFFDataCollator:
    """
    Data collator for TwiFF-format data.

    Mirrors FutureL1DataCollator but calls twiff_cot_preprocess_function to
    extract frames from video files on the fly instead of loading pre-saved images.
    """

    def __init__(self, processor, args):
        self.processor = processor
        self.args = args
        self.system_message = resolve_system_message(args)

        self.latent_token_idx = processor.tokenizer("<|latent|>", return_tensors="pt")["input_ids"][0]
        self.latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        self.latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        self.pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
        self.answer_start_token_pattern = processor.tokenizer(
            "<|im_start|>assistant", return_tensors="pt"
        )["input_ids"][0]
        self.recon_max_text_len = int(getattr(args, "decoder_recon_max_text_len", 0) or 0)
        self._debug_ce_batches_left = _parse_debug_ce_budget()
        if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) != 0:
            self._debug_ce_batches_left = 0

    def _debug_print_ce_tokens(self, batch):
        """Print tokens where labels != -100 (matching CE targets after HF shift)."""
        if self._debug_ce_batches_left <= 0:
            return
        tokenizer = self.processor.tokenizer
        labels = batch["labels"]
        bsz = labels.size(0)
        print("\n[FUTURE_L1_DEBUG_CE_TOKENS] batch ce-token dump (sample 0 of {})".format(bsz), flush=True)
        row_l = labels[0]
        row_shift = row_l[1:]
        ce_mask = row_shift != -100
        ce_labels = row_shift[ce_mask]
        n_raw = int((row_l != -100).sum().item())
        n = int(ce_mask.sum().item())
        print(
            f"  seq_len={row_l.size(0)} labels!=-100_count={n_raw} "
            f"ce_targets_after_shift(i>=1)={n}",
            flush=True,
        )
        if n > 0:
            tid_list = ce_labels.detach().cpu().tolist()
            preview_n = min(96, len(tid_list))
            print(f"  first_{preview_n}_ce_target_token_ids: {tid_list[:preview_n]}", flush=True)
            try:
                text = tokenizer.decode(tid_list, skip_special_tokens=False)
                cap = 3000
                shown = text if len(text) <= cap else text[:cap] + f"... [truncated, total_chars={len(text)}]"
                print(f"  decoded_ce_targets:\n{shown}", flush=True)
            except Exception as e:
                print(f"  decode_failed: {e}", flush=True)
        else:
            print("  (no CE targets in sample 0 after shift — check labels)", flush=True)
        print("[/FUTURE_L1_DEBUG_CE_TOKENS]\n", flush=True)
        self._debug_ce_batches_left -= 1

    def _collate_twiff_frames(self, raw_examples):
        examples = [
            twiff_cot_preprocess_function(
                ex,
                self.args.image_max_pixels,
                self.args.image_min_pixels,
                self.args.max_latent_token * 32 * 32,
                self.system_message,
                getattr(self.args, "shuffle_latent_images", False),
                getattr(self.args, "random_seed", None),
            )
            for ex in raw_examples
        ]
        examples = [ex for ex in examples if ex is not None]
        if not examples:
            return {}

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in examples]

        merge_latent = bool(getattr(self.args, "merge_latent_segments", False))
        if merge_latent:
            texts = replace_visual_spectial_tokens_merged(texts)
        else:
            texts = replace_visual_spectial_tokens(texts)

        image_inputs, _ = process_vision_info(examples, image_patch_size=16)

        user_examples = remove_assistant_images(examples)
        user_texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in user_examples]
        user_image_inputs, _ = process_vision_info(user_examples, image_patch_size=16)

        assistant_examples = remove_user_images(examples)
        assistant_texts = [
            self.processor.apply_chat_template(ex, tokenize=False) for ex in assistant_examples
        ]
        if merge_latent:
            assistant_texts = replace_visual_spectial_tokens_merged(assistant_texts)
        else:
            assistant_texts = replace_visual_spectial_tokens(assistant_texts)
        assistant_image_inputs, _ = process_vision_info(assistant_examples, image_patch_size=16)

        user_batch = self.processor(
            text=user_texts, images=user_image_inputs, return_tensors="pt", padding=True
        )
        assistant_batch = self.processor(
            text=assistant_texts, images=assistant_image_inputs, return_tensors="pt", padding=True
        )
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        batch["pixel_values"] = user_batch.get("pixel_values", None)
        batch["image_grid_thw"] = user_batch.get("image_grid_thw", None)
        batch["pixel_values_latent"] = assistant_batch.get("pixel_values", None)
        batch["image_grid_thw_latent"] = assistant_batch.get("image_grid_thw", None)

        new_input_ids, new_attention_mask, new_mm = collator_replace_latent(
            batch["input_ids"],
            batch["attention_mask"],
            self.latent_start_idx,
            self.latent_end_idx,
            self.latent_token_idx,
            self.answer_start_token_pattern,
            self.pad_token_idx,
            self.args,
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )

        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention_mask
        if new_mm is not None:
            batch["mm_token_type_ids"] = new_mm

        labels = generate_labels_after_multi_token_start(
            batch["input_ids"],
            self.answer_start_token_pattern,
            self.pad_token_idx,
            self.latent_token_idx,
        )
        batch["labels"] = labels
        if self._debug_ce_batches_left > 0:
            self._debug_print_ce_tokens(batch)

        if batch["pixel_values_latent"] is not None:
            image_out_mask = mask_image_output_tokens(
                batch["input_ids"], self.latent_start_idx, self.latent_token_idx
            )
            batch["image_out_mask"] = image_out_mask

        return batch

    def _process_video_vision_info(self, examples):
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                examples,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            image_inputs, video_inputs = process_vision_info(examples, image_patch_size=16)
            return image_inputs, video_inputs, None, {}

        video_metadata = None
        if video_inputs:
            first_video = video_inputs[0]
            if isinstance(first_video, tuple) and len(first_video) == 2:
                video_inputs, video_metadata = map(list, zip(*video_inputs))
        return image_inputs, video_inputs, video_metadata, video_kwargs or {}

    def _collate_video_distill(self, raw_examples):
        examples = [
            video_distill_preprocess_function(
                ex,
                getattr(self.args, "video_max_pixels", self.args.image_max_pixels),
                getattr(self.args, "video_min_pixels", self.args.image_min_pixels),
                self.system_message,
                getattr(self.args, "fps", None),
                getattr(self.args, "nframes", None),
                getattr(self.args, "video_resized_width", None),
                getattr(self.args, "video_resized_height", None),
            )
            for ex in raw_examples
        ]
        examples = [ex for ex in examples if ex is not None]
        if not examples:
            return {}

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in examples]
        image_inputs, video_inputs, video_metadata, video_kwargs = self._process_video_vision_info(examples)

        processor_kwargs = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "return_tensors": "pt",
            "padding": True,
        }
        if video_metadata is not None:
            processor_kwargs["video_metadata"] = video_metadata
            processor_kwargs["do_resize"] = False
        processor_kwargs.update(video_kwargs)

        batch = self.processor(**processor_kwargs)
        batch["labels"] = generate_labels_after_multi_token_start(
            batch["input_ids"],
            self.answer_start_token_pattern,
            pad_token_idx=self.pad_token_idx,
            img_token_indices=[],
        )
        if self._debug_ce_batches_left > 0:
            self._debug_print_ce_tokens(batch)
        return batch

    def _collate_mixed_formats(self, raw_examples):
        examples = []
        twiff_indices = []
        for idx, ex in enumerate(raw_examples):
            if self._is_twiff_frame_sample(ex):
                processed = twiff_cot_preprocess_function(
                    ex,
                    self.args.image_max_pixels,
                    self.args.image_min_pixels,
                    self.args.max_latent_token * 32 * 32,
                    self.system_message,
                    getattr(self.args, "shuffle_latent_images", False),
                    getattr(self.args, "random_seed", None),
                )
                if processed is not None:
                    twiff_indices.append(len(examples))
                    examples.append(processed)
            else:
                processed = video_distill_preprocess_function(
                    ex,
                    getattr(self.args, "video_max_pixels", self.args.image_max_pixels),
                    getattr(self.args, "video_min_pixels", self.args.image_min_pixels),
                    self.system_message,
                    getattr(self.args, "fps", None),
                    getattr(self.args, "nframes", None),
                    getattr(self.args, "video_resized_width", None),
                    getattr(self.args, "video_resized_height", None),
                )
                if processed is not None:
                    examples.append(processed)

        if not examples:
            return {}

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in examples]
        merge_latent = bool(getattr(self.args, "merge_latent_segments", False))
        if twiff_indices:
            if merge_latent:
                texts = replace_visual_spectial_tokens_merged(texts)
            else:
                texts = replace_visual_spectial_tokens(texts)

        image_inputs, video_inputs, video_metadata, video_kwargs = self._process_video_vision_info(examples)
        processor_kwargs = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "return_tensors": "pt",
            "padding": True,
        }
        if video_metadata is not None:
            processor_kwargs["video_metadata"] = video_metadata
            processor_kwargs["do_resize"] = False
        processor_kwargs.update(video_kwargs)
        batch = self.processor(**processor_kwargs)

        if twiff_indices:
            assistant_examples = remove_user_images([examples[i] for i in twiff_indices])
            assistant_texts = [
                self.processor.apply_chat_template(ex, tokenize=False) for ex in assistant_examples
            ]
            if merge_latent:
                assistant_texts = replace_visual_spectial_tokens_merged(assistant_texts)
            else:
                assistant_texts = replace_visual_spectial_tokens(assistant_texts)
            assistant_image_inputs, _ = process_vision_info(assistant_examples, image_patch_size=16)
            assistant_batch = self.processor(
                text=assistant_texts, images=assistant_image_inputs, return_tensors="pt", padding=True
            )
            batch["pixel_values_latent"] = assistant_batch.get("pixel_values", None)
            batch["image_grid_thw_latent"] = assistant_batch.get("image_grid_thw", None)

            new_input_ids, new_attention_mask, new_mm = collator_replace_latent(
                batch["input_ids"],
                batch["attention_mask"],
                self.latent_start_idx,
                self.latent_end_idx,
                self.latent_token_idx,
                self.answer_start_token_pattern,
                self.pad_token_idx,
                self.args,
                mm_token_type_ids=batch.get("mm_token_type_ids"),
            )
            batch["input_ids"] = new_input_ids
            batch["attention_mask"] = new_attention_mask
            if new_mm is not None:
                batch["mm_token_type_ids"] = new_mm
            batch["image_out_mask"] = mask_image_output_tokens(
                batch["input_ids"], self.latent_start_idx, self.latent_token_idx
            )

        batch["labels"] = generate_labels_after_multi_token_start(
            batch["input_ids"],
            self.answer_start_token_pattern,
            self.pad_token_idx,
            self.latent_token_idx,
        )
        if self._debug_ce_batches_left > 0:
            self._debug_print_ce_tokens(batch)
        return batch

    def _is_twiff_frame_sample(self, ex) -> bool:
        source_format = ex.get("source_format")
        if source_format:
            return source_format == "twiff_frames"
        return "reasoning_image" in ex

    def __call__(self, raw_examples):
        if all(self._is_twiff_frame_sample(ex) for ex in raw_examples):
            return self._collate_twiff_frames(raw_examples)
        if all(not self._is_twiff_frame_sample(ex) for ex in raw_examples):
            return self._collate_video_distill(raw_examples)
        return self._collate_mixed_formats(raw_examples)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def make_supervised_data_module_twiff(processor, args):
    """Make dataset and collator for TwiFF-format video-frame training."""
    dataset = TwiFFSFTDataset(data_root=args.data_path)

    raw_dataset = dataset.raw_dataset

    start_ratio = float(getattr(args, "start_ratio", 0.0) or 0.0)
    if start_ratio != 0.0:
        if not 0.0 <= start_ratio < 1.0:
            raise ValueError(f"start_ratio must be in [0.0, 1.0), got {start_ratio}")
        total = len(raw_dataset)
        start_idx = int(total * start_ratio)
        if start_idx >= total:
            raise ValueError(
                f"start_ratio={start_ratio} results in empty dataset "
                f"(len={total}, start_idx={start_idx})."
            )
        raw_dataset = raw_dataset.select(list(range(start_idx, total)))

    twiff_cot_ratio = float(getattr(args, "twiff_cot_ratio", 0.0) or 0.0)
    if not 0.0 <= twiff_cot_ratio <= 1.0:
        raise ValueError(f"twiff_cot_ratio must be in [0.0, 1.0], got {twiff_cot_ratio}")
    if twiff_cot_ratio > 0.0:
        total = len(raw_dataset)
        cot_count = int(round(total * twiff_cot_ratio))
        seed = int(getattr(args, "random_seed", 0) or 0)
        indices = list(range(total))
        random.Random(seed).shuffle(indices)
        cot_index_set = set(indices[:cot_count])
        response_modes = ["cot" if i in cot_index_set else "twiff" for i in range(total)]
        raw_dataset = raw_dataset.add_column("twiff_response_mode", response_modes)
        logging.info(
            f"TwiFF mix: assigned {cot_count}/{total} samples ({twiff_cot_ratio:.3f}) to vanilla CoT text-only format"
        )

    dataset.raw_dataset = raw_dataset
    data_collator = TwiFFDataCollator(processor=processor, args=args)
    return dict(train_dataset=dataset, eval_dataset=None, data_collator=data_collator)
