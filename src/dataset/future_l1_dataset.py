import re
import os
import random
import hashlib
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset, Subset
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset, Features, Value, Sequence
from qwen_vl_utils import process_vision_info
from .data_utils import *
from src.constants import SYSTEM_MESSAGE, ROT_SYSTEM_MESSAGE, PLAIN_SYSTEM_MESSAGE


def resolve_system_message(args):
    mode = getattr(args, "system_message_mode", "default")
    mode = (mode or "default").strip().lower()

    if mode == "default":
        return SYSTEM_MESSAGE
    if mode == "rot":
        return ROT_SYSTEM_MESSAGE
    if mode == "plain":
        return PLAIN_SYSTEM_MESSAGE
    if mode == "none":
        return ""
    if mode == "custom":
        custom_path = getattr(args, "custom_system_message_path", None)
        if not custom_path:
            raise ValueError("system_message_mode=custom requires --custom_system_message_path")
        if not os.path.exists(custom_path):
            raise FileNotFoundError(f"Custom system message file not found: {custom_path}")
        with open(custom_path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError(
        f"Unsupported system_message_mode: {mode}. Choices: default, rot, none, custom."
    )


# ========== Dataset Class ==========
class FutureL1SFTDataset(Dataset):
    def __init__(self, data_root: Union[str, List[str]]):
        super().__init__()
        self.raw_dataset = self._load_from_source(data_root)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, i: int):
        """Returns the raw sample at the given index, without preprocessing."""
        return self.raw_dataset[i]

    def _collect_json_files(self, path: Path) -> List[str]:
        """Helper to recursively find .json files from a given path."""
        if not path.exists():
            logging.warning(f"Path does not exist, skipping: {path}")
            return []

        if path.is_dir():
            # Find all .json files in the directory.
            found_files = [str(p) for p in path.glob('*.json') if p.is_file()]
            if not found_files:
                logging.warning(f"No .json files found in directory: {path}")
            return found_files

        if path.is_file() and path.suffix == '.json':
            return [str(path)]

        logging.warning(f"Path is not a valid .json file or directory, skipping: {path}")
        return []

    def _load_from_source(self, data_root: Union[str, List[str]]):
        """Main method to parse the data source and load the dataset."""
        # 1. Normalize the input into a list of strings
        if isinstance(data_root, str):
            # Split if it's a comma-separated string, otherwise wrap in a list
            paths_to_process = data_root.split(',') if ',' in data_root and not os.path.exists(data_root) else [
                data_root]
        elif isinstance(data_root, list):
            paths_to_process = data_root
        else:
            raise TypeError(f"Unsupported data_root type: {type(data_root)}. Must be str or list.")

        # 2. Collect all JSON files from all paths
        all_json_files = []
        for path_str in paths_to_process:
            path = Path(path_str.strip())
            all_json_files.extend(self._collect_json_files(path))

        # 3. Ensure we found at least one file
        if not all_json_files:
            raise ValueError("No valid .json files were found in any of the provided sources.")

        unique_files = sorted(list(set(all_json_files)))
        logging.info(f"Loading data from {len(unique_files)} unique JSON file(s).")

        # 4. Define the Generator Function
        # This function reads files one by one and yields standardized dictionaries
        def gen():
            for file_path in unique_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    base_dir = Path(file_path).parent

                    # Handle case where JSON is a list of objects or a single object
                    if isinstance(data, dict):
                        data = [data]

                    def _resolve_paths(paths):
                        """Resolve image paths to absolute paths based on the JSON file location."""
                        resolved = []
                        for p in paths or []:
                            p = str(p).strip()
                            if not p:
                                continue
                            if not os.path.isabs(p):
                                resolved.append(str(base_dir / p))
                            else:
                                resolved.append(p)
                        return resolved

                    for item in data:
                        # Extract only necessary fields and handle missing ones
                        yield {
                            "conversations": item.get("conversations", []),
                            "image": _resolve_paths(item.get("image", [])),  # Default to empty list
                            "reasoning_image": _resolve_paths(item.get("reasoning_image", [])),  # Default to empty list
                            "answer": item.get("answer", ""),
                            "cot": item.get("cot", ""),
                            "sketch": item.get("sketch", ""),
                        }
                except Exception as e:
                    logging.warning(f"Error reading file {file_path}: {e}")
                    continue

        # 5. Define Explicit Features (Schema)
        # This prevents errors if the first example has an empty list and Arrow can't infer the type
        features = Features({
            'conversations': Sequence(feature={
                'from': Value('string'),
                'value': Value('string')
            }),
            'image': Sequence(Value('string')),
            'reasoning_image': Sequence(Value('string')),
            'answer': Value('string'),
            'cot': Value('string'),
            'sketch': Value('string'),
        })

        # 6. Create Dataset from Generator
        try:
            combined_dataset = HFDataset.from_generator(gen, features=features)
            logging.info(f"Successfully loaded a total of {len(combined_dataset)} samples.")
            return combined_dataset
        except Exception as e:
            raise IOError(f"Failed to load dataset from JSON files.") from e

# ========== data processing ==========
def _deterministic_shuffle_list(items: List[str], seed: int, key: str) -> List[str]:
    if not items:
        return items
    # Stable per-sample shuffle: seed + md5(key) -> PRNG -> shuffle(copy)
    digest = hashlib.md5(key.encode("utf-8")).digest()
    salt = int.from_bytes(digest[:4], byteorder="little", signed=False)
    rng = random.Random(seed + salt)
    out = list(items)
    rng.shuffle(out)
    return out


def _is_render_image(path: str) -> bool:
    """Return True if the image path belongs to the 'render' family.

    Rule: after lowercasing the full path, if it contains substring "render" => render,
    otherwise => orig.
    """
    p = (path or "").lower()
    return "render" in p


def cot_preprocess_function(
    example,
    max_pixels=5120 * 32 * 32,
    min_pixels=128 * 32 * 32,
    latent_max_pixels=64 * 32 * 32,
    system_message=SYSTEM_MESSAGE,
    shuffle_latent_images: bool = False,
    shuffle_seed: int | None = None,
    latent_max_pixels_orig=None,
    latent_max_pixels_render=None,
):
    """
    Converts the JSON format to the required format for multimodal training.

    Input format (example):
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "Question text with <image> placeholders..."},
            {"from": "gpt", "value": "Reasoning text with <image> placeholders..."}
        ],
        "image": ["path/to/question_image_1.png", ...],
        "reasoning_image": ["path/to/reasoning_image_1.png", ...],
        "answer": "Final answer string"
    }

    Output format:
    [
        {"role": "user", "content": [...]},
        {"role": "assistant", "content": [...]}
    ]
    """
    conversations = example.get('conversations', []) 
    if isinstance(conversations, dict):
        try:
            keys = list(conversations.keys()) 
            length = len(conversations[keys[0]]) 
            new_conversations = [] 
            for i in range(length): 
                turn = {k: conversations[k][i] for k in keys}
                new_conversations.append(turn)
            conversations = new_conversations 
        except Exception as e: 
            print(example)
            logging.error(f"Failed to normalize conversations for ID {example.get('id', 'N/A')}: {e}")
            return None

    # 1. Separate human and gpt content from 'conversations'
    human_turn = None
    gpt_turn = None
    for turn in conversations:
        if turn.get('from') == 'human':
            human_turn = turn
        elif turn.get('from') == 'gpt':
            gpt_turn = turn

    if not human_turn or not gpt_turn:
        logging.warning(f"Sample {example.get('id', 'N/A')} is missing 'human' or 'gpt' turn, skipping.")
        return None # Return None to be filtered out later

    # ==================== 2. Process User Content ====================
    user_content = []
    question_text = human_turn.get('value', '')
    question_image_paths = example.get('image', [])
    
    # Use re.split and capture the delimiter to easily interleave text and images.
    # e.g., "text1<image>text2" -> ['text1', '<image>', 'text2']
    question_parts = re.split(r'(<image>)', question_text)
    
    question_image_idx = 0
    for part in question_parts:
        part = part.strip()
        if not part:
            continue
        
        if part == '<image>':
            # This is an image placeholder
            if question_image_idx < len(question_image_paths):
                img_path = question_image_paths[question_image_idx]
                try:
                    # The JSON contains image paths, so we need to load them with Pillow.
                    #image_data = Image.open(img_path).convert('RGB')
                    user_content.append({
                        "type": "image",
                        "image": img_path,
                        "max_pixels": max_pixels,
                        "min_pixels": min_pixels
                    })
                    question_image_idx += 1
                except FileNotFoundError:
                    logging.warning(f"User image not found: {img_path}")
                except Exception as e:
                    logging.error(f"Error loading user image {img_path}: {e}")
            else:
                logging.warning(f"An <image> tag was found in text, but there are not enough image paths in the 'image' list.")
        else:
            # This is a text part
            user_content.append({"type": "text", "text": part})

    # ================== 3. Process Assistant Content ==================
    assistant_content = []
    reasoning_text = gpt_turn.get('value', '')
    reasoning_image_paths = example.get('reasoning_image', [])
    if shuffle_latent_images and reasoning_image_paths:
        base_seed = int(shuffle_seed) if shuffle_seed is not None else 0
        shuffle_key = reasoning_text + "\n" + "|".join(map(str, reasoning_image_paths))
        reasoning_image_paths = _deterministic_shuffle_list(reasoning_image_paths, seed=base_seed, key=shuffle_key)

    reasoning_parts = re.split(r'(<image>)', reasoning_text)
    reasoning_image_idx = 0

    for part in reasoning_parts:
        part = part.strip()
        if not part:
            continue
        
        if part == '<image>':
            # This is a reasoning image placeholder
            if reasoning_image_idx < len(reasoning_image_paths):
                img_path = reasoning_image_paths[reasoning_image_idx]
                try:
                    # Per-image latent_max_pixels when dual-MSE orig/render are specified
                    if latent_max_pixels_orig is not None and latent_max_pixels_render is not None:
                        img_latent_max = latent_max_pixels_render if _is_render_image(img_path) else latent_max_pixels_orig
                    else:
                        img_latent_max = latent_max_pixels
                    # Keep latent min_pixels <= max_pixels for very small latent budgets (e.g., max_latent_token=1).
                    img_latent_min = min(min_pixels, img_latent_max)
                    assistant_content.append({
                        "type": "image",
                        "image": img_path,
                        "max_pixels": img_latent_max,
                        "min_pixels": img_latent_min,
                    })
                    # assistant_content.insert(-1, {"type": "text", "text": "\n"})
                    assistant_content.append({"type": "text", "text": "\n"})
                    reasoning_image_idx += 1
                except FileNotFoundError:
                    logging.warning(f"Reasoning image not found: {img_path}")
                except Exception as e:
                    logging.error(f"Error loading reasoning image {img_path}: {e}")
            else:
                logging.warning(f"An <image> tag was found in the GPT response, but there are not enough image paths in the 'reasoning_image' list.")
        else:
            # This is a reasoning text part
            # Remove "THOUGHT x: " tags
            cleaned_text = re.sub(r'THOUGHT \d+:\s*', '', part).strip()
            # cleaned_text = cleaned_text.replace('\n\n', '\n')
            if cleaned_text:
                assistant_content.append({
                    "type": "text",
                    # Wrap with <reason> tag
                    "text": f"<reason>{cleaned_text}</reason>\n"
                })

    # 4. Append the final answer
    final_answer = example.get('answer', '')
    if final_answer:
        assistant_content.append({
            "type": "text",
            "text": f"<answer>{final_answer}</answer>"
        })

    # 5. Assemble and return the final result
    if not user_content or not assistant_content:
        logging.warning(f"user_content or assistant_content is empty after processing, skipping sample {example.get('id', 'N/A')}")
        return None
        
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


def _parse_debug_ce_budget() -> int:
    """FUTURE_L1_DEBUG_CE_TOKENS=1|true 时打印前 FUTURE_L1_DEBUG_CE_TOKEN_BATCHES 个 batch（默认 2）；也可设为数字表示批次数。"""
    raw = os.environ.get("FUTURE_L1_DEBUG_CE_TOKENS", "").strip().lower()
    if raw in ("", "0", "false", "no"):
        return 0
    if raw in ("1", "true", "yes"):
        return max(0, int(os.environ.get("FUTURE_L1_DEBUG_CE_TOKEN_BATCHES", "2")))
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


# ==========  Collator ==========
class FutureL1DataCollator:
    
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args
        self.system_message = resolve_system_message(args)

        # Precompute token IDs once
        self.latent_token_idx = processor.tokenizer("<|latent|>", return_tensors="pt")["input_ids"][0]
        self.latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        self.latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        self.pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
        self.answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
        # Aux-decoder reconstruction: how many CoT tokens to keep per sample
        # (0 disables CoT tokenization entirely; original pipeline unchanged).
        self.recon_max_text_len = int(getattr(args, "decoder_recon_max_text_len", 0) or 0)
        self._debug_ce_batches_left = _parse_debug_ce_budget()
        if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) != 0:
            self._debug_ce_batches_left = 0

    def _debug_print_ce_tokens(self, batch):
        """Print tokens where labels != -100（与 HF CausalLM 中参与 CE 的 label 位置一致）。"""
        if self._debug_ce_batches_left <= 0:
            return
        tokenizer = self.processor.tokenizer
        labels = batch["labels"]
        bsz = labels.size(0)
        print("\n[FUTURE_L1_DEBUG_CE_TOKENS] batch ce-token dump (sample 0 of {})".format(bsz), flush=True)
        row_l = labels[0]
        # HF CausalLM: shift_labels = labels[..., 1:], 故 labels[0] 从不作为 CE 目标
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

    def __call__(self, raw_examples):
        """Process batch of raw examples."""
        # 调试打印已关闭，如需再次打开可恢复此前的随机 sample 日志代码
        examples = [
            cot_preprocess_function(
                ex, 
                self.args.image_max_pixels, 
                self.args.image_min_pixels, 
                self.args.max_latent_token * 32 * 32, 
                self.system_message,
                getattr(self.args, "shuffle_latent_images", False),
                getattr(self.args, "random_seed", None),
                # self.args.pattern
            ) 
            for ex in raw_examples
        ]

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in examples]

        # Optional "third path": merge consecutive reasoning-image blocks in
        # the assistant turn into a single <|latent_start|>...<|latent_end|>
        # envelope. Only takes effect when the flag is on AND a sample has
        # multiple consecutive assistant-side image blocks — the regex naturally
        # no-ops on single-image samples so it is safe to always dispatch here.
        merge_latent = bool(getattr(self.args, "merge_latent_segments", False))
        if merge_latent:
            texts = replace_visual_spectial_tokens_merged(texts)
        else:
            texts = replace_visual_spectial_tokens(texts)

        image_inputs, _ = process_vision_info(examples,image_patch_size=16)
        
        user_examples = remove_assistant_images(examples)
        user_texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in user_examples]
        user_image_inputs, _ = process_vision_info(user_examples,image_patch_size=16)
        
        assistant_examples = remove_user_images(examples)
        assistant_texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in assistant_examples]
        if merge_latent:
            assistant_texts = replace_visual_spectial_tokens_merged(assistant_texts)
        else:
            assistant_texts = replace_visual_spectial_tokens(assistant_texts)
        assistant_image_inputs, _ = process_vision_info(assistant_examples,image_patch_size=16)
        
        # Step 6: Tokenize and create batches
        user_batch = self.processor(text=user_texts, images=user_image_inputs, return_tensors="pt", padding=True)
        assistant_batch = self.processor(text=assistant_texts, images=assistant_image_inputs, return_tensors="pt", padding=True)
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        
        # Step 7: Combine pixel values
        batch['pixel_values'] = user_batch.get('pixel_values', None)
        batch['image_grid_thw'] = user_batch.get('image_grid_thw', None)
        batch['pixel_values_latent'] = assistant_batch.get('pixel_values', None)
        batch['image_grid_thw_latent'] = assistant_batch.get('image_grid_thw', None)
       
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
            batch["input_ids"], self.answer_start_token_pattern, 
            self.pad_token_idx, self.latent_token_idx
        )
        batch["labels"] = labels

        if self._debug_ce_batches_left > 0:
            self._debug_print_ce_tokens(batch)
        
        if batch['pixel_values_latent'] is not None:
            image_out_mask = mask_image_output_tokens(
                batch["input_ids"], self.latent_start_idx, self.latent_token_idx
            )
            batch["image_out_mask"] = image_out_mask

        # Optionally tokenize the raw CoT text for the auxiliary reconstruction
        # decoder (SIM-CoT style). No-op when disabled (recon_max_text_len <= 0).
        if self.recon_max_text_len > 0:
            cot_texts = []
            for ex in raw_examples:
                cot_str = ex.get("cot") if isinstance(ex, dict) else None
                if not cot_str:
                    # Fallback: use the gpt turn text with <image> stripped.
                    try:
                        gpt_val = ""
                        convs = ex.get("conversations", []) if isinstance(ex, dict) else []
                        if isinstance(convs, dict):
                            vals = convs.get("value", [])
                            froms = convs.get("from", [])
                            for f, v in zip(froms, vals):
                                if f == "gpt":
                                    gpt_val = v
                                    break
                        else:
                            for t in convs:
                                if t.get("from") == "gpt":
                                    gpt_val = t.get("value", "")
                                    break
                        cot_str = re.sub(r"<image>", " ", gpt_val).strip()
                    except Exception:
                        cot_str = ""
                cot_texts.append(cot_str or "")

            tok = self.processor.tokenizer(
                cot_texts,
                padding=True,
                truncation=True,
                max_length=self.recon_max_text_len,
                return_tensors="pt",
                add_special_tokens=False,
            )
            batch["recon_cot_input_ids"] = tok["input_ids"]
            batch["recon_cot_attention_mask"] = tok["attention_mask"]

        return batch


def make_supervised_data_module(processor, args):
    """Make dataset and collator for SwimBrid training."""
    
    dataset = FutureL1SFTDataset(data_root=args.data_path)

    # Optionally skip the first portion of the dataset, e.g. start_ratio=0.6 keeps only the last 40%.
    start_ratio = float(getattr(args, "start_ratio", 0.0) or 0.0)
    if start_ratio != 0.0:
        if not 0.0 <= start_ratio < 1.0:
            raise ValueError(f"start_ratio must be in [0.0, 1.0), got {start_ratio}")
        total = len(dataset)
        start_idx = int(total * start_ratio)
        if start_idx >= total:
            raise ValueError(
                f"start_ratio={start_ratio} results in empty dataset "
                f"(len={total}, start_idx={start_idx})."
            )
        indices = list(range(start_idx, total))
        dataset = Subset(dataset, indices)
    
    data_collator = FutureL1DataCollator(
        processor=processor,
        args=args
    )
    
    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator
    )

