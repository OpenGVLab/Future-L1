import torch
import logging
from typing import List, Optional

import numpy as np
import json
import random
# from datasets import Dataset

def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, torch, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _token_id_to_int(x) -> int:
    """Tokenizer outputs are often length-1 tensors; torch.full needs a Python scalar fill_value."""
    if isinstance(x, torch.Tensor):
        t = x.detach().reshape(-1)
        if t.numel() != 1:
            raise ValueError(f"Expected a single token id, got tensor shape {tuple(x.shape)}")
        return int(t.item())
    return int(x)


def remove_user_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "user":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples


def remove_assistant_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "assistant":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples


def replace_visual_spectial_tokens(texts):

    update_texts = []
    for i, text in enumerate(texts):
        prev, after = text.split("<|im_start|>assistant")
        update_texts.append(prev + "<|im_start|>assistant" + after.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>"))

    return update_texts


def replace_visual_spectial_tokens_merged(texts):
    """Same as replace_visual_spectial_tokens, but MERGES runs of consecutive
    vision blocks in the assistant turn into ONE latent envelope.

    A "run" = two or more ``<|vision_start|><|image_pad|>+<|vision_end|>`` blocks
    separated only by whitespace / newlines. Each run collapses to::

        <|latent_start|><|image_pad|>...<|image_pad|><|latent_end|>

    where the inner ``<|image_pad|>`` tokens from every block are concatenated
    (preserving order). Each image still produces its own pixel_values_latent
    entry (unchanged ViT forward) — only the assistant-side *text* boundary is
    merged so the LLM sees a single contiguous latent segment.

    Isolated single vision blocks fall back to the original per-image wrap.
    When a sample has exactly one reasoning image this function behaves
    identically to ``replace_visual_spectial_tokens``.
    """
    import re

    # A single vision block, with one or more image_pad tokens inside.
    pad_group = r"(?:<\|image_pad\|>)+"
    block_re = re.compile(rf"<\|vision_start\|>({pad_group})<\|vision_end\|>")
    # A run of 2+ blocks separated only by whitespace (incl. newlines).
    run_re = re.compile(
        rf"<\|vision_start\|>{pad_group}<\|vision_end\|>"
        rf"(?:\s*<\|vision_start\|>{pad_group}<\|vision_end\|>)+",
        flags=re.DOTALL,
    )

    def _merge_run(match: "re.Match") -> str:
        run_text = match.group(0)
        # Collect interior image_pad groups from every block in the run.
        pads = "".join(m.group(1) for m in block_re.finditer(run_text))
        return f"<|latent_start|>{pads}<|latent_end|>"

    out = []
    for text in texts:
        if "<|im_start|>assistant" not in text:
            out.append(text)
            continue
        prev, after = text.split("<|im_start|>assistant", 1)
        # 1) Merge any run of consecutive vision blocks.
        after = run_re.sub(_merge_run, after)
        # 2) Convert remaining isolated single blocks (original behaviour).
        after = block_re.sub(
            lambda m: f"<|latent_start|>{m.group(1)}<|latent_end|>", after
        )
        out.append(prev + "<|im_start|>assistant" + after)
    return out


def replace_visual_spectial_tokens_dual(texts, kinds_per_example):
    """
    DUAL mode: Replace vision placeholders with orig tokens (<|latent_start|>...) or
    render tokens (<|text_start|>...) per occurrence based on kinds_per_example.
    kinds_per_example[b][i] = 'orig' or 'render' for the i-th reasoning image in sample b.
    """
    import re
    # Pattern: <|vision_start|> <|image_pad|> (possibly repeated) <|vision_end|>
    pattern = r'(<\|vision_start\|>)(<\|image_pad\|>)+(<\|vision_end\|>)'
    update_texts = []
    for b, text in enumerate(texts):
        kinds = kinds_per_example[b] if b < len(kinds_per_example) else []
        kind_idx = [0]

        def replacer(m):
            orig_mid = m.group(2)  # the image_pad part (may be repeated)
            k = kinds[kind_idx[0]] if kind_idx[0] < len(kinds) else "orig"
            kind_idx[0] += 1
            if k == "render":
                return f"<|text_start|>{orig_mid}<|text_end|>"
            return f"<|latent_start|>{orig_mid}<|latent_end|>"

        if "<|im_start|>assistant" in text:
            prev, after = text.split("<|im_start|>assistant", 1)
            after_new = re.sub(pattern, replacer, after)
            update_texts.append(prev + "<|im_start|>assistant" + after_new)
        else:
            update_texts.append(re.sub(pattern, replacer, text))
    return update_texts


def replace_latent_dual(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_start: int,
    latent_end: int,
    latent_token: int,
    text_start: int,
    text_end: int,
    text_token: int,
    assistant_pattern: torch.Tensor,
    pad_token: int = 0,
):
    """
    DUAL mode: Replace tokens between (latent_start, latent_end) with latent_token,
    and between (text_start, text_end) with text_token.
    """
    latent_start = _token_id_to_int(latent_start)
    latent_end = _token_id_to_int(latent_end)
    latent_token = _token_id_to_int(latent_token)
    text_start = _token_id_to_int(text_start)
    text_end = _token_id_to_int(text_end)
    text_token = _token_id_to_int(text_token)
    pad_token = _token_id_to_int(pad_token)

    batch_size = input_ids.shape[0]
    processed_sequences = []
    pattern_len = len(assistant_pattern)

    for b in range(batch_size):
        seq = input_ids[b][attention_mask[b] == 1]
        seq_len = len(seq)

        assistant_pos = -1
        for i in range(seq_len - pattern_len, -1, -1):
            if torch.equal(seq[i : i + pattern_len], assistant_pattern):
                assistant_pos = i + pattern_len - 1
                break

        if assistant_pos == -1:
            processed_sequences.append(seq)
            continue

        new_seq = seq.clone()
        after_assistant = torch.arange(seq_len, device=seq.device) > assistant_pos

        # Process latent pairs
        lat_s = (seq == latent_start) & after_assistant
        lat_e = (seq == latent_end) & after_assistant
        lat_s_pos = lat_s.nonzero().squeeze(-1)
        lat_e_pos = lat_e.nonzero().squeeze(-1)
        for s_pos in lat_s_pos:
            idx = torch.searchsorted(lat_e_pos, s_pos, right=True)
            if idx < len(lat_e_pos):
                e_pos = lat_e_pos[idx].item()
                new_seq[s_pos.item() + 1 : e_pos] = latent_token

        # Process text pairs
        txt_s = (seq == text_start) & after_assistant
        txt_e = (seq == text_end) & after_assistant
        txt_s_pos = txt_s.nonzero().squeeze(-1)
        txt_e_pos = txt_e.nonzero().squeeze(-1)
        for s_pos in txt_s_pos:
            idx = torch.searchsorted(txt_e_pos, s_pos, right=True)
            if idx < len(txt_e_pos):
                e_pos = txt_e_pos[idx].item()
                new_seq[s_pos.item() + 1 : e_pos] = text_token

        processed_sequences.append(new_seq)

    new_max_len = max(seq.size(0) for seq in processed_sequences)
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    new_attention_mask = torch.zeros((batch_size, new_max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    for b, seq in enumerate(processed_sequences):
        seq_len = seq.size(0)
        new_input_ids[b, :seq_len] = seq
        new_attention_mask[b, :seq_len] = 1
    return new_input_ids, new_attention_mask


def _assistant_start_index(seq: torch.Tensor, assistant_pattern: torch.Tensor) -> int:
    """Return index of last token of the last assistant_pattern match, or -1."""
    pattern_len = len(assistant_pattern)
    seq_len = len(seq)
    for i in range(seq_len - pattern_len, -1, -1):
        if torch.equal(seq[i : i + pattern_len], assistant_pattern):
            return i + pattern_len - 1
    return -1


def _latent_span_pairs_after_assistant(
    seq: torch.Tensor, assistant_pos: int, start_token: int, end_token: int
):
    """Return list of (s_pos, e_pos) for each start..end pair after assistant_pos."""
    seq_len = len(seq)
    start_mask = (seq == start_token) & (torch.arange(seq_len, device=seq.device) > assistant_pos)
    end_mask = (seq == end_token) & (torch.arange(seq_len, device=seq.device) > assistant_pos)
    start_positions = start_mask.nonzero().squeeze(-1)
    end_positions = end_mask.nonzero().squeeze(-1)
    valid_pairs = []
    for s_pos in start_positions:
        idx = torch.searchsorted(end_positions, s_pos, right=True)
        if idx < len(end_positions):
            valid_pairs.append((int(s_pos.item()), int(end_positions[idx].item())))
    return valid_pairs


def replace_latent_fixed(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    fixed_token_count: int,
    assistant_pattern: torch.Tensor,
    pad_token: int = 0,
    mm_token_type_ids: Optional[torch.Tensor] = None,
    inserted_mm_token_type: int = 1,
):
    """
    After the last assistant_pattern, replace each (start_token, end_token) span's interior
    with exactly fixed_token_count copies of replacement_token (sequence length may change).

    When ``mm_token_type_ids`` is provided (Qwen3.5 M-RoPE), it is transformed in parallel so
    each inserted latent token is labeled with ``inserted_mm_token_type`` (default 1 = image).
    """
    if fixed_token_count <= 0:
        raise ValueError(f"fixed_token_count must be positive, got {fixed_token_count}")

    start_token = _token_id_to_int(start_token)
    end_token = _token_id_to_int(end_token)
    replacement_token = _token_id_to_int(replacement_token)
    pad_token = _token_id_to_int(pad_token)

    batch_size = input_ids.shape[0]
    processed_sequences: List[torch.Tensor] = []
    processed_mm_sequences: List[Optional[torch.Tensor]] = []
    pattern_len = len(assistant_pattern)

    for b in range(batch_size):
        seq = input_ids[b][attention_mask[b] == 1]
        mm_seq: Optional[torch.Tensor] = None
        if mm_token_type_ids is not None:
            mm_seq = mm_token_type_ids[b][attention_mask[b] == 1]
            if mm_seq.shape[0] != seq.shape[0]:
                raise ValueError(
                    "mm_token_type_ids valid length must match input_ids valid length: "
                    f"got {mm_seq.shape[0]} vs {seq.shape[0]} (batch row {b})."
                )
        assistant_pos = _assistant_start_index(seq, assistant_pattern)
        if assistant_pos == -1:
            processed_sequences.append(seq)
            processed_mm_sequences.append(mm_seq)
            continue

        valid_pairs = _latent_span_pairs_after_assistant(seq, assistant_pos, start_token, end_token)
        if not valid_pairs:
            processed_sequences.append(seq)
            processed_mm_sequences.append(mm_seq)
            continue

        new_pieces: List[torch.Tensor] = []
        new_mm_pieces: List[torch.Tensor] = []
        prev = 0
        for s_pos, e_pos in valid_pairs:
            new_pieces.append(seq[prev : s_pos + 1])
            new_pieces.append(
                torch.full(
                    (fixed_token_count,),
                    replacement_token,
                    dtype=seq.dtype,
                    device=seq.device,
                )
            )
            if mm_seq is not None:
                new_mm_pieces.append(mm_seq[prev : s_pos + 1])
                new_mm_pieces.append(
                    torch.full(
                        (fixed_token_count,),
                        int(inserted_mm_token_type),
                        dtype=mm_seq.dtype,
                        device=mm_seq.device,
                    )
                )
            prev = e_pos
        if prev < len(seq):
            new_pieces.append(seq[prev:])
            if mm_seq is not None:
                new_mm_pieces.append(mm_seq[prev:])
        processed_sequences.append(torch.cat(new_pieces, dim=0))
        if mm_seq is not None:
            processed_mm_sequences.append(torch.cat(new_mm_pieces, dim=0))
        else:
            processed_mm_sequences.append(None)

    new_max_len = max(seq.size(0) for seq in processed_sequences)
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    new_attention_mask = torch.zeros((batch_size, new_max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    new_mm_out: Optional[torch.Tensor] = None
    if mm_token_type_ids is not None:
        new_mm_out = mm_token_type_ids.new_zeros((batch_size, new_max_len), dtype=mm_token_type_ids.dtype)
    for b, seq in enumerate(processed_sequences):
        seq_len_b = seq.size(0)
        new_input_ids[b, :seq_len_b] = seq
        new_attention_mask[b, :seq_len_b] = 1
        if new_mm_out is not None and processed_mm_sequences[b] is not None:
            mm_row = processed_mm_sequences[b]
            assert mm_row is not None
            new_mm_out[b, : mm_row.size(0)] = mm_row
    if mm_token_type_ids is None:
        return new_input_ids, new_attention_mask
    return new_input_ids, new_attention_mask, new_mm_out


def replace_latent_dual_fixed(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_start: int,
    latent_end: int,
    latent_token: int,
    text_start: int,
    text_end: int,
    text_token: int,
    fixed_latent_token_count: int,
    assistant_pattern: torch.Tensor,
    pad_token: int = 0,
    mm_token_type_ids: Optional[torch.Tensor] = None,
    latent_inserted_mm_token_type: int = 1,
    text_inserted_mm_token_type: int = 0,
):
    """
    DUAL mode: latent spans become exactly fixed_latent_token_count latent_token;
    text spans keep the same behavior as replace_latent_dual (interior length preserved).

    Optional ``mm_token_type_ids`` (Qwen3.5): latent inserts use ``latent_inserted_mm_token_type``
    (default 1 = image), text inserts use ``text_inserted_mm_token_type`` (default 0 = text).
    """
    if fixed_latent_token_count <= 0:
        raise ValueError(f"fixed_latent_token_count must be positive, got {fixed_latent_token_count}")

    latent_start = _token_id_to_int(latent_start)
    latent_end = _token_id_to_int(latent_end)
    latent_token = _token_id_to_int(latent_token)
    text_start = _token_id_to_int(text_start)
    text_end = _token_id_to_int(text_end)
    text_token = _token_id_to_int(text_token)
    pad_token = _token_id_to_int(pad_token)

    batch_size = input_ids.shape[0]
    processed_sequences: List[torch.Tensor] = []
    processed_mm_sequences: List[Optional[torch.Tensor]] = []

    for b in range(batch_size):
        seq = input_ids[b][attention_mask[b] == 1]
        seq_len = len(seq)
        mm_seq: Optional[torch.Tensor] = None
        if mm_token_type_ids is not None:
            mm_seq = mm_token_type_ids[b][attention_mask[b] == 1]
            if mm_seq.shape[0] != seq.shape[0]:
                raise ValueError(
                    "mm_token_type_ids valid length must match input_ids valid length: "
                    f"got {mm_seq.shape[0]} vs {seq.shape[0]} (batch row {b})."
                )
        assistant_pos = _assistant_start_index(seq, assistant_pattern)
        if assistant_pos == -1:
            processed_sequences.append(seq)
            processed_mm_sequences.append(mm_seq)
            continue

        after_assistant = torch.arange(seq_len, device=seq.device) > assistant_pos
        events = []

        lat_s = (seq == latent_start) & after_assistant
        lat_e = (seq == latent_end) & after_assistant
        lat_s_pos = lat_s.nonzero().squeeze(-1)
        lat_e_pos = lat_e.nonzero().squeeze(-1)
        for s_pos in lat_s_pos:
            idx = torch.searchsorted(lat_e_pos, s_pos, right=True)
            if idx < len(lat_e_pos):
                e_pos = lat_e_pos[idx].item()
                events.append((int(s_pos.item()), int(e_pos), "latent"))

        txt_s = (seq == text_start) & after_assistant
        txt_e = (seq == text_end) & after_assistant
        txt_s_pos = txt_s.nonzero().squeeze(-1)
        txt_e_pos = txt_e.nonzero().squeeze(-1)
        for s_pos in txt_s_pos:
            idx = torch.searchsorted(txt_e_pos, s_pos, right=True)
            if idx < len(txt_e_pos):
                e_pos = txt_e_pos[idx].item()
                events.append((int(s_pos.item()), int(e_pos), "text"))

        if not events:
            processed_sequences.append(seq)
            processed_mm_sequences.append(mm_seq)
            continue

        events.sort(key=lambda x: x[0])
        new_pieces: List[torch.Tensor] = []
        new_mm_pieces: List[torch.Tensor] = []
        prev = 0
        for s_pos, e_pos, kind in events:
            new_pieces.append(seq[prev : s_pos + 1])
            if mm_seq is not None:
                new_mm_pieces.append(mm_seq[prev : s_pos + 1])
            if kind == "latent":
                new_pieces.append(
                    torch.full(
                        (fixed_latent_token_count,),
                        latent_token,
                        dtype=seq.dtype,
                        device=seq.device,
                    )
                )
                if mm_seq is not None:
                    new_mm_pieces.append(
                        torch.full(
                            (fixed_latent_token_count,),
                            int(latent_inserted_mm_token_type),
                            dtype=mm_seq.dtype,
                            device=mm_seq.device,
                        )
                    )
            else:
                inner_len = e_pos - s_pos - 1
                if inner_len > 0:
                    new_pieces.append(
                        torch.full((inner_len,), text_token, dtype=seq.dtype, device=seq.device)
                    )
                    if mm_seq is not None:
                        new_mm_pieces.append(
                            torch.full(
                                (inner_len,),
                                int(text_inserted_mm_token_type),
                                dtype=mm_seq.dtype,
                                device=mm_seq.device,
                            )
                        )
            prev = e_pos
        if prev < seq_len:
            new_pieces.append(seq[prev:])
            if mm_seq is not None:
                new_mm_pieces.append(mm_seq[prev:])
        processed_sequences.append(torch.cat(new_pieces, dim=0))
        if mm_seq is not None:
            processed_mm_sequences.append(torch.cat(new_mm_pieces, dim=0))
        else:
            processed_mm_sequences.append(None)

    new_max_len = max(seq.size(0) for seq in processed_sequences)
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    new_attention_mask = torch.zeros((batch_size, new_max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    new_mm_out: Optional[torch.Tensor] = None
    if mm_token_type_ids is not None:
        new_mm_out = mm_token_type_ids.new_zeros((batch_size, new_max_len), dtype=mm_token_type_ids.dtype)
    for b, seq in enumerate(processed_sequences):
        seq_len_b = seq.size(0)
        new_input_ids[b, :seq_len_b] = seq
        new_attention_mask[b, :seq_len_b] = 1
        if new_mm_out is not None and processed_mm_sequences[b] is not None:
            mm_row = processed_mm_sequences[b]
            assert mm_row is not None
            new_mm_out[b, : mm_row.size(0)] = mm_row
    if mm_token_type_ids is None:
        return new_input_ids, new_attention_mask
    return new_input_ids, new_attention_mask, new_mm_out


def collator_replace_latent(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_start: int,
    latent_end: int,
    latent_token: int,
    assistant_pattern: torch.Tensor,
    pad_token: int,
    args,
    *,
    use_dual_latent_tokens: bool = False,
    text_start: int = 0,
    text_end: int = 0,
    text_token: int = 0,
    mm_token_type_ids: Optional[torch.Tensor] = None,
):
    """Dispatch fixed-budget vs dynamic replace_latent (+ optional DUAL text/latent).

    Always returns ``(input_ids, attention_mask, mm_token_type_ids_out)``.
    For **fixed latent budget**, ``mm_token_type_ids_out`` is recomputed to match the
    lengthened sequences (Qwen3.5 M-RoPE). If the input had no ``mm_token_type_ids``,
    the third value is ``None``. For non-fixed replaces, the third value is the same
    tensor passed in (or ``None``), still aligned because sequence length is unchanged.
    """
    n_fix = getattr(args, "fixed_latent_budget", None)
    if n_fix is not None and int(n_fix) > 0:
        n_fix_i = int(n_fix)
        if use_dual_latent_tokens:
            r = replace_latent_dual_fixed(
                input_ids,
                attention_mask,
                latent_start,
                latent_end,
                latent_token,
                text_start,
                text_end,
                text_token,
                n_fix_i,
                assistant_pattern,
                pad_token,
                mm_token_type_ids=mm_token_type_ids,
            )
        else:
            r = replace_latent_fixed(
                input_ids,
                attention_mask,
                latent_start,
                latent_end,
                latent_token,
                n_fix_i,
                assistant_pattern,
                pad_token,
                mm_token_type_ids=mm_token_type_ids,
            )
        if len(r) == 2:
            return r[0], r[1], None
        return r

    if use_dual_latent_tokens:
        new_ids, new_mask = replace_latent_dual(
            input_ids,
            attention_mask,
            latent_start,
            latent_end,
            latent_token,
            text_start,
            text_end,
            text_token,
            assistant_pattern,
            pad_token,
        )
        return new_ids, new_mask, mm_token_type_ids

    new_ids, new_mask = replace_latent(
        input_ids,
        attention_mask,
        latent_start,
        latent_end,
        latent_token,
        assistant_pattern,
        pad_token,
    )
    return new_ids, new_mask, mm_token_type_ids


def maybe_flat_latent_pool_head_kinds(
    kinds_per_example: List[List[str]],
    fixed_latent_budget,
    image_grid_thw_latent: Optional[torch.Tensor],
) -> Optional[List[str]]:
    """
    When train_fixed_latent_budget > 0, forward needs one 'orig'/'render' label per row of
    image_grid_thw_latent (same order as assistant-side latent images in the batch).
    """
    if fixed_latent_budget is None or int(fixed_latent_budget) <= 0:
        return None
    if image_grid_thw_latent is None:
        return None
    flat: List[str] = []
    for kinds in kinds_per_example:
        flat.extend(kinds)
    n_rows = int(image_grid_thw_latent.shape[0])
    if len(flat) != n_rows:
        raise ValueError(
            f"latent_pool_head_kinds: total reasoning images ({len(flat)}) != "
            f"image_grid_thw_latent rows ({n_rows})"
        )
    return flat


def replace_latent(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    assistant_pattern: torch.Tensor,
    pad_token: int = 0
):
    """
    Replace tokens between start_token and end_token with replacement_token,
    keeping original length, for all valid pairs after assistant pattern.
    """
    start_token = _token_id_to_int(start_token)
    end_token = _token_id_to_int(end_token)
    replacement_token = _token_id_to_int(replacement_token)
    pad_token = _token_id_to_int(pad_token)

    batch_size = input_ids.shape[0]
    processed_sequences = []
    pattern_len = len(assistant_pattern)

    for b in range(batch_size):
        seq = input_ids[b][attention_mask[b] == 1]
        seq_len = len(seq)
        
        # Find last occurrence of assistant pattern
        assistant_pos = -1
        for i in range(seq_len - pattern_len, -1, -1):
            if torch.equal(seq[i:i+pattern_len], assistant_pattern):
                assistant_pos = i + pattern_len - 1
                break
        
        if assistant_pos == -1:
            processed_sequences.append(seq)
            continue
        
        # Find all start/end positions
        start_mask = (seq == start_token) & (torch.arange(seq_len, device=seq.device) > assistant_pos)
        end_mask = (seq == end_token) & (torch.arange(seq_len, device=seq.device) > assistant_pos)
        start_positions = start_mask.nonzero().squeeze(-1)
        end_positions = end_mask.nonzero().squeeze(-1)
        
        if len(start_positions) == 0 or len(end_positions) == 0:
            processed_sequences.append(seq)
            continue
        
        # Match each start with its first following end
        # Using searchsorted: for each start, find first end that comes after it
        valid_pairs = []
        for s_pos in start_positions:
            idx = torch.searchsorted(end_positions, s_pos, right=True)
            if idx < len(end_positions):
                valid_pairs.append((s_pos.item(), end_positions[idx].item()))
        
        if not valid_pairs:
            processed_sequences.append(seq)
            continue
        
        # Create replacement mask and new values
        new_seq = seq.clone()
        for s_pos, e_pos in valid_pairs:
            # Replace tokens between start and end (exclusive)
            new_seq[s_pos+1:e_pos] = replacement_token
        
        processed_sequences.append(new_seq)
    
    # Re-pad sequences
    new_max_len = max(seq.size(0) for seq in processed_sequences)
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    new_attention_mask = torch.zeros((batch_size, new_max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    
    for b, seq in enumerate(processed_sequences):
        seq_len = seq.size(0)
        new_input_ids[b, :seq_len] = seq
        new_attention_mask[b, :seq_len] = 1
    
    return new_input_ids, new_attention_mask



def find_subsequence(row: torch.Tensor, pattern: torch.Tensor) -> int:

    seq_len = row.size(0)
    pat_len = pattern.size(0)
    
    # Naive scan over all possible start positions
    for start_idx in range(seq_len - pat_len + 1):
        # Compare row[start_idx : start_idx + pat_len] to pattern
        if torch.all(row[start_idx : start_idx + pat_len] == pattern):
            return start_idx
    return -1


def generate_labels_after_multi_token_start(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
    img_token_idx: int = 151655,
    img_token_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    For each row in `input_ids`, find the *first* occurrence of `start_sequence`
    (a 1D tensor of multiple token IDs). Mask all tokens up to and including
    that entire sub-sequence (set them to -100), and also mask any padding tokens
    anywhere in the row. The remainder (tokens *after* the sub-sequence) are kept.

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      start_sequence: 1D tensor of shape [k], the multi-token "start" pattern.
      pad_token_id: which ID is used as padding (default=0).
      img_token_idx: single latent/image token to mask (legacy).
      img_token_indices: list of token IDs to mask as image/latent (e.g. [latent_id, text_id] for DUAL).
    
    Returns:
      labels: a new 2D tensor [batch_size, seq_len], where tokens before (and
              including) the sub-sequence are -100, as well as any pad tokens,
              and tokens after the sub-sequence are kept as in `input_ids`.
    """
    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone()
    tokens_to_mask = img_token_indices if img_token_indices is not None else [img_token_idx]

    for b in range(batch_size):
        row = labels[b]
        start_idx = find_subsequence(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
        else:
            sub_len = start_sequence.size(0)
            end_of_subseq = start_idx + sub_len
            row[:end_of_subseq] = -100
        row[row == pad_token_idx] = -100
        for tid in tokens_to_mask:
            row[row == tid] = -100
    return labels


def generate_labels_answer_only(
    input_ids: torch.Tensor,
    assistant_start_sequence: torch.Tensor,
    answer_start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
    img_token_idx: int = 151655,
    include_token_indices=None,
) -> torch.Tensor:
    """
    只对 <answer>...</answer> 部分计算 CE loss，reason 和 latent 部分全部 mask 为 -100。
    用于对比实验：img latent loss + answer CE loss，无 reason CE loss。

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      assistant_start_sequence: <|im_start|>assistant 的 token ids.
      answer_start_sequence: <answer> 的 token ids（可能多 token）.
      pad_token_idx: padding token id.
      img_token_idx: latent token id，需要 mask。
      include_token_indices: tokens that should always be included in CE labels.

    Returns:
      labels: 只有 <answer> 内的 token 参与 CE loss，其余为 -100。
    """
    if include_token_indices is None:
        include_token_indices = []
    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone()
    labels[:] = -100

    asst_pat_len = assistant_start_sequence.size(0)
    ans_pat_len = answer_start_sequence.size(0)

    for b in range(batch_size):
        row = input_ids[b]
        # 1. 找到 assistant 开始位置
        asst_idx = find_subsequence(row, assistant_start_sequence)
        if asst_idx == -1:
            continue
        # 2. 在 assistant 之后找 <answer> 的首次出现
        search_start = asst_idx + asst_pat_len
        ans_idx = -1
        for i in range(search_start, seq_len - ans_pat_len + 1):
            if torch.all(row[i : i + ans_pat_len] == answer_start_sequence):
                ans_idx = i
                break
        if ans_idx == -1:
            continue
        # 3. 从 <answer> 开始到序列末尾，保留 label（pad 和 img_token 仍 mask）
        for j in range(ans_idx, seq_len):
            if row[j] != pad_token_idx and row[j] != img_token_idx:
                labels[b, j] = row[j].item()

        # Force selected special tokens to participate in CE even in answer-only mode.
        if include_token_indices:
            for tid in include_token_indices:
                mask = row == tid
                labels[b, mask] = row[mask]

    return labels


def mask_image_output_tokens(
    input_ids: torch.Tensor,
    image_start_token: int,
    image_token: int
) -> torch.Tensor:
    """
    Creates a mask of the same shape as `input_ids`, with 1's wherever we want to
    'mask out' <image_token> after the first <image_start_token> has appeared,
    and 0's everywhere else.

    Args:
      input_ids: shape [batch_size, seq_len]
      image_start_token: the token ID that marks the start of an image chunk
      image_token: the token ID for image tokens

    Returns:
      A mask (torch.Tensor of the same shape) containing 0/1:
        - 1 = this position should be masked
        - 0 = this position is kept
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids)

    for i in range(batch_size):
        seq = input_ids[i]
        # Find first occurrence of image_start_token
        first_start_pos = -1
        for j in range(seq_len):
            if seq[j] == image_start_token:
                first_start_pos = j
                break
        
        if first_start_pos == -1:
            continue
        
        # For every position after the first <image_start_token>,
        # if the token is <image_token>, set mask = 1
        for k in range(first_start_pos + 1, seq_len):
            if seq[k] == image_token:
                mask[i, k] = 1

    return mask