# Unified API calling utilities (OpenAI-compatible interface)

import json
import base64
import os
import time
import threading
import logging

from openai import OpenAI

for name in ["openai", "openai._client", "httpx", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def _build_client(api_url=None, api_key=None):
    """Build an OpenAI-compatible client from arguments or environment variables."""
    if api_url is None:
        api_url = os.environ.get("OPENAI_BASE_URL", os.environ.get("API_URL"))
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "EMPTY"))

    base_url = api_url
    if base_url and "/v1/chat/completions" in base_url:
        base_url = base_url.replace("/v1/chat/completions", "")
    if base_url and not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    return OpenAI(base_url=base_url, api_key=api_key)


# Module-level default client (lazily created)
_default_client = None
_client_lock = threading.Lock()


def _get_default_client():
    global _default_client
    if _default_client is None:
        with _client_lock:
            if _default_client is None:
                _default_client = _build_client()
    return _default_client


def encode_image_to_base64(image_path):
    """Encode a local image file to a base64 string."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def ask_question_with_image(
    question, image_path, model="gpt-5", sys_prompt=None,
    temperature=0.3, max_tokens=8000, max_retries=3,
    api_url=None, api_key=None
):
    """
    Ask a question about an image using an OpenAI-compatible API.

    Args:
        question:    The question to ask.
        image_path:  Path to the image file.
        model:       Model name.
        sys_prompt:  System prompt (optional).
        temperature: Sampling temperature.
        max_tokens:  Maximum output tokens.
        max_retries: Number of retry attempts.
        api_url:     Custom API base URL.
        api_key:     Custom API key.

    Returns:
        Model response string, or None on failure.
    """
    image_base64 = encode_image_to_base64(image_path)

    if sys_prompt is None:
        sys_prompt = (
            'A conversation between User and Assistant. The user asks a question, '
            'and the Assistant solves it. The assistant first thinks about the reasoning '
            'process in the mind and then provides the user with the answer. The reasoning '
            'process and answer are enclosed within <reason></reason> (or <think></think>) and <answer></answer> '
            'tags, respectively.'
        )

    client = _build_client(api_url, api_key) if api_url else _get_default_client()

    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        },
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                wait_time = (attempt + 1) * 2
                print(f"[API Rate Limit] 429 error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            print(f"[API calling error] {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None

    return None


def get_api_response(
    api_model_name, sys_prompt, user_prompts, client=None,
    temperature=0.3, image_paths=None, max_retries=3,
    api_url=None, api_key=None
):
    """
    Get API responses for a list of user prompts (text and/or image).

    Args:
        api_model_name: Model name (e.g. "gpt-4o-mini", "deepseek-chat").
        sys_prompt:     System prompt.
        user_prompts:   List of user prompt strings.
        client:         Optional pre-built OpenAI client.
        temperature:    Sampling temperature.
        image_paths:    Optional list of image paths (parallel to user_prompts).
        max_retries:    Number of retry attempts per prompt.
        api_url:        Custom API base URL.
        api_key:        Custom API key.

    Returns:
        List of response strings.
    """
    if client is None:
        client = _build_client(api_url, api_key) if api_url else _get_default_client()

    responses = []

    for i, user_prompt in enumerate(user_prompts):
        image_path = None
        if image_paths is not None and i < len(image_paths):
            image_path = image_paths[i]

        if image_path is not None:
            response = ask_question_with_image(
                question=user_prompt,
                image_path=image_path,
                model=api_model_name,
                sys_prompt=sys_prompt,
                temperature=temperature,
                max_retries=max_retries,
                api_url=api_url,
                api_key=api_key,
            )
            if response is None:
                response = f"[API calling error] Failed to get response for prompt {i}"
            responses.append(response)
        else:
            success = False
            for attempt in range(max_retries):
                try:
                    kwargs = dict(
                        model=api_model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=8000,
                        temperature=temperature,
                    )
                    # Some models require extra body parameters
                    if "qwen" in api_model_name.lower() or "local" in api_model_name.lower():
                        kwargs["max_tokens"] = 100
                        kwargs["temperature"] = 0.0
                        kwargs["extra_body"] = {
                            "top_k": 20,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }

                    resp = client.chat.completions.create(**kwargs)
                    answer = resp.choices[0].message.content
                    responses.append(answer)
                    success = True
                    break
                except Exception as e:
                    if hasattr(e, 'status_code') and e.status_code == 429:
                        wait_time = (attempt + 1) * 2
                        print(f"[API Rate Limit] 429 error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    print(f"[API calling error] {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    responses.append(f"[API calling error] {str(e)}")
                    break

            if not success and len(responses) <= i:
                responses.append(f"[API calling error] Failed after {max_retries} retries")

    return responses


# Legacy compatibility wrappers

def build_gemini_client():
    """Kept for backward compatibility; returns None."""
    return None


def build_deepseek_client():
    """Kept for backward compatibility; returns None."""
    return None


def get_gemini_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="gemini-2.5-pro"):
    """Kept for backward compatibility; delegates to get_api_response."""
    return get_api_response(model_name, sys_prompt, user_prompts, temperature=temperature)


def get_deepseek_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="deepseek-chat"):
    """Kept for backward compatibility; delegates to get_api_response."""
    return get_api_response(model_name, sys_prompt, user_prompts, temperature=temperature)
