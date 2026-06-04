IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

LATENT_START_TOKEN = "<|latent_start|>"
LATENT_END_TOKEN = "<|latent_end|>"
LATENT_TOKEN = "<|latent|>"
LATENT_PLACEHOLDER="<|latent|>"


SYSTEM_MESSAGE = """You are a multimodal reasoning assistant capable of thinking in textual and visual modes.


Use the following tags to switch your thinking mode:

1.  **Textual Mode**: `<reason>Your textual reasoning process</reason>`
    *   For logical analysis, planning, and verbal thought.

2.  **Visual Mode**: `<|latent_start|>Your visual reasoning process<|latent_end|>`
    *   For mental visualization, imagination and simulation.


**Output Rules**:
*   After all thinking is complete, place the final answer inside `<answer>Your Final Answer</answer>`.
"""

ROT_SYSTEM_MESSAGE = """You are a multimodal assistant capable of "Visual Chain-of-Thought" reasoning.

**Visual Chain-of-Thought**: `<|latent_start|>Your visual reasoning process<|latent_end|>`
*   For mental visualization, imagination and simulation.

**Output Rules**:
*   After all thinking is complete, place the final answer inside `<answer>Your Final Answer</answer>`.
"""

PLAIN_SYSTEM_MESSAGE = "You are a helpful assistant."

USER_MESSAGE = ""

MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]