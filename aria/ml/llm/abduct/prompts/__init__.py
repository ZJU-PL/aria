"""Prompt templates for LLM-based abduction."""

from .basic import create_basic_prompt
from .feedback import create_feedback_prompt
from .cot import create_cot_prompt
from .few_shot import create_few_shot_prompt

__all__ = [
    "create_basic_prompt",
    "create_feedback_prompt",
    "create_cot_prompt",
    "create_few_shot_prompt",
]
