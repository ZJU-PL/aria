"""LLM-based abduction module.

This module provides tools for evaluating LLMs on abductive reasoning tasks
using SMT constraints as premises and conclusions.
"""

# Core functionality
from .data_structures import AbductionProblem, AbductionResult, AbductionIterationResult, FeedbackAbductionResult
from .base_abductor import LLMAbductor
from .feedback_abductor import FeedbackLLMAbductor
from .validation import validate_hypothesis, generate_counterexample
from .prompts import create_basic_prompt, create_feedback_prompt
from .utils import extract_smt_from_llm_response, parse_smt2_string

__all__ = [
    'AbductionProblem', 'AbductionResult', 'AbductionIterationResult', 'FeedbackAbductionResult',
    'LLMAbductor', 'FeedbackLLMAbductor',
    'validate_hypothesis', 'generate_counterexample',
    'create_basic_prompt', 'create_feedback_prompt',
    'extract_smt_from_llm_response', 'parse_smt2_string'
]
