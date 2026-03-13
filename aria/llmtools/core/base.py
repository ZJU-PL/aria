# pylint: disable=invalid-name
"""Base provider interface and inference result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class InferenceResult:
    """Normalized inference result across providers."""

    content: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    error: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference with the provider."""
        raise NotImplementedError
