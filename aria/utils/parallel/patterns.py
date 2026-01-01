"""Compatibility module re-exporting parallel patterns."""

from __future__ import annotations

from .fork_join import fork_join
from .pipeline import pipeline, PipelineStage
from .producer_consumer import producer_consumer
from .master_slave import master_slave

__all__ = [
    "fork_join",
    "pipeline",
    "PipelineStage",
    "producer_consumer",
    "master_slave",
]
