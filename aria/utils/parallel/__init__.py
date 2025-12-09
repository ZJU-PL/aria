"""Unified, concise parallel execution utilities and patterns."""

from .executor import ParallelExecutor, parallel_map, run_tasks
from .fork_join import fork_join
from .pipeline import pipeline, PipelineStage
from .producer_consumer import producer_consumer
from .master_slave import master_slave
from .actor import ActorSystem, spawn, ActorRef, ActorHandle
from .stream import Stream
from .dataflow import Dataflow, Node

__all__ = [
    "ParallelExecutor",
    "parallel_map",
    "run_tasks",
    # patterns
    "fork_join",
    "pipeline",
    "PipelineStage",
    "producer_consumer",
    "master_slave",
    # actor
    "ActorSystem",
    "spawn",
    "ActorRef",
    "ActorHandle",
    # streaming
    "Stream",
    # dataflow
    "Dataflow",
    "Node",
]
