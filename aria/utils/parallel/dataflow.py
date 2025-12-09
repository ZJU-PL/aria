"""Tiny dataflow graph runtime.

Users can define nodes with functions and connect them via named edges.
Execution pushes data along edges; nodes run with per-node parallelism.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .executor import ParallelExecutor


@dataclass
class Node:
    name: str
    func: Callable[[Any], Any]
    parallelism: int = 1
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


class Dataflow:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.queues: Dict[str, queue.Queue] = {}
        self.threads: List[threading.Thread] = []
        self._logger = logging.getLogger("aria.parallel.dataflow")
        self._sentinel = object()

    def add_queue(self, name: str, maxsize: int = 1024) -> None:
        self.queues[name] = queue.Queue(maxsize=maxsize)

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node
        # ensure queues exist for outputs
        for qn in node.outputs:
            if qn not in self.queues:
                self.add_queue(qn)

    def connect_source(self, queue_name: str, items: List[Any]) -> None:
        q = self.queues[queue_name]
        for it in items:
            q.put(it)
        q.put(self._sentinel)

    def connect_sink(self, queue_name: str, collector: List[Any]) -> None:
        def sink() -> None:
            q = self.queues[queue_name]
            while True:
                item = q.get()
                if item is self._sentinel:
                    break
                collector.append(item)
        t = threading.Thread(target=sink, daemon=True)
        self.threads.append(t)
        t.start()

    def run(self) -> None:
        # Start each node thread that pulls from inputs and pushes to outputs
        for node in self.nodes.values():
            def node_loop(n: Node = node) -> None:
                in_qs = [self.queues[qn] for qn in n.inputs]
                out_qs = [self.queues[qn] for qn in n.outputs]
                with ParallelExecutor(kind="threads", max_workers=n.parallelism) as ex:
                    # simple fan-in: round-robin across input queues
                    active = len(in_qs)
                    idx = 0
                    while active > 0:
                        q = in_qs[idx % len(in_qs)]
                        idx += 1
                        item = q.get()
                        if item is self._sentinel:
                            active -= 1
                            continue
                        try:
                            res = ex.run(n.func, [item])[0]
                        except Exception as exc:
                            self._logger.exception("dataflow node failed name=%s err=%s", n.name, exc)
                            continue
                        for oq in out_qs:
                            oq.put(res)
                    # propagate termination
                    for oq in out_qs:
                        oq.put(self._sentinel)

            t = threading.Thread(target=node_loop, daemon=True)
            self.threads.append(t)
            t.start()

        # Wait for all threads to finish briefly (non-blocking design)
        for t in self.threads:
            t.join()
