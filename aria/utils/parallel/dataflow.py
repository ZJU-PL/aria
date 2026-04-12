"""Tiny dataflow graph runtime.

Users can define nodes with functions and connect them via named edges.
Execution pushes data along edges; nodes run with per-node parallelism.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, List, Sequence, Tuple

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
        self.queues: Dict[str, int] = {}
        self.threads: List[threading.Thread] = []
        self._logger = logging.getLogger("aria.parallel.dataflow")
        self._sentinel = object()
        self._sources: DefaultDict[str, List[List[Any]]] = defaultdict(list)
        self._sinks: List[Tuple[str, List[Any]]] = []

    def add_queue(self, name: str, maxsize: int = 1024) -> None:
        self.queues[name] = maxsize

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node
        for qn in [*node.inputs, *node.outputs]:
            if qn not in self.queues:
                self.add_queue(qn)

    def connect_source(self, queue_name: str, items: List[Any]) -> None:
        if queue_name not in self.queues:
            self.add_queue(queue_name)
        self._sources[queue_name].append(list(items))

    def connect_sink(self, queue_name: str, collector: List[Any]) -> None:
        if queue_name not in self.queues:
            self.add_queue(queue_name)
        self._sinks.append((queue_name, collector))

    def _build_runtime(
        self,
    ) -> Tuple[
        Dict[str, List[queue.Queue]],
        Dict[str, List[queue.Queue]],
        Dict[str, int],
        List[Tuple[str, List[Any], queue.Queue]],
    ]:
        subscribers: DefaultDict[str, List[queue.Queue]] = defaultdict(list)
        node_inputs: Dict[str, List[queue.Queue]] = {}
        sinks: List[Tuple[str, List[Any], queue.Queue]] = []

        for node in self.nodes.values():
            inboxes: List[queue.Queue] = []
            for qn in node.inputs:
                inbox = queue.Queue(maxsize=self.queues.get(qn, 1024))
                subscribers[qn].append(inbox)
                inboxes.append(inbox)
            node_inputs[node.name] = inboxes

        for queue_name, collector in self._sinks:
            inbox = queue.Queue(maxsize=self.queues.get(queue_name, 1024))
            subscribers[queue_name].append(inbox)
            sinks.append((queue_name, collector, inbox))

        producer_counts: DefaultDict[str, int] = defaultdict(int)
        for queue_name, sources in self._sources.items():
            producer_counts[queue_name] += len(sources)
        for node in self.nodes.values():
            for queue_name in node.outputs:
                producer_counts[queue_name] += 1

        return dict(subscribers), node_inputs, dict(producer_counts), sinks

    def _publish(
        self, subscribers: Dict[str, List[queue.Queue]], queue_name: str, item: Any
    ) -> None:
        for inbox in subscribers.get(queue_name, []):
            inbox.put(item)

    def _start_source_threads(self, subscribers: Dict[str, List[queue.Queue]]) -> None:
        for queue_name, source_batches in self._sources.items():
            for items in source_batches:

                def source_loop(name: str = queue_name, batch: Sequence[Any] = items) -> None:
                    for item in batch:
                        self._publish(subscribers, name, item)
                    self._publish(subscribers, name, self._sentinel)

                t = threading.Thread(target=source_loop, daemon=True)
                self.threads.append(t)
                t.start()

    def _start_sink_threads(
        self,
        sinks: List[Tuple[str, List[Any], queue.Queue]],
        producer_counts: Dict[str, int],
    ) -> None:
        for queue_name, collector, inbox in sinks:

            def sink_loop(
                name: str = queue_name,
                sink_collector: List[Any] = collector,
                sink_inbox: queue.Queue = inbox,
            ) -> None:
                remaining = producer_counts.get(name, 0)
                while remaining > 0:
                    item = sink_inbox.get()
                    if item is self._sentinel:
                        remaining -= 1
                        continue
                    sink_collector.append(item)

            t = threading.Thread(target=sink_loop, daemon=True)
            self.threads.append(t)
            t.start()

    def run(self) -> None:
        subscribers, node_inputs, producer_counts, sinks = self._build_runtime()
        self._start_sink_threads(sinks, producer_counts)

        # Start each node thread that pulls from inputs and pushes to outputs.
        for node in self.nodes.values():

            def node_loop(n: Node = node) -> None:
                in_qs = node_inputs[n.name]
                remaining = [producer_counts.get(qn, 0) for qn in n.inputs]
                closed = [count == 0 for count in remaining]
                pending = set()
                idx = 0

                with ParallelExecutor(kind="threads", max_workers=n.parallelism) as ex:
                    while not all(closed) or pending:
                        if not all(closed) and in_qs:
                            for _ in range(len(in_qs)):
                                pos = idx % len(in_qs)
                                idx += 1
                                if closed[pos]:
                                    continue
                                try:
                                    item = in_qs[pos].get(timeout=0.05)
                                except queue.Empty:
                                    continue

                                if item is self._sentinel:
                                    remaining[pos] -= 1
                                    if remaining[pos] == 0:
                                        closed[pos] = True
                                else:
                                    pending.add(ex.submit(n.func, item))
                                break

                        done = {future for future in pending if future.done()}
                        for future in done:
                            try:
                                result = future.result()
                            except Exception as exc:
                                self._logger.exception(
                                    "dataflow node failed name=%s err=%s", n.name, exc
                                )
                            else:
                                for queue_name in n.outputs:
                                    self._publish(subscribers, queue_name, result)
                            pending.remove(future)

                for queue_name in n.outputs:
                    self._publish(subscribers, queue_name, self._sentinel)

            t = threading.Thread(target=node_loop, daemon=True)
            self.threads.append(t)
            t.start()

        self._start_source_threads(subscribers)

        for t in self.threads:
            t.join()
