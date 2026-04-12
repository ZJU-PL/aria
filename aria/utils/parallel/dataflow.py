"""Tiny dataflow graph runtime.

Users can define nodes with functions and connect them via named edges.
Execution pushes data along edges; nodes run with per-node parallelism.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, wait
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple

from .executor import ParallelExecutor


class _ClosableQueue(queue.Queue):
    """Queue that can interrupt blocked publishers during shutdown."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._closed = False

    def close(self) -> None:
        with self.mutex:
            self._closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()

    def put(
        self, item: Any, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        with self.not_full:
            if self._closed:
                raise RuntimeError("dataflow queue closed")

            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise queue.Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        if self._closed:
                            raise RuntimeError("dataflow queue closed")
                        self.not_full.wait(timeout=0.05)
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time.monotonic() + timeout
                    while self._qsize() >= self.maxsize:
                        if self._closed:
                            raise RuntimeError("dataflow queue closed")
                        remaining = endtime - time.monotonic()
                        if remaining <= 0.0:
                            raise queue.Full
                        self.not_full.wait(timeout=min(remaining, 0.05))

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


@dataclass
class Node:
    name: str
    func: Callable[..., Any]
    parallelism: int = 1
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


def _invoke_source_node(fn: Callable[..., Any]) -> Any:
    return fn()


class Dataflow:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.queues: Dict[str, int] = {}
        self._logger = logging.getLogger("aria.parallel.dataflow")
        self._sentinel = object()
        self._sources: DefaultDict[str, List[List[Any]]] = defaultdict(list)
        self._sinks: List[Tuple[str, List[Any]]] = []

    def add_queue(self, name: str, maxsize: int = 1024) -> None:
        self.queues[name] = maxsize

    def add_node(self, node: Node) -> None:
        if node.parallelism <= 0:
            raise ValueError("dataflow node parallelism must be positive")
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
        Dict[str, List[_ClosableQueue]],
        Dict[str, List[_ClosableQueue]],
        Dict[str, int],
        List[Tuple[str, List[Any], _ClosableQueue]],
    ]:
        subscribers: DefaultDict[str, List[_ClosableQueue]] = defaultdict(list)
        node_inputs: Dict[str, List[_ClosableQueue]] = {}
        sinks: List[Tuple[str, List[Any], _ClosableQueue]] = []

        for node in self.nodes.values():
            inboxes: List[_ClosableQueue] = []
            for qn in node.inputs:
                inbox = _ClosableQueue(maxsize=self.queues.get(qn, 1024))
                subscribers[qn].append(inbox)
                inboxes.append(inbox)
            node_inputs[node.name] = inboxes

        for queue_name, collector in self._sinks:
            inbox = _ClosableQueue(maxsize=self.queues.get(queue_name, 1024))
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
        self,
        subscribers: Dict[str, List[_ClosableQueue]],
        queue_name: str,
        item: Any,
    ) -> None:
        for inbox in subscribers.get(queue_name, []):
            try:
                inbox.put(item)
            except RuntimeError:
                continue

    def _start_source_threads(
        self,
        subscribers: Dict[str, List[_ClosableQueue]],
        threads: List[threading.Thread],
        stop_event: threading.Event,
    ) -> None:
        for queue_name, source_batches in self._sources.items():
            for items in source_batches:

                def source_loop(name: str = queue_name, batch: Sequence[Any] = items) -> None:
                    for item in batch:
                        if stop_event.is_set():
                            return
                        self._publish(subscribers, name, item)
                    if stop_event.is_set():
                        return
                    self._publish(subscribers, name, self._sentinel)

                t = threading.Thread(target=source_loop, daemon=True)
                threads.append(t)
                t.start()

    def _start_sink_threads(
        self,
        sinks: List[Tuple[str, List[Any], _ClosableQueue]],
        producer_counts: Dict[str, int],
        threads: List[threading.Thread],
        stop_event: threading.Event,
    ) -> None:
        for queue_name, collector, inbox in sinks:

            def sink_loop(
                name: str = queue_name,
                sink_collector: List[Any] = collector,
                sink_inbox: _ClosableQueue = inbox,
            ) -> None:
                remaining = producer_counts.get(name, 0)
                while remaining > 0 and not stop_event.is_set():
                    try:
                        item = sink_inbox.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if item is self._sentinel:
                        remaining -= 1
                        continue
                    sink_collector.append(item)

            t = threading.Thread(target=sink_loop, daemon=True)
            threads.append(t)
            t.start()

    def run(self) -> None:
        threads: List[threading.Thread] = []
        node_error: List[BaseException] = []
        error_lock = threading.Lock()
        stop_event = threading.Event()
        subscribers, node_inputs, producer_counts, sinks = self._build_runtime()

        def stop_flow(exc: BaseException) -> None:
            with error_lock:
                if node_error:
                    return
                node_error.append(exc)
                stop_event.set()
            for subscriber_group in subscribers.values():
                for inbox in subscriber_group:
                    inbox.close()

        self._start_sink_threads(sinks, producer_counts, threads, stop_event)

        # Start each node thread that pulls from inputs and pushes to outputs.
        for node in self.nodes.values():

            def node_loop(n: Node = node) -> None:
                try:
                    in_qs = node_inputs[n.name]
                    remaining = [producer_counts.get(qn, 0) for qn in n.inputs]
                    closed = [count == 0 for count in remaining]
                    pending = set()
                    idx = 0
                    source_pending = not n.inputs

                    with ParallelExecutor(
                        kind="threads", max_workers=n.parallelism
                    ) as ex:
                        def drain_completed(block: bool) -> None:
                            if not pending:
                                return

                            wait_timeout = 0.05 if block else 0
                            done, _ = wait(
                                pending,
                                timeout=wait_timeout,
                                return_when=FIRST_COMPLETED,
                            )
                            for future in done:
                                try:
                                    result = future.result()
                                except Exception as exc:
                                    self._logger.exception(
                                        "dataflow node failed name=%s err=%s",
                                        n.name,
                                        exc,
                                    )
                                    ex._fast_shutdown = True
                                    stop_flow(exc)
                                else:
                                    for queue_name in n.outputs:
                                        self._publish(subscribers, queue_name, result)
                                finally:
                                    pending.remove(future)

                        while (
                            source_pending or not all(closed) or pending
                        ) and not stop_event.is_set():
                            if pending and (
                                (not source_pending and all(closed))
                                or len(pending) >= n.parallelism
                            ):
                                drain_completed(block=True)
                                continue

                            if source_pending and not stop_event.is_set():
                                pending.add(ex.submit(_invoke_source_node, n.func))
                                source_pending = False
                                drain_completed(block=False)
                                continue

                            if not all(closed) and in_qs and not stop_event.is_set():
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

                            drain_completed(block=False)

                        ex._fast_shutdown = ex._fast_shutdown or stop_event.is_set()
                except BaseException as exc:
                    stop_flow(exc)
                else:
                    if not stop_event.is_set():
                        for queue_name in n.outputs:
                            self._publish(subscribers, queue_name, self._sentinel)

            t = threading.Thread(target=node_loop, daemon=True)
            threads.append(t)
            t.start()

        self._start_source_threads(subscribers, threads, stop_event)

        for t in threads:
            t.join()

        if node_error:
            raise node_error[0]
