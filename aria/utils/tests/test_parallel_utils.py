import time
import threading

import pytest

from aria.utils.parallel import (
    ParallelExecutor,
    PipelineStage,
    Stream,
    fork_join,
    pipeline,
    producer_consumer,
)
from aria.utils.parallel.actor import spawn
from aria.utils.parallel.dataflow import Dataflow, Node


def _return_value() -> int:
    return 7


def test_parallel_executor_preserves_order():
    items = list(range(6))

    def work(x: int) -> int:
        # Reverse sleep to encourage out-of-order completion
        time.sleep(0.01 * (len(items) - x))
        return x

    with ParallelExecutor(max_workers=3) as ex:
        assert ex.run(work, items) == items


def test_parallel_executor_times_out_and_cancels():
    def slow(_: int) -> int:
        time.sleep(0.2)
        return 1

    with ParallelExecutor(max_workers=1) as ex:
        with pytest.raises(TimeoutError):
            ex.run(slow, [1, 2], timeout=0.05)


def test_actor_ask_raises_exception():
    def bad(_: str) -> str:
        raise ValueError("boom")

    handle = spawn(bad, name="test-actor")
    try:
        with pytest.raises(ValueError):
            handle.ref.ask("hi", timeout=0.1)
    finally:
        handle.stop()


def test_actor_ask_timeout_raises_timeout_error():
    def slow(_: str) -> str:
        time.sleep(0.2)
        return "done"

    handle = spawn(slow, name="slow-actor")
    try:
        with pytest.raises(TimeoutError):
            handle.ref.ask("hi", timeout=0.01)
    finally:
        handle.stop()


def test_fork_join_process_pool_supports_picklable_callables():
    assert fork_join([_return_value], kind="processes") == [7]


def test_pipeline_processes_all_items():
    stages = [
        PipelineStage(worker=lambda x: x * 2, parallelism=2),
        PipelineStage(worker=lambda x: x + 1, parallelism=2),
    ]

    results = pipeline(stages, inputs=list(range(5)), queue_size=8)
    assert sorted(results) == [x * 2 + 1 for x in range(5)]


def test_stream_map_collects_results():
    data = list(range(4))
    assert Stream(data).map(lambda x: x + 1).collect() == [1, 2, 3, 4]


def test_producer_consumer_runs_multiple_consumers_concurrently():
    active = 0
    peak = 0
    lock = threading.Lock()

    def produce(q):
        for item in range(4):
            q.put(item)

    def consume(item: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            return item
        finally:
            with lock:
                active -= 1

    results = producer_consumer(produce, consume, consumer_parallelism=4)

    assert sorted(results) == [0, 1, 2, 3]
    assert peak >= 2


def test_dataflow_broadcasts_shared_inputs_to_multiple_nodes():
    collector = []
    flow = Dataflow()
    flow.add_node(Node("left", lambda x: ("left", x), inputs=["src"], outputs=["out"]))
    flow.add_node(Node("right", lambda x: ("right", x), inputs=["src"], outputs=["out"]))
    flow.connect_source("src", [1, 2])
    flow.connect_sink("out", collector)

    flow.run()

    assert sorted(collector) == [
        ("left", 1),
        ("left", 2),
        ("right", 1),
        ("right", 2),
    ]


def test_dataflow_waits_for_all_upstream_producers_before_closing_sink():
    collector = []
    flow = Dataflow()
    flow.add_node(Node("left", lambda x: x, inputs=["in_left"], outputs=["out"]))
    flow.add_node(Node("right", lambda x: x, inputs=["in_right"], outputs=["out"]))
    flow.connect_source("in_left", [1])
    flow.connect_source("in_right", [2])
    flow.connect_sink("out", collector)

    flow.run()

    assert sorted(collector) == [1, 2]
