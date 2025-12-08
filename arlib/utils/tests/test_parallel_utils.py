import time

import pytest

from arlib.utils.parallel import ParallelExecutor, PipelineStage, Stream, pipeline
from arlib.utils.parallel.actor import spawn


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
