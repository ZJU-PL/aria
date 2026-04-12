import time
import threading
import asyncio
import io
import subprocess
import sys

import pytest

from aria.utils.parallel import (
    ParallelExecutor,
    PipelineStage,
    Stream,
    fork_join,
    pipeline,
    producer_consumer,
)
from aria.utils.parallel.async_utils import (
    check_call_async,
    forward_and_return_data,
    kill_process_async,
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


def test_parallel_executor_timeout_returns_promptly():
    def slow(_: int) -> int:
        time.sleep(0.3)
        return 1

    start = time.time()
    with pytest.raises(TimeoutError):
        with ParallelExecutor(max_workers=1) as ex:
            ex.run(slow, [1], timeout=0.05)
    assert time.time() - start < 0.2


def test_parallel_executor_error_returns_promptly():
    def work(x: int) -> int:
        if x == 0:
            raise ValueError("boom")
        time.sleep(0.3)
        return x

    start = time.time()
    with pytest.raises(ValueError, match="boom"):
        with ParallelExecutor(max_workers=2) as ex:
            ex.run(work, [0, 1])
    assert time.time() - start < 0.2


def test_parallel_executor_timeout_with_return_exceptions_returns_promptly():
    def slow(_: int) -> int:
        time.sleep(0.3)
        return 1

    start = time.time()
    with ParallelExecutor(max_workers=1) as ex:
        results = ex.run(slow, [1], timeout=0.05, return_exceptions=True)
    assert time.time() - start < 0.2
    assert len(results) == 1
    assert isinstance(results[0], TimeoutError)


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
    try:
        result = fork_join([_return_value], kind="processes")
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"process pools unavailable in this environment: {exc}")
    assert result == [7]


def test_pipeline_processes_all_items():
    stages = [
        PipelineStage(worker=lambda x: x * 2, parallelism=2),
        PipelineStage(worker=lambda x: x + 1, parallelism=2),
    ]

    results = pipeline(stages, inputs=list(range(5)), queue_size=8)
    assert sorted(results) == [x * 2 + 1 for x in range(5)]


def test_pipeline_preserves_none_payloads():
    results = pipeline([PipelineStage(worker=lambda x: x)], inputs=[1, None, 2])
    assert results == [1, None, 2]


def test_pipeline_raises_stage_failures():
    def work(x: int) -> int:
        if x == 1:
            raise ValueError("boom")
        return x

    with pytest.raises(ValueError, match="boom"):
        pipeline([PipelineStage(worker=work)], inputs=[0, 1, 2])


def test_pipeline_raises_stage_failures_promptly():
    def work(x: int) -> int:
        if x == 0:
            raise ValueError("boom")
        time.sleep(0.3)
        return x

    start = time.time()
    with pytest.raises(ValueError, match="boom"):
        pipeline([PipelineStage(worker=work, parallelism=2)], inputs=[0, 1])
    assert time.time() - start < 0.2


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


def test_producer_consumer_raises_promptly_on_consumer_failure():
    def produce(q):
        for item in range(3):
            q.put(item)

    def consume(item: int) -> int:
        if item == 0:
            raise ValueError("boom")
        time.sleep(0.3)
        return item

    start = time.time()
    with pytest.raises(ValueError, match="boom"):
        producer_consumer(produce, consume, consumer_parallelism=3)
    assert time.time() - start < 0.15


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


def test_dataflow_raises_node_failures():
    collector = []
    flow = Dataflow()

    def work(x: int) -> int:
        if x == 1:
            raise ValueError("boom")
        return x

    flow.add_node(Node("bad", work, inputs=["src"], outputs=["out"]))
    flow.connect_source("src", [0, 1])
    flow.connect_sink("out", collector)

    with pytest.raises(ValueError, match="boom"):
        flow.run()


def test_dataflow_raises_node_failures_promptly():
    flow = Dataflow()

    def work(x: int) -> int:
        if x == 0:
            raise ValueError("boom")
        time.sleep(0.3)
        return x

    flow.add_node(Node("bad", work, parallelism=2, inputs=["src"], outputs=["out"]))
    flow.connect_source("src", [0, 1])
    flow.connect_sink("out", [])

    start = time.time()
    with pytest.raises(ValueError, match="boom"):
        flow.run()
    assert time.time() - start < 0.2


def test_actor_stop_does_not_block_on_full_mailbox():
    started = threading.Event()
    release = threading.Event()

    def slow(_: int) -> None:
        started.set()
        release.wait()

    handle = spawn(slow, mailbox_size=1, name="bounded-actor")
    try:
        handle.ref.tell(1)
        assert started.wait(0.1)
        handle.ref.tell(2)

        def delayed_release() -> None:
            time.sleep(0.3)
            release.set()

        threading.Thread(target=delayed_release, daemon=True).start()
        start = time.time()
        handle.stop(timeout=0.01)
        assert time.time() - start < 0.1
    finally:
        release.set()
        handle.thread.join(timeout=0.5)


def test_actor_tell_fails_fast_on_full_mailbox():
    started = threading.Event()
    release = threading.Event()

    def slow(_: int) -> None:
        started.set()
        release.wait()

    handle = spawn(slow, mailbox_size=1, name="bounded-actor")
    try:
        handle.ref.tell(1)
        assert started.wait(0.1)
        handle.ref.tell(2)

        start = time.time()
        with pytest.raises(TimeoutError, match="mailbox full"):
            handle.ref.tell(3)
        assert time.time() - start < 0.1
    finally:
        release.set()
        handle.thread.join(timeout=0.5)


def test_actor_ask_timeout_includes_mailbox_enqueue_wait():
    started = threading.Event()
    release = threading.Event()

    def slow(_: int) -> int:
        started.set()
        release.wait()
        return 1

    handle = spawn(slow, mailbox_size=1, name="bounded-actor")
    try:
        handle.ref.tell(1)
        assert started.wait(0.1)
        handle.ref.tell(2)

        start = time.time()
        with pytest.raises(TimeoutError, match="mailbox full"):
            handle.ref.ask(3, timeout=0.05)
        assert time.time() - start < 0.15
    finally:
        release.set()
        handle.thread.join(timeout=0.5)


def test_forward_and_return_data_handles_non_utf8_output():
    async def run_test() -> tuple[bytes, str]:
        reader = asyncio.StreamReader()
        reader.feed_data(b"\xff\n")
        reader.feed_eof()
        output = io.StringIO()
        data = await forward_and_return_data(output, "worker", reader)
        return data, output.getvalue()

    data, rendered = asyncio.run(run_test())
    assert data == b"\xff\n"
    assert "[worker]" in rendered


def test_kill_process_async_kills_after_terminate_timeout():
    async def run_test() -> int:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            (
                "import signal, time;"
                "signal.signal(signal.SIGTERM, lambda *_: None);"
                "time.sleep(60)"
            ),
        )
        await kill_process_async(proc, wait_before_kill_secs=0.05)
        assert proc.returncode is not None
        return proc.returncode

    returncode = asyncio.run(run_test())
    assert returncode != 0


def test_check_call_async_preserves_output_with_input_and_prefix(monkeypatch):
    async def run_test() -> subprocess.CalledProcessError:
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            await check_call_async(
                sys.executable,
                "-c",
                (
                    "import sys;"
                    "data = sys.stdin.buffer.read();"
                    "sys.stdout.buffer.write(data.upper());"
                    "sys.stderr.buffer.write(b'err:' + data);"
                    "raise SystemExit(3)"
                ),
                input=b"hello",
                prefix="worker",
            )
        return exc_info.value

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    error = asyncio.run(run_test())

    assert error.returncode == 3
    assert error.output == b"HELLO"
    assert error.stderr == b"err:hello"
    assert "[worker stdout]" in stdout.getvalue()
    assert "HELLO" in stdout.getvalue()
    assert "[worker stderr]" in stderr.getvalue()
    assert "err:hello" in stderr.getvalue()


def test_producer_consumer_stops_producer_after_consumer_failure():
    producer_stopped = threading.Event()

    def produce(q):
        try:
            for item in range(10_000):
                q.put(item)
        finally:
            producer_stopped.set()

    def consume(item: int) -> int:
        if item == 0:
            raise ValueError("boom")
        time.sleep(0.2)
        return item

    with pytest.raises(ValueError, match="boom"):
        producer_consumer(
            produce,
            consume,
            consumer_parallelism=1,
            queue_size=1,
        )

    assert producer_stopped.wait(0.6)
