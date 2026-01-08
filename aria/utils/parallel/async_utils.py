"""Async utilities for subprocess management and output handling.

This module provides asynchronous wrappers around subprocess execution,
including proper output forwarding, process termination, and error handling.
It's designed to safely manage long-running subprocesses with configurable
timeouts and output prefixes for multi-process scenarios.
"""

import asyncio
import collections
import functools
import subprocess
import sys
from typing import Any, TextIO, cast


async def forward_and_return_data(
    output: TextIO | None, prefix: str, input: asyncio.StreamReader
) -> bytes:
    """Forward data from a stream reader to an output file while collecting it.

    This function reads lines from an async stream reader (typically from a
    subprocess stdout/stderr), optionally forwards them to an output stream
    with a prefix for identification, and collects all the data to return it.

    Args:
        output: Optional output stream to forward lines to (e.g., sys.stdout).
                If None, lines are collected but not forwarded.
        prefix: Prefix string to prepend to each forwarded line for identification
                (e.g., "worker-1 stdout" or "solver stderr").
        input: Async stream reader to read data from (typically from subprocess).

    Returns:
        All bytes read from the input stream, concatenated together.

    Example:
        >>> proc = await asyncio.create_subprocess_exec(...)
        >>> data = await forward_and_return_data(
        ...     sys.stdout, "my-process", proc.stdout
        ... )
    """
    blocks: collections.deque[bytes] = collections.deque()
    # Calculate spacing to align prefixes nicely (target width ~30 chars)
    space = " " * max(1, (30 - 2 - len(prefix)))
    while not input.at_eof():
        line = await input.readline()
        if len(line) == 0:
            break
        blocks.append(line)
        if output is not None:
            # Ensure line has newline for consistent formatting
            if not line.endswith(b"\n"):
                line += b"\n"
            output.write(f"[{prefix}]{space}{line.decode('utf-8')}")
    return b"".join(blocks)


async def kill_process_async(
    proc: asyncio.subprocess.Process, wait_before_kill_secs: float = 5.0
) -> None:
    """Gracefully terminate or forcefully kill an async subprocess.

    This function implements a two-stage termination process:
    1. First, sends SIGTERM to allow the process to clean up gracefully
    2. After a timeout, sends SIGKILL to forcefully terminate if needed

    Includes a workaround for a Python 3.10 bug where transport cleanup
    is needed before process termination.

    Args:
        proc: The asyncio subprocess process to terminate/kill.
        wait_before_kill_secs: Number of seconds to wait after SIGTERM before
                              sending SIGKILL. If 0 or negative, skips graceful
                              termination and kills immediately.

    Note:
        The transport.close() call is a workaround for Python 3.10 issue #88050,
        which is fixed in Python 3.11+.

    Raises:
        Does not raise exceptions - silently handles ProcessLookupError if
        the process has already terminated.
    """
    # Process already terminated, nothing to do
    if proc.returncode is not None:
        return

    # https://github.com/python/cpython/issues/88050
    # This is fixed in Python 3.11, but we're still using 3.10 in prod.
    # Close transport first to avoid issues with process termination
    transport: asyncio.SubprocessTransport | None = proc._transport  # type: ignore
    if transport is not None:
        transport.close()

    # Graceful termination: send SIGTERM and wait for process to exit
    if wait_before_kill_secs > 0:
        try:
            proc.terminate()
        except ProcessLookupError:
            # Process already gone
            return
        # Wait up to wait_before_kill_secs for process to exit gracefully
        await asyncio.wait_for(proc.wait(), wait_before_kill_secs)

        # Check if process exited during the wait
        if cast(int | None, proc.returncode) is not None:
            return

    # Forceful termination: send SIGKILL if graceful termination didn't work
    try:
        proc.kill()
    except ProcessLookupError:
        # Process already terminated, ignore
        pass


@functools.wraps(asyncio.create_subprocess_exec)
async def run_subprocess_async(
    *cmd: str,
    input: bytes | None = None,
    wait_before_kill_secs: float = 5.0,
    capture_output: bool = True,
    **kwargs: Any,
) -> tuple[int, bytes, bytes]:
    """Run a subprocess asynchronously and return exit code and output.

    This is an async wrapper around asyncio.create_subprocess_exec that
    handles input/output capture and ensures proper process cleanup via
    graceful termination.

    Args:
        *cmd: Command and arguments to execute (e.g., "python", "-c", "print('hi')").
        input: Optional bytes to send to stdin.
        wait_before_kill_secs: Seconds to wait after SIGTERM before SIGKILL.
        capture_output: If True, capture stdout/stderr. If False, let them
                       stream to parent process (None is passed to subprocess).
        **kwargs: Additional keyword arguments passed to create_subprocess_exec.

    Returns:
        A tuple of (exit_code, stdout_data, stderr_data) where:
        - exit_code: Integer exit code of the process (0 = success, non-zero = failure).
        - stdout_data: Bytes captured from stdout (empty bytes if capture_output=False).
        - stderr_data: Bytes captured from stderr (empty bytes if capture_output=False).

    Example:
        >>> exit_code, stdout, stderr = await run_subprocess_async(
        ...     "python", "-c", "print('Hello')"
        ... )
        >>> print(f"Exit: {exit_code}, Output: {stdout.decode()}")
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=(asyncio.subprocess.PIPE if capture_output else None),
        stderr=(asyncio.subprocess.PIPE if capture_output else None),
        **kwargs,
    )
    try:
        stdout, stderr = await proc.communicate(input=input)
    finally:
        # Always ensure process is properly terminated, even if communicate() fails
        await kill_process_async(proc, wait_before_kill_secs=wait_before_kill_secs)

    assert proc.returncode is not None
    return (proc.returncode, stdout or b"", stderr or b"")


@functools.wraps(run_subprocess_async)
async def check_output_async(*cmd: str, **kwargs: Any) -> tuple[bytes, bytes]:
    """Run a subprocess and return output only if it succeeds.

    Similar to subprocess.check_output(), but async and returns both
    stdout and stderr. Raises CalledProcessError if the process exits
    with a non-zero status code.

    Args:
        *cmd: Command and arguments to execute.
        **kwargs: Additional keyword arguments passed to run_subprocess_async.

    Returns:
        A tuple of (stdout_data, stderr_data) if the process succeeds.

    Raises:
        subprocess.CalledProcessError: If the process exits with non-zero status.

    Example:
        >>> stdout, stderr = await check_output_async("echo", "hello")
        >>> print(stdout.decode())  # "hello\\n"
    """
    retcode, stdout, stderr = await run_subprocess_async(*cmd, **kwargs)
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmd, stdout, stderr)
    return (stdout, stderr)


@functools.wraps(asyncio.create_subprocess_exec)
async def check_call_async(
    *args: str | bytes,
    prefix: str | None = None,
    input: bytes | None = None,
    wait_before_kill_secs: float = 5.0,
    suppress_stdout: bool = False,
    **kwargs,
) -> None:
    """Run a subprocess and raise an error if it fails (async check_call).

    Similar to subprocess.check_call(), but async. Supports three modes:
    1. With input: Sends input to stdin and waits for completion
    2. With prefix: Forwards stdout/stderr to parent with prefix labels
    3. Default: Lets process inherit parent's stdout/stderr or suppresses them

    Args:
        *args: Command and arguments to execute (can be str or bytes).
        prefix: Optional prefix for forwarded output lines (e.g., "worker-1").
                When provided, stdout/stderr are forwarded to sys.stdout/stderr
                with prefixed labels. If None, output behavior depends on
                suppress_stdout and subprocess defaults.
        input: Optional bytes to send to stdin. If provided, process waits
               for input completion before proceeding.
        wait_before_kill_secs: Seconds to wait after SIGTERM before SIGKILL.
        suppress_stdout: If True and prefix is None, suppress stdout output
                        (redirect to DEVNULL).
        **kwargs: Additional keyword arguments passed to create_subprocess_exec.

    Returns:
        None if the process succeeds.

    Raises:
        subprocess.CalledProcessError: If the process exits with non-zero status.

    Example:
        # Run with prefixed output forwarding
        >>> await check_call_async("python", "script.py", prefix="worker-1")
        # Output: [worker-1 stdout]    Script output here...
        #         [worker-1 stderr]    Error output here...

        # Run with suppressed stdout
        >>> await check_call_async("python", "script.py", suppress_stdout=True)
    """
    # Determine subprocess output modes based on parameters
    stderr_mode = asyncio.subprocess.PIPE if prefix else None
    stdout_mode = asyncio.subprocess.DEVNULL if suppress_stdout else stderr_mode
    input_mode = asyncio.subprocess.PIPE if input is not None else None

    proc = await asyncio.create_subprocess_exec(
        *args, stdin=input_mode, stdout=stdout_mode, stderr=stderr_mode, **kwargs
    )

    try:
        if input is not None:
            # Mode 1: Send input and wait for completion
            stdout = b""
            stderr = b""
            await proc.communicate(input=input)
            retcode = proc.returncode
        elif prefix:
            # Mode 2: Forward output with prefixes (concurrent forwarding)
            assert proc.stdout is not None
            assert proc.stderr is not None
            # Run forwarding and waiting concurrently
            stdout, stderr, retcode = await asyncio.gather(
                forward_and_return_data(sys.stdout, f"{prefix} stdout", proc.stdout),
                forward_and_return_data(sys.stderr, f"{prefix} stderr", proc.stderr),
                proc.wait(),
            )
        else:
            # Mode 3: Default - let process handle its own I/O or suppress it
            assert proc.stdout is None
            assert proc.stderr is None
            stdout = b""
            stderr = b""
            retcode = await proc.wait()
    finally:
        # Always ensure proper cleanup, even if an exception occurs
        await kill_process_async(proc, wait_before_kill_secs=wait_before_kill_secs)

    assert retcode is not None
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, args, stdout, stderr)
