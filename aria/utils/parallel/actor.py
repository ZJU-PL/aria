"""Minimal actor model utilities.

Provides lightweight actors with mailboxes, tell/ask, and a system supervisor.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


logger = logging.getLogger("aria.parallel.actor")


class ActorStoppedError(RuntimeError):
    """Raised when an actor stops before processing a request."""


class _ClosableMailbox(queue.Queue):
    """Queue that wakes blocked producers when the actor stops."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._closed = False

    def close(self) -> None:
        with self.mutex:
            self._closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()

    def put(
        self,
        item: tuple[Any, Optional["queue.Queue[Any]"]],
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        with self.not_full:
            if self._closed:
                raise ActorStoppedError("actor mailbox closed")

            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise queue.Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        if self._closed:
                            raise ActorStoppedError("actor mailbox closed")
                        self.not_full.wait(timeout=0.05)
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time.monotonic() + timeout
                    while self._qsize() >= self.maxsize:
                        if self._closed:
                            raise ActorStoppedError("actor mailbox closed")
                        remaining = endtime - time.monotonic()
                        if remaining <= 0.0:
                            raise queue.Full
                        self.not_full.wait(timeout=min(remaining, 0.05))

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class ActorRef:
    """Reference to an actor's mailbox."""

    def __init__(
        self,
        name: str,
        mailbox: _ClosableMailbox,
        stop_event: threading.Event,
    ) -> None:
        self.name = name
        self._mailbox = mailbox
        self._stop_event = stop_event

    def _raise_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise ActorStoppedError(f"actor '{self.name}' is stopped")

    def _put(
        self,
        item: tuple[Any, Optional["queue.Queue[Any]"]],
        timeout: Optional[float],
        *,
        block_forever: bool,
    ) -> None:
        self._raise_if_stopped()
        try:
            if timeout is None:
                if block_forever:
                    self._mailbox.put(item)
                else:
                    self._mailbox.put_nowait(item)
            elif timeout <= 0:
                self._mailbox.put_nowait(item)
            else:
                self._mailbox.put(item, timeout=timeout)
        except ActorStoppedError as exc:
            raise ActorStoppedError(f"actor '{self.name}' is stopped") from exc
        except queue.Full as exc:
            raise TimeoutError(f"actor '{self.name}' mailbox full") from exc

    def tell(self, message: Any, timeout: Optional[float] = None) -> None:
        """Fire-and-forget send.

        By default this fails fast if the mailbox is full instead of blocking the
        caller indefinitely. Pass a timeout to wait for available mailbox space.
        """
        self._put((message, None), timeout, block_forever=False)

    def ask(
        self,
        message: Any,
        timeout: Optional[float] = None,
        *,
        raise_on_error: bool = True,
    ) -> Any:
        """Send and wait for a reply, raising on timeout.

        The timeout covers both mailbox enqueue and reply wait time.
        """
        reply_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        deadline = None if timeout is None else time.monotonic() + timeout
        enqueue_timeout = None
        if deadline is not None:
            enqueue_timeout = max(0.0, deadline - time.monotonic())
        self._put((message, reply_q), enqueue_timeout, block_forever=timeout is None)
        while True:
            try:
                remaining = 0.05 if deadline is None else max(0.0, deadline - time.monotonic())
                res = reply_q.get(timeout=remaining)
                break
            except queue.Empty as exc:
                self._raise_if_stopped()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"actor '{self.name}' timed out waiting for reply"
                    ) from exc
        if raise_on_error and isinstance(res, BaseException):
            raise res
        return res


@dataclass
class ActorHandle:
    """Handle for controlling a spawned actor."""

    ref: ActorRef
    thread: threading.Thread
    stop_event: threading.Event

    def _reply_stopped_to_pending_messages(self) -> None:
        while True:
            try:
                _, reply_q = self.ref._mailbox.get_nowait()
            except queue.Empty:
                return
            if reply_q is not None:
                try:
                    reply_q.put_nowait(
                        ActorStoppedError(
                            f"actor '{self.ref.name}' stopped before replying"
                        )
                    )
                except queue.Full:
                    pass

    def stop(self, timeout: Optional[float] = 1.0) -> None:
        self.stop_event.set()
        self.ref._mailbox.close()
        self._reply_stopped_to_pending_messages()
        # poke mailbox to unblock
        # pylint: disable=protected-access
        try:
            self.ref._mailbox.put_nowait((None, None))
        except (queue.Full, ActorStoppedError):
            pass
        self.thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()


def spawn(
    handler: Callable[[Any], Any],
    *,
    name: Optional[str] = None,
    mailbox_size: int = 1024,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> ActorHandle:
    """Spawn a thread-based actor with the given message handler.

    The handler is called for each message; if a reply channel is present,
    its return value is sent back (exceptions are propagated to the asker).
    """
    mbox = _ClosableMailbox(maxsize=mailbox_size)
    stop_event = threading.Event()
    actor_name = name or f"actor-{id(mbox)}"
    ref = ActorRef(actor_name, mbox, stop_event)

    def stop_actor(exc: Optional[BaseException] = None) -> None:
        stop_event.set()
        mbox.close()
        while True:
            try:
                _, reply_q = mbox.get_nowait()
            except queue.Empty:
                break
            if reply_q is not None:
                try:
                    message = f"actor '{actor_name}' stopped before replying"
                    if exc is not None:
                        message = f"actor '{actor_name}' stopped: {type(exc).__name__}"
                    reply_q.put_nowait(ActorStoppedError(message))
                except queue.Full:
                    pass

        if exc is None:
            return

        if isinstance(exc, Exception):
            logger.exception("actor.loop error name=%s err=%s", actor_name, exc)
            if on_error:
                try:
                    on_error(exc)
                except Exception:
                    logger.exception("actor.on_error failed name=%s", actor_name)
        else:
            logger.error(
                "actor.loop fatal name=%s type=%s", actor_name, type(exc).__name__
            )

    def loop() -> None:
        while not stop_event.is_set():
            try:
                try:
                    msg, reply_q = mbox.get(timeout=0.05)
                except queue.Empty:
                    continue
                if stop_event.is_set():
                    if reply_q is not None:
                        reply_q.put(
                            ActorStoppedError(
                                f"actor '{actor_name}' stopped before replying"
                            )
                        )
                    break
                if reply_q is None:
                    # fire-and-forget
                    try:
                        handler(msg)
                    except BaseException as exc:
                        stop_actor(exc)
                        break
                else:
                    try:
                        res = handler(msg)
                        reply_q.put(res)
                    except BaseException as exc:  # return failure to asker
                        reply_q.put(exc)
            except Exception as exc:
                stop_actor(exc)
                break

    t = threading.Thread(target=loop, name=actor_name, daemon=True)
    t.start()
    return ActorHandle(ref=ref, thread=t, stop_event=stop_event)


class ActorSystem:
    """Manage a set of actors and stop them gracefully."""

    def __init__(self) -> None:
        self._actors: list[ActorHandle] = []

    def spawn(
        self, handler: Callable[[Any], Any], *, name: Optional[str] = None
    ) -> ActorRef:
        h = spawn(handler, name=name)
        self._actors.append(h)
        return h.ref

    def stop_all(self, timeout: Optional[float] = 1.0) -> None:
        for h in self._actors:
            h.stop(timeout=timeout)
        self._actors.clear()


__all__ = ["ActorHandle", "ActorRef", "ActorStoppedError", "ActorSystem", "spawn"]
