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


class ActorRef:
    """Reference to an actor's mailbox."""

    def __init__(self, name: str, mailbox: "queue.Queue[tuple]") -> None:
        self.name = name
        self._mailbox = mailbox

    def _put(
        self,
        item: tuple[Any, Optional["queue.Queue[Any]"]],
        timeout: Optional[float],
        *,
        block_forever: bool,
    ) -> None:
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
        try:
            remaining = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
            res = reply_q.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError(f"actor '{self.name}' timed out waiting for reply") from exc
        if raise_on_error and isinstance(res, BaseException):
            raise res
        return res


@dataclass
class ActorHandle:
    """Handle for controlling a spawned actor."""

    ref: ActorRef
    thread: threading.Thread
    stop_event: threading.Event

    def stop(self, timeout: Optional[float] = 1.0) -> None:
        self.stop_event.set()
        # poke mailbox to unblock
        # pylint: disable=protected-access
        try:
            self.ref._mailbox.put_nowait((None, None))
        except queue.Full:
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
    mbox: "queue.Queue[tuple]" = queue.Queue(maxsize=mailbox_size)
    stop_event = threading.Event()
    actor_name = name or f"actor-{id(mbox)}"
    ref = ActorRef(actor_name, mbox)

    def loop() -> None:
        while not stop_event.is_set():
            try:
                try:
                    msg, reply_q = mbox.get(timeout=0.05)
                except queue.Empty:
                    continue
                if stop_event.is_set():
                    break
                if reply_q is None:
                    # fire-and-forget
                    handler(msg)
                else:
                    try:
                        res = handler(msg)
                        reply_q.put(res)
                    except BaseException as exc:  # return failure to asker
                        reply_q.put(exc)
            except Exception as exc:
                logger.exception("actor.loop error name=%s err=%s", actor_name, exc)
                if on_error:
                    try:
                        on_error(exc)
                    except Exception:
                        logger.exception("actor.on_error failed name=%s", actor_name)

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
