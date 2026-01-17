# coding: utf-8
"""Producer/consumer scheduler for Dissolve's worker architecture."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from typing import Dict, List, Set

from .models import DilemmaQuery

logger = logging.getLogger(__name__)


# ------------------------------ Scheduler Architecture ------------------------------ #


class Scheduler:
    """
    Scheduler component that coordinates the producer/consumer architecture.

    Implements the queue-based approach from Algorithm 3 of the paper.
    """

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers
        self.query_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.idle_workers: Set[int] = set(range(num_workers))
        self.active_queries: Dict[int, DilemmaQuery] = {}

        # Synchronization
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def producer_loop(
        self, dissolve_instance: "Dissolve", clauses: List[List[int]]
    ) -> None:
        """Producer loop that generates dilemma queries."""
        round_id = 0

        while not self.stop_event.is_set():
            if len(self.idle_workers) < self.num_workers:
                # Wait for more workers to become idle
                time.sleep(0.01)
                continue

            # Generate queries for this round
            # pylint: disable=protected-access
            queries = dissolve_instance._generate_dilemma_queries(round_id)

            if not queries:
                break

            # Submit queries to workers
            for query in queries:
                self.query_queue.put((query.query_id, query, clauses))

            # Wait for results or timeout
            results_collected = 0
            expected_results = len(queries)

            while results_collected < expected_results and not self.stop_event.is_set():
                try:
                    result = self.result_queue.get(timeout=1.0)
                    results_collected += 1
                    # Process result
                    # pylint: disable=protected-access
                    dissolve_instance._process_worker_result(result)
                except queue.Empty:
                    continue

            round_id += 1

    def worker_loop(self, worker_id: int, dissolve_instance: "Dissolve") -> None:
        """Worker loop that processes SAT queries."""
        while not self.stop_event.is_set():
            try:
                query_id, query, clauses = self.query_queue.get(timeout=1.0)

                with self.lock:
                    self.idle_workers.discard(worker_id)
                    self.active_queries[query_id] = query

                # Process the query
                # pylint: disable=protected-access
                result = dissolve_instance._solve_dilemma_query(query, clauses)

                # Return result
                self.result_queue.put((query_id, result))

                with self.lock:
                    self.idle_workers.add(worker_id)
                    del self.active_queries[query_id]

            except queue.Empty:
                continue
            except (RuntimeError, ValueError) as e:
                logger.exception("Worker %d error: %s", worker_id, e)
                with self.lock:
                    self.idle_workers.add(worker_id)
