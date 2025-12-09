"""
Tests for the pervasives (utility functions) module.

This module tests basic utility functions that are commonly used throughout SRK.
"""

import unittest
from aria.srk.util import (
    IntSet, IntMap, Counter, Stack, Queue, PriorityQueue,
    make_int_set, make_int_map, binary_search, merge_arrays
)


class TestIntSet(unittest.TestCase):
    """Test integer set functionality."""

    def test_creation(self):
        """Test creating integer sets."""
        s = make_int_set()
        self.assertIsInstance(s, IntSet)

    def test_insertion(self):
        """Test inserting elements into integer sets."""
        s = make_int_set()
        s.add(1)
        s.add(2)
        s.add(1)  # Duplicate

        # Should contain exactly 2 elements
        self.assertEqual(len(s), 2)

    def test_membership(self):
        """Test membership testing."""
        s = make_int_set()
        s.add(42)

        self.assertTrue(42 in s)
        self.assertFalse(43 in s)


class TestIntMap(unittest.TestCase):
    """Test integer map functionality."""

    def test_creation(self):
        """Test creating integer maps."""
        m = make_int_map()
        self.assertIsInstance(m, IntMap)

    def test_insertion_and_lookup(self):
        """Test inserting and looking up values."""
        m = make_int_map()
        m[1] = "hello"
        m[2] = "world"

        self.assertEqual(m[1], "hello")
        self.assertEqual(m[2], "world")
        self.assertEqual(len(m), 2)

    def test_missing_key(self):
        """Test behavior with missing keys."""
        m = make_int_map()

        with self.assertRaises(KeyError):
            _ = m[42]


class TestCounter(unittest.TestCase):
    """Test counter functionality."""

    def test_creation(self):
        """Test creating counters."""
        c = Counter()
        self.assertEqual(len(c), 0)

    def test_increment(self):
        """Test incrementing counters."""
        c = Counter()
        c.inc("test")
        c.inc("test")
        c.inc("other")

        self.assertEqual(c["test"], 2)
        self.assertEqual(c["other"], 1)


class TestStack(unittest.TestCase):
    """Test stack functionality."""

    def test_creation(self):
        """Test creating stacks."""
        s = Stack()
        self.assertTrue(s.empty())

    def test_push_pop(self):
        """Test push and pop operations."""
        s = Stack()
        s.push(1)
        s.push(2)

        self.assertFalse(s.empty())
        self.assertEqual(s.pop(), 2)
        self.assertEqual(s.pop(), 1)
        self.assertTrue(s.empty())


class TestQueue(unittest.TestCase):
    """Test queue functionality."""

    def test_creation(self):
        """Test creating queues."""
        q = Queue()
        self.assertTrue(q.empty())

    def test_enqueue_dequeue(self):
        """Test enqueue and dequeue operations."""
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)

        self.assertFalse(q.empty())
        self.assertEqual(q.dequeue(), 1)
        self.assertEqual(q.dequeue(), 2)
        self.assertTrue(q.empty())


class TestPriorityQueue(unittest.TestCase):
    """Test priority queue functionality."""

    def test_creation(self):
        """Test creating priority queues."""
        pq = PriorityQueue()
        self.assertTrue(pq.empty())

    def test_insert_and_extract(self):
        """Test insert and extract operations."""
        pq = PriorityQueue()
        pq.insert(5, "low")
        pq.insert(1, "high")
        pq.insert(3, "medium")

        # Should extract in priority order (lowest first)
        self.assertEqual(pq.extract_min(), (1, "high"))
        self.assertEqual(pq.extract_min(), (3, "medium"))
        self.assertEqual(pq.extract_min(), (5, "low"))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_binary_search(self):
        """Test binary search function."""
        arr = [1, 3, 5, 7, 9]
        self.assertEqual(binary_search(arr, 5), 2)
        self.assertEqual(binary_search(arr, 1), 0)
        self.assertEqual(binary_search(arr, 9), 4)
        self.assertEqual(binary_search(arr, 6), -1)  # Not found

    def test_merge_arrays(self):
        """Test array merging."""
        arr1 = [1, 3, 5]
        arr2 = [2, 4, 6]
        result = merge_arrays(arr1, arr2)

        # Should be sorted merge
        self.assertEqual(result, [1, 2, 3, 4, 5, 6])


if __name__ == '__main__':
    unittest.main()
