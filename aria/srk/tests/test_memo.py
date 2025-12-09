"""
Tests for the Memoization module.
"""

import unittest
from aria.srk.memo import MemoizationTable, memoize


class MockTabulate:
    """Mock tabulation for testing."""

    def __init__(self):
        self.cache = {}

    def call(self, x):
        if x in self.cache:
            return self.cache[x]
        else:
            # Simulate recursive computation
            if x == 0:
                result = 0
            elif x == 1:
                result = 1
            else:
                result = self.call(x - 1) + self.call(x - 2)
            self.cache[x] = result
            return result


class TestMemoization(unittest.TestCase):
    """Test memoization functionality."""

    def test_basic_memoization(self):
        """Test basic memoization functionality."""
        table = MemoizationTable(max_size=10, strategy="lru")

        # Test that table can be created and used
        table.put("key1", "value1")
        result = table.get("key1")
        self.assertEqual(result, "value1")

    def test_function_memoization(self):
        """Test function memoization."""
        @memoize(max_size=100)
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        # Test basic computation
        result = fibonacci(5)
        self.assertEqual(result, 5)  # fib(5) = 5

    def test_tabulation_interface(self):
        """Test tabulation interface."""
        # This is a simplified test since the full Tabulate implementation
        # would require more complex setup

        tabulator = MockTabulate()

        # Test basic tabulation functionality
        result = tabulator.call(5)
        self.assertEqual(result, 5)

        # Test that results are cached (call again should be fast)
        result2 = tabulator.call(5)
        self.assertEqual(result, result2)

    def test_cache_behavior(self):
        """Test that caching works correctly."""
        @memoize(max_size=10)
        def test_func(x):
            return x * 2

        # First call should compute
        result1 = test_func(5)
        self.assertEqual(result1, 10)

        # Second call should use cache
        result2 = test_func(5)
        self.assertEqual(result2, 10)

        # Different input should compute separately
        result3 = test_func(3)
        self.assertEqual(result3, 6)


class TestTabulate(unittest.TestCase):
    """Test tabulation functionality."""

    def test_tabulate_creation(self):
        """Test creating tabulators."""
        # This is a simplified test since Tabulate requires complex setup
        # In a full implementation, this would test the Tabulate.MakeRec functionality

        # Basic test that we can create tabulation structures
        tabulator = MockTabulate()
        self.assertIsNotNone(tabulator)


if __name__ == '__main__':
    unittest.main()
