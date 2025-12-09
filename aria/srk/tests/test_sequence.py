"""
Tests for the sequence analysis module.
"""

import unittest
from aria.srk.sequence import (
    UltimatelyPeriodicSequence, SequenceAnalyzer,
    fibonacci_sequence, arithmetic_sequence, geometric_sequence
)


class TestUltimatelyPeriodicSequence(unittest.TestCase):
    """Test ultimately periodic sequence operations."""

    def test_creation(self):
        """Test sequence creation."""
        seq = UltimatelyPeriodicSequence((1, 2, 3), (4, 5))
        self.assertEqual(seq.transient, (1, 2, 3))
        self.assertEqual(seq.periodic, (4, 5))
        self.assertEqual(seq.length(), 3)
        self.assertEqual(seq.period(), 2)

    def test_invalid_creation(self):
        """Test creation with empty periodic part."""
        with self.assertRaises(ValueError):
            UltimatelyPeriodicSequence((1, 2), ())

    def test_get_element(self):
        """Test element access."""
        seq = UltimatelyPeriodicSequence((1, 2), (3, 4))

        # Transient part
        self.assertEqual(seq.get(0), 1)
        self.assertEqual(seq.get(1), 2)

        # Periodic part
        self.assertEqual(seq.get(2), 3)  # First element of periodic
        self.assertEqual(seq.get(3), 4)  # Second element of periodic
        self.assertEqual(seq.get(4), 3)  # Back to first element

    def test_indexing(self):
        """Test indexing with [] notation."""
        seq = UltimatelyPeriodicSequence((10,), (20, 30))

        self.assertEqual(seq[0], 10)
        self.assertEqual(seq[1], 20)
        self.assertEqual(seq[2], 30)
        self.assertEqual(seq[3], 20)

    def test_take(self):
        """Test taking first n elements."""
        seq = UltimatelyPeriodicSequence((1, 2), (3, 4))
        first_5 = seq.take(5)

        expected = [1, 2, 3, 4, 3]
        self.assertEqual(first_5, expected)

    def test_enum(self):
        """Test infinite enumeration."""
        seq = UltimatelyPeriodicSequence((1,), (2, 3))
        enum = seq.enum()

        # Take first few elements
        elements = []
        for i, elem in enumerate(enum):
            elements.append(elem)
            if i >= 4:
                break

        expected = [1, 2, 3, 2, 3]
        self.assertEqual(elements, expected)

    def test_map(self):
        """Test mapping function over sequence."""
        seq = UltimatelyPeriodicSequence((1, 2), (3, 4))
        doubled = seq.map(lambda x: x * 2)

        self.assertEqual(doubled.transient, (2, 4))
        self.assertEqual(doubled.periodic, (6, 8))

    def test_filter(self):
        """Test filtering sequence elements."""
        seq = UltimatelyPeriodicSequence((1, 2, 3), (4, 5, 6))
        filtered = seq.filter(lambda x: x % 2 == 0)

        # Should filter both transient and periodic parts
        self.assertEqual(filtered.transient, (2,))
        self.assertEqual(filtered.periodic, (4, 6))

    def test_concatenation(self):
        """Test sequence concatenation."""
        seq1 = UltimatelyPeriodicSequence((1, 2), (3, 4))
        seq2 = UltimatelyPeriodicSequence((5, 6), (7, 8))

        concatenated = seq1 + seq2
        self.assertEqual(concatenated.transient, (1, 2, 5, 6))
        self.assertEqual(concatenated.periodic, (3, 4))  # Uses first sequence's periodic


class TestSequenceAnalyzer(unittest.TestCase):
    """Test sequence analysis tools."""

    def test_detect_period_simple(self):
        """Test period detection on simple periodic sequence."""
        sequence = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        period = SequenceAnalyzer.detect_period(sequence)

        self.assertEqual(period, 3)

    def test_detect_period_non_periodic(self):
        """Test period detection on non-periodic sequence."""
        sequence = [1, 2, 3, 4, 5, 6]
        period = SequenceAnalyzer.detect_period(sequence)

        self.assertIsNone(period)

    def test_find_repeating_pattern(self):
        """Test finding repeating patterns."""
        sequence = [1, 2, 3, 1, 2, 3, 1, 2]
        result = SequenceAnalyzer.find_repeating_pattern(sequence)

        self.assertIsNotNone(result)
        start_index, pattern = result
        self.assertEqual(start_index, 0)
        self.assertEqual(pattern, [1, 2, 3])

    def test_autocorrelation(self):
        """Test autocorrelation computation."""
        sequence = [1, 2, 3, 1, 2, 3]
        correlations = SequenceAnalyzer.compute_autocorrelation(sequence)

        self.assertEqual(len(correlations), 5)  # Lags 1 through 5
        # Should have high correlation at lag 3
        self.assertGreater(correlations[2], 0.5)  # Index 2 corresponds to lag 3

    def test_detect_ultimately_periodic(self):
        """Test ultimately periodic detection."""
        # Sequence: 1, 2, 3, 4, 3, 4, 3, 4, ...
        sequence = [1, 2, 3, 4, 3, 4, 3, 4]
        result = SequenceAnalyzer.detect_ultimately_periodic(sequence)

        self.assertIsNotNone(result)
        if result:
            self.assertEqual(result.transient, (1, 2))
            self.assertEqual(result.periodic, (3, 4))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility sequence functions."""

    def test_arithmetic_sequence(self):
        """Test arithmetic sequence generation."""
        seq = arithmetic_sequence(1, 2, 5)
        expected = [1, 3, 5, 7, 9]
        self.assertEqual(seq, expected)

    def test_geometric_sequence(self):
        """Test geometric sequence generation."""
        seq = geometric_sequence(2, 3, 4)
        expected = [2, 6, 18, 54]
        self.assertEqual(seq, expected)

    def test_fibonacci_sequence(self):
        """Test Fibonacci sequence creation."""
        fib = fibonacci_sequence()

        # Check some elements
        self.assertEqual(fib[0], 1)
        self.assertEqual(fib[1], 1)
        self.assertEqual(fib[2], 2)
        self.assertEqual(fib[3], 3)
        self.assertEqual(fib[4], 5)
        self.assertEqual(fib[8], 34)  # First periodic element


if __name__ == '__main__':
    unittest.main()
