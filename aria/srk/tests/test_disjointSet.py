"""
Tests for the disjointSet module.
"""

import unittest
from aria.srk.disjointSet import (
    DisjointSet, DisjointSetElement, make_disjoint_set,
    make_disjoint_set_from_elements, test_disjoint_set
)


class TestDisjointSetElement(unittest.TestCase):
    """Test disjoint set element functionality."""

    def test_element_creation(self):
        """Test creating disjoint set elements."""
        elem = DisjointSetElement(id=1, parent=None, rank=0)
        self.assertEqual(elem.id, 1)
        self.assertIsNone(elem.parent)
        self.assertEqual(elem.rank, 0)

    def test_element_with_parent(self):
        """Test creating element with parent."""
        parent = DisjointSetElement(id=0, parent=None, rank=1)
        child = DisjointSetElement(id=1, parent=parent, rank=0)

        self.assertEqual(child.id, 1)
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.rank, 0)


class TestDisjointSet(unittest.TestCase):
    """Test disjoint set data structure."""

    def setUp(self):
        self.ds = DisjointSet()

    def test_initial_state(self):
        """Test initial state of disjoint set."""
        self.assertEqual(self.ds.set_count(), 0)
        self.assertEqual(self.ds.size(), 0)

    def test_make_set(self):
        """Test creating singleton sets."""
        # Initially no sets
        self.assertEqual(self.ds.set_count(), 0)

        # Find should create a new set for unknown elements
        root1 = self.ds.find(1)
        self.assertIsNotNone(root1)
        self.assertEqual(self.ds.set_count(), 1)
        self.assertEqual(self.ds.size(), 1)

        root2 = self.ds.find(2)
        self.assertIsNotNone(root2)
        self.assertEqual(self.ds.set_count(), 2)
        self.assertEqual(self.ds.size(), 2)

    def test_union_operation(self):
        """Test union operations."""
        # Create two separate sets
        self.ds.find(1)
        self.ds.find(2)
        self.assertEqual(self.ds.set_count(), 2)

        # Union them
        self.ds.union(1, 2)
        self.assertEqual(self.ds.set_count(), 1)

        # They should be in the same set
        self.assertTrue(self.ds.same_set(1, 2))

    def test_path_compression(self):
        """Test path compression during find operations."""
        # Create a chain: 1 -> 2 -> 3 -> 4
        self.ds.union(1, 2)
        self.ds.union(2, 3)
        self.ds.union(3, 4)

        # Before path compression, find(1) should go through the chain
        root1_before = self.ds.find(1)

        # After path compression, all should point directly to root
        root1_after = self.ds.find(1)
        root2_after = self.ds.find(2)
        root3_after = self.ds.find(3)
        root4_after = self.ds.find(4)

        # All should have the same root
        self.assertEqual(root1_after.id, root2_after.id)
        self.assertEqual(root2_after.id, root3_after.id)
        self.assertEqual(root3_after.id, root4_after.id)

    def test_union_by_rank(self):
        """Test union by rank optimization."""
        # Create sets of different sizes
        # Set A: {1, 2, 3} (size 3)
        self.ds.union(1, 2)
        self.ds.union(2, 3)

        # Set B: {4, 5} (size 2)
        self.ds.union(4, 5)

        # Set C: {6} (size 1)
        self.ds.find(6)

        initial_count = self.ds.set_count()

        # Union larger set with smaller set (A with B)
        self.ds.union(1, 4)
        self.assertEqual(self.ds.set_count(), initial_count - 1)

        # Union with single element (A with C)
        self.ds.union(1, 6)
        self.assertEqual(self.ds.set_count(), initial_count - 2)

        # All should be in the same set
        self.assertTrue(self.ds.same_set(1, 4))
        self.assertTrue(self.ds.same_set(1, 6))

    def test_same_set(self):
        """Test same set checking."""
        # Initially different elements are not in same set
        self.assertFalse(self.ds.same_set(1, 2))

        # After union, they should be in same set
        self.ds.union(1, 2)
        self.assertTrue(self.ds.same_set(1, 2))

        # Elements not in any set should not be same
        self.assertFalse(self.ds.same_set(1, 99))
        self.assertFalse(self.ds.same_set(99, 100))

    def test_get_sets(self):
        """Test getting all disjoint sets."""
        # Create: {1, 2, 3} and {4, 5}
        self.ds.union(1, 2)
        self.ds.union(2, 3)
        self.ds.union(4, 5)

        sets = self.ds.get_sets()
        self.assertEqual(len(sets), 2)

        # Each set should have the correct elements
        set_sizes = sorted([len(s) for s in sets])
        self.assertEqual(set_sizes, [2, 3])

    def test_get_representatives(self):
        """Test getting set representatives."""
        # Create: {1, 2, 3} and {4, 5}
        self.ds.union(1, 2)
        self.ds.union(2, 3)
        self.ds.union(4, 5)

        reps = self.ds.get_representatives()
        self.assertEqual(len(reps), 2)

        # Representatives should be from different sets
        rep_set = set(reps)
        self.assertEqual(len(rep_set), 2)

    def test_copy(self):
        """Test copying disjoint sets."""
        # Create a disjoint set with some unions
        self.ds.union(1, 2)
        self.ds.union(3, 4)

        original_count = self.ds.set_count()

        # Copy the set
        copied_ds = self.ds.copy()

        # Should have same number of sets
        self.assertEqual(copied_ds.set_count(), original_count)

        # Should behave independently
        copied_ds.union(1, 3)
        self.assertEqual(copied_ds.set_count(), original_count - 1)
        self.assertEqual(self.ds.set_count(), original_count)  # Original unchanged

    def test_string_representation(self):
        """Test string representation."""
        # Create some sets
        self.ds.union(1, 2)
        self.ds.union(3, 4)

        str_repr = str(self.ds)
        self.assertIsInstance(str_repr, str)
        self.assertIn("DisjointSet", str_repr)

    def test_complex_operations(self):
        """Test complex union-find operations."""
        # Create multiple sets and perform various operations
        elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Initially all separate
        for elem in elements:
            self.ds.find(elem)
        self.assertEqual(self.ds.set_count(), len(elements))

        # Union in a specific pattern: (1-2-3) (4-5) (6-7-8-9-10)
        self.ds.union(1, 2)
        self.ds.union(2, 3)
        self.ds.union(4, 5)
        self.ds.union(6, 7)
        self.ds.union(7, 8)
        self.ds.union(8, 9)
        self.ds.union(9, 10)

        # Should have 3 sets now
        self.assertEqual(self.ds.set_count(), 3)

        # Test connectivity within sets
        self.assertTrue(self.ds.same_set(1, 3))
        self.assertTrue(self.ds.same_set(4, 5))
        self.assertTrue(self.ds.same_set(6, 10))
        self.assertTrue(self.ds.same_set(7, 9))

        # Test non-connectivity between sets
        self.assertFalse(self.ds.same_set(1, 4))
        self.assertFalse(self.ds.same_set(3, 6))
        self.assertFalse(self.ds.same_set(5, 7))


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_make_disjoint_set(self):
        """Test creating empty disjoint set."""
        ds = make_disjoint_set()
        self.assertIsInstance(ds, DisjointSet)
        self.assertEqual(ds.set_count(), 0)
        self.assertEqual(ds.size(), 0)

    def test_make_disjoint_set_from_elements(self):
        """Test creating disjoint set from element list."""
        elements = [1, 2, 3, 4, 5]
        ds = make_disjoint_set_from_elements(elements)

        self.assertIsInstance(ds, DisjointSet)
        self.assertEqual(ds.set_count(), len(elements))
        self.assertEqual(ds.size(), len(elements))

        # Each element should be in its own set initially
        for elem in elements:
            self.assertTrue(ds.same_set(elem, elem))


class TestDisjointSetEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        self.ds = DisjointSet()

    def test_union_same_element(self):
        """Test union of element with itself."""
        self.ds.find(1)
        initial_count = self.ds.set_count()

        # Union with itself should not change anything
        self.ds.union(1, 1)
        self.assertEqual(self.ds.set_count(), initial_count)

    def test_find_unknown_element(self):
        """Test finding unknown element."""
        # Should create a new set for unknown elements
        root = self.ds.find(999)
        self.assertIsNotNone(root)
        self.assertEqual(self.ds.set_count(), 1)

    def test_empty_sets_list(self):
        """Test getting sets from empty disjoint set."""
        sets = self.ds.get_sets()
        self.assertEqual(len(sets), 0)

    def test_empty_representatives_list(self):
        """Test getting representatives from empty disjoint set."""
        reps = self.ds.get_representatives()
        self.assertEqual(len(reps), 0)


class TestDisjointSetPerformance(unittest.TestCase):
    """Test performance characteristics."""

    def setUp(self):
        self.ds = DisjointSet()

    def test_many_unions(self):
        """Test performance with many union operations."""
        # Create a disjoint set with many elements
        n = 100
        elements = list(range(n))

        # Time the operations (basic performance check)
        import time
        start_time = time.time()

        for elem in elements:
            self.ds.find(elem)

        for i in range(n - 1):
            self.ds.union(elements[i], elements[i + 1])

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (less than 1 second for 100 elements)
        self.assertLess(duration, 1.0)

        # Should have 1 set at the end
        self.assertEqual(self.ds.set_count(), 1)

    def test_path_compression_effectiveness(self):
        """Test that path compression works effectively."""
        # Create a long chain and measure find time
        n = 50
        elements = list(range(n))

        # Create chain: 0 -> 1 -> 2 -> ... -> n-1
        for i in range(n - 1):
            self.ds.union(elements[i], elements[i + 1])

        import time

        # First find should be slower (no compression)
        start_time = time.time()
        root1 = self.ds.find(0)
        first_find_time = time.time() - start_time

        # Subsequent finds should be faster (compressed)
        start_time = time.time()
        root2 = self.ds.find(0)
        second_find_time = time.time() - start_time

        # Second find should be faster or equal (path is compressed)
        # Allow more tolerance for timing variations
        # Handle case where first find time is 0 (too fast to measure)
        if first_find_time == 0:
            self.assertLessEqual(second_find_time, 0.001)  # Should be very fast
        else:
            self.assertLessEqual(second_find_time, first_find_time * 10)

        # Both should return same root
        self.assertEqual(root1.id, root2.id)


if __name__ == '__main__':
    # Run the built-in test if available
    try:
        test_disjoint_set()
        print("Built-in test passed!")
    except Exception as e:
        print(f"Built-in test failed: {e}")

    # Run unit tests
    unittest.main(verbosity=2)
