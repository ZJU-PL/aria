"""
Tests for the feature tree module.
"""

import unittest
from aria.srk.featureTree import (
    FeatureTree, FeatureVector, empty, of_list, insert, find_leq,
    find_leq_map, remove, rebalance, enum, features
)


class TestFeatureTree(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Feature function that extracts coordinates
        def coord_features(obj):
            return [obj[0], obj[1]]

        self.features = coord_features

    def test_empty_feature_tree(self):
        """Test creating an empty feature tree."""
        ft = empty(self.features)
        self.assertIsNotNone(ft)

        # Empty tree should have no elements
        all_elements = enum(ft)
        self.assertEqual(len(all_elements), 0)

    def test_feature_tree_from_list(self):
        """Test creating a feature tree from a list."""
        elements = [(1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]
        ft = of_list(self.features, elements)

        # Should contain all elements
        all_elements = enum(ft)
        self.assertEqual(len(all_elements), 5)

        # Check that specific elements are present
        self.assertIn((1, 2), all_elements)
        self.assertIn((1, 3), all_elements)
        self.assertIn((2, 1), all_elements)
        self.assertIn((2, 2), all_elements)
        self.assertIn((3, 1), all_elements)

    def test_insert_element(self):
        """Test inserting elements into a feature tree."""
        # Start with empty tree
        ft = empty(self.features)

        # Insert an element
        ft = insert((1, 1), ft)
        self.assertEqual(len(enum(ft)), 1)

        # Insert another element
        ft = insert((2, 2), ft)
        self.assertEqual(len(enum(ft)), 2)

        # Insert element with same feature vector
        ft = insert((1, 1), ft)  # Duplicate
        self.assertEqual(len(enum(ft)), 3)  # Should have two (1,1) elements

    def test_find_leq(self):
        """Test finding elements with feature vectors <= query."""
        elements = [(1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]
        ft = of_list(self.features, elements)

        # Find elements with features <= [1, 3]
        result = find_leq([1, 3], lambda x: x[0] == 1, ft)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 1)  # Should find (1, 2) or (1, 3)

        # Find elements with features <= [2, 1]
        result = find_leq([2, 1], lambda x: x[1] == 1, ft)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 1)  # Should find (2, 1)

        # No element should satisfy this predicate
        result = find_leq([0, 0], lambda x: x[0] > 10, ft)
        self.assertIsNone(result)

    def test_find_leq_map(self):
        """Test finding elements and applying a function."""
        elements = [(1, 2), (1, 3), (2, 1)]
        ft = of_list(self.features, elements)

        # Find element and apply function
        def get_first_coord(obj):
            return obj[0] if obj[0] == 1 else None

        result = find_leq_map([1, 3], get_first_coord, ft)
        self.assertEqual(result, 1)

        # No element should satisfy the condition
        def impossible_func(obj):
            return obj[0] if obj[0] > 10 else None

        result = find_leq_map([1, 3], impossible_func, ft)
        self.assertIsNone(result)

    def test_remove_element(self):
        """Test removing elements from a feature tree."""
        elements = [(1, 2), (1, 3), (2, 1), (2, 2)]
        ft = of_list(self.features, elements)

        # Remove an element
        def element_equal(a, b):
            return a == b

        ft = remove(element_equal, (1, 2), ft)
        all_elements = enum(ft)

        # Should have 3 elements now
        self.assertEqual(len(all_elements), 3)
        self.assertNotIn((1, 2), all_elements)
        self.assertIn((1, 3), all_elements)

    def test_rebalance(self):
        """Test rebalancing a feature tree."""
        elements = [(1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]
        ft = of_list(self.features, elements)

        # Rebalance should preserve all elements
        ft_rebalanced = rebalance(ft)
        original_elements = set(enum(ft))
        rebalanced_elements = set(enum(ft_rebalanced))

        self.assertEqual(original_elements, rebalanced_elements)

    def test_features_function(self):
        """Test the features extraction function."""
        ft = empty(self.features)

        # Test feature extraction
        element = (2, 3)
        fv = features(ft, element)
        self.assertEqual(fv, [2, 3])

    def test_complex_feature_vectors(self):
        """Test with feature vectors of different lengths."""
        # Feature function that returns vectors of different lengths
        def variable_features(obj):
            if obj[0] == 1:
                return [obj[0]]  # Shorter vector
            else:
                return [obj[0], obj[1]]  # Longer vector

        elements = [(1, 2), (2, 3), (3, 4)]
        ft = of_list(variable_features, elements)

        # Should handle different vector lengths
        all_elements = enum(ft)
        self.assertEqual(len(all_elements), 3)


if __name__ == '__main__':
    unittest.main()
