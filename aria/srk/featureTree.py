"""
Feature tree for organizing elements by feature vectors.

This module provides a tree data structure that organizes a collection of elements
by their feature vectors, allowing efficient one-sided range queries.
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Union, Optional, Callable, Any, TypeVar, Generic
from dataclasses import dataclass
import random
from enum import Enum

# Type variable for elements
T = TypeVar('T')

# Feature vector type
FeatureVector = List[int]


@dataclass
class FeatureTreeNode:
    """Node in a feature tree."""
    value: int  # Split value
    feature: int  # Feature index used for splitting
    left: FeatureTree  # Left subtree
    right: FeatureTree  # Right subtree


@dataclass
class FeatureTree:
    """Feature tree data structure."""
    features: Callable[[T], FeatureVector]  # Function to extract features from elements
    tree: Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]  # Tree structure

    def __post_init__(self):
        if isinstance(self.tree, list):
            # Sort buckets by feature vector
            self.tree.sort(key=lambda x: x[0])


def compare_feature_vectors(fv1: FeatureVector, fv2: FeatureVector) -> int:
    """Compare two feature vectors lexicographically."""
    for i in range(min(len(fv1), len(fv2))):
        if fv1[i] < fv2[i]:
            return -1
        elif fv1[i] > fv2[i]:
            return 1
    return len(fv1) - len(fv2)


def feature_vector_leq(fv1: FeatureVector, fv2: FeatureVector) -> bool:
    """Check if fv1 <= fv2 component-wise."""
    return all(fv1[i] <= fv2[i] for i in range(min(len(fv1), len(fv2))))


def empty(features: Callable[[T], FeatureVector]) -> FeatureTree[T]:
    """Create an empty feature tree."""
    return FeatureTree(features, [])


def insert_bucket(fv: FeatureVector, elt: T, buckets: List[Tuple[FeatureVector, List[T]]]) -> List[Tuple[FeatureVector, List[T]]]:
    """Insert an element into a list of buckets."""
    for i, (fv_bucket, elts) in enumerate(buckets):
        cmp = compare_feature_vectors(fv, fv_bucket)
        if cmp == 0:
            # Same feature vector, add to existing bucket
            buckets[i] = (fv_bucket, elts + [elt])
            return buckets
        elif cmp < 0:
            # Insert new bucket before this one
            buckets.insert(i, (fv, [elt]))
            return buckets

    # Insert at the end
    buckets.append((fv, [elt]))
    return buckets


def make_tree(num_features: int, bucket: List[Tuple[FeatureVector, List[T]]]) -> Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]:
    """Build a tree from a bucket of elements."""
    if len(bucket) <= 4:  # bucket_size = 4
        return bucket

    # Choose a split feature and value
    if not bucket:
        return bucket

    # Pick a random element to use as reference
    ref_idx = random.randint(0, len(bucket) - 1)
    ref_fv, _ = bucket[ref_idx]

    # Count how many elements are <= ref element for each feature
    counts = [0] * num_features
    for fv, _ in bucket:
        for i in range(num_features):
            if i < len(fv) and fv[i] <= ref_fv[i]:
                counts[i] += 1

    # Choose the feature that gives the most balanced split
    mid = (len(bucket) + 1) // 2
    best_feature = 0
    best_balance = abs(mid - counts[0])

    for i in range(num_features):
        balance = abs(mid - counts[i])
        if balance < best_balance or (balance == best_balance and random.random() < 0.5):
            best_balance = balance
            best_feature = i

    # Split the bucket
    left_bucket = []
    right_bucket = []

    for fv, elts in bucket:
        if fv[best_feature] <= ref_fv[best_feature]:
            left_bucket.append((fv, elts))
        else:
            right_bucket.append((fv, elts))

    # Recursively build subtrees
    left_tree = make_tree(num_features, left_bucket)
    right_tree = make_tree(num_features, right_bucket)

    return FeatureTreeNode(
        value=ref_fv[best_feature],
        feature=best_feature,
        left=FeatureTree(bucket[0][1][0].__class__() if bucket else None, left_tree),  # type: ignore
        right=FeatureTree(bucket[0][1][0].__class__() if bucket else None, right_tree)  # type: ignore
    )


def of_list(features: Callable[[T], FeatureVector], elements: List[T]) -> FeatureTree[T]:
    """Create a feature tree from a list of elements."""
    if not elements:
        return empty(features)

    # Extract feature vectors and sort
    feature_elements = [(features(elt), elt) for elt in elements]
    feature_elements.sort(key=lambda x: x[0])

    # Group by feature vector
    buckets = []
    current_fv = None
    current_elts = []

    for fv, elt in feature_elements:
        if current_fv is None or compare_feature_vectors(fv, current_fv) != 0:
            if current_fv is not None:
                buckets.append((current_fv, current_elts))
            current_fv = fv
            current_elts = [elt]
        else:
            current_elts.append(elt)

    if current_fv is not None:
        buckets.append((current_fv, current_elts))

    # Build tree
    if not buckets:
        return empty(features)

    first_fv, _ = buckets[0]
    num_features = len(first_fv)
    tree_structure = make_tree(num_features, buckets)

    return FeatureTree(features, tree_structure)


def insert(element: T, tree: FeatureTree[T]) -> FeatureTree[T]:
    """Insert an element into a feature tree."""
    fv = tree.features(element)
    num_features = len(fv)

    def insert_into_tree(t: Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]) -> Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]:
        if isinstance(t, list):
            # Leaf node - insert into buckets
            return insert_bucket(fv, element, t)
        else:
            # Internal node
            if fv[t.feature] <= t.value:
                return FeatureTreeNode(
                    value=t.value,
                    feature=t.feature,
                    left=FeatureTree(tree.features, insert_into_tree(t.left.tree)),
                    right=t.right
                )
            else:
                return FeatureTreeNode(
                    value=t.value,
                    feature=t.feature,
                    left=t.left,
                    right=FeatureTree(tree.features, insert_into_tree(t.right.tree))
                )

    new_tree = insert_into_tree(tree.tree)
    return FeatureTree(tree.features, new_tree)


def find_leq(fv: FeatureVector, predicate: Callable[[T], bool], tree: FeatureTree[T]) -> Optional[T]:
    """Find an element with features <= fv that satisfies the predicate."""
    def find_in_buckets(buckets: List[Tuple[FeatureVector, List[T]]]) -> Optional[T]:
        for fv_bucket, elts in buckets:
            if feature_vector_leq(fv_bucket, fv):
                for elt in elts:
                    if predicate(elt):
                        return elt
        return None

    def find_in_tree(t: Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]) -> Optional[T]:
        if isinstance(t, list):
            return find_in_buckets(t)
        else:
            # Internal node
            if fv[t.feature] <= t.value:
                # Check left subtree first
                result = find_in_tree(t.left.tree)
                if result is not None:
                    return result
                # Then check right subtree
                return find_in_tree(t.right.tree)
            else:
                # Only check right subtree
                return find_in_tree(t.right.tree)

    return find_in_tree(tree.tree)


def find_leq_map(fv: FeatureVector, f: Callable[[T], Optional[Any]], tree: FeatureTree[T]) -> Optional[Any]:
    """Find an element with features <= fv where f is defined, return f(element)."""
    def find_in_buckets(buckets: List[Tuple[FeatureVector, List[T]]]) -> Optional[Any]:
        for fv_bucket, elts in buckets:
            if feature_vector_leq(fv_bucket, fv):
                for elt in elts:
                    result = f(elt)
                    if result is not None:
                        return result
        return None

    def find_in_tree(t: Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]) -> Optional[Any]:
        if isinstance(t, list):
            return find_in_buckets(t)
        else:
            # Internal node
            if fv[t.feature] <= t.value:
                # Check left subtree first
                result = find_in_tree(t.left.tree)
                if result is not None:
                    return result
                # Then check right subtree
                return find_in_tree(t.right.tree)
            else:
                # Only check right subtree
                return find_in_tree(t.right.tree)

    return find_in_tree(tree.tree)


def remove(equal: Callable[[T, T], bool], element: T, tree: FeatureTree[T]) -> FeatureTree[T]:
    """Remove an element from the feature tree."""
    fv = tree.features(element)

    def remove_from_buckets(buckets: List[Tuple[FeatureVector, List[T]]]) -> List[Tuple[FeatureVector, List[T]]]:
        result = []
        for fv_bucket, elts in buckets:
            if compare_feature_vectors(fv_bucket, fv) == 0:
                # Same feature vector, remove matching elements
                new_elts = [elt for elt in elts if not equal(elt, element)]
                if new_elts:
                    result.append((fv_bucket, new_elts))
            else:
                result.append((fv_bucket, elts))
        return result

    def remove_from_tree(t: Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]) -> Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]:
        if isinstance(t, list):
            return remove_from_buckets(t)
        else:
            # Internal node
            if fv[t.feature] <= t.value:
                return FeatureTreeNode(
                    value=t.value,
                    feature=t.feature,
                    left=FeatureTree(tree.features, remove_from_tree(t.left.tree)),
                    right=t.right
                )
            else:
                return FeatureTreeNode(
                    value=t.value,
                    feature=t.feature,
                    left=t.left,
                    right=FeatureTree(tree.features, remove_from_tree(t.right.tree))
                )

    new_tree = remove_from_tree(tree.tree)
    return FeatureTree(tree.features, new_tree)


def rebalance(tree: FeatureTree[T]) -> FeatureTree[T]:
    """Rebalance a feature tree."""
    def collect_buckets(t: Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]) -> List[Tuple[FeatureVector, List[T]]]:
        if isinstance(t, list):
            return t
        else:
            left_buckets = collect_buckets(t.left.tree)
            right_buckets = collect_buckets(t.right.tree)
            return left_buckets + right_buckets

    buckets = collect_buckets(tree.tree)

    if not buckets:
        return empty(tree.features)

    first_fv, _ = buckets[0]
    num_features = len(first_fv)
    new_tree = make_tree(num_features, buckets)

    return FeatureTree(tree.features, new_tree)


def enum(tree: FeatureTree[T]) -> List[T]:
    """Convert the feature tree to a list of all elements."""
    def collect_elements(t: Union[List[Tuple[FeatureVector, List[T]]], FeatureTreeNode]) -> List[T]:
        if isinstance(t, list):
            result = []
            for _, elts in t:
                result.extend(elts)
            return result
        else:
            left_elts = collect_elements(t.left.tree)
            right_elts = collect_elements(t.right.tree)
            return left_elts + right_elts

    return collect_elements(tree.tree)


# Convenience functions
def features(tree: FeatureTree[T], element: T) -> FeatureVector:
    """Get the feature vector for an element."""
    return tree.features(element)


# Example usage and testing
if __name__ == "__main__":
    # Example feature function
    def simple_features(obj: Tuple[int, int]) -> FeatureVector:
        return [obj[0], obj[1]]

    # Create a feature tree
    elements = [(1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]
    ft = of_list(simple_features, elements)

    # Insert an element
    ft = insert((1, 1), ft)

    # Find elements
    result = find_leq([1, 2], lambda x: x[0] == 1, ft)
    print(f"Found element: {result}")

    # List all elements
    all_elements = enum(ft)
    print(f"All elements: {all_elements}")
