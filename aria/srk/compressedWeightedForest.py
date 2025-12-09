"""
Compressed weighted forest data structure.

This module implements a union-find (disjoint-set) data structure with weights,
using path compression for efficiency. Each node can have a weighted path to its
parent, where weights come from an arbitrary monoid.

The implementation follows the OCaml compressedWeightedForest.ml module.
"""

from __future__ import annotations
from typing import Generic, TypeVar, Callable, Optional, List
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class Link(Generic[T]):
    """A link from a node to its parent with associated weight."""
    parent: int
    weight: T


class CompressedWeightedForest(Generic[T]):
    """
    Compressed weighted forest using union-find with path compression.
    
    Each node is identified by an integer index. Nodes can be linked together
    with weighted edges, where weights come from a monoid (multiplication
    operation and identity element).
    
    Attributes:
        links: List of optional links (None for roots, Link for non-roots)
        one: Identity element of the monoid
        mul: Multiplication operation for combining weights
    """
    
    def __init__(self, mul: Callable[[T, T], T], one: T):
        """
        Initialize a compressed weighted forest.
        
        Args:
            mul: Monoid multiplication operation (associative)
            one: Identity element of the monoid
        """
        self.links: List[Optional[Link[T]]] = []
        self.one: T = one
        self.mul: Callable[[T, T], T] = mul
    
    def compress(self, i: int) -> tuple[int, T]:
        """
        Find the root of the tree containing node i and compress the path.
        
        This implements path compression: as we traverse from i to the root,
        we update each node to point directly to the root, combining weights
        along the way.
        
        Args:
            i: Node index
            
        Returns:
            Tuple of (root_index, combined_weight_from_i_to_root)
        """
        link = self.links[i]
        
        if link is None:
            # i is already a root
            return (i, self.one)
        
        # Recursively find root and compress path
        root, weight = self.compress(link.parent)
        
        # Update link to point directly to root if not already
        if root != link.parent:
            # Combine the weight to parent with weight from parent to root
            new_weight = self.mul(link.weight, weight)
            self.links[i] = Link(parent=root, weight=new_weight)
        
        # Return root and combined weight
        return (root, self.mul(link.weight, weight))
    
    def root(self) -> int:
        """
        Create a new root node.
        
        Returns:
            Index of the new root node
        """
        pos = len(self.links)
        self.links.append(None)
        return pos
    
    def find(self, i: int) -> int:
        """
        Find the root of the tree containing node i.
        
        Args:
            i: Node index
            
        Returns:
            Index of the root node
        """
        root, _ = self.compress(i)
        return root
    
    def eval(self, i: int) -> T:
        """
        Get the combined weight from node i to its root.
        
        Args:
            i: Node index
            
        Returns:
            Combined weight along the path from i to root
        """
        _, weight = self.compress(i)
        return weight
    
    def link(self, child: int, weight: T, parent: int) -> None:
        """
        Link a child node to a parent with a given weight.
        
        The child must not already have a parent (must be a root).
        The child and parent must be different nodes.
        
        Args:
            child: Index of child node (must be a root)
            weight: Weight of the edge from child to parent
            parent: Index of parent node
            
        Raises:
            ValueError: If child == parent
            ValueError: If child already has a parent
        """
        if child == parent:
            raise ValueError("Cannot link: child and parent must be different")
        
        if self.links[child] is not None:
            raise ValueError("Cannot link: child already has a parent")
        
        self.links[child] = Link(parent=parent, weight=weight)
    
    def size(self) -> int:
        """Get the number of nodes in the forest."""
        return len(self.links)
    
    def is_root(self, i: int) -> bool:
        """Check if node i is a root."""
        return self.links[i] is None
    
    def __len__(self) -> int:
        """Get the number of nodes in the forest."""
        return len(self.links)


# Factory function for common use cases

def create_forest(mul: Callable[[T, T], T], one: T) -> CompressedWeightedForest[T]:
    """
    Create a new compressed weighted forest.
    
    Args:
        mul: Monoid multiplication operation
        one: Identity element
        
    Returns:
        New CompressedWeightedForest instance
    """
    return CompressedWeightedForest(mul, one)


# Example: Integer addition monoid
def integer_addition_forest() -> CompressedWeightedForest[int]:
    """Create a forest with integer addition as the monoid operation."""
    return CompressedWeightedForest(mul=lambda x, y: x + y, one=0)


# Example: Integer multiplication monoid
def integer_multiplication_forest() -> CompressedWeightedForest[int]:
    """Create a forest with integer multiplication as the monoid operation."""
    return CompressedWeightedForest(mul=lambda x, y: x * y, one=1)


# Example: String concatenation monoid
def string_concatenation_forest() -> CompressedWeightedForest[str]:
    """Create a forest with string concatenation as the monoid operation."""
    return CompressedWeightedForest(mul=lambda x, y: x + y, one="")


# Utility functions

def union_by_rank(forest: CompressedWeightedForest[T], x: int, y: int, 
                  weight: T, ranks: dict[int, int]) -> int:
    """
    Union two sets by rank heuristic.
    
    Args:
        forest: The weighted forest
        x: Node in first set
        y: Node in second set
        weight: Weight for the edge
        ranks: Dictionary tracking rank of each root
        
    Returns:
        Root of the unified set
    """
    root_x = forest.find(x)
    root_y = forest.find(y)
    
    if root_x == root_y:
        return root_x
    
    rank_x = ranks.get(root_x, 0)
    rank_y = ranks.get(root_y, 0)
    
    if rank_x < rank_y:
        forest.link(root_x, weight, root_y)
        return root_y
    elif rank_x > rank_y:
        forest.link(root_y, forest.one, root_x)
        return root_x
    else:
        forest.link(root_y, forest.one, root_x)
        ranks[root_x] = rank_x + 1
        return root_x
