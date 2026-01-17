# coding: utf-8
"""Unlimited Branching Tree (UBTree) for clause storage and subsumption."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ------------------------------ UBTree ---------------------------------- #


class UBTreeNode:
    """Node in the Unlimited Branching Tree."""

    def __init__(self, literal: int, parent: Optional["UBTreeNode"] = None) -> None:
        self.literal = literal  # The literal this node represents
        self.parent = parent
        self.children: Dict[int, "UBTreeNode"] = {}  # literal -> child node
        self.subsumed_by: Optional["UBTreeNode"] = None  # Node that subsumes this one
        self.subsumes: Set["UBTreeNode"] = set()  # Nodes subsumed by this one
        self.clause: Optional[List[int]] = None  # The clause this node represents
        self.flag: bool = False  # Boolean flag for subsumption checking

    def add_child(self, literal: int) -> "UBTreeNode":
        """Add a child node for the given literal."""
        if literal not in self.children:
            self.children[literal] = UBTreeNode(literal, self)
        return self.children[literal]

    def find_or_create_path(self, literals: List[int]) -> "UBTreeNode":
        """Find or create a path through the tree for the given literals."""
        current = self
        for literal in literals:
            current = current.add_child(literal)
        return current

    def is_subsumed_by(self, other: "UBTreeNode") -> bool:
        """Check if this node is subsumed by another node."""
        # Two nodes are equivalent if they represent the same clause
        if (
            self.clause is not None
            and other.clause is not None
            and set(abs(l) for l in self.clause) == set(abs(l) for l in other.clause)
        ):
            return True
        return False

    def mark_subsumed(self, by: "UBTreeNode") -> None:
        """Mark this node as subsumed by another node."""
        self.subsumed_by = by
        by.subsumes.add(self)


class UBTree:
    """
    Complete Unlimited Branching Tree implementation with subsumption.

    As described in the paper, the UBTree organizes clauses into a tree structure
    where each path from root to leaf represents a clause. The tree supports:
    - Efficient clause insertion with subsumption checking
    - Retrieval of highest-quality clauses for sharing
    - Multiple quality tiers (T_t,1, T_t,2, T_t,3) per round
    """

    def __init__(self) -> None:
        self.root = UBTreeNode(0)  # Root node
        self.nodes_by_clause: Dict[Tuple[int, ...], UBTreeNode] = {}
        self.tiers: Dict[int, Dict[int, List[UBTreeNode]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # tiers[round][tier] = list of nodes

    def _clause_key(self, clause: List[int]) -> Tuple[int, ...]:
        """Create a normalized key for a clause."""
        # Sort by absolute value, maintain sign relationships
        abs_literals = [abs(l) for l in clause]
        sorted_indices = sorted(range(len(clause)), key=lambda i: abs_literals[i])
        return tuple(clause[i] for i in sorted_indices)

    def _calculate_lbd(self, clause: List[int], assignment: Dict[int, int]) -> int:
        """Calculate Literal Block Distance for a clause."""
        if not clause:
            return 0

        # LBD measures how many distinct decision levels the literals span
        levels = set()
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                levels.add(assignment[var])
        return len(levels) if levels else 1

    def insert_clause(
        self,
        clause: List[int],
        round_id: int,
        assignment: Optional[Dict[int, int]] = None,
    ) -> UBTreeNode:
        """
        Insert a clause into the UBTree with subsumption checking.

        Returns the node representing the clause (or the subsuming node if subsumed).
        """
        if not clause:
            return self.root

        # Normalize clause for key generation
        clause_key = self._clause_key(clause)

        # Check if we already have this clause
        if clause_key in self.nodes_by_clause:
            existing_node = self.nodes_by_clause[clause_key]
            if existing_node.clause is None:
                existing_node.clause = clause
            return existing_node

        # Create path for the clause
        current_node = self.root.find_or_create_path(clause)

        # Set the clause for the leaf node
        current_node.clause = clause

        # Calculate quality metrics
        size = len(clause)
        lbd = self._calculate_lbd(clause, assignment or {})

        # Determine tier based on size and LBD
        if size == 1:
            tier = 1  # Unit clauses
        elif size <= 3 or lbd <= 2:
            tier = 2  # Short or low-LBD clauses
        else:
            tier = 3  # Other clauses

        # Store in appropriate tier for this round
        self.tiers[round_id][tier].append(current_node)
        self.nodes_by_clause[clause_key] = current_node

        # Check for subsumption with existing clauses
        self._check_subsumption(current_node, round_id)

        return current_node

    def _check_subsumption(self, new_node: UBTreeNode, round_id: int) -> None:
        """Check if the new node subsumes or is subsumed by existing nodes."""
        if new_node.clause is None:
            return

        new_clause = set(abs(l) for l in new_node.clause)

        # Check against nodes in higher tiers first (better quality)
        for tier in [1, 2, 3]:
            if tier not in self.tiers[round_id]:
                continue

            for existing_node in self.tiers[round_id][tier]:
                if existing_node.clause is None:
                    continue

                existing_clause = set(abs(l) for l in existing_node.clause)

                # Check if new clause subsumes existing clause
                if new_clause.issubset(existing_clause):
                    existing_node.mark_subsumed(new_node)

                # Check if existing clause subsumes new clause
                elif existing_clause.issubset(new_clause):
                    new_node.mark_subsumed(existing_node)

    def get_best_clauses_for_round(
        self, round_id: int, max_clauses: int = 1000
    ) -> List[List[int]]:
        """
        Get the best clauses for sharing in the next round.

        Returns clauses in order of quality, respecting subsumption relationships.
        """
        if round_id not in self.tiers:
            return []

        result = []
        tier_order = [1, 2, 3]  # Process tiers in quality order

        for tier in tier_order:
            if tier not in self.tiers[round_id]:
                continue

            # Sort nodes in this tier by quality (smaller LBD/size is better)
            nodes = self.tiers[round_id][tier]
            nodes.sort(
                key=lambda n: (
                    len(n.clause) if n.clause else 999,
                    self._calculate_lbd(n.clause, {}),
                )
            )

            for node in nodes:
                # Skip subsumed nodes
                if node.subsumed_by is not None:
                    continue

                if node.clause is not None:
                    result.append(node.clause)
                    if len(result) >= max_clauses:
                        return result

        return result

    def clear_round(self, round_id: int) -> None:
        """Clear clauses from a specific round."""
        if round_id in self.tiers:
            del self.tiers[round_id]
