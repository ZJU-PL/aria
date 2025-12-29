# import numpy as np
"""
DNNF (Decomposable Negation Normal Form) is a form of propositional logic formula
that provides efficient support for many logical operations.
"""
import copy
from typing import List, Optional, Dict, Union, Tuple


class DNF_Node:
    """Represents a node in a DNNF (Decomposable Negation Normal Form) tree."""

    def __init__(self, node_type: str, left_child: Optional['DNF_Node'] = None,
                 right_child: Optional['DNF_Node'] = None, literal: Optional[int] = None,
                 conflict_atom: Optional[int] = None) -> None:
        """
        Initialize a DNNF node.

        Args:
            node_type: Type of node ('A', 'O', or 'L')
            left_child: Left child node
            right_child: Right child node
            literal: Literal value (for leaf nodes)
            conflict_atom: Conflict atom (for OR nodes)
        """
        assert node_type in ('A', 'O', 'L')
        self.type = node_type  # A, O or L
        self.left_child = left_child
        self.right_child = right_child
        self.literal = literal
        self.conflict_atom = conflict_atom

        self.explore_id: Optional[int] = None

        self.atoms: Optional[List[int]] = None
        self.models: Optional[List[Dict[int, bool]]] = None

        if self.type == 'L':
            assert self.literal is not None
            assert self.left_child is None
            assert self.right_child is None
            self.atoms = [abs(literal)]
        elif self.type in ('O', 'A'):
            assert self.literal is None
            assert self.left_child is not None
            assert self.right_child is not None
            self.atoms = list(set(self.left_child.atoms).union(self.right_child.atoms))

    def count_node(self, current_id: int) -> int:
        """
        Count nodes in the DNNF.

        Args:
            current_id: Current node ID

        Returns:
            Next available node ID
        """
        if self.explore_id is not None:
            return current_id
        if self.type != 'L':
            current_id = self.left_child.count_node(current_id)
            current_id = self.right_child.count_node(current_id)
        self.explore_id = current_id
        return current_id + 1

    def count_edge(self) -> int:
        """
        Count edges in the DNNF.

        Returns:
            Number of edges
        """
        if self.type == 'L':
            return 0
        return self.left_child.count_edge() + self.right_child.count_edge() + 2

    def collect_var(self) -> List[int]:
        """
        Collect variables in the DNNF.

        Returns:
            List of variable indices
        """
        if self.type == 'L':
            return [abs(self.literal)]
        return list(set(self.left_child.collect_var()).union(self.right_child.collect_var()))

    def print_nnf(self, current_id: int, output_file: Optional[str] = None) -> int:
        """
        Print DNNF in NNF format.

        Args:
            current_id: Current node ID
            output_file: Optional output file path

        Returns:
            Next available node ID
        """
        if self.explore_id is not None:
            return current_id

        if self.type == 'L':
            if output_file is not None:
                with open(output_file, 'a', encoding='utf-8') as out:
                    out.write(f'L {self.literal}\n')
            else:
                print(f'{current_id} L {self.literal}')
        else:
            current_id = self.left_child.print_nnf(current_id, output_file)
            current_id = self.right_child.print_nnf(current_id, output_file)
            if self.type == 'A':
                if output_file is not None:
                    with open(output_file, 'a', encoding='utf-8') as out:
                        out.write(f'A 2 {self.left_child.explore_id} '
                                  f'{self.right_child.explore_id}\n')
                else:
                    print(f'{current_id} A 2 {self.left_child.explore_id} '
                          f'{self.right_child.explore_id}')
            if self.type == 'O':
                if output_file is not None:
                    with open(output_file, 'a', encoding='utf-8') as out:
                        out.write(f'O {self.conflict_atom} 2 '
                                  f'{self.left_child.explore_id} '
                                  f'{self.right_child.explore_id}\n')
                else:
                    print(f'{current_id} O {self.conflict_atom} 2 '
                          f'{self.left_child.explore_id} '
                          f'{self.right_child.explore_id}')
        self.explore_id = current_id
        return current_id + 1

    def reset(self) -> None:
        """Reset exploration IDs."""
        self.explore_id = None
        if self.type != 'L':
            self.left_child.reset()
            self.right_child.reset()



class DNNF_Compiler:
    """Compiler for converting CNF to DNNF format."""
    def __init__(self, dtree: 'Node') -> None:
        """
        Initialize DNNF compiler.

        Args:
            dtree: Decision tree node
        """
        self.dtree = dtree
        self.cache: Dict[str, DNF_Node] = {}
        self.cache_lit: Dict[int, DNF_Node] = {}
        self.ddnnf: Optional[DNF_Node] = None


    def bcp(self, dtree: 'Node', literal: int) -> Union['Node', int]:
        """
        Perform Boolean Constraint Propagation (BCP) on the given dtree with the given literal.

        Args:
            dtree: The decision tree to perform BCP on.
            literal: The literal to propagate through the dtree.

        Returns:
            A modified decision tree after BCP, or -1 if conflict.
        """

        modified = copy.deepcopy(dtree)
        if modified.is_leaf():
            # print(literal)
            # print(dtree.clauses)
            if len(modified.clauses) == 0:
                return modified

            leaf_clause = modified.clauses[0]
            if literal in leaf_clause:
                modified.clauses = []
                modified.atoms = []
                modified.clause_key = [1]
                modified.lit_key += 2 ** (abs(literal) - 1)
            elif -literal in leaf_clause:
                modified_clause = [lit for lit in leaf_clause if lit != -literal]
                modified.clauses[0] = modified_clause
                if len(modified.clauses[0]) == 0:
                    modified.atoms = []
                    return -1  # CONFLICT !!!
                modified.atoms = [abs(lit) for lit in modified.clauses[0]]
                modified.clause_key = [0]
                modified.lit_key += 2 ** (abs(literal) - 1)
        else:
            modified.left_child = self.bcp(modified.left_child, literal)
            modified.right_child = self.bcp(modified.right_child, literal)
            modified.atoms = list(set(modified.left_child.atoms).union(
                modified.right_child.atoms))
            modified.separators = list(set(modified.left_child.atoms).intersection(
                modified.right_child.atoms))
            modified.clauses = modified.left_child.clauses + modified.right_child.clauses
            modified.clause_key = modified.left_child.clause_key + modified.right_child.clause_key
            modified.lit_key += 2 ** (abs(literal) - 1)
        return modified

    # def pure_literals(dtree):
    #     counter = dtree.get_counter()
    #     pure_assignment = []
    #     for l in counter:
    #         if -l not in counter:
    #             pure_assignment.append(l)

    #     for l in pure_assignment:
    #         dtree = bcp(dtree, l)
    #     return dtree, pure_assignment

    def unit_propagation(self, dtree: 'Node') -> Tuple[Union['Node', int], List[int]]:
        """
        Perform unit propagation on the decision tree.

        Args:
            dtree: Decision tree to propagate

        Returns:
            Tuple of (modified_tree, unit_assignments)
        """
        modified = copy.deepcopy(dtree)
        unit_assignment: List[int] = []
        unit_clauses = [c for c in modified.clauses if len(c) == 1]
        while len(unit_clauses) > 0:
            unit = unit_clauses[0][0]
            modified = self.bcp(modified, unit)
            unit_assignment.append(unit)
            if modified == -1:
                return -1, []
            if len(modified.clauses) == 0:
                return modified, unit_assignment
            unit_clauses = [c for c in modified.clauses if len(c) == 1]
        return modified, unit_assignment


    def compose(self, node_type: str, list_tree: List[DNF_Node],
                conflict: Optional[List[int]] = None) -> Optional[DNF_Node]:
        """
        Compose nodes into a tree with specified node type.

        Args:
            node_type: Type of node ('A' or 'O')
            list_tree: List of nodes to compose
            conflict: Optional conflict atoms for OR nodes

        Returns:
            Composed node or None
        """
        assert node_type != 'L'
        # Filter out None values first
        list_tree = [t for t in list_tree if t is not None]
        if len(list_tree) == 0:
            return None

        if len(list_tree) == 1:
            return list_tree[0]

        if conflict is not None:
            right_composed_node = self.compose(
                node_type, list_tree[1:], conflict[1:]
            )
            composed_node = DNF_Node(
                node_type=node_type, left_child=list_tree[0],
                right_child=right_composed_node, conflict_atom=conflict[0])
        else:
            right_composed_node = self.compose(node_type, list_tree[1:])
            composed_node = DNF_Node(
                node_type=node_type, left_child=list_tree[0],
                right_child=right_composed_node)
        return composed_node

    def create_term_node(self, term: List[int]) -> Optional[DNF_Node]:
        """
        Create a term node from a list of literals.

        Args:
            term: List of literals

        Returns:
            Term node or None
        """
        if len(term) == 0:
            return None
        leaves: List[DNF_Node] = []
        for literal in term:
            if literal not in self.cache_lit:
                leaf_node = DNF_Node(node_type='L', literal=literal)
                leaves.append(leaf_node)
                self.cache_lit[literal] = leaf_node
            else:
                leaves.append(self.cache_lit[literal])
        return self.compose(node_type='A', list_tree=leaves)

    def clause2ddnnf(self, dtree: 'Node') -> Optional[DNF_Node]:
        """
        Convert a clause to DNNF.

        Args:
            dtree: Decision tree node representing a clause

        Returns:
            DNNF node or None
        """
        if len(dtree.atoms) == 0:
            return None
        clause = dtree.clauses[0]
        assert len(clause) > 0
        nodes: List[DNF_Node] = []
        conflict: List[int] = []

        for i, _ in enumerate(clause):
            # li= [DNF_Node(node_type='L',literal=clause[i])]
            # not_lj = [DNF_Node(node_type='L',literal=-clause[j]) for j in range(i)]
            li: List[DNF_Node] = []
            list_not_lj: List[DNF_Node] = []
            if clause[i] in self.cache_lit:
                li.append(self.cache_lit[clause[i]])
            else:
                li.append(DNF_Node(node_type='L', literal=clause[i]))
                self.cache_lit[clause[i]] = li[0]

            for j in range(i):
                if -clause[j] in self.cache_lit:
                    not_lj = self.cache_lit[-clause[j]]
                else:
                    not_lj = DNF_Node(node_type='L', literal=-clause[j])
                    self.cache_lit[-clause[j]] = not_lj
                list_not_lj.append(not_lj)

            choice = self.compose(node_type='A', list_tree=li + list_not_lj)
            nodes.append(choice)
            conflict.append(clause[i])
        return self.compose(node_type='O', list_tree=nodes, conflict=conflict)

    def cnf2aux(self, dtree: 'Node') -> Optional[DNF_Node]:
        """
        Convert CNF to auxiliary DNNF with caching.

        Args:
            dtree: Decision tree node

        Returns:
            DNNF node or None
        """
        if dtree.is_leaf():
            return self.clause2ddnnf(dtree)
        l_key = dtree.lit_key
        c_key = 0
        for i, v in enumerate(dtree.clause_key):
            c_key += v * (2 ** i)
        if l_key in self.cache and c_key in self.cache[l_key]:
            print('Using cache !')
            return self.cache[l_key][c_key]
        r = self.cnf2ddnnf(dtree)
        if r is not False and r is not None:
            if self.cache is None:
                self.cache = {}
            if l_key not in self.cache:
                self.cache[l_key] = {}
            self.cache[l_key][c_key] = r
        return r


    def cnf2ddnnf(self, dtree: 'Node') -> Optional[DNF_Node]:
        """
        Convert CNF to DNNF.

        Args:
            dtree: Decision tree node

        Returns:
            DNNF node or None
        """
        # if dtree.is_leaf():
        #     return clause2ddnnf(dtree)
        dtree, unit_assignment = self.unit_propagation(dtree)
        if dtree == -1:
            return None
        term_node = self.create_term_node(unit_assignment)
        sep = dtree.separators
        if sep is None or len(sep) == 0:
            left_node = self.cnf2aux(dtree.left_child)
            right_node = self.cnf2aux(dtree.right_child)
            return self.compose(node_type='A', list_tree=[term_node, left_node, right_node])
        else:
            v = dtree.pick_most()
            print('Pick ', v)
            p = self.cnf2ddnnf(self.bcp(dtree, v))
            if not p:
                return self.cnf2ddnnf(self.bcp(dtree, -v))
            print('Pick ', -v)
            n = self.cnf2ddnnf(self.bcp(dtree, -v))
            if not n:
                return self.cnf2ddnnf(self.bcp(dtree, v))

            if v in self.cache_lit:
                v_node = self.cache_lit[v]
            else:
                v_node = DNF_Node('L', literal=v)
                self.cache_lit[v] = v_node
            if -v in self.cache_lit:
                not_v_node = self.cache_lit[-v]
            else:
                not_v_node = DNF_Node('L', literal=-v)
                self.cache_lit[-v] = not_v_node
            p_node = self.compose(node_type='A', list_tree=[v_node, p])
            n_node = self.compose(node_type='A', list_tree=[not_v_node, n])
            t_node = DNF_Node(
                node_type='O', left_child=p_node, right_child=n_node,
                conflict_atom=abs(v))
            # t_node = compose(node_type='O', list_tree=[p_node, n_node])
            return self.compose(node_type='A', list_tree=[term_node, t_node])

    def compile(self) -> Optional[DNF_Node]:
        """
        Compile the decision tree to DNNF.

        Returns:
            Compiled DNNF node or None
        """
        self.ddnnf = self.cnf2ddnnf(self.dtree)
        return copy.deepcopy(self.ddnnf)


    def conditioning(self, dnnf: DNF_Node, instanciation: List[int]) -> DNF_Node:
        """
        Apply conditioning to DNNF.

        Args:
            dnnf: DNNF node
            instanciation: List of literals to condition on

        Returns:
            Conditioned DNNF node
        """
        if dnnf.explore_id is None:
            assert not isinstance(dnnf.literal, bool)
            if dnnf.type == 'L':
                if dnnf.literal in instanciation:
                    dnnf.literal = True
                elif -dnnf.literal in instanciation:
                    dnnf.literal = False
            else:
                dnnf.left_child = self.conditioning(dnnf.left_child, instanciation)
                dnnf.right_child = self.conditioning(dnnf.right_child, instanciation)
            dnnf.explore_id = 1
        return dnnf

    def conjoin(self, dnnf: DNF_Node, instanciation: List[int]) -> DNF_Node:
        """
        Conjoin DNNF with instantiation.

        Args:
            dnnf: DNNF node
            instanciation: List of literals

        Returns:
            Conjoined DNNF node
        """
        return DNF_Node(
            node_type='A',
            left_child=self.simplify(self.conditioning(dnnf, instanciation)),
            right_child=self.create_term_node(instanciation))

    def simplify(self, dnnf: DNF_Node) -> DNF_Node:
        """
        Simplify DNNF.

        Args:
            dnnf: DNNF node to simplify

        Returns:
            Simplified DNNF node
        """
        if dnnf.type == 'L':
            return dnnf
        if dnnf.type == 'O':
            dnnf.left_child = self.simplify(dnnf.left_child)
            dnnf.right_child = self.simplify(dnnf.right_child)
            if dnnf.left_child.literal:
                return dnnf.left_child
            if dnnf.right_child.literal:
                return dnnf.right_child
            if not dnnf.left_child.literal:
                return dnnf.right_child
            if not dnnf.right_child.literal:
                return dnnf.left_child
            return dnnf
        if dnnf.type == 'A':
            dnnf.left_child = self.simplify(dnnf.left_child)
            dnnf.right_child = self.simplify(dnnf.right_child)
            if dnnf.left_child.literal and dnnf.right_child.literal:
                return dnnf.left_child
            if not dnnf.left_child.literal:
                return dnnf.left_child
            if not dnnf.right_child.literal:
                return dnnf.right_child
            return dnnf
        return dnnf

    def is_sat(self, dnnf: DNF_Node) -> bool:
        """
        Check if DNNF is satisfiable.

        Args:
            dnnf: DNNF node

        Returns:
            True if satisfiable, False otherwise
        """
        if dnnf.type == 'L':
            return bool(dnnf.literal)
        if dnnf.type == 'O':
            return self.is_sat(dnnf.left_child) or self.is_sat(dnnf.right_child)
        if dnnf.type == 'A':
            return self.is_sat(dnnf.left_child) and self.is_sat(dnnf.right_child)
        return False

    def project(self, dnnf: DNF_Node, atoms: List[int]) -> DNF_Node:
        """
        Project DNNF onto specified atoms.

        Args:
            dnnf: DNNF node
            atoms: List of atoms to project onto

        Returns:
            Projected DNNF node
        """
        if dnnf.type == 'L':
            if not isinstance(dnnf.literal, bool):
                if abs(dnnf.literal) not in atoms:
                    dnnf.literal = True
        else:
            dnnf.left_child = self.project(dnnf.left_child, atoms)
            dnnf.right_child = self.project(dnnf.right_child, atoms)
        return dnnf

    def m_card(self, dnnf: DNF_Node) -> float:
        """
        Compute minimum cardinality of DNNF.

        Args:
            dnnf: DNNF node

        Returns:
            Minimum cardinality
        """
        if dnnf.type == 'L':
            if isinstance(dnnf.literal, bool):
                if dnnf.literal:
                    return 0
                return float('inf')
            if dnnf.literal > 0:
                return 0
            return 1
        if dnnf.type == 'O':
            return min(self.m_card(dnnf.left_child), self.m_card(dnnf.right_child))
        if dnnf.type == 'A':
            return self.m_card(dnnf.left_child) + self.m_card(dnnf.right_child)
        return float('inf')

    def minimize(self, dnnf):
        """Minimize DNNF by removing unnecessary nodes."""
        if dnnf.type == 'L':
            return dnnf
        if dnnf.type == 'A':
            dnnf.left_child = self.minimize(dnnf.left_child)
            dnnf.right_child = self.minimize(dnnf.right_child)
            if dnnf.left_child is None:
                return dnnf.right_child
            if dnnf.right_child is None:
                return dnnf.left_child
            return dnnf
        if dnnf.type == 'O':
            mcard = self.m_card(dnnf)
            left_mcard = self.m_card(dnnf.left_child)
            right_mcard = self.m_card(dnnf.right_child)
            if mcard not in (left_mcard, right_mcard):
                return None
            if left_mcard == mcard and right_mcard != mcard:
                return self.minimize(dnnf.left_child)
            if left_mcard != mcard and right_mcard == mcard:
                return self.minimize(dnnf.right_child)
            dnnf.left_child = self.minimize(dnnf.left_child)
            dnnf.right_child = self.minimize(dnnf.right_child)
            return dnnf
        return dnnf

    def create_trivial_node(self, atom):
        """Create a trivial OR node for an atom."""
        if atom in self.cache_lit:
            p = self.cache_lit[atom]
        else:
            p = DNF_Node('L', literal=atom)
            self.cache_lit[atom] = p
        if -atom in self.cache_lit:
            n = self.cache_lit[-atom]
        else:
            n = DNF_Node('L', literal=-atom)
            self.cache_lit[-atom] = n
        return DNF_Node('O', left_child=p, right_child=n, conflict_atom=abs(atom))

    def smooth(self, dnnf):
        """Smooth DNNF by ensuring all children have the same atoms."""
        if dnnf.type == 'L':
            pass
        elif dnnf.type == 'A':
            dnnf.left_child = self.smooth(dnnf.left_child)
            dnnf.right_child = self.smooth(dnnf.right_child)
        elif dnnf.type == 'O':
            atoms = dnnf.atoms
            not_atoms_left = list(set(dnnf.left_child.atoms) ^ set(atoms))
            not_atoms_right = list(set(dnnf.right_child.atoms) ^ set(atoms))
            if len(not_atoms_left) > 0:
                print('Left node is not smooth')
                trivial_nodes = [self.create_trivial_node(lit) for lit in not_atoms_left]
                dnnf.left_child = self.compose(
                    node_type='A', list_tree=[dnnf.left_child] + trivial_nodes)
                dnnf.left_child.atoms = atoms
            if len(not_atoms_right) > 0:
                print('Right node is not smooth')
                trivial_nodes = [self.create_trivial_node(lit) for lit in not_atoms_right]
                dnnf.right_child = self.compose(
                    node_type='A', list_tree=[dnnf.right_child] + trivial_nodes)
                dnnf.right_child.atoms = atoms
            dnnf.left_child = self.smooth(dnnf.left_child)
            dnnf.right_child = self.smooth(dnnf.right_child)
        return dnnf

    def enumerate_models(self, dnnf):
        """Enumerate all models of the DNNF."""
        if dnnf.type == 'L':
            if isinstance(dnnf.literal, bool):
                if dnnf.literal:
                    return [[]]
                return []
            return [[dnnf.literal]]
        if dnnf.type == 'O':
            return self.union_models(
                self.enumerate_models(dnnf.left_child),
                self.enumerate_models(dnnf.right_child))
        if dnnf.type == 'A':
            return self.multiply_models(
                self.enumerate_models(dnnf.left_child),
                self.enumerate_models(dnnf.right_child))
        return []

    def union_models(self, l1, l2):
        """Union two lists of models."""
        model = l1.copy()
        for item in l2:
            if item not in l1:
                model.append(item)
        return model

    def multiply_models(self, l1, l2):
        """Multiply two lists of models (cartesian product)."""
        model = []
        for item_a in l1:
            for item_b in l2:
                model.append(list(set(item_a).union(item_b)))
        return model
