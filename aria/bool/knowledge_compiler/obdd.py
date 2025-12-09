"""
OBDD: Ordered Binary Decision Diagrams
"""
import copy
import itertools
from typing import List, Dict, Optional, Tuple, Any


class BDD:
    def __init__(self, var: int, low: Optional['BDD'], high: Optional['BDD']) -> None:
        """
        Initialize a BDD node.

        Args:
            var: Variable index (0 for sink nodes)
            low: Low branch (False branch)
            high: High branch (True branch)
        """
        self.var = var
        self.low = low
        self.high = high
        self.explore_id = 0

    def is_sink(self) -> bool:
        """Check if this is a sink node (terminal)."""
        return self.low is None and self.high is None

    def _print_info(self, current_id: int, rank: List[List[int]],
                   output_file: Optional[str] = None) -> Tuple[int, List[List[int]]]:
        """
        Internal method to print BDD information.

        Args:
            current_id: Current node ID
            rank: List of node ranks
            output_file: Optional output file path

        Returns:
            Tuple of (next_id, updated_rank)
        """
        if self.explore_id > 0:
            return current_id, rank

        if self.is_sink():
            if output_file is not None:
                out = open(output_file, 'a')
                if self.var:
                    out.write('     {} [label="True", color=green, shape=square];\n'.format(current_id + 1))
                elif not self.var:
                    out.write('     {} [label="False", color=red, shape=square];\n'.format(current_id + 1))
                out.close()
            else:
                print('{}-SINK : {}'.format(current_id + 1, self.var))
        else:
            left_current_id, rank = self.low._print_info(current_id, rank, output_file)
            current_id, rank = self.high._print_info(left_current_id, rank, output_file)
            if output_file is not None:
                out = open(output_file, 'a')
                out.write('     {} [label="{}"];\n'.format(current_id + 1, self.var))
                out.write('     {} -> {} [style=dotted];\n'.format(current_id + 1, self.low.explore_id))
                out.write('     {} -> {};\n'.format(current_id + 1, self.high.explore_id))
                out.close()
            else:
                print('{}-Var: {}'.format(current_id + 1, self.var))
        self.explore_id = current_id + 1
        if self.is_sink():
            rank[0].append(self.explore_id)
        else:
            rank[self.var].append(self.explore_id)
        return current_id + 1, rank

    def print_info(self, nvars: int, output_file: Optional[str] = None) -> List[List[int]]:
        """
        Print BDD information.

        Args:
            nvars: Number of variables
            output_file: Optional output file path

        Returns:
            List of node ranks
        """
        rank = [[] for _ in range(nvars + 1)]
        _, rank = copy.deepcopy(self)._print_info(0, rank, output_file)
        # for i in range(len(rank)):
        #     print(i, ': ', rank[i])
        return rank


class BDD_Compiler:
    def __init__(self, n_vars: int, clausal_form: List[List[int]]) -> None:
        """
        Initialize BDD compiler.

        Args:
            n_vars: Number of variables
            clausal_form: CNF formula as list of clauses
        """
        self.clausal_form = clausal_form
        self.n_vars = n_vars
        self.unique: Dict[Tuple[int, Optional[BDD], Optional[BDD]], BDD] = {}
        self.cache: Dict[int, Dict[int, BDD]] = {}
        for i in range(n_vars + 1):
            self.cache[i] = {}
        self.cutset_cache = self._generate_cutset_cache()
        self.separator_cache = self._generate_separator_cache()

        self.F_SINK = BDD(False, None, None)
        self.T_SINK = BDD(True, None, None)

    def bcp(self, formula: List[List[int]], literal: int) -> List[List[int]]:
        """
        Boolean Constraint Propagation.

        Args:
            formula: CNF formula
            literal: Literal to propagate

        Returns:
            Modified formula after BCP
        """
        modified = []
        for clause in formula:
            if literal in clause:
                modified.append([])
            elif -literal in clause:
                c = [x for x in clause if x != -literal]
                if len(c) == 0:
                    return -1
                modified.append(c)
            else:
                modified.append(clause)
        return modified

    '''
    Functions used for computing cutset key and cache
    '''

    def _compute_cutset(self, clausal_form: List[List[int]], var: int) -> List[int]:
        """
        Compute cutset for a variable.

        Args:
            clausal_form: CNF formula
            var: Variable index

        Returns:
            List of clause indices in cutset
        """
        cutset = []
        for i, clause in enumerate(clausal_form):
            if len(clause) == 0:
                continue
            atoms = [abs(lit) for lit in clause]
            if min(atoms) <= var < max(atoms):
                cutset.append(i)
        return cutset

    def _generate_cutset_cache(self) -> List[List[int]]:
        """Generate cutset cache for all variables."""
        cutset_cache = []
        print('CUTSET CACHE:')
        for i in range(self.n_vars):
            cutset_i = self._compute_cutset(self.clausal_form, i + 1)
            cutset_cache.append(cutset_i)
            print('-cutset {} : {}'.format(i + 1, cutset_i))
        return cutset_cache

    def compute_cutset_key(self, clausal_form: List[List[int]], var: int) -> int:
        """
        Compute cutset key for caching.

        Args:
            clausal_form: CNF formula
            var: Variable index

        Returns:
            Cutset key
        """
        cutset_var = self.cutset_cache[var - 1]
        cutset_key = 0
        for i, c in enumerate(cutset_var):
            if len(clausal_form[c]) == 0:
                cutset_key += 2 ** i
        return cutset_key

    '''
    Functions used for compute separator key and cache
    '''

    def _compute_separator(self, clausal_form: List[List[int]], var: int) -> List[int]:
        """
        Compute separator for a variable.

        Args:
            clausal_form: CNF formula
            var: Variable index

        Returns:
            List of variables in separator
        """
        sep = []
        for ci in self.cutset_cache[var - 1]:
            sep += self.clausal_form[ci]
        sep = [abs(lit) for lit in sep if abs(lit) <= var]
        sep = list(set(sep))
        return sep

    def _generate_separator_cache(self) -> List[List[int]]:
        """Generate separator cache for all variables."""
        sep_cache = []
        print('SEPARATOR CACHE:')
        for i in range(self.n_vars):
            sep_i = self._compute_separator(self.clausal_form, i + 1)
            sep_cache.append(sep_i)
            print('-sep {} : {}'.format(i + 1, sep_i))
        return sep_cache

    def compute_separator_key(self, clausal_form: List[List[int]], var: int) -> int:
        """
        Compute separator key for caching.

        Args:
            clausal_form: CNF formula
            var: Variable index

        Returns:
            Separator key
        """
        sep_var = self.separator_cache[var - 1]
        sep_key = 0
        for v in sep_var:
            sep_key += 2 ** v
        return sep_key

    '''
    Core functions
    '''

    def get_nodes(self, var: int, low: BDD, high: BDD) -> BDD:
        """
        Get or create a BDD node.

        Args:
            var: Variable index
            low: Low branch
            high: High branch

        Returns:
            BDD node
        """
        if low == high:
            return low
        if (var, low, high) in self.unique:  # and low == self.unique[i].low and high == self.unique[i].high:
            # print('Unique node {} found!'.format(var))
            return self.unique[(var, low, high)]
        result = BDD(var, low, high)
        self.unique[(var, low, high)] = result
        return result

    def cnf2obdd(self, clausal_form: List[List[int]], i: int, key_type: str = 'cutset') -> BDD:
        """
        Convert CNF to OBDD.

        Args:
            clausal_form: CNF formula
            i: Variable index
            key_type: Type of key for caching ('cutset' or 'separator')

        Returns:
            BDD representing the formula
        """
        assert key_type == 'cutset' or key_type == 'separator'

        if clausal_form == -1:
            return self.F_SINK
        elif len(list(itertools.chain(*clausal_form))) == 0:
            return self.T_SINK

        assert i <= self.n_vars + 1

        if key_type == 'cutset':
            key = self.compute_cutset_key(clausal_form, i - 1)
        elif key_type == 'separator':
            key = self.compute_separator_key(clausal_form, i - 1)

        if key in self.cache[i - 1]:
            print('This node is already in cache {} with key {}'.format(i - 1, key))
            return self.cache[i - 1][key]

        low = self.cnf2obdd(self.bcp(clausal_form, -i), i + 1)
        high = self.cnf2obdd(self.bcp(clausal_form, i), i + 1)
        result = self.get_nodes(i, low, high)

        self.cache[i - 1][key] = result
        # print('This node is stored in cache {} with key {}'.format(i-1, key))
        return result

    def compile(self, key_type: str = 'cutset') -> BDD:
        """
        Compile CNF formula to OBDD.

        Args:
            key_type: Type of key for caching

        Returns:
            Compiled BDD
        """
        return self.cnf2obdd(self.clausal_form, 1, key_type)
