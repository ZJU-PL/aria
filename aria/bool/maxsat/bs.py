"""
Implement Nadel's algorithm for OMT(BV) "Bit-Vector Optimization (TACAS'16)"

Key idea: OMT on unsigned BV can be seen as lexicographic optimization over the bits in the
bitwise representation of the objective, ordered from the most-significant bit (MSB)
to the least-significant bit (LSB).

Notice that, in this domain, this corresponds to a binary search over the space of the values of the objective

NOTE: we assume that each element in self.soft is a unary clause, i.e., self.soft is [[l1], [l2], ...]
"""

from pysat.solvers import Solver
from typing import List


def obv_bs(clauses: List[List[int]], literals: List[int]) -> List[int]:
    """
    This is a binary search algorithm of bit-vector optimization.
    Args:
        clauses: the given constraints
        literals: literals listed in priority

    Returns: the maximum assignment of literals

    """
    result = []
    s = Solver(bootstrap_with=clauses)
    if s.solve():
        m = s.get_model()
        # print(m)
    else:
        print('UNSAT')
        return result
    l = len(m)
    for lit in literals:
        if lit > l:
            '''If 'lit' is not in m, 'lit' can be assigned 0 or 1, to maximum the result, 'lit' is assigned 1.'''
            result.append(lit)
        else:
            # if lit is true in m, then lit is assigned 1
            if m[lit - 1] > 0:
                result.append(lit)
            else:
                # if lit is false in m, then lit is assigned 0
                result.append(lit)
                # update the assumptions and solve again
                if s.solve(assumptions=result):
                    m = s.get_model()
                else:
                    result.pop()
                    result.append(-lit)
    # print(result)
    return result



def obv_bs_anytime(clauses: List[List[int]], literals: List[int], time_limit: float = 60.0, conflict_limit: int = 1000) -> List[int]:
    """
    An anytime version of the binary search algorithm of bit-vector optimization.

    The algorithm will return the best solution found so far when it is interrupted.
    Args:
        clauses: the given constraints
        literals: literals listed in priority
        time_limit: maximum time in seconds (default: 60s)
        conflict_limit: maximum number of conflicts per SAT call (default: 1000)

    Returns:
        best_result: the best assignment found within the time limit
    """
    import time
    start_time = time.time()

    # Initialize solver with conflict limit
    s = Solver(bootstrap_with=clauses)
    s.conf_budget(conflict_limit)

    # Try to get initial solution
    if not s.solve_limited():
        print('UNSAT')
        return []

    best_result = []  # Store best result found so far
    current_result = []
    m = s.get_model()
    l = len(m)

    try:
        for lit in literals:
            # Check time limit
            if time.time() - start_time > time_limit:
                print('Time limit reached')
                return best_result if best_result else current_result

            if lit > l:
                # If literal not in model, try setting it to 1 first
                current_result.append(lit)
            else:
                # Try setting current bit to 1 first
                current_result.append(lit)

                # Set new conflict budget for this SAT call
                s.conf_budget(conflict_limit)

                if s.solve_limited(assumptions=current_result):
                    m = s.get_model()
                else:
                    # If UNSAT or conflict limit reached, try with 0
                    current_result.pop()
                    current_result.append(-lit)

                    # Try one more time with opposite value
                    s.conf_budget(conflict_limit)
                    if not s.solve_limited(assumptions=current_result):
                        # If both values fail, backtrack to previous best result
                        current_result.pop()
                        if best_result:
                            return best_result
                        continue
                    m = s.get_model()

            # Update best result if current solution is valid
            if s.solve_limited(assumptions=current_result):
                best_result = current_result.copy()

    except KeyboardInterrupt:
        # Handle external interruption
        print('Interrupted - returning best result found')
        return best_result if best_result else current_result

    return best_result if best_result else current_result
