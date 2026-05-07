from typing import List, Optional, Set, Tuple

from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import And, Not, Or, Solver, BOOL, EqualsOrIff


# ── Internal solver helper ───────────────────────────────────────────────────

def _check(solver: Solver, solver_calls: list[int]) -> str:
    """
    Normalize solver status to sat/unsat/unknown/timeout.
    """
    solver_calls[0] += 1
    try:
        res = solver.solve()
    except SolverReturnedUnknownResultError:
        return "timeout"

    if res is True:
        return "sat"
    if res is False:
        return "unsat"
    
    return "unknown"

# ── Algorithm: CoreLitFilter ───────────────────────────────────────────────

def _get_top_level_literals(formula) -> List:
    """
    Extract top-level literals from a conjunction, returned as a list.
    For  (a ∧ b ∧ c), returns [a, b, c].
    For  a single atom/literal, returns [formula].
    Does NOT recurse into disjunctions, ite, or implications — only And nodes.
    """
    lits: List = []
    try:
        if formula.is_and():
            for arg in formula.args():
                lits.extend(_get_top_level_literals(arg))
        else:
            lits.append(formula)
    except Exception:
        pass
    return lits


def _is_blocked(formula, forbidden_ids: Set[int]) -> bool:
    """
    Return True if `formula` contains a top-level literal whose node_id
    is in `forbidden_ids`.
    """
    for lit in _get_top_level_literals(formula):
        try:
            if id(lit) in forbidden_ids or lit.node_id() in forbidden_ids:
                return True
        except Exception:
            pass
    return False


def core_lit_filter(precond, cnt_list: List, timeout_ms: int = 0, solver_calls: Optional[list] = None) -> Tuple[List[int], int]:
    """
    Accumulate forbidden literals from UNSAT results and use them to pre-screen future predicates at zero solver cost.
    """
    if solver_calls is None:
        solver_calls = [0]
        
    n = len(cnt_list)
    results = [2] * n

    # forbidden_ids: set of node_id() values of literals ℓ where φ ⊨ ¬ℓ
    forbidden_ids: Set[int] = set()
    FORBIDDEN_BUDGET = 64  # max literals to verify, limits extra solver calls

    with Solver(name="z3", solver_options={"timeout": timeout_ms}) as solver:
        solver.add_assertion(precond)

        for i, cnt in enumerate(cnt_list):
            if results[i] != 2:
                continue

            # Free pre-screening (zero solver calls), the solver confirmed φ ∧ ℓ is UNSAT in a clean φ-only scope.
            if forbidden_ids and _is_blocked(cnt, forbidden_ids):
                results[i] = 0
                continue

            # Standard incremental check
            solver.push()
            solver.add_assertion(cnt)
            status = _check(solver, solver_calls)

            if status == "timeout" or status == "unknown":
                solver.pop()
                return results, solver_calls[0]

            if status == "sat":
                results[i] = 1
                model = solver.get_model()
                for j in range(i + 1, n):
                    if results[j] == 2:
                        try:
                            if model.get_value(cnt_list[j]).is_true():
                                results[j] = 1
                        except Exception:
                            pass
                solver.pop()

            else:  # unsat
                results[i] = 0
                # Collect candidate literals from p_i for forbidden verification.
                candidates = []
                if len(forbidden_ids) < FORBIDDEN_BUDGET:
                    for lit in _get_top_level_literals(cnt):
                        try:
                            nid = lit.node_id()
                        except Exception:
                            nid = id(lit)
                        if nid not in forbidden_ids:
                            candidates.append((nid, lit))

                solver.pop()

                # Now verify candidates under φ alone, UNSAT here genuinely means φ ⊨ ¬ℓ.
                for nid, lit in candidates:
                    if len(forbidden_ids) >= FORBIDDEN_BUDGET:
                        break
                    solver.push()
                    solver.add_assertion(lit)
                    lit_status = _check(solver, solver_calls)
                    solver.pop()
                    if lit_status == "unsat":
                        forbidden_ids.add(nid)

    return results, solver_calls[0]