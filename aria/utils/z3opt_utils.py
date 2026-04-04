"""Helper functions for Z3 optimization tasks."""

from typing import List

import z3


def optimize(fml: z3.ExprRef, obj: z3.ExprRef, minimize=False, timeout: int = 0):
    """Optimize a single objective under a formula."""
    solver = z3.Optimize()
    solver.add(fml)
    if timeout > 0:
        solver.set("timeout", timeout)
    if minimize:
        obj_handle = solver.minimize(obj)
    else:
        obj_handle = solver.maximize(obj)
    if solver.check() == z3.sat:
        return obj_handle.value()
    return None


def box_optimize(
    fml: z3.ExprRef, minimize: List, maximize: List, timeout: int = 0
):
    """Optimize multiple objectives with box priority."""
    solver = z3.Optimize()
    solver.set("opt.priority", "box")
    solver.add(fml)
    if timeout > 0:
        solver.set("timeout", timeout)
    min_objectives = [solver.minimize(expr) for expr in minimize]
    max_objectives = [solver.maximize(expr) for expr in maximize]
    if solver.check() == z3.sat:
        min_res = [obj.value() for obj in min_objectives]
        max_res = [obj.value() for obj in max_objectives]
        return min_res, max_res
    return None, None


def pareto_optimize(
    fml: z3.ExprRef, minimize: List, maximize: List, timeout: int = 0
):
    """Optimize multiple objectives with pareto priority."""
    solver = z3.Optimize()
    solver.set("opt.priority", "pareto")
    solver.add(fml)
    if timeout > 0:
        solver.set("timeout", timeout)
    min_objectives = [solver.minimize(expr) for expr in minimize]
    max_objectives = [solver.maximize(expr) for expr in maximize]
    if solver.check() == z3.sat:
        min_res = [obj.value() for obj in min_objectives]
        max_res = [obj.value() for obj in max_objectives]
        return min_res, max_res
    return None, None


def maxsmt(
    hard: z3.BoolRef, soft: List[z3.BoolRef], weight: List[int], timeout=0
) -> int:
    """Solve a weighted MaxSMT instance and return unsatisfied soft weight."""
    cost = 0
    solver = z3.Optimize()
    solver.add(hard)
    if timeout > 0:
        solver.set("timeout", timeout)
    for i, clause in enumerate(soft):
        solver.add_soft(clause, weight=weight[i])
    if solver.check() == z3.sat:
        model = solver.model()
        for i, clause in enumerate(soft):
            if z3.is_false(model.eval(clause, True)):
                cost += weight[i]
    return cost
