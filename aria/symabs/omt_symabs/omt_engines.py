"""
OMT (Optimization Modulo Theories) engines for symbolic abstraction.

TODO: use utils.z3_plus_smtlib_solver to integrate third-party engine
"""
from enum import Enum
from typing import List, Optional, Any

import z3

from aria.utils.z3_plus_smtlib_solver import Z3SolverPlus
from aria.symabs.omt_symabs.z3opt_util import box_optimize, optimize


class OMTEngineType(Enum):
    """Types of OMT engines available."""
    LINEAR_SEARCH = 0
    BINARY_SEARCH = 1  # not implemented
    MIXED_LINEAR_BINARY_SEARCH = 2  # not implemented
    QUANTIFIED_SATISFACTION = 3  # has bugs??
    Z3OPT = 4
    OPTIMATHSAT = 5  # external?
    CVC5 = 6  # does it support int and real?
    PYSMT = 7  # what does pysmt support


class OMTEngine:
    """Engine for Optimization Modulo Theories."""
    def __init__(self) -> None:
        self.initialized: bool = False
        self.formula: Optional[z3.ExprRef] = None
        self.compact_opt: bool = True
        self.engine_type: OMTEngineType = OMTEngineType.Z3OPT

    def maximize_with_linear_search(self, obj: z3.ExprRef) -> Any:
        """
        Linear Search based OMT
        """
        s = z3.Solver()
        s.add(self.formula)
        maximum = None
        if s.check() == z3.sat:
            maximum = s.model().eval(obj)
        while True:
            assumption = obj > maximum
            if s.check(assumption) == z3.unsat:
                break
            maximum = s.model().eval(obj)
            # print("current: ", maximum)
        return maximum

    def maximize_with_compact_linear_search(self, objs: List[z3.ExprRef]) -> List[Any]:
        """
        Linear Search for Boxed-OMT
        Essentially, this is equivalent to the algorithm from Li Yi's POPL 14
        """
        # print(objs)
        maximum_list: List[Any] = []
        s = z3.Solver()
        s.add(self.formula)
        if s.check() == z3.sat:
            for obj in objs:
                maximum_list.append(s.model().eval(obj))

        while True:
            assumption = z3.BoolVal(False)
            for i, obj in enumerate(objs):
                assumption = z3.Or(assumption, obj > maximum_list[i])
            if s.check(assumption) == z3.unsat:
                # stop here
                break
            cur_model = s.model()
            for i, obj in enumerate(objs):
                value_of_ith_obj = cur_model.eval(obj)
                if value_of_ith_obj.as_long() >= maximum_list[i].as_long():
                    maximum_list[i] = value_of_ith_obj
            print("current: ", maximum_list)

        return maximum_list

    def opt_with_qsat(self, exp: z3.ExprRef, minimize: bool) -> Any:
        """
        Quantified Satisfaction based OMT
        TODO: currently only works when exp is a variable (need to handle a term)? how to handle unbounded objectives? (seems not work??)
        """
        if z3.is_real(exp):
            exp_misc = z3.Real(str(exp) + "_m")
        else:
            exp_misc = z3.Int(str(exp) + "m")
        s = z3.Solver()
        new_fml = z3.substitute(self.formula, (exp, exp_misc))
        if minimize:
            qfml = z3.And(self.formula, z3.ForAll([exp_misc], z3.Implies(new_fml, exp <= exp_misc)))
        else:
            # TODO: why not working when x can be +oo????
            qfml = z3.And(
                self.formula,
                z3.ForAll([exp_misc], z3.Implies(new_fml, exp_misc <= exp))
            )
        s.add(qfml)
        if s.check() == z3.sat:
            tt = s.model().eval(exp)
            return tt
        print(s.to_smt2())
        print("UNSAT")
        return None

    def max_with_qe(self, exp: z3.ExprRef):
        """
        Quantifier Elimination based OMT
        """
        raise NotImplementedError

    def minimize_with_z3opt(self, obj: z3.ExprRef):
        """Minimize using Z3 optimizer."""
        return optimize(self.formula, obj, minimize=True)

    def maximize_with_z3opt(self, obj: z3.ExprRef):
        """Maximize using Z3 optimizer."""
        return optimize(self.formula, obj, minimize=False)

    def minimize_with_optimathsat(self, obj: z3.ExprRef):
        """Minimize using OptiMathSAT."""
        s = Z3SolverPlus()
        return s.optimize(self.formula, obj, minimize=True)

    def maximize_with_optimathsat(self, obj: z3.ExprRef):
        """
        Use OptiMathSAT (currently via pipe)
        """
        s = Z3SolverPlus()
        return s.optimize(self.formula, obj, minimize=False)

    def init_from_file(self, filename: str):
        """Initialize from an SMT2 file."""
        try:
            self.formula = z3.And(z3.parse_smt2_file(filename))
            self.initialized = True
        except z3.Z3Exception as ex:
            print("error when initialization")
            print(ex)

    def init_from_fml(self, fml: z3.BoolRef):
        """Initialize from a Z3 formula."""
        try:
            self.formula = fml
            self.initialized = True
        except z3.Z3Exception as ex:
            print("error when initialization")
            print(ex)

    def min_once(self, exp):
        """
        Minimize the objective exp
        """
        if self.engine_type == OMTEngineType.Z3OPT:
            return -self.maximize_with_z3opt(-exp)
        if self.engine_type == OMTEngineType.LINEAR_SEARCH:
            return -self.maximize_with_linear_search(-exp)
        if self.engine_type == OMTEngineType.BINARY_SEARCH:
            raise NotImplementedError
        if self.engine_type == OMTEngineType.MIXED_LINEAR_BINARY_SEARCH:
            raise NotImplementedError
        if self.engine_type == OMTEngineType.QUANTIFIED_SATISFACTION:
            return self.opt_with_qsat(exp, minimize=True)
        if self.engine_type == OMTEngineType.OPTIMATHSAT:
            return self.minimize_with_optimathsat(exp)
        return self.minimize_with_z3opt(exp)

    def max_once(self, exp: z3.ExprRef):
        """
        Maximize the objective exp
        """
        if self.engine_type == OMTEngineType.Z3OPT:
            return self.maximize_with_z3opt(exp)
        if self.engine_type == OMTEngineType.LINEAR_SEARCH:
            return self.maximize_with_linear_search(exp)
        if self.engine_type == OMTEngineType.BINARY_SEARCH:
            raise NotImplementedError
        if self.engine_type == OMTEngineType.MIXED_LINEAR_BINARY_SEARCH:
            raise NotImplementedError
        if self.engine_type == OMTEngineType.QUANTIFIED_SATISFACTION:
            return self.opt_with_qsat(exp, minimize=False)
        if self.engine_type == OMTEngineType.OPTIMATHSAT:
            return self.maximize_with_optimathsat(exp)
        return self.maximize_with_z3opt(exp)

    def min_max_many(self, multi_queries):
        """
        Boxed-OMT: compute the maximum AND minimum of queries in multi_queries
        """
        if self.engine_type == OMTEngineType.Z3OPT:
            print("fml: ", self.formula)
            print("objs: ", multi_queries, multi_queries)
            min_res, max_res = box_optimize(
                self.formula, minimize=multi_queries, maximize=multi_queries
            )
            print("box res: ", min_res, max_res)
            cnts = []
            for i, query in enumerate(multi_queries):
                vmin = min_res[i]
                vmax = max_res[i]
                str_vmin = str(vmin)
                str_vmax = str(vmax)
                print("vmin: ", str(vmin))
                print("vmax: ", str(vmax))
                # FIXME: how to efficiently identify oo and epsilon (the following is ugly)
                if "oo" not in str_vmin:
                    if "eps" in str_vmin:
                        cnts.append(query > z3.RealVal(vmin.children()[0]))
                        # cnts.append(query > vmin.children()[0])
                    else:
                        cnts.append(query >= z3.RealVal(vmin))
                        # cnts.append(query >= vmin)
                if "oo" not in str_vmax:
                    if "eps" in str_vmax:
                        cnts.append(query < z3.RealVal(vmax.children()[0]))
                        # cnts.append(query < vmax.children()[0])
                    else:
                        cnts.append(query <= z3.RealVal(vmax))
                        # cnts.append(query <= vmax)
            return z3.And(cnts)
        raise NotImplementedError
