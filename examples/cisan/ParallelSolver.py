from Rules import RuleContext
from Utility import *
from multiprocessing import Manager, Process, Pool
from typing import List, Tuple

INCONSISTENT_KB = "INCONSISTENT_KB"


class ParallelSlicingSolver:
    """
    A class to solve the problem of checking consistency and pruning in parallel by applying different rules.
    """

    rule_set = [
        "symmetric_rule",
        "decomposition_rule",
        "weak_union_rule",
        "contraction_rule",
        "intersection_rule",
        "composition_rule",
        #    "weak_transitivity_rule",
        "chordality_rule",
    ]

    def __init__(
        self,
        var_num: int,
        ci_facts: List[CIStatement],
        incoming_ci: CIStatement,
        timeout: int,
        max_workers=32,
    ):
        self.max_workers = max_workers
        self.var_num = var_num
        self.ci_facts = ci_facts
        self.incoming_ci = incoming_ci
        self.timeout = timeout
        self.manager = Manager()

    def check_consistency(self):
        self.return_dict = self.manager.dict()
        jobs: List[Process] = []
        for idx, rule_name in enumerate(ParallelSlicingSolver.rule_set):
            p = Process(
                target=ParallelSlicingSolver.worker,
                args=(
                    idx,
                    rule_name,
                    self.var_num,
                    self.ci_facts + [self.incoming_ci],
                    self.timeout,
                    self.return_dict,
                ),
            )
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        return unsat not in self.return_dict.values()

    def check_pruning(self):
        self.return_dict = self.manager.dict()
        jobs: List[Process] = []
        for idx, rule_name in enumerate(ParallelSlicingSolver.rule_set):
            p1 = Process(
                target=ParallelSlicingSolver.worker,
                args=(
                    idx + 1,
                    rule_name,
                    self.var_num,
                    self.ci_facts + [self.incoming_ci],
                    self.timeout,
                    self.return_dict,
                ),
            )
            p2 = Process(
                target=ParallelSlicingSolver.worker,
                args=(
                    -idx - 1,
                    rule_name,
                    self.var_num,
                    self.ci_facts + [self.incoming_ci.get_negation()],
                    self.timeout,
                    self.return_dict,
                ),
            )
            jobs.append(p1)
            jobs.append(p2)
            p1.start()
            p2.start()
        for proc in jobs:
            proc.join()
        for key in self.return_dict.keys():
            if key > 0:
                if self.return_dict[key] == unsat:
                    print("Pruning by rule:", ParallelSlicingSolver.rule_set[key - 1])
                    return self.incoming_ci.get_negation()
            else:
                if self.return_dict[key] == unsat:
                    print("Pruning by rule:", ParallelSlicingSolver.rule_set[-key - 1])
                    return self.incoming_ci
        return None

    @staticmethod
    def worker(
        index: int,
        rule_name: str,
        var_num: int,
        ci_facts: List[CIStatement],
        timeout: int,
        return_dict,
    ):
        solver = SolverFor("QF_UFBV")
        ci_euf = Function(
            "ci_euf",
            BitVecSort(var_num),
            BitVecSort(var_num),
            BitVecSort(var_num),
            BitVecSort(2),
        )
        rule_ctx = RuleContext(var_num, ci_euf)
        solver.add(rule_ctx.constraints["initial_validity_condition"])
        solver.add(rule_ctx.constraints[rule_name])
        for ci in ci_facts:
            solver.add(ci.generate_constraint(ci_euf, var_num))
        solver.set("timeout", timeout)
        return_dict[index] = solver.check()


class ParallelHybridEDSanSolver:
    rule_set = [
        "symmetric_rule",
        "decomposition_rule",
        "weak_union_rule",
        "contraction_rule",
        "intersection_rule",
        "composition_rule",
        "chordality_rule",
        "full",
    ]

    dump_unsat_core = True

    def __init__(
        self,
        var_num: int,
        ci_facts: List[CIStatement],
        incoming_ci: CIStatement,
        slicing_timeout: int,
        full_timeout: int,
    ):
        self.var_num = var_num
        self.ci_facts = ci_facts
        self.incoming_ci = incoming_ci
        self.slicing_timeout = slicing_timeout
        self.full_timeout = full_timeout

    def check_consistency(self) -> Tuple[bool, bool]:

        pool = Pool(processes=48)

        for i in pool.imap_unordered(
            ParallelHybridEDSanSolver.worker_helper,
            [
                (
                    rule,
                    self.var_num,
                    self.ci_facts + [self.incoming_ci],
                    self.full_timeout if rule == "full" else self.slicing_timeout,
                )
                for rule in ParallelHybridEDSanSolver.rule_set
            ],
            chunksize=1,
        ):
            if i[1] == unsat:
                pool.terminate()
                return False, i[0] != "full"
        return True, False

    @staticmethod
    def worker(rule_name: str, var_num: int, ci_facts: List[CIStatement], timeout: int):
        solver = SolverFor("QF_UFBV")
        if ParallelHybridEDSanSolver.dump_unsat_core:
            solver.set(unsat_core=True)
        ci_euf = Function(
            "ci_euf",
            BitVecSort(var_num),
            BitVecSort(var_num),
            BitVecSort(var_num),
            BitVecSort(2),
        )
        rule_ctx = RuleContext(var_num, ci_euf)
        if rule_name == "full":
            for rule in rule_ctx.constraints.items():
                if ParallelHybridEDSanSolver.dump_unsat_core:
                    solver.assert_and_track(rule[1], rule[0])
                else:
                    solver.add(rule[1])
        else:
            if ParallelHybridEDSanSolver.dump_unsat_core:
                solver.assert_and_track(rule_ctx.constraints[rule_name], rule_name)
                solver.add(rule_ctx.constraints["initial_validity_condition"])
            else:
                solver.add(rule_ctx.constraints[rule_name])
                solver.add(rule_ctx.constraints["initial_validity_condition"])

        for ci in ci_facts:
            if rule_name == "full" or ci_facts[-1].has_overlap(ci):
                if ParallelHybridEDSanSolver.dump_unsat_core and rule_name == "full":
                    solver.assert_and_track(
                        ci.generate_constraint(ci_euf, var_num), str(ci)
                    )
                else:
                    solver.add(ci.generate_constraint(ci_euf, var_num))
        solver.set("timeout", timeout)
        try:
            rlt = solver.check()
        except Z3Exception:
            rlt = unknown
        if ParallelHybridEDSanSolver.dump_unsat_core and len(solver.unsat_core()) != 0:
            print("Unsat core:", solver.unsat_core())
        return rule_name, rlt

    @staticmethod
    def worker_helper(args: Tuple[str, int, List[CIStatement], int]):
        return ParallelHybridEDSanSolver.worker(*args)


class ParallelGraphoidEDSanSolver:

    def __init__(
        self, var_num: int, ci_facts: List[CIStatement], incoming_ci: CIStatement
    ):
        self.var_num = var_num
        self.new_facts = ci_facts.copy()
        self.new_facts.append(incoming_ci)
        self.variables = psitip.rv(*[f"X{i}" for i in range(self.var_num)])
        self.ci_statements = [
            fact.graphoid_expr(self.variables) for fact in self.new_facts if fact.ci
        ]
        self.source_expr = psitip.alland(self.ci_statements)
        self.cd_terms = [
            fact.graphoid_term(self.variables) for fact in self.new_facts if not fact.ci
        ]

    def check_consistency(self):
        # pool = Pool(processes=48)
        # all CI statements are dependent in this case; no need to check
        if self.source_expr is None:
            return True
        source_bn = self.source_expr.get_bayesnet()
        # for i in pool.imap_unordered(ParallelGraphoidEDSanSolver.worker_helper, [(cd_term, source_bn) for cd_term in self.cd_terms], chunksize=1):
        #     if i:
        #         pool.terminate()
        #         return False
        for input_term in [(cd_term, source_bn) for cd_term in self.cd_terms]:
            if ParallelGraphoidEDSanSolver.worker_helper(input_term):
                return False
        return True

    @staticmethod
    def worker(cd_term: psitip.Expr, source_bn: psitip.BayesNet):
        return source_bn.check_ic(cd_term)

    @staticmethod
    def worker_helper(args: Tuple[psitip.Expr, psitip.BayesNet]):
        return ParallelGraphoidEDSanSolver.worker(*args)


class ParallelPSanFullSolver:

    def __init__(
        self,
        var_num: int,
        ci_facts: List[CIStatement],
        incoming_ci: CIStatement,
        timeout: int,
        max_workers=32,
    ):
        self.max_workers = max_workers
        self.var_num = var_num
        self.ci_facts = ci_facts
        self.incoming_ci = incoming_ci
        self.timeout = timeout
        self.manager = Manager()

    def check_pruning(self):
        self.return_dict = self.manager.dict()

        p1 = Process(
            target=ParallelPSanFullSolver.worker,
            args=(
                self.incoming_ci,
                self.var_num,
                self.ci_facts,
                self.timeout,
                self.return_dict,
            ),
        )
        p2 = Process(
            target=ParallelPSanFullSolver.worker,
            args=(
                self.incoming_ci.get_negation(),
                self.var_num,
                self.ci_facts,
                self.timeout,
                self.return_dict,
            ),
        )
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        if (
            self.return_dict[str(self.incoming_ci)] == unsat
            and self.return_dict[str(self.incoming_ci.get_negation())] == unsat
        ):
            return INCONSISTENT_KB
        if self.return_dict[str(self.incoming_ci)] == unsat:
            return self.incoming_ci.get_negation()
        if self.return_dict[str(self.incoming_ci.get_negation())] == unsat:
            return self.incoming_ci
        return None

    @staticmethod
    def worker(
        incoming_ci: CIStatement,
        var_num: int,
        ci_facts: List[CIStatement],
        timeout: int,
        return_dict,
    ):
        solver = SolverFor("QF_UFBV")
        ci_euf = Function(
            "ci_euf",
            BitVecSort(var_num),
            BitVecSort(var_num),
            BitVecSort(var_num),
            BitVecSort(2),
        )
        rule_ctx = RuleContext(var_num, ci_euf)
        for name, constraint in rule_ctx.constraints.items():
            solver.add(constraint)
        for ci in ci_facts:
            solver.add(ci.generate_constraint(ci_euf, var_num))
        solver.add(incoming_ci.generate_constraint(ci_euf, var_num))
        solver.set("timeout", timeout)
        return_dict[str(incoming_ci)] = solver.check()
