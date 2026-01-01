import random
from typing import List
from p_tqdm import p_umap
from functools import partial
from itertools import chain
from multiprocessing import Pool
from Rules import RuleContext
from Utility import *
from ParallelSolver import (
    ParallelSlicingSolver,
    ParallelPSanFullSolver,
    ParallelHybridEDSanSolver,
    INCONSISTENT_KB,
    ParallelGraphoidEDSanSolver,
)
from copy import deepcopy
from datetime import datetime
import functools

FUNCTION_TIME_DICT = {}

MARGINAL_COUNT = 0
CONSTRAINT_SLICING = True
MAX_BACKTRACK_THRESHOLD = 10
ENABLE_PARALLEL = True
ENABLE_GRAPHOID = True
ENABLE_MARGINAL_OMITTING = True  # if true, we will omit Psan and EDsan if the pool only contains marginal statements
# psitip.PsiOpts.setting(solver = "pyomo.glpk")


class EDsanAssertError(Exception):
    def __init__(self, method_name, message):
        self.method_name = method_name
        self.message = message
        super().__init__(self.message)


def time_statistic(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        cost = end_time - start_time
        if func.__name__ not in FUNCTION_TIME_DICT:
            FUNCTION_TIME_DICT[func.__name__] = (cost, 1)
        else:
            FUNCTION_TIME_DICT[func.__name__] = (
                FUNCTION_TIME_DICT[func.__name__][0] + cost,
                FUNCTION_TIME_DICT[func.__name__][1] + 1,
            )
        # print("Time cost of", func.__name__, "is", end_time - start_time)
        return result

    return wrapper


class KnowledgeBase:
    def __init__(self, global_facts: List[CIStatement], var_num: int, do_trace=True):
        self.facts = global_facts
        self.var_num = var_num
        self.do_track = do_trace
        self.backtrack_count = 0

    def copy(self):
        return KnowledgeBase(deepcopy(self.facts), self.var_num, self.do_track)

    def AddFact(self, fact: CIStatement):
        self.facts.append(fact)

    def AddFacts(self, facts: List[CIStatement]):
        self.facts += facts

    # Done: Add a function to randomly flip x% of the facts, (and return the flipped facts list)
    def Perturb(self, ratio: float, seed: int):
        # perturbed_facts = []
        random.seed(seed)
        select_index = random.sample(
            range(len(self.facts)), int(len(self.facts) * ratio)
        )
        for ind in select_index:
            self.facts[ind].ci = not self.facts[ind].ci
            # perturbed_facts.append(self.facts[ind])
        return select_index

    def FlipOne(self, seed: int):
        # Set the random seed to ensure reproducibility
        random.seed(seed)

        # Generate a random 32-bit integer
        random_int = random.randint(0, 0xFFFFFFFF)

        # Randomly choose which type of constraint to flip
        which_to_flip = random.randint(0, 1)

        if which_to_flip == 0:
            # If flipping a CD constraint, choose a random index from the list of CD constraints
            cd_idx = [idx for idx, fact in enumerate(self.facts) if not fact.ci]
            flip_idx = cd_idx[random_int % len(cd_idx)]
        else:
            # If flipping a CI constraint, choose a random index from the list of CI constraints
            ci_idx = [idx for idx, fact in enumerate(self.facts) if fact.ci]
            flip_idx = ci_idx[random_int % len(ci_idx)]

        # Flip the constraint at the selected index
        print("Flip", flip_idx, self.facts[flip_idx])
        self.facts[flip_idx].ci = not self.facts[flip_idx].ci

        # Return the index of the flipped constraint
        return flip_idx

    def FlipSome(self, seed: int, size: int = 10, handle_symmetric=True):
        # Set the random seed to ensure reproducibility
        random.seed(seed)

        # Generate a random 32-bit integer
        flip_idx_list = [
            random.randint(0, 0xFFFFFFFF) % len(self.facts) for _ in range(size)
        ]
        # random_int = random.randint(0, 0xffffffff)
        # # Randomly choose which type of constraint to flip
        # which_to_flip = random.randint(0, 1)

        # if which_to_flip == 0:
        #     # If flipping a CD constraint, choose a random index from the list of CD constraints
        #     cd_idx = [idx for idx, fact in enumerate(self.facts) if not fact.ci]
        #     flip_idx_list = [cd_idx[(random_int+offset) % len(cd_idx)] for offset in range(size)]
        # else:
        #     # If flipping a CI constraint, choose a random index from the list of CI constraints
        #     ci_idx = [idx for idx, fact in enumerate(self.facts) if fact.ci]
        #     flip_idx_list = [ci_idx[(random_int+offset) % len(ci_idx)] for offset in range(size)]

        # Flip the constraint at the selected index
        print("Flip", "\t".join([str(self.facts[idx]) for idx in flip_idx_list]))
        for idx in flip_idx_list:
            self.facts[idx].ci = not self.facts[idx].ci
            if handle_symmetric:
                for idx2 in range(len(self.facts)):
                    if idx2 != idx and self.facts[idx].is_form_equal(self.facts[idx2]):
                        self.facts[idx2].ci = self.facts[idx].ci

    # def ConstructKB(self):
    #     converted_facts = []
    #     for ind in self.facts:
    #         x, y, z, is_ind = ind
    #         converted_facts.append(
    #             (
    #                 KnowledgeBase.GenerateBitVal(x, self.var_num),
    #                 KnowledgeBase.GenerateBitVal(y, self.var_num),
    #                 KnowledgeBase.GenerateBitVal(z, self.var_num),
    #                 is_ind
    #             )
    #         )
    #     return converted_facts

    # @staticmethod
    # def Involves(raw_fact, node):
    #     return node in raw_fact[0] or node in raw_fact[1] or node in raw_fact[2]

    # @staticmethod
    # def StronglyInvolves(raw_fact : List[set], local_nodes: set):
    #     return local_nodes.intersection(raw_fact[0]) and local_nodes.intersection(raw_fact[1])

    # def ConstructMiniKB(self, local_nodes: set):
    #     local_facts = []
    #     for raw_fact in self.raw_facts:
    #         for node in local_nodes:
    #             if KnowledgeBase.Involves(raw_fact, node):
    #                 local_facts.append(raw_fact)
    #                 break
    #     return self.ConstructKB(local_facts)

    @staticmethod
    def GenerateConstraints(
        facts: List[CIStatement], ci_euf: FuncDeclRef, var_num: int
    ):
        constraints = []
        for fact in facts:
            constraints.append(fact.generate_constraint(ci_euf, var_num))
        return And(constraints)

    @staticmethod
    def GenerateLightWeightConstraints(
        facts: List[CIStatement], target: CIStatement, ci_euf: FuncDeclRef, var_num: int
    ):
        constraints = []
        for fact in facts:
            if fact.has_overlap(target):
                constraints.append(fact.generate_constraint(ci_euf, var_num))
        return And(constraints)

    def degenerate_check(self, incoming_ci: CIStatement):
        if any(map(lambda ci: ci.is_negation(incoming_ci), self.facts)):
            return False
        return True

    @time_statistic
    def marginal_omitting(self, incoming_ci: CIStatement):
        if incoming_ci.is_marginal():
            if all(map(lambda x: x.is_marginal(), self.facts)):
                return True
        return False

    @time_statistic
    def marginal_pruning(self, incoming_ci: CIStatement):
        if incoming_ci.is_marginal():
            if all(map(lambda x: x.is_marginal(), self.facts)) == True:
                return True
        return False

        # return None
        # if incoming_ci.is_marginal():
        #     if all(map(lambda x: x.is_marginal(), self.facts)) == True:
        #         if any(map(lambda ci: ci.is_negation(incoming_ci), self.facts)):
        #             return incoming_ci.get_negation()
        #         else:
        #             return "SKIP"
        # return None

    # the first return value denotes whether we obtain meaningful result
    # the second return value denotes whether the hyp holds
    def PSanFullLegacy(self, hyp: CIStatement, prune_neg=False):
        # return: [int,bool] first bool variable denotes whether hyp is true (1)
        # or false (0) or non-deterministic (-1); second bool variable denotes
        # whether the solver returns unknown

        # if local_nodes != None:
        #     facts, var_map = self.ConstructMiniKB(local_nodes)
        # else:
        # facts, var_map = self.ConstructKB()

        ci_euf = Function(
            "ci_euf",
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(2),
        )

        hyp_constraint = hyp.generate_constraint(ci_euf, self.var_num)
        neg_hyp_constraint = hyp.get_negation().generate_constraint(
            ci_euf, self.var_num
        )

        kb_constraint = KnowledgeBase.GenerateConstraints(
            self.facts, ci_euf, self.var_num
        )

        rule_ctx = RuleContext(self.var_num, ci_euf)

        timeout = self.compute_timeout("psan_full")

        # pos_solver = SolverFor("QF_UFBV")
        pos_solver = Solver()
        pos_solver.set("timeout", timeout)
        # neg_solver = SolverFor("QF_UFBV")
        neg_solver = Solver()
        neg_solver.set("timeout", timeout)

        if not self.do_track:
            pos_solver.add(hyp_constraint)
            neg_solver.add(neg_hyp_constraint)
            pos_solver.add(kb_constraint)
            neg_solver.add(kb_constraint)
        else:
            pos_solver.assert_and_track(hyp_constraint, "hyp")
            print(pos_solver.check(), "hyp_constraint")
            neg_solver.assert_and_track(neg_hyp_constraint, "not hyp")
            pos_solver.assert_and_track(kb_constraint, "kb")
            print(pos_solver.check(), "kb_constraint")
            neg_solver.assert_and_track(kb_constraint, "kb")

        for rule in rule_ctx.constraints.items():
            name, constraint = rule
            if not self.do_track:
                pos_solver.add(constraint)
                neg_solver.add(constraint)
            else:
                pos_solver.assert_and_track(constraint, name)
                neg_solver.assert_and_track(constraint, name)
        pos_rlt = pos_solver.check()
        if pos_rlt == unsat and prune_neg:
            return 0, True
        neg_rlt = neg_solver.check()
        if pos_rlt == unsat and neg_rlt == sat:
            return 0, False
        elif pos_rlt == sat and neg_rlt == unsat:
            return 1, False
        elif pos_rlt == unsat and neg_rlt == unknown:
            return 0, True
        elif pos_rlt == unknown and neg_rlt == unsat:
            return 1, True
        elif pos_rlt == unknown and neg_rlt == unknown:
            return -1, True
        elif pos_rlt == sat and neg_rlt == sat:
            return -1, False
        elif pos_rlt == sat and neg_rlt == unknown:
            return -1, True
        elif pos_rlt == sat and neg_rlt == unknown:
            return -1, True
        else:
            return -2, False

    # the first return value denotes whether we obtain meaningful result
    # the second return value denotes whether the hyp holds
    def PSanSlicing(self, hyp: CIStatement, prune_neg=False):  # Legacy code
        # return: [int] first variable denotes whether hyp is true (1)
        # or false (0) or non-deterministic (-1)

        ci_euf = Function(
            "ci_euf",
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(2),
        )

        hyp_constraint = hyp.generate_constraint(ci_euf, self.var_num)
        neg_hyp_constraint = hyp.get_negation().generate_constraint(
            ci_euf, self.var_num
        )

        kb_constraint = KnowledgeBase.GenerateLightWeightConstraints(
            self.facts, hyp, ci_euf, self.var_num
        )

        rule_ctx = RuleContext(self.var_num, ci_euf)

        timeout = self.compute_timeout("psan_slicing")

        pos_solvers = {i: SolverFor("QF_UFBV") for i in rule_ctx.constraints}
        neg_solvers = {i: SolverFor("QF_UFBV") for i in rule_ctx.constraints}

        indistinguishable = [sat, unknown]

        for rule in rule_ctx.constraints.items():
            name, constraint = rule
            pos_solvers[name].set("timeout", timeout)
            neg_solvers[name].set("timeout", timeout)
            pos_solvers[name].add(constraint)
            neg_solvers[name].add(constraint)
            pos_solvers[name].add(hyp_constraint)
            neg_solvers[name].add(neg_hyp_constraint)
            pos_solvers[name].add(kb_constraint)
            neg_solvers[name].add(kb_constraint)
            pos_rlt = pos_solvers[name].check()
            if pos_rlt == unsat and prune_neg:
                return 0, True
            neg_rlt = neg_solvers[name].check()
            if pos_rlt in indistinguishable and neg_rlt in indistinguishable:
                continue
            if pos_rlt == unsat:
                return 0
            if neg_rlt == unsat:
                return 1
        return -1

    def PSanFullLegacy(self, hyp: CIStatement):
        ci_euf = Function(
            "ci_euf",
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(2),
        )

        hyp_constraint = hyp.generate_constraint(ci_euf, self.var_num)
        neg_hyp_constraint = hyp.get_negation().generate_constraint(
            ci_euf, self.var_num
        )

        kb_constraint = KnowledgeBase.GenerateConstraints(
            self.facts, ci_euf, self.var_num
        )

        rule_ctx = RuleContext(self.var_num, ci_euf)

        timeout = self.compute_timeout("psan_full")

        # pos_solver = SolverFor("QF_UFBV")
        pos_solver = Solver()
        pos_solver.set("timeout", timeout)
        # neg_solver = SolverFor("QF_UFBV")
        neg_solver = Solver()
        neg_solver.set("timeout", timeout)
        pos_solver.add(hyp_constraint)
        neg_solver.add(neg_hyp_constraint)
        pos_solver.add(kb_constraint)
        neg_solver.add(kb_constraint)
        for rule in rule_ctx.constraints.items():
            name, constraint = rule
            pos_solver.add(constraint)
            neg_solver.add(constraint)
        pos_rlt = pos_solver.check()
        if pos_rlt == unsat:
            return hyp.get_negation()
        neg_rlt = neg_solver.check()
        if neg_rlt == unsat:
            return hyp
        return None

    @time_statistic
    def PSanFullParallel(self, hyp: CIStatement):
        ps = ParallelPSanFullSolver(
            self.var_num, self.facts, hyp, self.compute_timeout("psan_full")
        )
        return ps.check_pruning()

    @time_statistic
    def PSanSlicingParallel(self, hyp: CIStatement):
        # return: [int] first variable denotes whether hyp is true (1)
        # or false (0) or non-deterministic (-1)
        ps = ParallelSlicingSolver(
            self.var_num,
            [fact for fact in self.facts if fact.has_overlap(hyp)],
            hyp,
            self.compute_timeout("psan_slicing"),
        )
        return ps.check_pruning()
        # if rlt is None:
        #     return -1
        # elif rlt.ci == hyp.ci:
        #     return 1
        # else:
        #     return 0

    @time_statistic
    def Graphoid(self, hyp: CIStatement):
        new_facts = self.facts.copy()
        new_facts.append(hyp)
        return self.graphoid_consistency_checking(new_facts)

    @time_statistic
    def CheckConsistency(
        self,
    ):  # ToM: if even the full EDsan fails to verify, should we still return True?
        # ci_euf = Function("ci_euf", BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(2))
        # kb_constraint = KnowledgeBase.GenerateConstraints(self.facts, ci_euf, self.var_num)
        # solver = SolverFor("QF_UFBV")
        # timeout = int(max(10_000, 1000 * self.var_num * 2))
        # solver.set("timeout", timeout)
        # solver.add(kb_constraint)
        # rule_ctx = RuleContext(self.var_num, ci_euf)
        # for rule in rule_ctx.constraints.items():
        #     solver.add(rule[1])
        # check_rlt = solver.check()
        # return check_rlt == sat or check_rlt == unknown

        # Done: refer to following code snippet to implement
        # Consistent: return True; Inconsistent: return False
        last_ci = self.facts[-1]
        if self.degenerate_check(last_ci) == False:
            return False
        if ENABLE_MARGINAL_OMITTING:
            # Done: implement marginal omittings
            if self.marginal_omitting(last_ci):
                return True
        if ENABLE_GRAPHOID:
            if self.Graphoid(last_ci) == False:
                return False
        if CONSTRAINT_SLICING:
            if self.EDSanSlicingParallel(last_ci) == False:
                return False
        return self.EDSanFull(last_ci)

    @time_statistic
    def EDSanFull(self, incoming_ci: CIStatement):
        ci_euf = Function(
            "ci_euf",
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(2),
        )
        kb_constraint = KnowledgeBase.GenerateConstraints(
            self.facts, ci_euf, self.var_num
        )
        incoming_constraint = incoming_ci.generate_constraint(ci_euf, self.var_num)
        solver = SolverFor("QF_UFBV")
        timeout = self.compute_timeout("edsan_full")
        solver.set("timeout", timeout)
        solver.add(kb_constraint)
        solver.add(incoming_constraint)
        rule_ctx = RuleContext(self.var_num, ci_euf)
        for rule in rule_ctx.constraints.items():
            solver.add(rule[1])
        check_rlt = solver.check()
        return check_rlt == sat or check_rlt == unknown

    def EDSanSlicing(self, incoming_ci: CIStatement):
        ci_euf = Function(
            "ci_euf",
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(self.var_num),
            BitVecSort(2),
        )
        kb_constraint = KnowledgeBase.GenerateLightWeightConstraints(
            self.facts, incoming_ci, ci_euf, self.var_num
        )
        incoming_constraint = incoming_ci.generate_constraint(ci_euf, self.var_num)
        rule_ctx = RuleContext(self.var_num, ci_euf)
        solvers = {i: SolverFor("QF_UFBV") for i in rule_ctx.constraints}
        timeout = self.compute_timeout("esdan_slicing")
        for rule in rule_ctx.constraints.items():
            name, constraint = rule
            solvers[name].add(constraint)
            solvers[name].add(incoming_constraint)
            solvers[name].add(kb_constraint)
            solvers[name].set("timeout", timeout)
            rlt = solvers[name].check()
            if rlt == unsat:
                return False
        return True

    @time_statistic
    def EDSanSlicingParallel(self, incoming_ci: CIStatement):
        ps = ParallelSlicingSolver(
            self.var_num,
            [fact for fact in self.facts if fact.has_overlap(incoming_ci)],
            incoming_ci,
            self.compute_timeout("edsan_slicing"),
        )
        return ps.check_consistency()

    @time_statistic
    def EDSanHybridParallel(self, incoming_ci: CIStatement):
        ps = ParallelHybridEDSanSolver(
            self.var_num,
            self.facts,
            incoming_ci,
            self.compute_timeout("edsan_slicing"),
            self.compute_timeout("edsan_full"),
        )
        return ps.check_consistency()

    @time_statistic
    def EDsanGraphoidParallel(self, incoming_ci: CIStatement):
        ps = ParallelGraphoidEDSanSolver(self.var_num, self.facts, incoming_ci)
        return ps.check_consistency()

    def EDSan(
        self, incoming_ci: CIStatement
    ):  # Done: implement another PC.py for EDSan
        # assert self.degenerate_check(incoming_ci), "There has been a degenerate case!"
        if ENABLE_MARGINAL_OMITTING:
            # Done: implement marginal omittings
            if self.marginal_omitting(incoming_ci):
                return
        if ENABLE_GRAPHOID:
            assert (
                self.Graphoid(incoming_ci) == True
            ), f"Graphoid find inconsistency on {incoming_ci}"
        if CONSTRAINT_SLICING:
            assert self.EDSanSlicingParallel(
                incoming_ci
            ), f"EDSanSlicing find inconsistency on {incoming_ci}"
        assert self.EDSanFull(
            incoming_ci
        ), f"EDSanFull find inconsistency on {incoming_ci}"

    def EDSan_ablation(
        self,
        incoming_ci: CIStatement,
        use_marginal: bool = True,
        use_graphoid: bool = False,
        use_slicing: bool = False,
        verbose: bool = True,
    ):
        if verbose:
            print(
                f"Current Optimization: marginal={use_marginal}, graphoid={use_graphoid}, slicing={use_slicing}"
            )
        if use_marginal:
            if self.marginal_omitting(incoming_ci):
                global MARGINAL_COUNT
                MARGINAL_COUNT += 1
                # print("marginal ommiting")
                return
        if use_graphoid:
            # assert self.Graphoid(
            #     incoming_ci) == True, f"Graphoid find inconsistency on {incoming_ci}"
            # if self.Graphoid(incoming_ci) == False:
            if self.EDsanGraphoidParallel(incoming_ci) == False:
                raise EDsanAssertError(
                    "Graphoid", f"Graphoid find inconsistency on {incoming_ci}"
                )

        if use_slicing:
            # assert self.EDSanSlicingParallel(
            #     incoming_ci), f"EDSanSlicing find inconsistency on {incoming_ci}"
            ret, if_slicing_find = self.EDSanHybridParallel(incoming_ci)
            if ret == False:
                if if_slicing_find:
                    raise EDsanAssertError(
                        "EDSanSlicing",
                        f"EDSanSlicing find inconsistency on {incoming_ci}",
                    )
                else:
                    raise EDsanAssertError(
                        "EDSanFull (with Slicing)",
                        f"EDSanFull (with Slicing) find inconsistency on {incoming_ci}",
                    )
        if not use_marginal and not use_graphoid and not use_slicing:
            # assert self.EDSanFull(incoming_ci), f"EDSanFull find inconsistency on {incoming_ci}"
            if self.EDSanFull(incoming_ci) == False:
                raise EDsanAssertError(
                    "EDSanFull", f"EDSanFull find inconsistency on {incoming_ci}"
                )

    def EDsan_singleM(self, incoming_ci: CIStatement, method_name: str):
        if self.marginal_omitting(incoming_ci):
            return
        if method_name == "Graphoid":
            assert (
                self.Graphoid(incoming_ci) == True
            ), f"Graphoid find inconsistency on {incoming_ci}"
        elif method_name == "Slicing":
            assert self.EDSanSlicingParallel(
                incoming_ci
            ), f"EDSanSlicing find inconsistency on {incoming_ci}"
        else:
            raise ValueError("method_name should be either Graphoid or Slicing")

    @time_statistic
    def Backtracking(self):
        print("start backtracking KB")
        self.backtrack_count += 1
        while not self.CheckConsistency():
            dropped = self.facts.pop()
            print("drop", str(dropped))

    def BatchPSan(self, hyps: List[CIStatement]):  # Legacy code
        confirmed_ci = []
        if ENABLE_GRAPHOID:
            for hyp in hyps:
                if self.Graphoid(hyp) == False:
                    confirmed_ci.append(hyp.get_negation())
                    hyps.remove(hyp)
                elif self.Graphoid(hyp.get_negation()) == False:
                    confirmed_ci.append(hyp)
                    hyps.remove(hyp)

        # lightweight checking
        if CONSTRAINT_SLICING:
            if ENABLE_PARALLEL:
                with Pool() as pool:
                    check_results = pool.map(partial(self.PSanSlicing), hyps)
            else:
                check_results = list(map(partial(self.PSanSlicing), hyps))
            results = {hyps[idx]: result for idx, result in enumerate(check_results)}
            confirmed_ci = [
                hyps[idx] if result == 1 else hyps[idx].get_negation()
                for idx, result in enumerate(check_results)
                if result != -1
            ]
            remained_hyps = [hyp for hyp in hyps if results[hyp] == -1]
        else:
            remained_hyps = hyps
        # complete checking
        if ENABLE_PARALLEL:
            with Pool() as pool:
                check_results = pool.map(partial(self.PSanFullLegacy), remained_hyps)
        else:
            check_results = list(map(partial(self.PSanFullLegacy), remained_hyps))
        results = {
            remained_hyps[idx]: result for idx, result in enumerate(check_results)
        }
        if len([r for r in results if results[r][0] == -2]) != 0:
            self.Backtracking()
            return []
        for ci in results:
            ci: CIStatement
            status, _ = results[ci]
            if status == 0:
                confirmed_ci.append(ci.get_negation())
            elif status == 1:
                confirmed_ci.append(ci)
        return confirmed_ci

    def SinglePSan(self, hyp: CIStatement):  # Implementation of PSan
        # Note: Assume the KB (proceeding facts set) is consistent, otherwise, this function may require revision
        if ENABLE_MARGINAL_OMITTING:
            # Done: implement marginal omitting
            if self.marginal_pruning(hyp):
                return None
            # if marginal_output is not None:
            #     if marginal_output == "SKIP":
            #         return None
            #     else:
            #         print("marginal pruning:", marginal_output, "is inferred")
            #         return marginal_output

        if ENABLE_GRAPHOID:
            graphoid_outcome = self.graphoid_pruning(hyp)
            if graphoid_outcome is not None:
                print("graphoid:", graphoid_outcome, "is inferred")
                return graphoid_outcome
        if CONSTRAINT_SLICING:
            # psanslicing_result = self.PSanSlicing(hyp)
            psanslicing_outcome = self.PSanSlicingParallel(hyp)
            # if psanslicing_result == 0:
            #     print("slicing:", hyp.get_negation(), "is inferred")
            #     return hyp.get_negation()
            # elif psanslicing_result == 1:
            #     print("slicing:", hyp, "is inferred")
            #     return hyp
            if psanslicing_outcome is not None:
                print("slicing:", psanslicing_outcome, "is inferred")
                return psanslicing_outcome
        # psanfull_result = self.PSanFullLegacy(hyp)
        # if psanfull_result == 0:
        #     print("full:", hyp.get_negation(), "is inferred")
        #     return hyp.get_negation()
        # elif psanfull_result == 1:
        #     print("full:", hyp, "is inferred")
        #     return hyp
        psanfull_outcome = self.PSanFullParallel(hyp)
        if psanfull_outcome == INCONSISTENT_KB:
            self.Backtracking()
            return None
        if psanfull_outcome is not None:
            print("full:", psanfull_outcome, "is inferred")
            return psanfull_outcome
        print("full:", hyp, "is not inferred")
        return None

    def RecursivePSan(
        self,
        hyps: List[CIStatement],
        ind_func,
        step_size: int = 1,
        early_stop: bool = False,
    ) -> List[CIStatement]:  # Legacy code
        if self.backtrack_count >= MAX_BACKTRACK_THRESHOLD:
            print("max backtrack count reached, fallback to CI tests")
            return []
        result: List[CIStatement] = []
        sigma_star = hyps

        def _stop():
            if early_stop:
                for ci in result:
                    if ci.ci:
                        return True
            return False

        while len(sigma_star) > 0:
            confirmed_ci = self.BatchPSan(sigma_star)
            self.AddFacts(confirmed_ci)
            result += confirmed_ci
            if _stop():
                return result
            new_sigma_star = []
            for hyp in sigma_star:
                iso_ci = [i for i in confirmed_ci if hyp.is_isomorphic(i)]
                if len(iso_ci) == 0:
                    new_sigma_star.append(hyp)
            sigma_star = new_sigma_star
            if len(sigma_star) == 0:
                break
            random.shuffle(sigma_star)
            for i in range(min(len(sigma_star), step_size)):
                ci: CIStatement = sigma_star.pop(0)
                assert len(ci.x) == 1 and len(ci.y) == 1
                x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                new_ci = CIStatement.createByXYZ(x, y, z, ind_func(x, y, z))
                self.AddFact(new_ci)
                result.append(new_ci)
                if _stop():
                    return result

        return result

    def graphoid_consistency_checking(self, facts: List[CIStatement]) -> bool:
        variables = psitip.rv(*[f"X{i}" for i in range(self.var_num)])
        ci_statements = [fact.graphoid_expr(variables) for fact in facts if fact.ci]
        if len(ci_statements) == 0:
            return True
        ci_facts = [fact for fact in facts if fact.ci]
        source_expr = psitip.alland(ci_statements)
        cd_term = [fact.graphoid_term(variables) for fact in facts if not fact.ci]
        cd_facts = [fact for fact in facts if not fact.ci]
        for idx, cd_term in enumerate(cd_term):
            # graphoid does not support disprove conditional independence
            # if target term is always independent given source statements, then there is an inconsistency in the graphoid
            if source_expr.get_bayesnet().check_ic(cd_term):
                print("input:", "\t".join([str(fact) for fact in ci_facts]))
                print("target:", cd_facts[idx])
                return False
        return True

    @time_statistic
    def graphoid_pruning(self, incoming_ci: CIStatement) -> bool:
        if not self.graphoid_consistency_checking(self.facts + [incoming_ci]):
            return incoming_ci.get_negation()
        elif not self.graphoid_consistency_checking(
            self.facts + [incoming_ci.get_negation()]
        ):
            return incoming_ci
        else:
            return None
        # return source_expr.get_bayesnet().check_ic(target_term)

    def compute_timeout(
        self, check_type: str
    ) -> int:  # This version can only be used for Earthquake dataset
        if check_type == "psan_full":
            return int(max(30_000, 1000 * self.var_num * 2))
        elif check_type == "psan_slicing":  # Todo: change coefficient and variable
            # return int(max(20_000, 1000 * len(self.facts)))
            return int(max(10_000, 1000 * self.var_num))
        elif check_type == "edsan_full":
            return int(max(120_000, 1000 * self.var_num * 2))
        elif check_type == "edsan_slicing":
            # return int(max(20_000, 1000 * self.var_num * 1.5))
            return int(max(25_000, 1000 * self.var_num))


if __name__ == "__main__":

    # set_param("parallel.enable", True)
    # set_param("euf", True)

    # facts = [
    #     ({0}, {1}, {}, False),
    #     ({0}, {2}, {}, False),
    #     ({1}, {2}, {}, True),
    #     ({1}, {2}, {}, True),
    #     # ({1}, {2}, {3}, True),
    #     # ({2}, {3}, {1}, True),
    #     # ({1}, {2}, {2}, False)
    # ]

    # hyp = [({0}, {1}, {2}, True), ({3}, {1}, {2}, True), ({0}, {3}, {2}, True)]

    # kb = KnowledgeBase(list(map(CIStatement.create, facts)), var_num=4)

    # # is_hyp, has_unknown = kb.VerifyCIHypothesis(CIRelation.create(hyp))
    # def _mock_ind(x,y,z):
    #     print(x,y,z)
    #     return True
    # rlt = kb.RecursivePSan(list(map(CIStatement.create, hyp)), _mock_ind)
    # for ci in rlt:
    #     print(ci)
    # psitip.PsiOpts.setting(lptype = "H")
    X0, X1, X2, X3, X4 = psitip.rv("X", "Y", "Z", "W", "U")
    facts = [
        psitip.I(X0 & X1) == 0,
        psitip.I(X0 & X2) < 0,
        psitip.I(X0 & X3) < 0,
        psitip.I(X0 & X4) < 0,
        psitip.I(X1 & X2) < 0,
        psitip.I(X1 & X3) < 0,
        psitip.I(X1 & X4) < 0,
        psitip.I(X2 & X3) < 0,
        psitip.I(X2 & X4) < 0,
        psitip.I(X3 & X4) < 0,
        psitip.I(X0 & X2 | X3) < 0,
        # psitip.I(X0&X2|X4) < 0,
        # psitip.I(X0&X3|X2) == 0,
        # psitip.I(X0&X4|X2) == 0,
        # psitip.I(X1&X2|X3) < 0,
        # psitip.I(X1&X2|X4) < 0,
        # psitip.I(X1&X3|X2) == 0,
        # psitip.I(X1&X4|X2) == 0,
        # psitip.I(X2&X0|X1) < 0,
    ]
    source_expr = psitip.alland(facts)
    print(source_expr.implies(psitip.I(X0 & X2 | X3) < 0))
