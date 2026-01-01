from z3 import *
from Utility import *
from typing import Dict


class RuleContext:

    INVALID = BitVecVal(3, 2)
    IND = BitVecVal(1, 2)
    DEP = BitVecVal(0, 2)

    def __init__(self, var_num, ci_euf, additional_rule_enabled=True):
        self.ci_euf = ci_euf  # Function("is_ind", BitVecSort(var_num*3), IntSort())
        self.var_num = var_num

        self.INVALID = RuleContext.INVALID
        self.IND = RuleContext.IND
        self.DEP = RuleContext.DEP

        self.constraints: Dict[str, ExprRef] = {}

        self.initial_validity_condition()
        self.symmetric_rule()
        self.decomposition_rule()
        self.weak_union_rule()
        self.contraction_rule()
        self.intersection_rule()
        if additional_rule_enabled:
            self.composition_rule()
            # self.weak_transitivity_rule()
            self.chordality_rule()

    def is_valid(self, x, y, z):
        # x, y, z is a valid input to ind(x,y,z) if the conditions hold
        return And(
            [
                IsEmpty(BV_Intersection(x, y)),
                IsEmpty(BV_Intersection(z, y)),
                IsEmpty(BV_Intersection(z, x)),
                Not(IsEmpty(x)),
                Not(IsEmpty(y)),
            ]
        )

    @staticmethod
    def static_validity_condition(x, y, z):
        return And(
            [
                IsEmpty(BV_Intersection(x, y)),
                IsEmpty(BV_Intersection(z, y)),
                IsEmpty(BV_Intersection(z, x)),
                Not(IsEmpty(x)),
                Not(IsEmpty(y)),
            ]
        )

    def initial_validity_condition(self):
        x = BitVec("initial_validity_condition_x", self.var_num)
        y = BitVec("initial_validity_condition_y", self.var_num)
        z = BitVec("initial_validity_condition_z", self.var_num)
        cond1 = ForAll(
            [x, y, z],
            Not(self.is_valid(x, y, z)) == (self.ci_euf(x, y, z) == self.INVALID),
        )

        cond2 = ForAll(
            [x, y, z],
            Or(
                [
                    self.ci_euf(x, y, z) == self.INVALID,
                    self.ci_euf(x, y, z) == self.DEP,
                    self.ci_euf(x, y, z) == self.IND,
                ]
            ),
        )

        cond3 = Distinct(x, y, z)

        self.constraints["initial_validity_condition"] = simplify(
            And([cond1, cond2, cond3])
        )

    @staticmethod
    def static_initial_validity_condition(var_num, ci_euf):
        x = BitVec("initial_validity_condition_x", var_num)
        y = BitVec("initial_validity_condition_y", var_num)
        z = BitVec("initial_validity_condition_z", var_num)
        cond1 = Not(RuleContext.static_validity_condition(x, y, z)) == (
            ci_euf(x, y, z) == RuleContext.INVALID
        )
        cond2 = Or(
            [
                ci_euf(x, y, z) == RuleContext.INVALID,
                ci_euf(x, y, z) == RuleContext.DEP,
                ci_euf(x, y, z) == RuleContext.IND,
            ]
        )
        cond3 = Distinct(x, y, z)
        return simplify(And([cond1, cond2, cond3]))

    # def completeness_rule(self):
    #     x = BitVec("completeness_rule_x", self.var_num)
    #     y = BitVec("completeness_rule_y", self.var_num)
    #     z = BitVec("completeness_rule_z", self.var_num)
    #     cond = ForAll(
    #         [x,y,z],
    #         And(
    #             [
    #                 Implies(
    #                     And([self.is_valid(x,y,z), self.is_ind_func(GenerateFuncInput(x,y,z)) == 1]),
    #                     Not(self.is_ind_func(GenerateFuncInput(x,y,z)) == 0)
    #                 ),
    #                 Implies(
    #                     And([self.is_valid(x,y,z), self.is_ind_func(GenerateFuncInput(x,y,z)) == 0]),
    #                     Not(self.is_ind_func(GenerateFuncInput(x,y,z)) == 1)
    #                 )
    #             ]
    #         )
    #     )

    #     self.constraints["completeness_rule"] = cond

    def symmetric_rule(self):
        x = BitVec("symmetric_rule_x", self.var_num)
        y = BitVec("symmetric_rule_y", self.var_num)
        z = BitVec("symmetric_rule_z", self.var_num)
        cond = ForAll([x, y, z], self.ci_euf(x, y, z) == self.ci_euf(y, x, z))
        self.constraints["symmetric_rule"] = cond

    @staticmethod
    def static_symmetric_rule(var_num, ci_euf):
        x = BitVec("symmetric_rule_x", var_num)
        y = BitVec("symmetric_rule_y", var_num)
        z = BitVec("symmetric_rule_z", var_num)
        return ForAll([x, y, z], ci_euf(x, y, z) == ci_euf(y, x, z))

    def decomposition_rule(self):
        x = BitVec("decomposition_rule_x", self.var_num)
        y = BitVec("decomposition_rule_y", self.var_num)
        w = BitVec("decomposition_rule_w", self.var_num)
        z = BitVec("decomposition_rule_z", self.var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        self.is_valid(x, y, z),
                        self.is_valid(x, w, z),
                        self.is_valid(x, BV_Union(y, w), z),
                        self.ci_euf(x, BV_Union(y, w), z) == self.IND,
                    ]
                ),
                And(
                    [
                        self.ci_euf(x, y, z) == self.IND,
                        self.ci_euf(x, w, z) == self.IND,
                    ]
                ),
            ),
        )

        self.constraints["decomposition_rule"] = simplify(cond)

    @staticmethod
    def static_decomposition_rule(var_num, ci_euf):
        x = BitVec("decomposition_rule_x", var_num)
        y = BitVec("decomposition_rule_y", var_num)
        w = BitVec("decomposition_rule_w", var_num)
        z = BitVec("decomposition_rule_z", var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        RuleContext.static_validity_condition(x, y, z),
                        RuleContext.static_validity_condition(x, w, z),
                        RuleContext.static_validity_condition(x, BV_Union(y, w), z),
                        ci_euf(x, BV_Union(y, w), z) == RuleContext.IND,
                    ]
                ),
                And(
                    [
                        ci_euf(x, y, z) == RuleContext.IND,
                        ci_euf(x, w, z) == RuleContext.IND,
                    ]
                ),
            ),
        )

        return simplify(cond)

    def weak_union_rule(self):
        x = BitVec("weak_union_rule_x", self.var_num)
        y = BitVec("weak_union_rule_y", self.var_num)
        w = BitVec("weak_union_rule_w", self.var_num)
        z = BitVec("weak_union_rule_z", self.var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        self.is_valid(x, y, BV_Union(z, w)),
                        self.is_valid(x, BV_Union(y, w), z),
                        self.ci_euf(x, BV_Union(y, w), z) == self.IND,
                    ]
                ),
                And(
                    self.ci_euf(x, y, BV_Union(z, w)) == self.IND,
                ),
            ),
        )

        self.constraints["weak_union_rule"] = cond

    @staticmethod
    def static_weak_union_rule(var_num, ci_euf):
        x = BitVec("weak_union_rule_x", var_num)
        y = BitVec("weak_union_rule_y", var_num)
        w = BitVec("weak_union_rule_w", var_num)
        z = BitVec("weak_union_rule_z", var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        RuleContext.static_validity_condition(x, y, BV_Union(z, w)),
                        RuleContext.static_validity_condition(x, BV_Union(y, w), z),
                        ci_euf(x, BV_Union(y, w), z) == RuleContext.IND,
                    ]
                ),
                And(
                    ci_euf(x, y, BV_Union(z, w)) == RuleContext.IND,
                ),
            ),
        )

        return cond

    def contraction_rule(self):
        x = BitVec("contraction_rule_x", self.var_num)
        y = BitVec("contraction_rule_y", self.var_num)
        w = BitVec("contraction_rule_w", self.var_num)
        z = BitVec("contraction_rule_z", self.var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        self.is_valid(x, y, z),
                        self.is_valid(x, w, BV_Union(z, y)),
                        self.is_valid(x, BV_Union(y, w), z),
                        self.ci_euf(x, y, z) == self.IND,
                        self.ci_euf(x, w, BV_Union(y, z)) == self.IND,
                    ]
                ),
                And(
                    self.ci_euf(x, BV_Union(y, w), z) == self.IND,
                ),
            ),
        )

        self.constraints["contraction_rule"] = cond

    @staticmethod
    def static_contraction_rule(var_num, ci_euf):
        x = BitVec("contraction_rule_x", var_num)
        y = BitVec("contraction_rule_y", var_num)
        w = BitVec("contraction_rule_w", var_num)
        z = BitVec("contraction_rule_z", var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        RuleContext.static_validity_condition(x, y, z),
                        RuleContext.static_validity_condition(x, w, BV_Union(z, y)),
                        RuleContext.static_validity_condition(x, BV_Union(y, w), z),
                        ci_euf(x, y, z) == RuleContext.IND,
                        ci_euf(x, w, BV_Union(y, z)) == RuleContext.IND,
                    ]
                ),
                And(
                    ci_euf(x, BV_Union(y, w), z) == RuleContext.IND,
                ),
            ),
        )

        return cond

    def intersection_rule(self):
        x = BitVec("intersection_rule_x", self.var_num)
        y = BitVec("intersection_rule_y", self.var_num)
        w = BitVec("intersection_rule_w", self.var_num)
        z = BitVec("intersection_rule_z", self.var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        self.is_valid(x, y, BV_Union(z, w)),
                        self.is_valid(x, w, BV_Union(z, y)),
                        self.is_valid(x, BV_Union(y, w), z),
                        self.ci_euf(x, y, BV_Union(z, w)) == self.IND,
                        self.ci_euf(x, w, BV_Union(z, y)) == self.IND,
                    ]
                ),
                And(
                    self.ci_euf(x, BV_Union(y, w), z) == self.IND,
                ),
            ),
        )

        self.constraints["intersection_rule"] = cond

    @staticmethod
    def static_intersection_rule(var_num, ci_euf):
        x = BitVec("intersection_rule_x", var_num)
        y = BitVec("intersection_rule_y", var_num)
        w = BitVec("intersection_rule_w", var_num)
        z = BitVec("intersection_rule_z", var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        RuleContext.static_validity_condition(x, y, BV_Union(z, w)),
                        RuleContext.static_validity_condition(x, w, BV_Union(z, y)),
                        RuleContext.static_validity_condition(x, BV_Union(y, w), z),
                        ci_euf(x, y, BV_Union(z, w)) == RuleContext.IND,
                        ci_euf(x, w, BV_Union(z, y)) == RuleContext.IND,
                    ]
                ),
                And(
                    ci_euf(x, BV_Union(y, w), z) == RuleContext.IND,
                ),
            ),
        )

        return cond

    def composition_rule(self):
        x = BitVec("composition_rule_x", self.var_num)
        y = BitVec("composition_rule_y", self.var_num)
        w = BitVec("composition_rule_w", self.var_num)
        z = BitVec("composition_rule_z", self.var_num)

        # no need to use Implication; because when composition holds, decomposition also holds
        cond = ForAll(
            [x, y, w, z],
            And(
                [
                    self.is_valid(x, y, z),
                    self.is_valid(x, w, z),
                    self.is_valid(x, BV_Union(y, w), z),
                    self.ci_euf(x, BV_Union(y, w), z) == self.IND,
                ]
            )
            == And(
                [
                    self.is_valid(x, y, z),
                    self.is_valid(x, w, z),
                    self.is_valid(x, BV_Union(y, w), z),
                    self.ci_euf(x, y, z) == self.IND,
                    self.ci_euf(x, w, z) == self.IND,
                ]
            ),
        )

        self.constraints["composition_rule"] = cond

    @staticmethod
    def static_composition_rule(var_num, ci_euf):
        x = BitVec("composition_rule_x", var_num)
        y = BitVec("composition_rule_y", var_num)
        w = BitVec("composition_rule_w", var_num)
        z = BitVec("composition_rule_z", var_num)

        # no need to use Implication; because when composition holds, decomposition also holds
        cond = ForAll(
            [x, y, w, z],
            And(
                [
                    RuleContext.static_validity_condition(x, y, z),
                    RuleContext.static_validity_condition(x, w, z),
                    RuleContext.static_validity_condition(x, BV_Union(y, w), z),
                    ci_euf(x, BV_Union(y, w), z) == RuleContext.IND,
                ]
            )
            == And(
                [
                    RuleContext.static_validity_condition(x, y, z),
                    RuleContext.static_validity_condition(x, w, z),
                    RuleContext.static_validity_condition(x, BV_Union(y, w), z),
                    ci_euf(x, y, z) == RuleContext.IND,
                    ci_euf(x, w, z) == RuleContext.IND,
                ]
            ),
        )

        return cond

    def weak_transitivity_rule(self):  # optional
        x = BitVec("weak_transitivity_rule_x", self.var_num)
        y = BitVec("weak_transitivity_rule_y", self.var_num)
        z = BitVec("weak_transitivity_rule_z", self.var_num)
        u = BitVec("weak_transitivity_rule_u", self.var_num)

        cond = ForAll(
            [x, y, z, u],
            Implies(
                And(
                    [
                        BV_IsSingle(u),
                        self.is_valid(x, y, z),
                        self.is_valid(x, y, BV_Union(z, u)),
                        self.is_valid(u, y, z),
                        self.is_valid(x, u, z),
                        self.ci_euf(x, y, z) == self.IND,
                        self.ci_euf(x, y, BV_Union(z, u)) == self.IND,
                    ]
                ),
                And(
                    [
                        BV_IsSingle(u),
                        self.is_valid(x, y, z),
                        self.is_valid(x, y, BV_Union(z, u)),
                        self.is_valid(u, y, z),
                        self.is_valid(x, u, z),
                        self.ci_euf(u, y, z) == self.IND,
                        self.ci_euf(x, u, z) == self.IND,
                    ]
                ),
            ),
        )

        self.constraints["weak_transitivity_rule"] = cond

    @staticmethod
    def static_weak_transitivity_rule(var_num, ci_euf):
        x = BitVec("weak_transitivity_rule_x", var_num)
        y = BitVec("weak_transitivity_rule_y", var_num)
        z = BitVec("weak_transitivity_rule_z", var_num)
        u = BitVec("weak_transitivity_rule_u", var_num)

        return ForAll(
            [x, y, z, u],
            Implies(
                And(
                    [
                        BV_IsSingle(u),
                        RuleContext.static_validity_condition(x, y, z),
                        RuleContext.static_validity_condition(x, y, BV_Union(z, u)),
                        RuleContext.static_validity_condition(u, y, z),
                        RuleContext.static_validity_condition(x, u, z),
                        ci_euf(x, y, z) == RuleContext.IND,
                        ci_euf(x, y, BV_Union(z, u)) == RuleContext.IND,
                    ]
                ),
                And(
                    [
                        BV_IsSingle(u),
                        RuleContext.static_validity_condition(x, y, z),
                        RuleContext.static_validity_condition(x, y, BV_Union(z, u)),
                        RuleContext.static_validity_condition(u, y, z),
                        RuleContext.static_validity_condition(x, u, z),
                        ci_euf(u, y, z) == RuleContext.IND,
                        ci_euf(x, u, z) == RuleContext.IND,
                    ]
                ),
            ),
        )

    def chordality_rule(self):
        x = BitVec("chordality_rule_x", self.var_num)
        y = BitVec("chordality_rule_y", self.var_num)
        w = BitVec("chordality_rule_w", self.var_num)
        z = BitVec("chordality_rule_z", self.var_num)

        cond = ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        BV_IsSingle(x),
                        BV_IsSingle(y),
                        BV_IsSingle(z),
                        BV_IsSingle(w),
                        x != y,
                        y != z,
                        z != w,
                        self.ci_euf(x, y, BV_Union(z, w)) == self.IND,
                        self.ci_euf(z, w, BV_Union(x, y)) == self.IND,
                    ]
                ),
                And(
                    [
                        BV_IsSingle(x),
                        BV_IsSingle(y),
                        BV_IsSingle(z),
                        BV_IsSingle(w),
                        x != y,
                        y != z,
                        z != w,
                        self.ci_euf(x, y, z) == self.IND,
                        self.ci_euf(x, y, w) == self.IND,
                    ]
                ),
            ),
        )

        self.constraints["chordality_rule"] = cond

    @staticmethod
    def static_chordality_rule(var_num, ci_euf):
        x = BitVec("chordality_rule_x", var_num)
        y = BitVec("chordality_rule_y", var_num)
        w = BitVec("chordality_rule_w", var_num)
        z = BitVec("chordality_rule_z", var_num)

        return ForAll(
            [x, y, w, z],
            Implies(
                And(
                    [
                        BV_IsSingle(x),
                        BV_IsSingle(y),
                        BV_IsSingle(z),
                        BV_IsSingle(w),
                        x != y,
                        y != z,
                        z != w,
                        ci_euf(x, y, BV_Union(z, w)) == RuleContext.IND,
                        ci_euf(z, w, BV_Union(x, y)) == RuleContext.IND,
                    ]
                ),
                And(
                    [
                        BV_IsSingle(x),
                        BV_IsSingle(y),
                        BV_IsSingle(z),
                        BV_IsSingle(w),
                        x != y,
                        y != z,
                        z != w,
                        ci_euf(x, y, z) == RuleContext.IND,
                        ci_euf(x, y, w) == RuleContext.IND,
                    ]
                ),
            ),
        )

    # def simple_tactic(self):
    #     x = BitVec("simple_tactic_x", self.var_num)
    #     y = BitVec("simple_tactic_y", self.var_num)
    #     z = BitVec("simple_tactic_z", self.var_num)

    #     cond = ForAll(
    #         [x,y,z],
    #         Implies(
    #             And(
    #                 [
    #                     BV_IsSingle(x),
    #                     BV_IsSingle(y),
    #                     BV_IsSingle(z),
    #                     x != y,
    #                     y != z,
    #                     self.ci_euf(GenerateFuncInput(x,y,BV_GetEmptySet(x.size()))) == 1,
    #                     self.ci_euf(GenerateFuncInput(z,x,BV_GetEmptySet(x.size()))) == 0,
    #                 ]
    #             ),
    #             And(
    #                 [
    #                     BV_IsSingle(x),
    #                     BV_IsSingle(y),
    #                     BV_IsSingle(z),
    #                     x != y,
    #                     y != z,
    #                     self.ci_euf(GenerateFuncInput(x,y,z)) == 0,
    #                 ]
    #             )
    #         )
    #     )

    #     self.constraints["simple_tactic"] = cond
