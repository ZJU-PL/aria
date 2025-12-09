# coding: utf-8
"""
Translate Exists-Forall Bit-Vector Instances to Boolean Formulas
- QBF: z3 expression, QDIMAS
- SAT: z3 expression, DIMACS
- BDD: the BDD in pyEDA?

For maintaining the correlations of bit-vec and Boolean world.

Some references of QDIMACS
  http://www.qbflib.org/qdimacs.html
  http://beyondnp.org/pages/solvers/qbf-solvers/
"""

import logging
from typing import List

import z3
# from z3.z3util import get_vars

from aria.smt.bv.mapped_blast import translate_smt2formula_to_numeric_clauses

logger = logging.getLogger(__name__)


class EFBV2BoolAux:
    def __init__(self):
        self.universal_bools = []
        self.existential_bools = []
        self.bool_clauses = []

    def flattening_qf_fml(self, fml: z3.ExprRef):
        """
        The flattening_af_fml function takes a bit-vector formula and translates it to a Boolean formula.
        """
        bv2bool, bool2id, header, bool_clauses = translate_smt2formula_to_numeric_clauses(fml)
        return bool2id, bool_clauses

    def flattening(self, fml: z3.ExprRef, existential_vars: List[z3.ExprRef], universal_vars: List[z3.ExprRef]):
        """
        The flattening function takes a bit-vector formula and translates it to a Boolean formula.
        It also initializes some fields of the class:
        """
        # TODO: should handle cases where fml is simplified to be true or false
        # bv2bool, bool2id, header, clauses = translate_smt2formula_to_cnf(fml)
        # for cls in clauses:
        #    self.bool_clauses.append([int(lit) for lit in cls.split(" ")])

        bv2bool, bool2id, header, self.bool_clauses = translate_smt2formula_to_numeric_clauses(fml)

        logger.debug("  from bv to bools: {}".format(bv2bool))
        logger.debug("  from bool to sat id: {}".format(bool2id))

        for bv_var in existential_vars:
            for bool_var_name in bv2bool[str(bv_var)]:
                self.existential_bools.append(bool2id[bool_var_name])

        for bv_var in universal_vars:
            # print(bv_var, ": corresponding bools")
            # print([id_table[bname] for bname in bv2bool[str(bv_var)]])
            for bool_var_name in bv2bool[str(bv_var)]:
                self.universal_bools.append(bool2id[bool_var_name])

        logger.debug("existential vars: {}".format(self.existential_bools))
        logger.debug("universal vars:   {}".format(self.universal_bools))
        logger.debug("boolean clauses:  {}".format(self.bool_clauses))

    def to_qbf_z3expr(self) -> z3.ExprRef:
        """ Translate to an EFSMT(BF) formula to a QBF formula
        FIXME:
         After bit-blasting and CNF transformation, we may have many auxiliary Boolean variables.
         If operating over the Boolean level, it seems that we need to solve the problem below:
             Exists BX ForAll BY Exists BZ . BF(BX, BY, BZ)
                 where BZ is the set of auxiliary variables)
         Instead of the following problem
             Exists X ForAll Y . F(X, Y)
                 where X and Y are the existential and universal quantified bit-vectors, resp.)
        """
        prefix = "q"
        int2var = {}
        expr_clauses = []
        universal_vars = []
        existential_vars = []
        aux_bool_vars = []

        for clause in self.bool_clauses:
            expr_cls = []
            for numeric_lit in clause:
                # if numeric_lit == 0: break
                numeric_var = abs(numeric_lit)
                if numeric_var in int2var:
                    z3_var = int2var[numeric_var]
                else:
                    # create new Boolean vars
                    z3_var = z3.Bool("{0}{1}".format(prefix, numeric_var))
                    int2var[numeric_var] = z3_var
                    if numeric_var in self.universal_bools:
                        universal_vars.append(z3_var)
                    elif numeric_var in self.existential_bools:
                        existential_vars.append(z3_var)
                    else:
                        if z3_var not in aux_bool_vars:
                            aux_bool_vars.append(z3_var)

                z3_lit = z3.Not(z3_var) if numeric_lit < 0 else z3_var
                expr_cls.append(z3_lit)
            expr_clauses.append(z3.Or(expr_cls))

        fml = z3.And(expr_clauses)
        # print("E vars: \n{}".format(existential_vars))
        # print("U vars: \n{}".format(universal_vars))
        # print("fml:    \n{}".format(fml))

        # { NOET a trick for eliminating a subset of aux variables that are
        #     equivalent with existential or universal variables
        # FIXME: I forgot what algorithms the following code follow
        """
        auxiliary_boolean_vars = []
        replace_mappings = []
        cared_vars_length = len(self.existential_bools) + len(self.universal_bools)

        for var_id in self.existential_bools:
            # a trick for identifying equivalent variables
            to_rep = z3.Bool("{0}{1}".format(prefix, var_id + cared_vars_length))
            var_z3 = z3.Bool("{0}{1}".format(prefix, var_id))
            replace_mappings.append((to_rep, var_z3))

        for var_id in self.universal_bools:
            to_rep = z3.Bool("{0}{1}".format(prefix, var_id + cared_vars_length))
            var_z3 = z3.Bool("{0}{1}".format(prefix, var_id))
            replace_mappings.append((to_rep, var_z3))
        # print("Rep mapping: \n {}".format(replace_mappings))
        # NOTE: the line below removes a subset of aux variables
        simplified_fml = z3.simplify(z3.substitute(fml, replace_mappings))
        # } End of the trick for eliminating a subset of aux variables.
        
        # the following loop collects the remaining aux variables
        for var in get_vars(simplified_fml):   # get_vars can be slow!!!!
            if not (var in universal_vars or var in existential_vars):
                auxiliary_boolean_vars.append(var)
        """
        simplified_fml = fml  # do not use the above "fancy simplification"
        # TODO: remove universal vars that do not appear in simplified_fml?
        if len(aux_bool_vars) >= 1:
            # cnt = z3.ForAll(universal_vars, z3.Exists(aux_bool_vars, simplified_fml))
            cnt = z3.Exists(existential_vars,
                            z3.ForAll(universal_vars, z3.Exists(aux_bool_vars, simplified_fml)))
        else:
            cnt = z3.ForAll(universal_vars, simplified_fml)
        print("Finishing generating QBF CNT...")
        return cnt

    def to_qbf_qdimacs(self) -> str:
        """ Translate to an EFSMT(BF) formula to a QBF formula
        FIXME:
         After bit-blasting and CNF transformation, we may have many auxiliary Boolean variables.
         If operating over the Boolean level, it seems that we need to solve the problem below:
             Exists BX ForAll BY Exists BZ . BF(BX, BY, BZ)
                 where BZ is the set of auxiliary variables)
         Instead of the following problem
             Exists X ForAll Y . F(X, Y)
                 where X and Y are the existential and universal quantified bit-vectors, resp.)
        """
        aux_bool_vars = []
        prefix = "q"
        int2var = {}

        for clause in self.bool_clauses:
            for numeric_lit in clause:
                # if numeric_lit == 0: break
                numeric_var = abs(numeric_lit)
                if numeric_var not in int2var:
                    # create new Boolean vars
                    z3_var = z3.Bool("{0}{1}".format(prefix, numeric_var))
                    int2var[numeric_var] = z3_var
                    if (numeric_var not in self.universal_bools) and \
                            (numeric_var not in self.existential_bools):
                        # should we add the following line to avoid duplicates?
                        if numeric_var not in aux_bool_vars:
                            aux_bool_vars.append(numeric_var)

        logger.debug("aux vars: {}".format(aux_bool_vars))
        num_vars = len(self.existential_bools) + len(self.universal_bools) + len(aux_bool_vars)
        num_clauses = len(self.bool_clauses)
        fml_str = ["c QBF from EFSMT(BV)",
                   "p cnf {0} {1}".format(str(num_vars), str(num_clauses)),
                   # FIXME: should we keep the next line or not?
                   #  i.e, Exits X . Forall Y. Exists Z . P(X, Y, Z)
                   # "e {} 0".format(" ".join([str(v) for v in self.existential_bools])),
                   "a {} 0".format(" ".join([str(v) for v in self.universal_bools])),
                   "e {} 0".format(" ".join([str(v) for v in aux_bool_vars]))]

        for cls in self.bool_clauses:
            cls_str = [str(lit) for lit in cls]
            cls_str.append(" 0")
            fml_str.append(" ".join(cls_str))

        print("Finishing generating QBF CNT...")
        return "\n".join(fml_str) + "\n"


class EFBVFormulaTranslator:
    """
    """

    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 1)  # random seed
        """
        qe_level: 
          To convert an EFSMT(BV)formula to a Boolean formula without quantifies, we need 
          to perform quantifier elimination (QE) somewhere, e.g., before bit-blasting (word-level QE),
          or after bit-blasting (bool-level QE)
        qe_tactic:
          To perform the QE, currently we rely on Z3's tactics.
          TODO: implement a native expansion-based qe procedure?
        """
        self.qe_level = "word"  # { "bool", "word" }
        self.qe_tactic = "qe2"  # {"qe", "qe2"}

    def to_z3_qbf(self, fml: z3.ExprRef, existential_vars: List[z3.ExprRef],
                  universal_vars: List[z3.ExprRef]) -> z3.ExprRef:
        """Translate an EFSMT(BV) formula to a QBF formula (in z3)
        :param fml: a quantifier-free bit-vector formula
        :param existential_vars: the set of existential quantified bit-vector variables
        :param universal_vars: the set of universal quantified bit-vector formulas
        :return: a quantified Boolean formula (in z3)
        """
        translator = EFBV2BoolAux()
        translator.flattening(fml, existential_vars, universal_vars)
        return translator.to_qbf_z3expr()

    def to_qdimacs_str(self, fml: z3.ExprRef, existential_vars: List[z3.ExprRef], universal_vars: List[z3.ExprRef]):
        translator = EFBV2BoolAux()
        translator.flattening(fml, existential_vars, universal_vars)
        return translator.to_qbf_qdimacs()

    def to_z3_sat(self, fml: z3.BoolRef, existential_vars: List[z3.ExprRef], universal_vars: List[z3.ExprRef]):
        """Translate an EFSMT(BV) formula to a SAT formula
        :return: a quantifier-free Boolean formula (in z3)
        """
        print("Translating to Boolean formula in Z3")
        if self.qe_level == "bool":
            # First, convert the EFBV formula to a QBF formula
            # Second, perform Boolean-level quantifier elimination to obtin SAT
            qbf_fml = self.to_z3_qbf(fml, existential_vars, universal_vars)
            sat_formula = z3.Then("simplify", self.qe_tactic)(qbf_fml).as_expr()
            # Finally, convert the SAT formula to CNF form
            return z3.Then("simplify", "tseitin-cnf")(sat_formula).as_expr()
        else:
            qbv_fml = z3.ForAll(universal_vars, fml)
            # First, use word-level QE to build a quantifier-free bit-vec formula
            qfbv_fml = z3.Then("simplify", self.qe_tactic)(qbv_fml).as_expr()
            # second, convert the bit-vec formula to CNF
            # return z3.Then("simplify", "bit-blast", "simplify", "tseitin-cnf")(qfbv_fml).as_expr()
            return z3.Then("simplify", "bit-blast")(qfbv_fml).as_expr()

    def to_dimacs_str(self, fml: z3.BoolRef, existential_vars: List[z3.ExprRef], universal_vars: List[z3.ExprRef]):
        """Translate an EFSMT(BV) formula to a SAT formula
        1. First, perform quantifier elimination (currently, at bit-vector level)
        2. Second, perform bit-blasting
        """
        print("Translating to DIMACS")
        z3_sat_fml = self.to_z3_sat(fml, existential_vars, universal_vars)
        translator = EFBV2BoolAux()
        bool2id, bool_clauses = translator.flattening_qf_fml(z3_sat_fml)
        num_vars = len(bool2id)
        num_clauses = len(bool_clauses)

        fml_str = ["c SAT from EFSMT(BV)",
                   "p cnf {0} {1}".format(str(num_vars), str(num_clauses)),
                   ]

        for cls in bool_clauses:
            cls_str = [str(lit) for lit in cls]
            cls_str.append(" 0")
            fml_str.append(" ".join(cls_str))

        print("Finishing generating SAT CNT...")
        return "\n".join(fml_str) + "\n"


def demo_efbv2bool():
    x, y, z = z3.BitVecs("x y z", 4)
    fml = z3.Or(z > 3, z3.And(x + y == 6, x - y == 3))
    fml_manager = EFBVFormulaTranslator()
    qdimacs = fml_manager.to_qdimacs_str(fml, existential_vars=[z], universal_vars=[x, y])
    return qdimacs


if __name__ == '__main__':
    res = demo_efbv2bool()
    print(res)
