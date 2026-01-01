import warnings
import z3

# model.compact should be turned off to get uncompressed models
z3.set_param("model.compact", False)

import aria.quant.fossil.naturalproofs.uct as uct
import aria.quant.fossil.naturalproofs.pfp as pfp
import aria.quant.fossil.naturalproofs.utils as nputils
import aria.quant.fossil.naturalproofs.extensions.finitemodel as finitemodel


def generate_pfp_constraint(
    rec_funcdeclref, lemma_args, finite_model, annctx, smt_simplify=False
):
    if smt_simplify:
        warnings.warn(
            "setting simplify to true causes problems in formula parse tree traversal. Ignoring simplify."
        )
        smt_simplify = False
    model_dict = finite_model.finitemodel
    lemma_arity = len(lemma_args)
    # Assuming that all arguments of the lemma are of the foreground sort
    lemma_rhs_macro = z3.Function(
        "lemma", *([z3.IntSort()] * lemma_arity), z3.BoolSort()
    )
    # Assuming that the arguments for the recursive definition on the lhs are the first 'k' variables in lemma_args
    lhs_args = lemma_args[: rec_funcdeclref.arity()]
    lemma = z3.Implies(rec_funcdeclref(*lhs_args), lemma_rhs_macro(*lemma_args))
    lemma_pfp = pfp.make_pfp_formula(lemma)
    # 'Evaluate' the pfp formula on the given finite model
    lemma_pfp_eval = _eval_vars(lemma_pfp, model_dict, annctx)
    if smt_simplify:
        lemma_pfp_eval = z3.simplify(lemma_pfp_eval)
    return lemma_pfp_eval


def _eval_vars(formula, model_dict, annctx):
    # Construct transformation condition/operation pairs for variables of different sorts
    # Handling all sorts uniformly for now
    cond = (
        lambda expr: expr.decl().arity() == 0
        and uct.get_uct_sort(expr, annctx) is not None
    )
    op = lambda expr: finitemodel.recover_value(
        model_dict[finitemodel.model_key_repr(expr.decl())][()],
        uct.get_uct_sort(expr, annctx),
    )
    return nputils.transform_expression(formula, [(cond, op)])
