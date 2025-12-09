# This module is the main module of the naturalproofs package. It defines a natural proofs solver and various
# configuration options, as well as the structure that the solver returns.
import warnings

import z3

from aria.quant.fossil.naturalproofs.AnnotatedContext import default_annctx
from aria.quant.fossil.naturalproofs.decl_api import get_recursive_definition, get_all_axioms, is_expr_fg_sort
from aria.quant.fossil.naturalproofs.utils import Implies_as_FuncDeclRef
import aria.quant.fossil.naturalproofs.proveroptions as proveroptions
from aria.quant.fossil.naturalproofs.prover_utils import make_recdef_unfoldings, get_foreground_terms, instantiate, get_recdef_applications


class NPSolution:
    """
    Class for representing solutions discovered by NPSolver instances.
    Such a representation is necessary because logging information can be attached to the NPSolution object, along with
    default outputs like a satisfying model.
    """
    def __init__(self, if_sat=None, model=None, extraction_terms=None, instantiation_terms=None, depth=None, options=None):
        """
        Explanation of attributes:
        - if_sat (bool): if there exists a satisfying model under the given configuration.
        - model (z3.ModelRef or None): satisfying model if exists.
        - extraction_terms (set of z3.ExprRef): logging attribute. Set of foreground terms in the formula
        given to the smt solver. A finite model can be extracted over these terms that preserves the failure of
        the proof attempt. Usually this is the set of all foreground terms in the quantifier-free formula given
        to the solver at the end of all instantiations.
        - instantiation_terms: the set of terms that were used for instantiating all axioms and recursive
        definitions. Only applicable when the instantiation is uniform for all axioms. Not applicable when
        instantiation mode is depth_one_stratified_instantiation or lean_instantiation.
        - depth (int): depth at which the solution object was created. Applicable when instantiation mode is
        bounded depth.
        - options (proveroptions.Options): logging attribute. Options used to configure the solver.
        """
        self.if_sat = if_sat
        self.model = model
        self.extraction_terms = extraction_terms
        self.instantiation_terms = instantiation_terms
        self.depth = depth
        self.options = options


class NPSolver:
    """
    Class for creating Natural Proofs solvers.
    Can be configured using the 'options' attribute, an instance of the naturalproofs.Options class.
    """
    def __init__(self, annctx=default_annctx):
        """
        Each solver instance must be created with an AnnotatedContext that stores the vocabulary, recursive definitions,
         and axioms --- essentially defining a theory for the solver.
        :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
        """
        self.annctx = annctx
        self.options = proveroptions.Options()

    def solve(self, goal, lemmas=None):
        """
        Primary function of the NPSolver class. Attempts to prove the goal with respect to given lemmas and the theory
        defined by the AnnotatedContext in self.annctx.
        :param goal: z3.BoolRef
        :param lemmas: set of z3.BoolRef
        :return: NPSolution
        """
        z3.set_param('smt.random_seed', 0)
        z3.set_param('sat.random_seed', 0)
        # TODO: check that the given lemmas are legitimate bound formula instances with their formal parameters from
        #  the foreground sort.
        options = self.options
        # Make recursive definition unfoldings
        recdefs = get_recursive_definition(None, alldefs=True, annctx=self.annctx)
        recdef_unfoldings = make_recdef_unfoldings(recdefs)
        # Sometimes we don't need the unfoldings indexed by recdefs
        untagged_unfoldings = set(recdef_unfoldings.values())
        # Add them to the set of axioms and lemmas to instantiate
        axioms = get_all_axioms(self.annctx)
        if lemmas is None:
            lemmas = set()
        else:
            # Check that each bound parameter in all the lemmas are of the foreground sort
            for lemma in lemmas:
                bound_vars, lemma_body = lemma
                if not all(is_expr_fg_sort(bound_var, annctx=self.annctx) for bound_var in bound_vars):
                    raise TypeError('Bound variables of lemma: {} must be of the foreground sort'.format(lemma_body))
        recdef_indexed_lemmas = _sort_by_trigger(lemmas, list(recdef_unfoldings.keys()))

        if options.instantiation_mode == proveroptions.lean_instantiation_with_lemmas:
            # Recdefs and lemmas need to be treated separately using 'lean' instantiation
            fo_abstractions = axioms
        if options.instantiation_mode == proveroptions.lean_instantiation:
            # Recdefs need to be treated separately using 'lean' instantiation
            fo_abstractions = axioms | lemmas
        else:
            # If the instantiation isn't the 'lean' kind then all defs are going to be instantiated with all terms
            fo_abstractions = axioms | untagged_unfoldings | lemmas

        # All parameters have been set appropriately. Begin constructing instantiations
        # Negate the goal
        neg_goal = z3.Not(goal)
        # Create a solver object and add the goal negation to it
        z3solver = z3.Solver()
        z3solver.add(neg_goal)
        # Keep track of terms in the quantifier-free problem given to the solver
        initial_terms = get_foreground_terms(neg_goal, annctx=self.annctx)
        extraction_terms = initial_terms
        recdef_application_terms = get_recdef_applications(neg_goal, annctx=self.annctx)
        instantiation_terms = set()
        # Instantiate and check for provability according to options
        # Handle manual instantiation mode first
        if options.instantiation_mode == proveroptions.manual_instantiation:
            terms_to_instantiate = options.terms_to_instantiate
            instantiations = instantiate(fo_abstractions, terms_to_instantiate)
            if instantiations != set():
                instantiation_terms = terms_to_instantiate
                extraction_terms = extraction_terms.union(get_foreground_terms(instantiations, annctx=self.annctx))
            z3solver.add(instantiations)
            if_sat = _solver_check(z3solver)
            model = z3solver.model() if if_sat else None
            return NPSolution(if_sat=if_sat, model=model, extraction_terms=extraction_terms,
                              instantiation_terms=instantiation_terms, options=options)
        # Automatic instantiation modes
        # stratified instantiation strategy
        if options.instantiation_mode == proveroptions.depth_one_stratified_instantiation:
            conservative_fo_abstractions = axioms | untagged_unfoldings
            tracked_instantiations = instantiate(conservative_fo_abstractions, initial_terms)
            if tracked_instantiations != set():
                instantiation_terms = initial_terms
                tracked_terms = get_foreground_terms(tracked_instantiations, annctx=self.annctx)
                extraction_terms = extraction_terms.union(tracked_terms)
            z3solver.add(tracked_instantiations)
            untracked_instantiations = instantiate(lemmas, extraction_terms)
            if untracked_instantiations != set():
                instantiation_terms = instantiation_terms.union(extraction_terms)
                untracked_terms = get_foreground_terms(untracked_instantiations, annctx=self.annctx)
                extraction_terms = extraction_terms.union(untracked_terms)
            other_instantiations = instantiate(conservative_fo_abstractions, extraction_terms)
            z3solver.add(untracked_instantiations)
            z3solver.add(other_instantiations)
            if_sat = _solver_check(z3solver)
            model = z3solver.model() if if_sat else None
            return NPSolution(if_sat=if_sat, model=model, extraction_terms=extraction_terms,
                              instantiation_terms=instantiation_terms, options=options)
        # Set up initial values of variables
        depth_counter = 0
        # Keep track of formulae produced by instantiation
        instantiations = set()
        # When the instantiation mode is infinite we realistically can't exceed 10^3 instantiations anyway
        target_depth = 1000 if options.instantiation_mode == proveroptions.infinite_depth else options.depth
        while depth_counter < target_depth:
            # Try to prove with available instantiations
            z3solver.add(instantiations)
            # If the instantiation mode is fixed depth we can continue instantiating until we get to that depth
            if options.instantiation_mode != proveroptions.fixed_depth:
                # Otherwise check satisfiability with current state of instantiations
                if_sat = _solver_check(z3solver)
                # If unsat, stop and return NPSolution instance
                if not if_sat:
                    return NPSolution(if_sat=if_sat, model=None, extraction_terms=extraction_terms,
                                      instantiation_terms=instantiation_terms, depth=depth_counter, options=options)
            # target depth not reached or unsat not reached
            # Do another round of instantiations.
            # TODO: optimise instantiations so repeated instantiation is not done. Currently all instantiations
            #  are done in every round. But optimisation is difficult in the presence of multiple arities.
            instantiation_terms = extraction_terms
            # Instantiate all basic abstractions
            instantiations = instantiate(fo_abstractions, instantiation_terms)
            # Instantiate other abstractions depending on instantiation mode. Typically recdefs and lemmas
            if options.instantiation_mode in {proveroptions.lean_instantiation, proveroptions.lean_instantiation_with_lemmas}:
                # Add recursive definition instantiations to the set of all instantiations
                for recdef, application_terms in recdef_application_terms.items():
                    lean_instantiations = instantiate(recdef_unfoldings[recdef], application_terms)
                    instantiations.update(lean_instantiations)
            if options.instantiation_mode == proveroptions.lean_instantiation_with_lemmas:
                # Add lemma instantiations to the set of all instantiations
                for recdef, application_terms in recdef_application_terms.items():
                    triggered_lemmas = recdef_indexed_lemmas.get(recdef, [])
                    triggered_instantiations = instantiate(triggered_lemmas, application_terms)
                    instantiations.update(triggered_instantiations)
            # If the set of instantiations is empty exit the loop
            if instantiations == set():
                instantiation_terms = set()
                break
            # Update the variables for the next round
            depth_counter = depth_counter + 1
            new_terms = get_foreground_terms(instantiations, annctx=self.annctx)
            recdef_application_terms = get_recdef_applications(instantiations, annctx=self.annctx)
            extraction_terms = extraction_terms.union(new_terms)
        # Reach this case when depth_counter = target depth, either in fixed_depth or bounded_depth mode.
        # Final attempt at proving goal
        z3solver.add(instantiations)
        if_sat = _solver_check(z3solver)
        model = z3solver.model() if if_sat else None
        return NPSolution(if_sat=if_sat, model=model, extraction_terms=extraction_terms,
                          instantiation_terms=instantiation_terms, depth=depth_counter, options=options)


# Helper function to check the satisfiability and throw exception if solver returns unknown
def _solver_check(z3solver):
    z3solution = z3solver.check()
    if z3solution == z3.sat:
        return True
    elif z3solution == z3.unsat:
        return False
    elif z3solution == z3.unknown:
        raise ValueError('Solver returned unknown. Something is wrong. Exiting.')


# Helper function to index lemmas by the recursive definitions they are triggered by
# WARNING: very limited in functionality
def _sort_by_trigger(bound_exprs, trigger_decls):
    result = dict()
    result[None] = []
    for bound_expr in bound_exprs:
        params, expr = bound_expr
        # The expression has to be an implication
        if expr.decl() != Implies_as_FuncDeclRef:
            result[None].append(bound_expr)
        lhs, rhs = expr.children()
        # antecedent must be an expression made from one of the triggers (like recdefs)
        lhs_decl = lhs.decl()
        if lhs_decl not in trigger_decls:
            result[None].append(bound_expr)
        # Otherwise, store the expression under the appropriate trigger
        if lhs_decl not in result:
            result[lhs_decl] = []
        result[lhs_decl].append(bound_expr)
    return result
