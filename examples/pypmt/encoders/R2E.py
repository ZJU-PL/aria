from copy import deepcopy
from collections import defaultdict, namedtuple

import z3

from unified_planning.shortcuts import EffectKind, FNode, Fraction
from unified_planning.plans import SequentialPlan
from unified_planning.plans import ActionInstance

from .basic import EncoderGrounded
from .utilities import str_repr
from ..planner.plan.smt_sequential_plan import SMTSequentialPlan

class EncoderRelaxed2Exists(EncoderGrounded):
    """
    This is a simple implementation of the R2E encoding from: 

    Bofill, M.; Espasa, J.; and Villaret, M. Relaxed Exists-Step Plans in Planning as SMT. 
    In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence,
    IJCAI 2017, Melbourne, Australia
    """

    def __init__(self, task):
        super().__init__("r2e", task, None, False)

        # The total order between actions needed in the encoding
        self.sorted_actions = self.get_action_order()

        # to create variables, you tipically iterate over the problem
        # representation in UP and create the corresponding variables to the
        # self.up_fluent_to_z3 dict. Chain variables also need to be
        # instantiated each step, so they also will end up in
        # self.up_fluent_to_z3.

        # ChainVar is a namedtuple, and chain_variables_ids is just a list of
        # names to be iterated over when calling the usual create_variables 
        self.ChainVar = namedtuple("ChainVar", ["name", "type"])
        self.chain_variables_ids = list() # just a collection of namedtuples (name, type)

        # This is a big lookup table, that will help to encode the problem's
        # actions.  each problem variable (row) has a list of length n where n
        # is the number of actions in the problem each column has the variable
        # that the action needs to use when encoding. Given action in position k
        # in the total order, The precondition and conditions of conditional
        # effects must use column k-1 while effects must use column k
        self.chain_lookup = defaultdict(list) 

    def get_action_order(self):
        """!
          This method is used to derive an order between actions for the encoding 
          @returns: a sorted list of grounded actions sorted
        """
        # TODO: do something meaningful in terms of the order here
        # Maybe use causal graphs?
        return sorted([action for action in self.task.actions], key=lambda x: x.name)

    def __iter__(self):
        return iter(self.sorted_actions)

    def create_variables(self, t):
        """! 
        Create the Z3 variables for step t 
        @param t: the step we want to encode
        @returns: None
        """
        def new_var(name, type):
            """! Given a name and a UP type, return an equivalent Z3 var """
            if type.is_bool_type():
                return z3.Bool(name, ctx=self.ctx)
            elif type.is_real_type():
                return z3.Real(name, ctx=self.ctx)
            elif type.is_int_type():
                return z3.Int(name, ctx=self.ctx)
            else:
                raise TypeError(f"UP type {type} still not supported")

        # for each chainvar, now create it
        for chain_var in self.chain_variables_ids:
            # - create a new chain variable and put it into the usual structure
            new_chain_var = new_var(self.chain_str_repr(chain_var.name, t), chain_var.type)
            self.up_fluent_to_z3[chain_var.name].append(new_chain_var)

        return super().create_variables(t)

    def get_substitutions(self, up_vars, idx, phase):
        """!
        Given a set of UP fluents we will need to substitute those for different vars.

        for each var:       
        If it is part of a precondition,
          or condition of conditional effect 
          or the rhs of an effect (phase=="pre"):
        we have to substitute it for the value in the previous layer
        
        If it is the lhs of an effect (phase=="eff"):
        we have to substitute it 

        @param up_vars: the UP vars for which we need to find the subsitutions
        @param idx: the idx of the action we're in
        @param phase: "pre/eff"
        """
        assert(phase == "pre" or phase == "eff")
        substitutions = []
        # for each var in up_var
        for var in up_vars:
            if phase == "pre":
                old_var = self.up_fluent_to_z3[str_repr(var)][0]
                new_var_id = self.chain_lookup[str_repr(var)][idx - 1]
                new_var = self.up_fluent_to_z3[new_var_id][0]
            else: # eff
                old_var = self.up_fluent_to_z3[str_repr(var)][1]
                new_var_id = self.chain_lookup[str_repr(var)][idx]

                if new_var_id in [x.name for x in self.chain_variables_ids]: # if the new var is a chain var, we need the current step
                    new_var = self.up_fluent_to_z3[new_var_id][0]
                else: # otherwise get the next step var directly
                    new_var = self.up_fluent_to_z3[new_var_id][1]
            if not z3.eq(old_var, new_var): # if they are not the same variable
                substitutions.append((old_var, new_var)) # add the subsitution to the list
        return substitutions

    def chain_substitutions(self, z3_expr, up_expr, idx, phase):
        """!
        Applies all the needed substitutions for a given idx in the lookup table

        z3_expr: the Z3 expr where to apply the substitution to
        up_vars: the UP vars for which we need to find the subsitutions
        idx: the idx of the action we're in
        phase: "pre/eff"
        """
        # is there a better way to get this than poking at the first one?
        vars_extractor = up_expr.environment.free_vars_extractor
        substitutions = self.get_substitutions(vars_extractor.get(up_expr), idx, phase)
        #print(f"for ({up_expr}) in {phase}: vars {vars_extractor.get(up_expr)} and sub {substitutions}")
        if len(substitutions) > 0:
            return z3.substitute(z3_expr, substitutions) # apply all the substitutions, thanks z3 :)
        else: # we don't have any substitution to do, so its the identity function
            return z3_expr 

    def initialise_chains(self):
        """! 
        Decide which auxiliary chain variables are needed and initialise
        the chain lookup table to easily encode actions.
        """

        # the precondition of the first action to be encoded should always apply
        # the identity substitution. That is, be encoded as normal. Therefore we initialise
        # the first column with the normal (same) variable
        grounded_fluents_str =[str_repr(f) for f in self.all_fluents]
        for fluent_str in grounded_fluents_str:
            self.chain_lookup[fluent_str] = [None] * (len(self.sorted_actions) + 1) # initialise table with Nones
            self.chain_lookup[fluent_str][0] = fluent_str # layer 0 should be the normal var

        # now, for each action (layer of the table), we update the variables it modifies
        for idx, grounded_action in enumerate(self.sorted_actions, start=1):
            # first at all we initialise all the column to the previous one
            for fluent in grounded_fluents_str:
                self.chain_lookup[fluent][idx] = self.chain_lookup[fluent][idx - 1]

            # each effect will modify a variable. For each one we need to:
            for eff in grounded_action.effects:
                var_modified  = str_repr(eff.fluent) # get str representation of the modified variable
                new_var_name = var_modified + f"_chain{idx}"
                # - append it to the variables to create
                # HACK: we need to guard this as some formulations will have actions
                # that modify the same variable twice ...
                if new_var_name not in [x.name for x in self.chain_variables_ids]:
                    self.chain_variables_ids.append(self.ChainVar(new_var_name, eff.fluent.type))
                # - update the lookup table
                self.chain_lookup[var_modified][idx] = new_var_name

    def chain_str_repr(self, name, t):
        """! 
        Given an ID (str name) of an abstract chain variable, return the name of
        the Z3 variable that corresponds to the step t

        @param name: name (id) of the chain variable
        @param t: timestep

        @returns: the corresponding string
        """
        return name + f"_{t}"

    def encode_actions(self):
        """!
        Encodes the transition function. That is, the actions with the
        chains already applied. 

        @returns: list of Z3 formulas asserting the actions
        """
        def translate_rhs(eff):
            # the problem is that we need to do this transformation in the expressions before 
            # we translate the expression, as eff.fluent has to be incorporated there

            # for example, an increase x, y lhs=x rhs=y, but when translating
            # that to z3 we get x' = x + y and x is not part of the variables
            # and therefore not substituted 
            if eff.kind == EffectKind.ASSIGN:
                return self._expr_to_z3(eff.value, 0)
            elif eff.kind == EffectKind.DECREASE:
                return self._expr_to_z3(eff.fluent, 0) - self._expr_to_z3(eff.value, 0)
            elif eff.kind == EffectKind.INCREASE:
                return self._expr_to_z3(eff.fluent, 0) + self._expr_to_z3(eff.value, 0)
            else:
                raise ValueError("effect type unknown")


        actions = []
        for idx, grounded_action in enumerate(self.sorted_actions, start=1):
            key = str_repr(grounded_action)
            action_var = self.up_actions_to_z3[key][0]

            # translate the action precondition
            action_pre = [ z3.BoolVal(True, ctx=self.ctx) ]
            for pre in grounded_action.preconditions:
                translated_pre = self._expr_to_z3(pre, 0) # naively translate pre
                # do the usual a -> pre
                action_pre.append(
                    z3.Implies(action_var,
                               self.chain_substitutions(
                                   translated_pre, pre, idx, "pre"), ctx=self.ctx))

            # translate the action effect. We have to do the following:
            # 1 - generate a formula for the effect that gets subsituted on the
            # prev variables on the chain
            # 2 - generate a formula for the non-execution to say it carries on the value
            action_eff = []
            for eff in grounded_action.effects:
                # we separately extract lhs and rhs of the effect, as we have to treat
                # them separately with regards to the substitutions
                lhs = self._expr_to_z3(eff.fluent, 1)
                rhs = translate_rhs(eff)
                final_lhs = self.chain_substitutions(lhs, eff.fluent, idx, "eff")
                final_rhs = self.chain_substitutions(rhs, eff.value, idx, "pre")
                final_rhs = self.chain_substitutions(final_rhs, eff.fluent, idx, "pre") # ADD
                tr_eff = final_lhs == final_rhs
                
                if eff.is_conditional():
                    # we do the usual of cond -> lhs = rhs
                    tr_eff = z3.Implies(
                        self.chain_substitutions(
                            self._expr_to_z3(eff.condition, 0),
                            eff.condition, idx, "pre"),
                        tr_eff, ctx=self.ctx)

                # now we encode the chain. the effect in positive (i.e., if the action happens)
                # 1 - a -> effect
                action_eff.append(z3.Implies(action_var, tr_eff, ctx=self.ctx))
                # the effect in negative (i.e., if the action does not happen)
                # 2 - not(a) -> things stay equal
                # note that we know that eff.fluent is just a var, so we can (hopefully) get away with this
                prev_lhs = self.chain_substitutions(self._expr_to_z3(eff.fluent, 0), eff.fluent, idx, "pre")
                action_eff.append(z3.Implies(z3.Not(action_var), final_lhs == prev_lhs, ctx=self.ctx))

            # the proper encoding
            # MA342: I'm adding this condition since this would throw a context mismatch if the preconditions 
            # or effects are empty. If this happens then we know where to look.
            assert len(action_pre) > 0 and len(action_eff) > 0, f"Empty action {grounded_action}"
            actions.append(z3.And(action_pre))
            actions.append(z3.And(action_eff))
        return z3.And(actions)

    def extract_plan(self, model, horizon):
        plan = SequentialPlan([])
        if not model: return SMTSequentialPlan(plan, self.task)
        ## linearize partial-order plan
        for t in range(0, horizon + 1):
            for action in self:
                if z3.is_true(model[self.up_actions_to_z3[action.name][t]]):
                    plan.actions.append(ActionInstance(action))

        return SMTSequentialPlan(plan, self.task)

    def encode(self, t):
        """!
        Builds and returns the formulas for a single transition step (from t to t+1).
        @param t: timestep where the goal is true
        @returns: A dict with the different parts of the formula encoded
        """

        # If we encode the first step, we return the "abstract" formula,
        # as this has already been encoded
        if t == 0:
            self.base_encode()
            return deepcopy(self.formula)

        # Otherwise we do the proper substitutions
        self.create_variables(t+1)  # base_encode added vars for step 0, so we only need to add t+1

        list_substitutions = [] # list of pairs (from,to)
        for key in self.up_actions_to_z3.keys():
            list_substitutions.append(
                (self.up_actions_to_z3[key][0],
                 self.up_actions_to_z3[key][t]))
        for key in self.up_fluent_to_z3.keys():
            list_substitutions.append(
                (self.up_fluent_to_z3[key][0],
                 self.up_fluent_to_z3[key][t]))
            list_substitutions.append(
                (self.up_fluent_to_z3[key][1],
                 self.up_fluent_to_z3[key][t + 1]))

        encoded_formula = dict()
        encoded_formula['initial']    = self.formula['initial']
        encoded_formula['actions']    = z3.substitute(self.formula['actions'], list_substitutions)
        encoded_formula['goal']       = z3.substitute(self.formula['goal'], list_substitutions)
        encoded_formula['chainlinks'] = z3.substitute(self.formula['chainlinks'], list_substitutions)

        return encoded_formula
    
    def base_encode(self):
        """!
        Builds the base R2E encoding.
        Populates the formula dictionary, where all the "raw" formulas are stored
        @return None
        """
        self.initialise_chains() # populate the big lookup table
        self.create_variables(0) # Create variables for the initial state
        self.create_variables(1)


        self.formula['initial']      = z3.And(self.encode_initial_state()) # Encode initial state axioms
        self.formula['actions']      = z3.And(self.encode_actions())       # Encode universal axioms
        self.formula['goal']         = z3.And(self.encode_goal_state(0))   # Encode goal state axioms
        self.formula['chainlinks']   = z3.And(self.encode_chain_links())   # Encode variables linking between time steps.

    def encode_chain_links(self):
        """!
        Encode the chain links between steps. Basically for each original state variable, get
        the value stored in the chain and make the x^t of the next step get its value

        if variable x gets modified twice we will have the following variables:
        x^t, xc1^t, xc2^t 
        we have to say that xc2^t = x^t+1

        @returns: a list of Z3 clauses
        """
        chain_links = []
        # for each original grounded var
        grounded_fluents_str = [str_repr(f) for f, _ in self.task.initial_values.items()]
        for grounded_fluent_str in grounded_fluents_str:
            # get where the last chain variable id that holds the value of grounded_fluent_str
            last_var_str = self.chain_lookup[grounded_fluent_str][-1] 
            # now get its real Z3 variable
            last_var_0  = self.up_fluent_to_z3[last_var_str][0]
            first_var_1 = self.up_fluent_to_z3[grounded_fluent_str][1]
            chain_links.append(last_var_0 == first_var_1)
        return chain_links
       
    def _expr_to_z3(self, expr, t, c=None):
        """!
        Traverses a UP AST in in-order and converts it to a Z3 expression.
        @param expr: The tree expression node. (Can be a value, variable name, or operator)
        @param t: The timestep for the Fluents to be considered 
        @param c: The context, which can be used to take into account free params
        @returns: An equivalent Z3 expression
        """
        if isinstance(expr, int): # A python Integer
            return z3.IntVal(expr, ctx=self.ctx)
        elif isinstance(expr, bool): # A python Boolean
            return z3.BoolVal(expr, ctx=self.ctx)
        elif isinstance(expr, float): 
            return z3.RealVal(expr, ctx=self.ctx)

        elif isinstance(expr, FNode): # A UP FNode ( can be anything really )
            if expr.is_object_exp(): # A UP object
                raise ValueError(f"{expr} should not be evaluated")
            elif expr.is_constant(): # A UP constant
                return self._expr_to_z3(expr.constant_value(), t, c)
            elif expr.is_or():  # A UP or
                return z3.Or([self._expr_to_z3(x, t, c) for x in expr.args])
            elif expr.is_and():  # A UP and
                return z3.And([self._expr_to_z3(x, t, c) for x in expr.args])
            elif expr.is_fluent_exp(): # A UP fluent
                return self.up_fluent_to_z3[str_repr(expr)][t]
            elif expr.is_parameter_exp():
                raise ValueError(f"{expr} should not be evaluated")
            elif expr.is_lt():
                return self._expr_to_z3(expr.args[0], t, c) < self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_le():
                return self._expr_to_z3(expr.args[0], t, c) <= self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_equals():
                return self._expr_to_z3(expr.args[0], t, c) == self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_times():
                return self._expr_to_z3(expr.args[0], t, c) * self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_div():
                return self._expr_to_z3(expr.args[0], t, c) / self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_plus():
                return z3.Sum([self._expr_to_z3(x, t, c) for x in expr.args])
            elif expr.is_minus():
                return self._expr_to_z3(expr.args[0], t, c) - self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_not():
                return z3.Not(self._expr_to_z3(expr.args[0], t, c))
            elif expr.is_equals():
                return self._expr_to_z3(expr.args[0], t, c) == self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_implies():
                return z3.Implies(self._expr_to_z3(expr.args[0], t, c), self._expr_to_z3(expr.args[1], t, c))
            else:
                raise TypeError(f"Unsupported expression: {expr} of type {type(expr)}")
        elif isinstance(expr, Fraction):
            return z3.RealVal(f"{expr.numerator}/{expr.denominator}", ctx=self.ctx)
        else:
            raise TypeError(f"Unsupported expression: {expr} of type {type(expr)}")
