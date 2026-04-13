import z3

from collections import defaultdict

from unified_planning.plans import SequentialPlan
from unified_planning.plans import ActionInstance

from unified_planning.shortcuts import Parameter, FNode, Effect, EffectKind, Fraction

from ..planner.plan.smt_sequential_plan import SMTSequentialPlan
from .base import Encoder

class EncoderSequentialLifted(Encoder):
    """
    Roughly uses the encoding idea of max-parameters from:
    Espasa Arxer, J., Miguel, I. J., Villaret, M., & Coll, J. (2019). Towards
    Lifted Encodings for Numeric Planning in Essence Prime. Paper presented at
    The 18th workshop on Constraint Modelling and Reformulation, Stamford,
    Connecticut, United States.

    It uses quantifiers to allow a very compact representation of the encoding
    but that makes in general non-decidable.  The encoding seems to be correct
    although has not been thoroughly tested as it is awfully slow for now.
    """
    def __init__(self, task):
        self.name = "uf"
        self.task = task # The UP problem
        self.ctx = z3.Context() # The context where we will store the problem
        # TODO: Check the fluents that are static and do not add a timestep parameter to it

        self.z3_timestep_sort = z3.IntSort(ctx=self.ctx) # for now, it's just an int
        self.z3_timestep_var = None # the var that stores the last step

        # Z3 EnumSort used to represent problem objects. This will be the type
        # for most of the action and fluent parameters (except the timestep)
        self.z3_objects_sort = None 
        # map from UP objects to Z3 objects and map from Z3 objects to UP objects
        self.up_objects_to_z3 = dict()
        self.z3_objects_to_up = dict()
        # Z3 EnumSort used to represent actions
        self.z3_actions_sort = None 

        # a function action(timestep) -> action object 
        # s.t. given a timestep, tells us which action is being executed
        self.z3_action_variable = None
        # this is a mapping from the UP actions (up.action) to an action object, encoding the selected action
        # and the other way: i.e., from the z3 action objects to the up action
        self.z3_actions_mapping = dict()
        self.up_actions_mapping = dict()
        # list of parameters used by the selected action. We need max(cardinality(A)) parameters.
        # Each one of them is a function param_k(timestep) -> object
        # The typing functions (self.z3_typing_functions) will be used to constraint the types.
        self.z3_action_parameters = []

        # The encoding of the state
        self.z3_fluents = dict() # mapping from up.fluent.name to Z3_UF function

        # from up.type.name to Z3_UF function that gets an object 
        # and returns a Bool saying if the object belongs to the type
        self.z3_typing_functions = dict()

        # frame index, indexing what actions can modify which fluent
        # TODO: incorporate conditional effects in the frame?
        self.frame_idx = defaultdict(list)

        # populate the action - fluent mapping
        self._populate_modifiers()

        # Store the "raw" formula that we will later instantiate
        self.formula  = defaultdict(list)  
        
        # Store the length of the formula
        self.formula_length = 0

    def __len__(self):
        return self.formula_length

    def _populate_modifiers(self):
        """!
        Populates an index on which grounded actions can modify which fluents. 
        These are used afterwards for encoding the frame.
        """
        # Creates in self.frame_idx a mapping from fluent name to a list of
        # the actions that modify it and their mapping of parameters
        # Note that actions can be repeated there, as it can modify two
        # fluents of the same name easily.
        # The pairs of numbers indicate the mapping of the parameters between
        # the fluent and the action.
        #    {'located': [{'board': [(0, 0), (1, 2)]},
        #                 {'fly-fast': [(0, 0), (1, 1)]},
        #                 {'fly-fast': [(0, 0), (1, 2)]}],
        # ...
        #     'total-fuel-used': [{'fly-slow': []},
        #                         {'fly-fast': []}],
        def param_to_str(p):
            if isinstance(p, Parameter):
                return p.name
            elif isinstance(p, FNode): # A UP FNode ( can be anything really )
                if p.is_parameter_exp():
                    return p.parameter().name
                elif p.is_constant(): # A UP constant
                    return p.constant_value()
                else:
                    raise TypeError(f"Unsupported thing: {p} of type {type(p)}")

        for action in self.task.actions:
            # we convert the list of parameters of the action to str
            action_str_parameters = list(map(lambda x: param_to_str(x), action.parameters))

            # now for each effect, we get which fluent we're modifying
            for effect in action.effects:
                fluent_name = effect.fluent.fluent().name

                fluent_modification = dict() # init the data structure
                # we need to store the empty case to reflect modifications of 
                # fluents that have no parameters. i.e., that are one variable
                fluent_modification[action.name] = []
                for idx_fluent_arg, fluent_arg in enumerate(effect.fluent.args):
                    # convert current arg of the fluent to str
                    fluent_arg_str = param_to_str(fluent_arg)
                    # and we look at which position it correspond on the action
                    idx_action_arg = action_str_parameters.index(fluent_arg_str)
                    # and we store it
                    fluent_modification[action.name].append((idx_fluent_arg, idx_action_arg))
                # when we have gathered all the mappings for the current fluent's appearance
                # we store that into the global index.
                self.frame_idx[fluent_name].append(fluent_modification)
        
    def _setup_typing(self):
        """!
        Map the objects and types in the UP problem to Z3 clauses.
        """
        # We create the Z3 EnumSort, with all problem objects in it
        # Then, we maintain the mapping between UP and Z3 objects
        self.z3_objects_sort, z3_objects = z3.EnumSort("object",
            list(map(lambda x: x.name, self.task.all_objects)),
            ctx=self.ctx)
        self.up_objects_to_z3 = dict(zip(self.task.all_objects, z3_objects))
        self.z3_objects_to_up = dict(zip(z3_objects, self.task.all_objects))

        # Now we create a function is_X for each type in the problem
        # This function maps from all possible objects to either True or False
        for type in self.task.user_types:
            self.z3_typing_functions[type.name] = z3.Function(f"is_{type.name}",
                                    self.z3_objects_sort, z3.BoolSort(ctx=self.ctx))

        # Finally, we create functions to enforce typing.
        # The functions ix_X will only map to true when we pass an object of that type
        # i.e., in a problem with two cars and three planes, we state:
        # Forall X . x = car1 \/ x = car2 <-> is_car(x) == True
        # Forall X . x = plane1 \/ x = plane2 \/ x = plane3 <-> is_plane(x) == True
        x = z3.Const('obj', self.z3_objects_sort)
        for type in self.task.user_types:
            # get all z3 objects of the type "type"
            z3_objects_of_given_type = list(map(lambda x: self.up_objects_to_z3[x], self.task.objects(type)))
            self.formula['typing'].append(
                z3.ForAll([x],
                   z3.Or([y == x for y in z3_objects_of_given_type])
                     == (self.z3_typing_functions[type.name](x) == z3.BoolVal(True, ctx=self.ctx))))

    def _up_type_to_z3_type(self, type):
        """ Given a UP type, return the Z3 sort """
        if type.is_bool_type():
            return z3.BoolSort(ctx=self.ctx)
        elif type.is_user_type():
            # All user types are represented with the "object"
            # sort and filtered by the is_X() functions
            return self.z3_objects_sort
        elif type.is_real_type():
            return z3.RealSort(ctx=self.ctx)
        elif type.is_int_type():
            return z3.IntSort(ctx=self.ctx)
        else:
            raise Exception(f"UP type {type} still not supported")

    def _setup_actions(self):
        """!
        Create all the action execution infrastructure for Z3.
        We will have a UF named Exec, that gets a timestep and returns
        which action is being executed at that timestep.
        
        To store the parameters for the actions in each timestep, we will define
        too a set of param_x uninterpreted functions. We will have a number of
        those variables equal to the maximum number of parameters between all
        actions. These, similarly to the Exec function will given a timestep, is
        going to tell us to which object that parameter is being mapped to.
        """
        # Define the actions sort
        self.z3_actions_sort, z3_actions = z3.EnumSort("action",
                list(map(lambda x: x.name, self.task.actions)), ctx=self.ctx)
        # Now map the UP actions to the corresponding Z3 object
        self.z3_actions_mapping = dict(zip(self.task.actions, z3_actions))
        self.up_actions_mapping = dict(zip(z3_actions, self.task.actions))

        # Define the function that will tell us which action is being executed in a timestep
        self.z3_action_variable = z3.Function("Exec", self.z3_timestep_sort, self.z3_actions_sort)

        # find the max cardinality for all actions
        max_card = 0
        for action in self.task.actions:
            if len(action.parameters) > max_card:
                max_card = len(action.parameters)

        # for each timestep, create a function that given a timestep, returns us an object
        # these will be the parameters of each action executed in a timestep.
        for i in range(0, max_card):
            action_parameter = z3.Function(f"param_{i}", self.z3_timestep_sort, self.z3_objects_sort)
            self.z3_action_parameters.append(action_parameter)

    def _setup_state(self):
        """!
        Creates a UF representation for each planning fluent in UP
        """
        for fluent in self.task.fluents:
            parameters = []
            # first add all the fluent parameters
            for p in fluent.signature:
                parameters.append(self._up_type_to_z3_type(p.type))
            # now add the timestep (int) and then return type
            parameters.append(self.z3_timestep_sort)
            parameters.append(self._up_type_to_z3_type(fluent.type))
            self.z3_fluents[fluent.name] = z3.Function(fluent.name, parameters)

    def encode_execution_semantics(self):
        """!
        Encodes execution semantics as specified by modifier class.
        @return axioms that specify execution semantics.
        """
        #TODO: remove?
        return None

    def create_variables(self, t):
        """!
        Creates variables needed in the encoding.
        """
        # TODO: check, I don't think we need any more variables as all is now UF
        # Functions that have a timestep parameter
        pass

    def encode_initial_state(self):
        """!
        Encodes formula defining initial state
        @return initial: Z3 formula asserting initial state
        """
        t = 0
        initial = []
        for FNode, initial_value in self.task.initial_values.items():
            name = FNode.fluent().name # we translate the FNode to a Fluent and get its name
            parameters = [] # now we translate the parameters to Z3 objects
            for arg in FNode.args:
                parameters.append(self._expr_to_z3(arg, t))
            parameters.append(z3.IntVal(0, ctx=self.ctx)) # add the timestep

            # Can be done faster without the FNode -> Fluent conversion I guess,
            # but we're then digging into the innards of FNode and I don't like that ...
            #print(f"initial state: {name}({parameters}) = {self._expr_to_z3(initial_value, t)}")
            initial.append(self.z3_fluents[name](parameters) == self._expr_to_z3(initial_value, t))
        
        initial.append(self.z3_timestep_var >= 0)
        return initial

    def encode_goal_state(self):
        """!
        Encodes formula defining goal state
        @return goal: Z3 formula asserting propositional and numeric subgoals
        """
        goal = []
        for goal_pred in self.task.goals:
            goal.append(self._expr_to_z3(goal_pred, self.z3_timestep_var))
        return goal

    def _expr_to_z3(self, expr, t, ctx=None):
        """
        Traverses a tree expression in-order and converts it to a Z3 expression.
        expr: The tree expression node. (Can be a value, variable name, or operator)
        t: The timestep for the Fluents to be considered 
        ctx: A context manager, as we need to take into account parameters from actions, fluents, etc ...
        Returns A Z3 expression or Z3 value.
        """
        if isinstance(expr, int): # A python Integer
            return z3.IntVal(expr, ctx=self.ctx)
        elif isinstance(expr, bool): # A python Boolean
            return z3.BoolVal(expr, ctx=self.ctx)
        elif isinstance(expr, float): 
            return z3.RealVal(expr, ctx=self.ctx)

        elif isinstance(expr, Effect): # A UP Effect
            eff = None
            if expr.kind == EffectKind.ASSIGN:
                eff = self._expr_to_z3(expr.fluent, t + 1, ctx) == self._expr_to_z3(expr.value, t, ctx)
            if expr.kind == EffectKind.DECREASE:
                eff = self._expr_to_z3(expr.fluent, t + 1, ctx) == self._expr_to_z3(expr.fluent, t, ctx) - self._expr_to_z3(expr.value, t, ctx)
            if expr.kind == EffectKind.INCREASE:
                eff = self._expr_to_z3(expr.fluent, t + 1, ctx) == self._expr_to_z3(expr.fluent, t, ctx) + self._expr_to_z3(expr.value, t, ctx)
            if expr.is_conditional():
                return z3.Implies(self._expr_to_z3(expr.condition, t, ctx) , eff)
            else:
                return eff

        # TODO: Many operations are missing, but are trivial to add once needed
        elif isinstance(expr, FNode): # A UP FNode ( can be anything really )
            if expr.is_object_exp(): # A UP object
                return self.up_objects_to_z3[expr.object()]
            elif expr.is_constant(): # A UP constant
                return expr.constant_value()
            elif expr.is_or():  # A UP or
                return z3.Or([self._expr_to_z3(x, t, ctx) for x in expr.args])
            elif expr.is_and():  # A UP and
                return z3.And([self._expr_to_z3(x, t, ctx) for x in expr.args])
            elif expr.is_fluent_exp(): # A UP fluent
                f = expr.fluent() # the fluent
                p = [self._expr_to_z3(x, t, ctx) for x in expr.args] # its parameters translated
                p.append(t) # finally the timestep, as we know its a fluent
                return self.z3_fluents[f.name](p) # return the application
            elif expr.is_parameter_exp():
                p = expr.parameter()
                return ctx[p] # recover the param depending on the expression we are in
            elif expr.is_lt():
                return self._expr_to_z3(expr.args[0], t, ctx) < self._expr_to_z3(expr.args[1], t, ctx)
            elif expr.is_le():
                return self._expr_to_z3(expr.args[0], t, ctx) <= self._expr_to_z3(expr.args[1], t, ctx)
            elif expr.is_times():
                return self._expr_to_z3(expr.args[0], t, ctx) * self._expr_to_z3(expr.args[1], t, ctx)
            elif expr.is_div():
                return self._expr_to_z3(expr.args[0], t, ctx) / self._expr_to_z3(expr.args[1], t, ctx)
            elif expr.is_plus():
                return z3.Sum([self._expr_to_z3(x, t, ctx) for x in expr.args])
            elif expr.is_minus():
                return self._expr_to_z3(expr.args[0], t, ctx) - self._expr_to_z3(expr.args[1], t, ctx)
            elif expr.is_not():
                return z3.Not(self._expr_to_z3(expr.args[0], t, ctx))
            elif expr.is_equals():
                return self._expr_to_z3(expr.args[0], t, ctx) == self._expr_to_z3(expr.args[1], t, ctx)
            elif expr.is_implies():
                return z3.Implies(self._expr_to_z3(expr.args[0], t, ctx), self._expr_to_z3(expr.args[1], t, ctx))
            else:
                raise TypeError(f"Unsupported expression: {expr} of type {type(expr)}")
        elif isinstance(expr, Fraction):
            return z3.RealVal(f"{expr.numerator}/{expr.denominator}", ctx=self.ctx)
        else:
            raise TypeError(f"Unsupported expression: {expr} of type {type(expr)}")


    def encode_actions(self):
        """!
        Encodes the Actions
        @return actions: list of Z3 formulas asserting the actions

        The main idea is to do:
        forall t - int
            (t >= 0 /\ t < t_goal /\ 
            is_plane(param_1(t)) /\ is_city(param_2(t)) /\ is_city(param_3(t)) /\ # typing
            exec(t) = fly) -> 
                    (    ... # precondition ... # effect)
        """
        t = z3.Const('t', self.z3_timestep_sort)
        actions = []
        for up_action in self.task.actions:
            ctx = {} # context for the translation
            # we add the mappings of the action parameter to the variable that selects the value
            for i in range(0, len(up_action.parameters)):
                action_parameter = self.z3_action_parameters[i]
                ctx[up_action.parameters[i]] = action_parameter(t) 

            # constraint that says the action executed is up_action
            action_matches = self.z3_action_variable(t) == self.z3_actions_mapping[up_action]

            # constraint that ensures the parameter types match
            action_typing = []
            for i in range(0, len(up_action.parameters)):
                type_str = up_action.parameters[i].type.name
                typing_function = self.z3_typing_functions[type_str]
                action_parameter = self.z3_action_parameters[i]
                action_typing.append(typing_function(action_parameter(t))) # is_plane(param1(t))
            action_typing = z3.And(action_typing)
            
            # translate the action precondition
            action_pre = []
            for pre in up_action.preconditions:
                action_pre.append(self._expr_to_z3(pre, t, ctx=ctx))
            action_pre = z3.And(action_pre)
               
            # translate the action effect
            action_eff = []
            for eff in up_action.effects:
               action_eff.append(self._expr_to_z3(eff, t, ctx=ctx))
            action_eff = z3.And(action_eff)

            # for an action to be executable, it needs to be selected
            # and the timestep needs to be in the correct bound
            action = z3.Implies(z3.And([t >= 0, t < self.z3_timestep_var, action_typing, action_matches]), 
                                z3.And([action_pre, action_eff]))
            actions.append(action)

        # now we bound the action executions inside the considered timesteps
        return z3.ForAll([t], z3.And(actions))

    def encode_frame(self):
        """!
        Encode explanatory frame axioms: a predicate retains its value unless
        it is modified by the effects of an action.

        f(x,y,z, t) != f(x,y,z, t+1) -> 
            (exec(t) = action1 /\ param2(t) = z) \/
            (exec(t) = action3 /\ param1(t) = x /\ param3(t) = y) \/ ...

        The empty disjunction on the RHS will evaluate to false if there are no
        actions that can change the value of f
        @returns: list of frame axioms
        """
        # for each possible action parameter, instantiate an fresh var
        # to then use to quantify all frame axioms
        frame_variables = []
        for i in range(0,len(self.z3_action_parameters)):
            frame_variables.append(z3.Const(f'frame_{i}', self.z3_objects_sort))

        # a timestep variable
        t = z3.Const(f"frame_t", self.z3_timestep_sort)

        frame = []
        for fluent_name, modifications_dict in self.frame_idx.items():
            z3_fluent = self.z3_fluents[fluent_name]
            fluent_arity = z3_fluent.arity()
            fluent_vars = frame_variables[:fluent_arity - 1] # we substract timestep
            
            # f(x,y,z, t) != f(x,y,z, t+1)
            difference = z3_fluent(fluent_vars + [t]) != z3_fluent(fluent_vars + [t + 1])
            or_actions = []
            for pairings in modifications_dict:
                for action_name, parameter_pairings in pairings.items():
                    # retrieve the z3 action object
                    z3_action_object = self.z3_actions_mapping[self.task.action(action_name)]

                    # action executed at step t is the correct one
                    and_pairing = [ self.z3_action_variable(t) == z3_action_object ]
                    for pairing in parameter_pairings:
                        idx_fluent = pairing[0]
                        idx_action = pairing[1]
                        name_parameter_type = self.task.action(action_name).parameters[idx_action].type.name
                        typing_function = self.z3_typing_functions[name_parameter_type]
                        #print(f"fluent {idx_fluent}, action {idx_action}")
                        # TODO: check if it is useful, as I think its not needed
                        # we enforce the typing of the parameter
                        and_pairing.append(typing_function(self.z3_action_parameters[idx_action](t)))
                        and_pairing.append(fluent_vars[idx_fluent] == self.z3_action_parameters[idx_action](t))

                    # and the pairings between parameters match
                    or_actions.append(z3.And(and_pairing))

            # difference -> big list of ors
            bounded_time = t < self.z3_timestep_var # we are inside time bounds
            frame.append(
                z3.ForAll(frame_variables + [t],
                z3.Implies(z3.And(bounded_time, difference), z3.Or(or_actions))
                ))
        return frame

    def extract_plan(self, model, horizon):
        """!
        Extracts plan from model of the formula.
        Plan returned is linearized.

        @param model: Z3 model of the planning formula.
        @param encoder: encoder object, contains maps variable/variable names.

        @returns: dictionary containing plan. Keys are steps, values are actions.
        """
        plan = SequentialPlan([])
        if not model: return SMTSequentialPlan(plan, self.task)
        ## linearize partial-order plan
        for step in range(0, horizon):
            # which action is in step "step?"
            action_selected = model.evaluate(self.z3_action_variable(step))
            up_action = self.up_actions_mapping[action_selected]

            action_parameters = []
            for i in range(0, len(up_action.parameters)):
                z3_object = model.evaluate(self.z3_action_parameters[i](step))
                up_object = self.z3_objects_to_up[z3_object]
                action_parameters.append(up_object)

            action_inst = ActionInstance(up_action, action_parameters)
            plan.actions.append(action_inst)
            
        return SMTSequentialPlan(plan, self.task)


    def encode(self, t:int):
        """!
        Builds and returns the formulas for a single transition step (from t to t+1).
        @param t: timestep where the goal is true
        @returns: A dict with the different parts of the formula encoded
        """
        # now let's init the Z3 context:
        self._setup_typing()
        self._setup_actions()
        self._setup_state()

        self.z3_timestep_var = z3.Int("t_goal", ctx=self.ctx) # the var that stores the last step

        encoded_formula = dict()
        encoded_formula['initial'] = z3.And(self.encode_initial_state())
        encoded_formula['goal'] = z3.And(self.encode_goal_state()) 
        encoded_formula['actions'] = z3.And(self.encode_actions())
        encoded_formula['frame'] = z3.And(self.encode_frame())
        encoded_formula['typing'] = z3.And(self.formula['typing'])
        self.formula = encoded_formula
        self.formula_length += 1
        return encoded_formula
