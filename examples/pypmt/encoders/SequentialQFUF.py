import z3

from collections import defaultdict

from unified_planning.plans import SequentialPlan
from unified_planning.plans import ActionInstance

from unified_planning.shortcuts import Parameter, FNode, Effect, EffectKind, Fraction
from unified_planning.model.fluent import get_all_fluent_exp
from unified_planning.engines.results import CompilerResult

from .utilities import flattern_list
from ..planner.plan.smt_sequential_plan import SMTSequentialPlan
from .base import Encoder

class EncoderSequentialQFUF(Encoder):
    """
    Roughly uses the encoding idea of max-parameters from:
    Espasa Arxer, J., Miguel, I. J., Villaret, M., & Coll, J. (2019). Towards
    Lifted Encodings for Numeric Planning in Essence Prime. Paper presented at
    The 18th workshop on Constraint Modelling and Reformulation, Stamford,
    Connecticut, United States.

    It derives from the SequentialLifted but removes all quantifiers 
    """
    # TODO: Check the fluents that are static and do not add a timestep parameter to it
    def __init__(self, task):
        self.name = "qfuf"
        self.task = task # The UP problem
        assert len(self.task.all_objects) > 0, "problem fully grounded, please use a different encoder"
        self.ctx = z3.Context() # The context where we will store the problem

        # cache all fluents in the problem.
        _task    = self.task.problem if isinstance(self.task, CompilerResult) else self.task
        self.all_fluents = flattern_list([list(get_all_fluent_exp(_task, f)) for f in _task.fluents])
        self._initialize_fluents(_task, self.all_fluents)

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
    
    def _initialize_fluents(self, _task, _fluentslist):
        initialized_fluents = list(_task.explicit_initial_values.keys())
        unintialized_fluents = list(filter(lambda x: not x in initialized_fluents, _fluentslist))
        for fe in unintialized_fluents:
            if fe.type.is_bool_type():
                _task.set_initial_value(fe, False) # we need this for plan validator.
            elif fe.type.is_real_type():
                _task.set_initial_value(fe, 0) # we need this for plan validator.
            else:
                raise TypeError

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
        #
        # That is: {'located': [{'board': [(0, 0), (1, 2)] ....
        # means that the located fluent with two parameters, is modified by the
        # board action. The first parameter of located (0) corresponds to the first of board (0),
        # while the second (1) corresponds to the third parameter of board (2). If there is a 
        # string there such as "plane1" means that was a constant parameter in the action.
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
                    fluent_arg_str = param_to_str(fluent_arg)
                    # if we have an object (partially instantiated action or a PDDL constant)
                    # we can continue on the next iteration as this does not need a matching. 
                    if fluent_arg.is_object_exp():
                        fluent_modification[action.name].append((idx_fluent_arg, fluent_arg_str))
                    elif fluent_arg.is_parameter_exp():
                        # convert current arg of the fluent to str
                        # and we look at which position it correspond on the action
                        idx_action_arg = action_str_parameters.index(fluent_arg_str)
                        # and we store it
                        fluent_modification[action.name].append((idx_fluent_arg, idx_action_arg))
                # when we have gathered all the mappings for the current fluent's appearance
                # we store that into the global index.
                self.frame_idx[fluent_name].append(fluent_modification)
        
    def _setup_typing(self):
        """!
        map the objects and types in the UP problem to Z3 clauses
        """
        # We create the Z3 EnumSort, with all problem objects in it
        # Then, we maintain the mapping between UP and Z3 objects
        self.z3_objects_sort, z3_objects = z3.EnumSort(
            "object", list(map(lambda x: x.name, self.task.all_objects)), ctx=self.ctx)
        self.up_objects_to_z3 = dict(zip(self.task.all_objects, z3_objects))
        self.z3_objects_to_up = dict(zip(z3_objects, self.task.all_objects))

        # Now we create a function is_X for each type in the problem
        # This function maps from all possible objects to either True or False
        for type in self.task.user_types:
            self.z3_typing_functions[type.name] = z3.Function(f"is_{type.name}",
                      self.z3_objects_sort, z3.BoolSort(ctx=self.ctx))

        # Finally, we create functions to enforce typing.
        # The functions ix_X will only map to true when we pass an object of that type
        # i.e., in a problem with two cars and one plane, we state:
        # is_car(car1) == True /\ is_car(car2) == True /\ is_car(plane1) == False
        for type in self.task.user_types:
            # get all z3 objects of the type "type"
            z3_objects_of_given_type = list(map(lambda x: self.up_objects_to_z3[x], self.task.objects(type)))
            self.formula['typing'].append(z3.And([
                    self.z3_typing_functions[type.name](x) == z3.BoolVal(True, ctx=self.ctx) 
                    if x in z3_objects_of_given_type else
                    self.z3_typing_functions[type.name](x) == z3.BoolVal(False, ctx=self.ctx)
                    for x in z3_objects
                ]))

    def _up_type_to_z3_type(self, type):
        """!
        Given a UP type, return the corresponding Z3 sort
        """
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
        self.z3_action_variable = z3.Function("Exec",
                                self.z3_timestep_sort, self.z3_actions_sort)

        # find the max cardinality for all actions
        max_card = 0
        for action in self.task.actions:
            if len(action.parameters) > max_card:
                max_card = len(action.parameters)

        # for each timestep, create a function that given a timestep, returns us an object
        # these will be the parameters of each action executed in a timestep.
        for i in range(0, max_card):
            action_parameter = z3.Function(f"param_{i}",
                        self.z3_timestep_sort, self.z3_objects_sort)
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

    def encode_initial_state(self):
        """!
        Encodes formula defining initial state
        @return initial: Z3 formula asserting initial state
        """
        # set default values for uninitialized fluents
        initialized_fluents = list(self.task.explicit_initial_values.keys())
        unintialized_fluents = list(filter(lambda x: not x in initialized_fluents, self.all_fluents))
        for fe in unintialized_fluents:
            if fe.type.is_bool_type():
                self.task.set_initial_value(fe, False)
                self.task.set_initial_value(fe, False) # we need this for plan validator.
            elif fe.type.is_real_type():
                self.task.set_initial_value(fe, 0)
                self.task.set_initial_value(fe, 0) # we need this for plan validator.
            else:
                raise TypeError
            
        t = 0
        initial = []
        for FNode, initial_value in self.task.initial_values.items():
            name = FNode.fluent().name # we translate the FNode to a Fluent and get its name
            parameters = [] # now we translate the parameters to Z3 objects
            for arg in FNode.args:
                parameters.append(self._expr_to_z3(arg, t))
            parameters.append(z3.IntVal(0, ctx=self.ctx)) # add the timestep

            fluent = self.z3_fluents[name](parameters)
            value = self._expr_to_z3(initial_value, t)

            # Can be done faster without the FNode -> Fluent conversion I guess,
            # but we're then digging into the innards of FNode and I don't like that ...
            initial.append(fluent == value)
        
        return initial

    def encode_goal_state(self):
        """!
        Encodes formula defining goal state
        @return goal: Z3 formula asserting propositional and numeric subgoals
        """
        t =  self.z3_timestep_var
        goal = []
        for goal_pred in self.task.goals:
            goal.append(self._expr_to_z3(goal_pred, t + 1))
        return goal

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
        t = self.z3_timestep_var
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
            # and the types need to be correct
            if any(len(action.children()) == 0 for action in [action_typing, action_pre, action_eff]):
                action = z3.Implies(action_matches, z3.BoolVal(True, ctx=self.ctx))
            else:
                action = z3.Implies(z3.And([action_matches]), 
                                    z3.And([action_typing, action_pre, action_eff]))
            actions.append(action)
        return z3.And(actions)

    def encode_frame(self):
        """!
        f(x,y,z, t) != f(x,y,z, t+1) -> 
            (exec(t) = action1 /\ param2(t) = z) \/
            (exec(t) = action3 /\ param1(t) = x /\ param3(t) = y) \/ ...
        for each possible grounding of f()
        """
        # the timestep variable
        t = self.z3_timestep_var
        frame = [] # the whole frame

        # for each grounded fluent, we say its different from t to t + 1
        grounded_up_fluents = [f for f, _ in self.task.initial_values.items()]
        for grounded_fluent in grounded_up_fluents:
            fluent_name = grounded_fluent.fluent().name
            z3_fluent = self.z3_fluents[fluent_name]
            # convert the UP parameters of the grounded fluent to a list of Z3 objects
            fluent_vars = list(map(lambda x: self.up_objects_to_z3[x.constant_value()], grounded_fluent.args))
            difference = z3_fluent(fluent_vars + [t]) != z3_fluent(fluent_vars + [t + 1])

            # for each possible modification
            or_actions = []
            for pairings in self.frame_idx[fluent_name]:
                for action_name, parameter_pairings in pairings.items():
                # retrieve the z3 action object
                    z3_action_object = self.z3_actions_mapping[self.task.action(action_name)]
                    and_pairing = [ self.z3_action_variable(t) == z3_action_object ] # and its parameter pairing

                    add = True
                    # and matching of the action parameters to the fluent parameters
                    for pairing in parameter_pairings:
                        idx_fluent = pairing[0]
                        idx_action = pairing[1]
                        if isinstance(pairing[1],int): 
                            # if our "action" index is an integer, then that is a proper action parameter
                            and_pairing.append(
                                self.z3_action_parameters[idx_action](t) == fluent_vars[idx_fluent])
                        elif idx_action.type:
                            # if our "action" index has a type, one of the parameters of the fluent is a constant
                            if idx_action != grounded_fluent.args[idx_fluent].object(): #fluent_vars[idx_fluent]:
                                add = False
                        else:
                            raise TypeError("Don't know what this is ...")
                    if add:
                        or_actions.append(z3.And(and_pairing))
            #print(f"fluent {grounded_fluent} can be changed by: {or_actions}")
            # simplify the list in case its empty
            if len(or_actions) == 0:
                who_can_change_fluent = z3.BoolVal(False, ctx=self.ctx)
            else:
                who_can_change_fluent = z3.Or(or_actions)
            frame.append(z3.Implies(difference, who_can_change_fluent, ctx=self.ctx))
        return frame

    def extract_plan(self, model, horizon):
        """!
        Extracts plan from model of the formula.
        Plan returned is linearized.

        @param model: Z3 model of the planning formula.
        @param encoder: encoder object, contains maps variable/variable names.

        @return  plan: dictionary containing plan. Keys are steps, values are actions.
        """
        plan = SequentialPlan([])
        if not model: return SMTSequentialPlan(plan, self.task)
        ## linearize partial-order plan
        for step in range(0, horizon+1):
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

    def encode(self, t):
        """!
        Builds and returns the formulas for a single transition step (from t to t+1).
        @param t: timestep where the goal is true
        @return encoded_formula: A dict with the different parts of the formula encoded
        """
        if t == 0:
        # now let's init the Z3 context:
            self._setup_typing()
            self._setup_actions()
            self._setup_state()
            self.base_encode()
        
        list_substitutions = (self.z3_timestep_var, z3.IntVal(t, ctx=self.ctx)) # list of pairs (from,to)
        encoded_formula = dict()
        encoded_formula['initial'] = self.formula['initial']
        encoded_formula['goal']    = z3.substitute(self.formula['goal'], list_substitutions)
        encoded_formula['actions'] = z3.substitute(self.formula['actions'], list_substitutions)
        encoded_formula['frame']   = z3.substitute(self.formula['frame'], list_substitutions)
        encoded_formula['typing']  = self.formula['typing']
        self.formula_length += 1
        return encoded_formula

    def base_encode(self):
        """!
        Builds the encoding. Populates the formula dictionary, where all the "raw" formulas are stored
        @return None
        """
        self.z3_timestep_var    = z3.Int("t", ctx=self.ctx) # the var that stores the last step
        self.formula['initial'] = z3.And(self.encode_initial_state())  # Encode initial state axioms
        self.formula['goal']    = z3.And(self.encode_goal_state())  # Encode goal state axioms
        self.formula['actions'] = z3.And(self.encode_actions())  # Encode universal axioms
        self.formula['frame']   = z3.And(self.encode_frame())  # Encode explanatory frame axioms
        self.formula['typing']  = z3.And(self.formula['typing'])  # Encode explanatory frame axioms
        self.formula_length += 1 # increment the formula length.

    # TODO abstract this away in utilities.py
    def _expr_to_z3(self, expr, t, ctx=None):
        """!
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
