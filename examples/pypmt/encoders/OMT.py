from copy import deepcopy
from collections import defaultdict

from z3 import *

from unified_planning.shortcuts import *

from .basic import EncoderGrounded
from .utilities import str_repr
from unified_planning.model.fluent import get_all_fluent_exp

from ..modifiers.modifierLinear import LinearModifier
from ..modifiers.modifierParallel import ParallelModifier

import networkx as nx
import itertools

class EncoderOMT(EncoderGrounded):

    """
    Add references 
    """
    def __init__(self, name, task, modifier):
        super().__init__(name, task, modifier, parallel=False)
        self.allow_multiple_actions_per_step=False

        # Auxiliary variables
        # this is a mapping from the UP ground actions to z3 and back
        self.aux_z3_actions_to_up = dict() # multiple z3 vars point to one grounded fluent
        self.up_actions_to_aux_z3 = defaultdict(list)
        
        # mapping from up fluent to Z3 var
        self.up_fluent_to_aux_z3 = defaultdict(list) 

    def create_aux_variables(self, t):
        """! 
        Create the Z3 auxiliary variables for step t.
        @param t: the step we want to encode
        @returns: None
        """

        # for actions
        for grounded_action in self.task.actions:
            key   = str_repr(grounded_action)
            keyt  = "abs_" + key
            act_var = z3.Bool(keyt, ctx=self.ctx)
            self.up_actions_to_aux_z3[key].append(act_var)
            self.aux_z3_actions_to_up[act_var] = key

        # for fluents
        for fe in self.all_fluents:
            key  = str_repr(fe)
            keyt = "mod_" + key
            if fe.type.is_real_type() or fe.type.is_bool_type():
                self.up_fluent_to_aux_z3[key].append(z3.Bool(keyt, ctx=self.ctx))
            else:
                raise TypeError

    def encode(self,t):

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
    
        # Start encoding basic building block using rename variables from substitution  
        encoded_formula = dict()
        encoded_formula['initial']    = self.formula['initial']
        encoded_formula['actions']    = substitute(self.formula['actions'], list_substitutions)
        encoded_formula['goal']    = z3.substitute(self.formula['goal'], list_substitutions)

        encoded_formula['frame'] = substitute(self.formula['frame'], list_substitutions)
        encoded_formula['sem'] = substitute(self.formula['sem'], list_substitutions)

        encoded_formula['abstract_actions']    = substitute(self.formula['abstract_actions'], list_substitutions)
        encoded_formula['abstract_goal']       = substitute(self.formula['abstract_goal'], list_substitutions)
        encoded_formula['loop_formulas']       = substitute(self.formula['loop_formulas'], list_substitutions)

        # TODO: this will probably have be adapted, depending on whether a metric is provided in the PDDL or not.
        encoded_formula['objective']       = self.formula['objective']


        return encoded_formula

    def base_encode(self):
        """!
        Builds based OMT encoding.
        Populates the formula dictionary, where all the "raw" formulas are stored
        @return None
        """
        self.create_variables(0) # Create variables for the initial state
        self.create_variables(1)
        self.create_aux_variables(1) 
        self._populate_modifiers() # do indices

        self.formula['initial'] = z3.And(self.encode_initial_state())  # Encode initial state axioms
        self.formula['actions'] = z3.And(self.encode_actions(0))  # Encode universal axioms
        self.formula['frame']   = z3.And(self.encode_frame(0))  # Encode explanatory frame axioms
        self.formula['sem']     = z3.And(self.encode_execution_semantics())  # Encode execution semantics (lin/par)    
        self.formula['goal']    = z3.And(self.encode_goal_state(0))  # Encode goal state axioms

        # Abstract encoding
        self.formula['abstract_actions'] = z3.And(self.encode_abstract_actions(0)) # Encode relaxed universal axioms
        self.formula['abstract_goal']    = z3.And(self.encode_abstract_goal_state(0))  # Encode abstract goal state axioms
        self.formula['loop_formulas']    = z3.And(self.encode_loop_formulas(0)) # Encode loop formula

        # Objective
        self.formula['objective'] = self.encode_objective()
       
     
    def _free_to_abs_variables(self,formula,t):
        fluents = formula.environment.free_vars_extractor.get(formula)
                
        vars = []
        for f in fluents:
            key = str_repr(f)
            var = self.up_fluent_to_aux_z3[key][t]
            vars.append(var)
        return vars
    
    def encode_abstract_actions(self, t):
        """!
        Encodes the abstract Actions
        @return actions: list of Z3 formulas asserting the abstract actions

        """
        abstract_actions = []

        # For each fluent, store actions that assign that fluent in their effects
        look_up_table = defaultdict(list, { k[t]:[] for k in self.up_fluent_to_aux_z3.values() })

        for grounded_action in self.task.actions:
            key = str_repr(grounded_action)
            action_var = self.up_actions_to_aux_z3[key][-1] # need the last one

            # create the action abstract precondition
            action_pre = []
            for pre in grounded_action.preconditions:
                vars = self._free_to_abs_variables(pre,-1)                
                action_pre.append(z3.Or(self._expr_to_z3(pre, t+1),z3.Or(vars)))
        
            
            # create the action abstract effect on auxiliary variables
            action_eff = []
            for eff in grounded_action.effects:
                key = str_repr(eff.fluent)
                var = self.up_fluent_to_aux_z3[key][-1]
                
                action_eff.append(var)

                look_up_table[var].append(action_var)
            
            
            # assert universal
            action_pre = z3.And(action_pre) if len(action_pre) > 0 else z3.BoolVal(True, ctx=self.ctx)
            abstract_actions.append(z3.Implies(action_var, z3.And(action_pre)))
            action_eff = z3.And(action_eff) if len(action_eff) > 0 else z3.BoolVal(True, ctx=self.ctx)
            abstract_actions.append(z3.Implies(action_var, z3.And(action_eff)))
            
        # Ensure that auxiliary variables are true only if corresponding abstract action is executed
        for key,val in look_up_table.items():
            if len(val) > 0:
                abstract_actions.append(z3.Implies(key, z3.Or(val)))
            else:
                abstract_actions.append(z3.Implies(key, False))

        return abstract_actions

    def encode_abstract_goal_state(self, t):
            """!
            Encodes formula defining abstract goal state
            @return goal: Z3 formula asserting propositional and numeric subgoals
            """
            # TODO: assuming conjunctive goals for now
            goal = []
            for goal_pred in self.task.goals:
                vars = self._free_to_abs_variables(goal_pred,-1)
                goal.append(z3.Or(self._expr_to_z3(goal_pred, t+1), z3.Or(vars)))
            return goal
    
    def encode_loop_formulas(self,t):
        """!
            Encodes loop formulas to forbid self-enabling abstract actions
            @return goal: Z3 formula asserting loop formulas
        """
        loop_formula = []

        dgraph_edges = self._build_d_graph_edges()

        loops = self._identify_scc(dgraph_edges)
        
        for loop in loops:
            l_vars = []
            r_vars = []


            for grounded_action in self.task.actions:
                    effects = set([eff.fluent for eff in grounded_action.effects])

                    if len(loop & effects) > 0:
                        
                        action_pre = []
                        for pre in grounded_action.preconditions:
                            flat_abstract_pre = self._free_to_abs_variables(pre,-1)    
                            flat_abstract_pre.append(self._expr_to_z3(pre, t+1))            
                            action_pre.append(flat_abstract_pre)
                        
                        
                        for l in list(loop):
                            key = str_repr(l)
                            var = self.up_fluent_to_aux_z3[key][-1]
                            if var not in l_vars:
                                l_vars.append(var)


                        if len(action_pre) == 1:
                            for cond in action_pre[0]:
                                if cond in l_vars:
                                    pass
                                else:
                                    r_vars.append(cond)

                        else:

                            dnf = list(itertools.product(*action_pre))
                            
                            for combo in dnf:
                                if len(set(l_vars) & set(combo)) > 0:
                                    pass
                                else:
                                    r_vars.append(z3.And(combo))


            loop_formula.append(z3.Implies(z3.Or(l_vars),z3.Or(r_vars)))
            
        return loop_formula
    
    def _build_d_graph_edges(self):

        edges = []

        # nodes in the dependency graph are either concrete preconditions (step #1) or
        # auxiliary variables corresponding to fluents appearing in the concrete preconditions (step #2)
        # For convenience step #2 below operates on fluents as opposed to auxiliary variables. 
        # The reasoning is still the same but it facilitates later steps - the fluent will be converted to
        # an auxiliary variable later when encoding the loop formula.

        for grounded_action in self.task.actions:
                     
            for eff in grounded_action.effects:

                for pre in grounded_action.preconditions:
                    edges.append((eff.fluent,pre))
            
                    for fluent in pre.environment.free_vars_extractor.get(pre):
                        edges.append((eff.fluent,fluent)) 
                    
                    

        return set(edges)
    
    def _identify_scc(self, edges):

        graph = nx.DiGraph()
        graph.add_edges_from(edges)

        self_loops = set(nx.nodes_with_selfloops(graph))

        scc = []
        for el in nx.strongly_connected_components(graph):
            if len(el) > 1:
                scc.append(el)
            elif len(el & self_loops):
                scc.append(el)
            else:
                continue # discard  indiviual nodes with no self-loops
        
        return scc 
                     
    def encode_objective(self):

        objective = []

        if len(self.task.quality_metrics) > 0:
            print(self.task.quality_metrics)
            raise
        else:
            
            for grounded_action in self.task.actions:
                key = str_repr(grounded_action)
                abs_action_var = self.up_actions_to_aux_z3[key][-1]
                objective.append(z3.If(abs_action_var,1.0,0.0))

                for a in self.up_actions_to_z3[key]:
                    objective.append(z3.If(a,1.0,0.0))
        
        return sum(objective)

   


class EncoderSequentialOMT(EncoderOMT):
    """
    Class that encodes a problem to a classical sequential encoding to SMT
    """
    def __init__(self, task):
        super().__init__("omtseq", task, LinearModifier())
        self.allow_multiple_actions_per_step = False

class EncoderParallelOMT(EncoderOMT):
    """
    Class that encodes a problem to a parallel encoding to SMT using the forall-step semantics
    TODO: Check this is really forall-step semantics
    """
    def __init__(self, task):
        raise NotImplementedError("Can't use parallel encoding for now.") #TODO: need mutexes for this

        super().__init__(task, ParallelModifier())
        self.allow_multiple_actions_per_step = True
        self.name = "omt-seqForall"  # set the planner name.



    def encodeASAP(self,t):

        actions = []
        for grounded_action in self.task.actions:
            key = str_repr(grounded_action)
            # Fetch variable for action at current step t
            action_current = self.up_actions_to_z3[key][t]

            # Fetch variable for action at previous step t-1
            action_previous = self.up_actions_to_z3[key][t-1]

            # Fetch action preconditions
            action_pre = []
            for pre in grounded_action.preconditions:
                action_pre.append(self._expr_to_z3(pre, t))
