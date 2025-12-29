"""
Use unit propagation to compute "probing features" of SAT instances
"""
import math
import random
import statistics

from aria.bool.features.enums import ClauseState, VarState
from aria.bool.features.stopwatch import Stopwatch


class DPLLProbing:
    """
    So the values(states) of the variables are stored in varStates
    The clauses themselves are not changed (I assume for storage reasons)
    So assignments of the variables are stored in varstate, if they are true, false, etc,
    and then clauses are iterated through and checked if they are unassigned, so that is the next unit clause
    lengths of clauses kept track of in clause_lengths
    First check is for unit clauses -> if there are unit clauses
    Within these clauses, the unassigned variable is found
    This variable is set to the value of the literal (positive or negative)
    and the clauses that contain this variable are reduced
    reduction
    first remove all of the instances of the negative variable...
    so reduce the size of clauses with this variable,
    if the new size is 0, then we have a problem (inconsistent)
    Then satisfy the consistent clauses (clauses that contain that literal)
    Code adapted from SATzilla implementation
    """

    def __init__(self, sat_instance):
        self.sat_instance = sat_instance
        self.verbose = sat_instance.verbose
        self.num_vars_to_try = 10
        self.num_probes = 5

        # still not sure what lob is
        self.num_lob_probe = 30000
        # not sure what 2 is in this case(perhaps 2 seconds?)
        self.time_limit = 2

        # two stacks
        self.num_reduced_clauses = []
        self.num_reduced_vars = []

        # stack of variables that have been reduced
        self.reduced_vars = []
        self.reduced_clauses = []
        self.unit_props_log_nodes_dict = {}

    # unit propagation

    # num_bin_clauses_with_var, int array containing the number of binary clauses with a certain variable (index),
    # this should change as the propagation happens

    def search_space_probe(self, halt_on_assignment=False):
        """
        Randomly choose a variable, propagate, and see how deep you get. This generates the mean depth to contradiction
        and an estimate on the number of nodes.
        Timeout of 2 seconds (can be changed, more runs means a more accurate result, as the algorithm is stochastic)
        :param halt_on_assignment:
        :return:
        """
        # lobjois probe in satzilla
        if self.verbose:
            print("search space estimate probe")

        sw = Stopwatch()
        sw.start()

        depths = []

        orig_num_active_vars = self.sat_instance.num_active_vars

        probe_num = 0

        # while probe_num < self.num_lob_probe and stopwatch.lap() < self.lobjois_tim_limit:
        while probe_num < self.num_lob_probe and sw.lap() < self.time_limit:

            # randomly choose an unassigned variable and a value propagate this,
            # and continue until a contradiction is reached. This result is the
            # depth to contradiction.
            # Do this while within the time limit, and within the number of probes

            var = 1
            val = False

            # another do while block
            while True:

                if self.sat_instance.num_active_vars == 0:
                    if self.sat_instance.num_active_clauses == 0 and halt_on_assignment:
                        print("finished")
                        return
                    break

                # chooses a random unassigned variable, should be between 1 and v
                var = 0
                randnum = random.randint(1, self.sat_instance.num_active_vars)
                for _ in range(randnum, 0, -1):
                    var += 1

                    while self.sat_instance.var_states[var] != VarState.UNASSIGNED:
                        var += 1
                        # This upper limit should be v, as the actual array of
                        # variables does not change, but the just states of the
                        # variables
                        if var == self.sat_instance.v:
                            var = 1

                # Choose a random value to propagate
                val = random.random() < 0.5

                if not self.set_var_and_prop(var, val):
                    break

            # print("reached bottom")
            depths.append(orig_num_active_vars - self.sat_instance.num_active_vars)

            # reset the problem
            while self.sat_instance.num_active_vars != orig_num_active_vars:
                self.backtrack()

            probe_num += 1

        mean_depth_to_con_over_vars = statistics.mean(depths) / self.sat_instance.v
        # print("mean depth over variables: ",
        #       mean_depth_to_contradiction_over_vars)

        self.unit_props_log_nodes_dict[
            "mean_depth_to_contradiction_over_vars"
        ] = mean_depth_to_con_over_vars

        max_depth = max(depths)

        res = 0.0
        for i in range(probe_num):
            res += math.pow(2, (depths[i] - max_depth))

        lobjois = max_depth + (math.log(res / probe_num) / math.log(2.0))
        if probe_num == 0:
            res = 0
        else:
            res = lobjois / self.sat_instance.v

        self.unit_props_log_nodes_dict[
            "estimate_log_number_nodes_over_vars"
        ] = res
        # print("lobjois log num nodes over vars",
        #       lobjois/self.sat_instance.v)

        # also should be timed
        if self.verbose:
            print("total time:", sw.lap())

    def unit_propagation_probe(self, halt_on_assignment=False):
        """
        Method to calculate the dpll probing features
        """
        if self.verbose:
            print("unit propagation probe")

        # the depths are manually set, multiples of 4 each time
        current_depth = 0
        orig_num_active_vars = self.sat_instance.num_active_vars
        reached_bottom = False

        sw = Stopwatch()
        sw.start()

        for probe_num in range(self.num_probes):
            # sets depth to 1, 4, 16, 64, 256

            # alternatively next_probe_depth = 4 ** probe_num
            next_probe_depth = 1
            for j in range(probe_num):
                next_probe_depth = next_probe_depth * 4

            if self.verbose:
                print("searching depth: ", next_probe_depth)
            # the actual searching
            while current_depth < next_probe_depth and not reached_bottom and sw.lap() < self.time_limit:
                # print("c depth", current_depth)

                vars_in_most_bin_clauses = [0] * self.num_vars_to_try
                num_bin = [0] * self.num_vars_to_try

                array_size = 0
                for var in range(1, self.sat_instance.v + 1):
                    # if the variable is not unassigned, skip it (it already has
                    # a value (true, false) or it is irrelevant
                    if self.sat_instance.var_states[var] != VarState.UNASSIGNED:
                        continue

                    if array_size < self.num_vars_to_try:
                        array_size += 1

                    j = 0
                    while (j < array_size - 1 and
                           self.sat_instance.num_bin_clauses_with_var[var] <
                           num_bin[j]):
                        j += 1

                    # what is this actually doing... somehow sorting and keeping
                    # track of the top 10 vars that occur in the most binary
                    # clauses??
                    for k in range(array_size - 1, j, -1):
                        vars_in_most_bin_clauses[k] = \
                            vars_in_most_bin_clauses[k - 1]
                        num_bin[k] = num_bin[k - 1]

                    vars_in_most_bin_clauses[j] = var
                    num_bin[j] = self.sat_instance.num_bin_clauses_with_var[var]

                max_props_var = 0
                max_props_val = False

                # if there are no binary clauses, just take the first
                # unassigned var
                if array_size == 0:
                    max_props_var = 1
                    while (self.sat_instance.var_states[max_props_var] !=
                           VarState.UNASSIGNED and
                           max_props_var < self.sat_instance.v):
                        max_props_var += 1

                    max_props_val = True

                else:
                    max_props = -1
                    for var_num in range(array_size):
                        # should be simplified, basically you try to set the
                        # variable to true, and to false
                        value = True
                        # do while
                        while True:
                            # do block
                            # for value = True and value = False
                            # print("vars in bin clauses",
                            #       vars_in_most_bin_clauses)

                            # print("before")
                            con = self.set_var_and_prop(
                                vars_in_most_bin_clauses[var_num], value
                            )
                            # print("con", con)
                            # print("active vars", self.feats.num_active_vars)
                            if con and self.sat_instance.num_active_vars <= 0:

                                if halt_on_assignment:
                                    print("solved")
                                    print(self.sat_instance.num_active_clauses)
                                    print(self.sat_instance.num_active_vars)
                                    # output_assignment()
                                    # DONE is just some number... still to be
                                    # seen what this does in the satzilla code
                                    return

                            num_props = (orig_num_active_vars -
                                         self.sat_instance.num_active_vars -
                                         current_depth)

                            if num_props > max_props:
                                max_props_var = \
                                    vars_in_most_bin_clauses[var_num]
                                max_props_val = value

                            self.backtrack()

                            value = not value

                            if value:
                                break

                assert max_props_var != 0

                if not self.set_var_and_prop(max_props_var, max_props_val):
                    reached_bottom = True

                elif self.sat_instance.num_active_clauses == 0:
                    # print("no more active variables, solved")
                    if halt_on_assignment:
                        # print("assignment solved")
                        print(self.sat_instance.num_active_clauses)
                        print(self.sat_instance.num_active_vars)
                        # outputAssignment()
                        return

                    reached_bottom = True

                current_depth += 1

            unit_props_str = "unit_props_at_depth_" + str(next_probe_depth)
            unit_props_res = (
                (orig_num_active_vars - self.sat_instance.num_active_vars -
                 current_depth) / self.sat_instance.v
            )
            self.unit_props_log_nodes_dict[unit_props_str] = unit_props_res
            # print("vars reduced depth ", next_probe_depth)
            # print((orig_num_active_vars - self.sat_instance.num_active_vars -
            #        current_depth) / self.sat_instance.v)

        while self.sat_instance.num_active_vars != orig_num_active_vars:
            self.backtrack()

    def set_var_and_prop(self, var, value):
        """
        Sets the variable to the value, and then propagates the assignment
        :param var: integer, the variable
        :param value: boolean, value that the variable should get set to
        :return:
        """
        # these need to be passed by reference, instead of passed by assignment
        num_clauses_reduced = 0
        num_vars_reduced = 1

        # can only set an unassigned variable to a value
        assert self.sat_instance.var_states[var] == VarState.UNASSIGNED

        # set the value of the variable
        if value:
            self.sat_instance.var_states[var] = VarState.TRUE_VAL
            literal = var
        else:
            self.sat_instance.var_states[var] = VarState.FALSE_VAL
            literal = -var

        self.reduced_vars.append(var)
        self.sat_instance.num_active_vars -= 1

        consistent, num_clauses_reduced, num_vars_reduced = \
            self.reduce_clauses(literal, num_clauses_reduced, num_vars_reduced)

        if consistent:
            consistent, num_clauses_reduced, num_vars_reduced = \
                self.unit_prop(num_clauses_reduced, num_vars_reduced)

        # stack to hold the number that have been reduced, used in backtracking
        self.num_reduced_clauses.append(num_clauses_reduced)
        self.num_reduced_vars.append(num_vars_reduced)

        return consistent

    def reduce_clauses(self, orig_literal, num_clauses_reduced, num_vars_reduced):
        """
        Reduce clauses
        :param orig_literal:
        :param num_clauses_reduced:
        :param num_vars_reduced:
        :return:
        """
        # we are trying to assign this literal value to true (in all clauses that contain it)
        # "remove" vars from inconsistent clauses
        # check which clauses contain the negative of this literal, and remove that negative literal from them

        for clause_num in self.sat_instance.clauses_with_literal(-orig_literal):
            # iterate through all of the clauses that contain this literal
            # if it is active
            if self.sat_instance.clause_states[clause_num] == ClauseState.ACTIVE:
                self.reduced_clauses.append(clause_num)
                num_clauses_reduced += 1

                # decrease the size (this length actually represents the number
                # of yet to be assigned variables within that clause)
                self.sat_instance.clause_lengths[clause_num] -= 1

                if self.sat_instance.clause_lengths[clause_num] == 2:
                    # 0 marked as the end of the clause
                    # for (int i=0; clauses[clause][i] != 0; i++)

                    # iterate through the clause
                    # for j in range(len(self.feats.clauses[clause_num])):
                    for literal in self.sat_instance.clauses[clause_num]:
                        # still a bit strange, does this not mark duplicates??
                        # self.feats.num_bin_clauses_with_var[abs(self.feats.clauses[clause_num][j])] += 1
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] += 1

                elif self.sat_instance.clause_lengths[clause_num] == 1:
                    # for (int i=0; clauses[clause][i] != 0; i++)
                    # clauses themselves are never actually edited, just the values in varstates, clausestates, clauselengths etc.
                    # for j in range(len(self.feats.clauses[clause_num])):
                    for literal in self.sat_instance.clauses[clause_num]:
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] -= 1

                    # now a unit clause
                    # print(clause_num, "is unit clause")
                    self.sat_instance.unit_clauses.append(clause_num)

                elif self.sat_instance.clause_lengths[clause_num] == 0:
                    # inconsistent, the last literal in the clause has been
                    # removed, and it has to be satisfied, as opposed to being
                    # removed
                    return False, num_clauses_reduced, num_vars_reduced

        # satisfy the consistent clauses
        # print("consisten clause",
        #       self.feats.clauses_with_literal(orig_literal))
        for clause_num in self.sat_instance.clauses_with_literal(orig_literal):
            if self.sat_instance.clause_states[clause_num] == ClauseState.ACTIVE:
                # print("pacify ", clause_num)

                self.sat_instance.clause_states[clause_num] = \
                    ClauseState.PASSIVE
                self.reduced_clauses.append(clause_num)
                self.sat_instance.num_active_clauses -= 1

                for literal in self.sat_instance.clauses[clause_num]:
                    curr_var = abs(literal)
                    self.sat_instance.num_active_clauses_with_var[curr_var] -= 1

                    if self.sat_instance.clause_lengths[clause_num] == 2:
                        self.sat_instance.num_bin_clauses_with_var[curr_var] -= 1

                    # is the variable now irrelevant (active, but existing in
                    # no clauses)
                    if (self.sat_instance.num_active_clauses_with_var[curr_var] == 0 and
                            self.sat_instance.var_states[curr_var] ==
                            VarState.UNASSIGNED):
                        self.sat_instance.var_states[curr_var] = \
                            VarState.IRRELEVANT
                        self.reduced_vars.append(curr_var)
                        self.sat_instance.num_active_vars -= 1
                        num_vars_reduced += 1

                num_clauses_reduced += 1

        return True, num_clauses_reduced, num_vars_reduced

    def unit_prop(self, num_clauses_reduced, num_vars_reduced):
        """
        Propagates the unit clauses
        :param num_clauses_reduced:
        :param num_vars_reduced:
        :return:
        """

        consistent = True

        # for each unit clause (if there are unit clauses)
        while (len(self.sat_instance.unit_clauses) > 0) and consistent:
            # get the next unit clause
            clause_number = self.sat_instance.unit_clauses.pop()

            # skip inactive clauses
            # print("cstate", self.feats.clause_states[clause_number])
            if self.sat_instance.clause_states[clause_number] != \
                    ClauseState.ACTIVE:
                continue
            # print("unit clause number", clause_number)

            lit_num = 0

            # while the current literal is not unassigned
            # get the next possible unassigned literal
            while (self.sat_instance.var_states[
                       abs(self.sat_instance.clauses[clause_number][lit_num])] != VarState.UNASSIGNED):
                lit_num += 1

            assert self.sat_instance.clause_lengths[clause_number] == 1

            # get the literal literal (excuse the pun)
            literal = self.sat_instance.clauses[clause_number][lit_num]

            if literal > 0:
                self.sat_instance.var_states[abs(literal)] = VarState.TRUE_VAL
            else:
                self.sat_instance.var_states[abs(literal)] = VarState.FALSE_VAL

            self.reduced_vars.append(abs(literal))
            self.sat_instance.num_active_vars -= 1
            num_vars_reduced += 1

            # now reduce the clauses with that literal value
            r_consistent, num_clauses_reduced, num_vars_reduced = \
                self.reduce_clauses(literal, num_clauses_reduced,
                                    num_vars_reduced)
            # consistent = consistent and self.reduce_clauses(literal,
            #                                                 num_clauses_reduced,
            #                                                 num_vars_reduced)
            consistent = consistent and r_consistent

        return consistent, num_clauses_reduced, num_vars_reduced

    def backtrack(self):
        """
        Should undo one call of setVar or unitprop
        :return:
        """
        # print("backtrack")

        num_vars_reduced = self.num_reduced_vars.pop()

        # for all the vars that were reduced, unassign them
        for _ in range(num_vars_reduced):
            var = self.reduced_vars.pop()
            self.sat_instance.var_states[var] = VarState.UNASSIGNED
            self.sat_instance.num_active_vars += 1

        # for all of the clauses that were reduced
        num_clauses_reduced = self.num_reduced_clauses.pop()
        for i in range(num_clauses_reduced):
            clause_num = self.reduced_clauses.pop()

            # re activate the clause
            if self.sat_instance.clause_states[clause_num] != ClauseState.ACTIVE:
                self.sat_instance.num_active_clauses += 1
                self.sat_instance.clause_states[clause_num] = ClauseState.ACTIVE

                if self.sat_instance.clause_lengths[clause_num] == 2:
                    for j in range(len(self.sat_instance.clauses[clause_num])):
                        literal = self.sat_instance.clauses[clause_num][j]
                        self.sat_instance.num_active_clauses_with_var[abs(literal)] += 1
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] += 1
                else:
                    for j in range(len(self.sat_instance.clauses[clause_num])):
                        literal = self.sat_instance.clauses[clause_num][j]
                        self.sat_instance.num_active_clauses_with_var[abs(literal)] += 1

            else:
                self.sat_instance.clause_lengths[clause_num] += 1

                if self.sat_instance.clause_lengths[clause_num] == 2:
                    for j in range(len(self.sat_instance.clauses[clause_num])):
                        literal = self.sat_instance.clauses[clause_num][j]
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] += 1

                elif self.sat_instance.clause_lengths[clause_num] == 3:
                    for j in range(len(self.sat_instance.clauses[clause_num])):
                        literal = self.sat_instance.clauses[clause_num][j]
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] -= 1

        # empty out unit clauses
        self.sat_instance.unit_clauses = []
