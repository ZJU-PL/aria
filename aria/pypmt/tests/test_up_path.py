import unified_planning
import unified_planning.model
from unified_planning.shortcuts import UserType, BoolType

from ..apis import solveUP, solveFile
from ..config import Config
from ..config import global_config  # Import the global configuration

import pytest

def create_robot_planning_task():
    """!
    The goal here is to create a dummy planning task such that we know its encodings.
    Then we use this task to check that the encoders are working correctly.
    """
    # Step one create a problem's objects.
    Location = UserType('Location')
    robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
    
    # Step two create the actions.
    move = unified_planning.model.InstantaneousAction('move', l_from=Location, l_to=Location)
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(robot_at(l_from))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)
    
    # Step three create the task.
    problem = unified_planning.model.Problem('robot')
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_action(move)

    NLOC = 2
    locations = [unified_planning.model.Object('l%s' % i, Location) for i in range(NLOC)]
    problem.add_objects(locations)

    # Set the initial state.
    problem.set_initial_value(robot_at(locations[0]), True)

    # Set the goal state.
    problem.add_goal(robot_at(locations[-1]))
    return problem


def test_UpProblem():
    conf = Config("seq")
    problem = create_robot_planning_task()
    plan = solveUP(problem, conf)
    if plan is not None:
        assert plan.isvalid, f"plan is invalid due to {plan.validation_fail_reason}"
    else:
        pytest.fail(f"UP-created domain is either unsolvable or has an invalid plan")
