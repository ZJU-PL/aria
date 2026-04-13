import os
import pytest
from collections import defaultdict

from ..apis import solveFile
from ..config import Config
from .utilities import read_tasks_files

# Retrieve all tasks and encodings for parameterization
pddldir = os.path.join(os.path.dirname(__file__), "pddl")
tasks = read_tasks_files(pddldir)
encodings = ["seq", "forall", "forall-lazy", "exists", "exists-lazy", "r2e", "qfuf"]

# Generate test cases for each combination of domain/problem and encoding
@pytest.mark.parametrize("domainname, domainfile, problemfile, encoding", [
    (domainname, domainfile, problemfile, encoding)
    for domainname, domainfile, problemfile in tasks
    for encoding in encodings
])
def test_planner(domainname, domainfile, problemfile, encoding):
    plan = solveFile(domainfile, problemfile, Config(encoding), validate_plan=True)
    if plan is not None:
        assert plan.isvalid, f"{domainname}: plan is invalid due to {plan.validation_fail_reason}"
    else:
        pytest.fail(f"{domainname}: Either unsolvable or Invalid plan")
