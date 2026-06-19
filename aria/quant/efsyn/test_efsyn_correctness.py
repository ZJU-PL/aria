from z3 import And, Int, IntVal, Or, Solver, sat

from aria.quant.efsyn.efsyn_int import LinearExistsForallCEGIS


def test_invalid_initial_x_cannot_be_certified():
    x = Int("x")
    y = Int("y")

    solver = LinearExistsForallCEGIS(
        x_vars=[x],
        y_vars=[y],
        predicate=x == 1,
        domain_x=x == 0,
        domain_y=y == 0,
        initial_x=[[1]],
        max_iters=2,
        verbose=False,
    )

    result = solver.solve()

    assert result.status != "valid"
    assert result.witness is None


def test_generalized_failure_guard_implies_failure():
    x = Int("x")
    y = Int("y")

    solver = LinearExistsForallCEGIS(
        x_vars=[x],
        y_vars=[y],
        predicate=x >= y,
        domain_x=And(0 <= x, x <= 10),
        domain_y=And(0 <= y, y <= 10),
        initial_x=[[0]],
        max_iters=1,
        verbose=False,
    )
    solver._initialize_memories(seed_x=(IntVal(0),))

    cand = solver.M_X[0]
    pred_x = solver._instantiate(solver.predicate, solver.x_vars, cand.vals)
    guard = solver._generalize_failure(cand, (IntVal(5),))

    check = Solver()
    check.add(solver.domain_y, guard, pred_x)

    assert check.check() != sat
    assert not solver._ground_holds(guard == True)


def test_budget_exhaustion_does_not_return_uncertified_witness():
    x = Int("x")
    y = Int("y")

    solver = LinearExistsForallCEGIS(
        x_vars=[x],
        y_vars=[y],
        predicate=x > y,
        domain_x=And(0 <= x, x <= 10),
        domain_y=And(0 <= y, y <= 10),
        max_iters=1,
        verbose=False,
    )

    result = solver.solve()

    assert result.status == "budget-exhausted"
    assert result.witness is None


def test_initial_x_must_satisfy_some_template():
    x = Int("x")
    y = Int("y")

    solver = LinearExistsForallCEGIS(
        x_vars=[x],
        y_vars=[y],
        predicate=x == 1,
        domain_x=Or(x == 0, x == 1),
        domain_y=y == 0,
        x_templates=[x == 0],
        initial_x=[[1]],
        max_iters=2,
        verbose=False,
    )

    result = solver.solve()

    assert result.status != "valid"
    assert result.witness is None


def test_empty_existential_vars_do_not_hang():
    y = Int("y")

    solver = LinearExistsForallCEGIS(
        x_vars=[],
        y_vars=[y],
        predicate=y >= 0,
        domain_y=And(0 <= y, y <= 2),
        max_iters=1,
        verbose=False,
    )

    result = solver.solve()

    assert result.status == "valid"
    assert result.witness == {}


def test_empty_universal_vars_do_not_hang():
    x = Int("x")

    solver = LinearExistsForallCEGIS(
        x_vars=[x],
        y_vars=[],
        predicate=x == 0,
        domain_x=x == 0,
        max_iters=1,
        verbose=False,
    )

    result = solver.solve()

    assert result.status == "valid"
    assert result.witness == {"x": 0}
