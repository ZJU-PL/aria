from aria.symabs.ai_symabs.domains.z3_variables import Z3VariablesState


def test_sign_concrete_state_creation_query():
    state1 = Z3VariablesState({
        "a": 1,
        "b": -1000,
        "c": 50,
        "d": 0
    })

    assert state1.value_of("a") == 1
    assert state1.value_of("b") == -1000
    assert state1.value_of("c") == 50
    assert state1.value_of("d") == 0


def test_sign_concrete_repr():
    state1 = Z3VariablesState({
        "a": 1,
        "b": -1000,
        "c": 50,
        "d": 0
    })

    string_repr = repr(state1)
    for name, value in state1.variable_values.items():
        assert f"{name}: {value}" in string_repr
    assert len(string_repr.split(",")) == 4
