import z3

from aria.ml.ematching import LLMTriggerGenerator, TriggerCandidate, TriggerSelector


class _StubLLM:
    """Deterministic stub that returns a fixed JSON payload."""

    def __init__(self, payload: str):
        self.payload = payload
        self.last_prompt = None

    def infer(self, message: str, is_measure_cost: bool = False):
        self.last_prompt = message
        return self.payload, 0, 0


def test_llm_trigger_generator_parses_indices():
    x, y = z3.Ints("x y")
    f = z3.Function("f", z3.IntSort(), z3.IntSort())
    g = z3.Function("g", z3.IntSort(), z3.IntSort())

    candidates = [
        TriggerCandidate(expr=f(x), text="(f x)", variables=["x"]),
        TriggerCandidate(expr=g(y), text="(g y)", variables=["y"]),
    ]

    quant = z3.ForAll([x, y], f(x) == g(y))
    generator = LLMTriggerGenerator(llm=_StubLLM('{"triggers": [[0], [1]]}'))

    groups = generator.suggest_trigger_groups(quant, candidates, ["x", "y"])

    assert len(groups) == 2
    assert all(groups)
    # Both bound variables are covered
    covered_vars = {str(v) for exprs in groups for v in _collect_vars(exprs)}
    assert covered_vars == {"x", "y"}


def test_trigger_selector_falls_back_to_heuristics():
    x = z3.Int("x")
    int_sort = z3.IntSort()
    f = z3.Function("f", int_sort, int_sort)
    quant = z3.ForAll([x], f(x) > 0)

    selector = TriggerSelector(quant, verbose=True)
    groups = selector.select_triggers(quant)
    assert groups, "Heuristic trigger selection should return at least one group"

    annotated = selector.annotate_with_triggers()
    assert z3.is_quantifier(annotated)
    assert annotated.num_patterns() >= 0  # Should not raise when inspecting patterns


def _collect_vars(exprs):
    for expr in exprs:
        for child in _walk(expr):
            if z3.is_const(child) and child.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                yield child


def _walk(expr):
    stack = [expr]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.children())
