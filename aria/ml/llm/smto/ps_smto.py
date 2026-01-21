"""PS_SMTO: SMT Solver with Synthesized Specifications.

This module extends the base SMTO solver with:
1. Specification synthesis from code/docs/examples
2. Bidirectional SAT/UNSAT search
3. CDCL-style conflict learning
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import z3

from aria.ml.llm.llmtool.LLM_utils import LLM
from aria.ml.llm.smto.oracles import OracleInfo, WhiteboxOracleInfo
from aria.ml.llm.smto.spec_synth.synthesizer import (
    SpecSynthesizer,
    SynthesizedSpec,
)
from aria.ml.llm.smto.utils import (
    ExplanationLogger,
    OracleCache,
    generate_cache_key,
    z3_value_to_python,
)


class SolvingMode(Enum):
    """Mode of operation for PS_SMTO."""
    SAT_FIRST = "sat_first"
    UNSAT_FIRST = "unsat_first"
    BIDIRECTIONAL = "bidirectional"
    UNSAT_ONLY = "unsat_only"
    SAT_ONLY = "sat_only"


class SolvingStatus(Enum):
    """Result status of solving."""
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class SolvingResult:
    """Result of a solving attempt."""
    status: SolvingStatus
    model: Optional[z3.ModelRef] = None
    unsat_core: Optional[List[z3.BoolRef]] = None
    learned_clauses: List[z3.BoolRef] = field(default_factory=list)
    spec_confidence: float = 0.0
    time_elapsed: float = 0.0
    iterations: int = 0
    message: str = ""


@dataclass
class PS_SMTOConfig:
    """Configuration for PS_SMTO solver."""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    provider: str = "openai"
    temperature: float = 0.1
    mode: SolvingMode = SolvingMode.BIDIRECTIONAL
    max_iterations: int = 50
    timeout_ms: int = 300000
    sat_timeout_ratio: float = 0.5
    enable_spec_synthesis: bool = True
    cache_dir: Optional[str] = None
    explanation_level: str = "basic"


class PS_SMTOSolver:
    """SMT Solver with Synthesized Specifications.

    Extends SMTO with:
    - Specification synthesis from code/docs/examples
    - Bidirectional SAT/UNSAT search
    - CDCL-style conflict learning
    """

    def __init__(
        self,
        config: Optional[PS_SMTOConfig] = None,
        llm: Optional[LLM] = None,
    ):
        self.config = config or PS_SMTOConfig()

        if llm is not None:
            self.llm = llm
        else:
            from aria.ml.llm.llmtool.LLM_utils import LLM  # type: ignore
            from aria.ml.llm.llmtool.logger import Logger  # type: ignore
            logger = Logger("ps_smto.log")
            self.llm = LLM(  # type: ignore
                online_model_name=self.config.model,
                logger=logger,
                temperature=self.config.temperature,
            )

        self.spec_synthesizer = SpecSynthesizer(self.llm)
        self.solver = z3.Solver()
        self.cache = OracleCache(cache_dir=self.config.cache_dir)  # type: ignore
        self.explanation_logger = ExplanationLogger(
            level=self.config.explanation_level
        )

        self.oracles: Dict[str, OracleInfo] = {}
        self.specs: Dict[str, SynthesizedSpec] = {}
        self.learned_clauses: List = []

    def register_oracle(self, oracle_info: OracleInfo):
        """Register an oracle with the solver."""
        self.oracles[oracle_info.name] = oracle_info

        if self.config.enable_spec_synthesis:
            self._synthesize_spec(oracle_info)

    def _synthesize_spec(self, oracle_info: OracleInfo):
        """Synthesize specification from oracle information."""
        self.explanation_logger.log(
            f"Synthesizing spec for {oracle_info.name}"
        )

        source_code = None
        if isinstance(oracle_info, WhiteboxOracleInfo):
            source_code = oracle_info.source_code

        examples = []
        if hasattr(oracle_info, 'examples'):
            examples = oracle_info.examples

        spec = self.spec_synthesizer.synthesize(
            oracle_name=oracle_info.name,
            input_types=oracle_info.input_types,
            output_type=oracle_info.output_type,
            nldesc=oracle_info.description,
            examples=examples,
            source_code=source_code,
        )

        self.specs[oracle_info.name] = spec

        self.explanation_logger.log(
            f"Spec for {oracle_info.name}: confidence={spec.confidence}, "
            f"constraints={len(spec.constraints)}"
        )

    def add_constraint(self, constraint: z3.BoolRef):
        """Add a constraint to the solver."""
        self.solver.add(constraint)

    def check(self, timeout_ms: Optional[int] = None) -> SolvingResult:
        """Main solving entry point."""
        start_time = time.time()
        timeout = timeout_ms or self.config.timeout_ms

        if timeout > 0:
            self.solver.set("timeout", timeout)

        mode = self.config.mode

        if mode == SolvingMode.SAT_ONLY:
            return self._solve_sat(start_time, timeout)
        elif mode == SolvingMode.UNSAT_ONLY:
            return self._solve_unsat(start_time, timeout)
        else:
            return self._solve_bidirectional(start_time, timeout)

    def _solve_bidirectional(
        self, start_time: float, timeout_ms: int
    ) -> SolvingResult:
        """Bidirectional search: alternate between SAT and UNSAT."""
        sat_time_budget = int(timeout_ms * self.config.sat_timeout_ratio)
        unsat_time_budget = timeout_ms - sat_time_budget

        phase = "SAT"
        iterations = 0

        while time_elapsed(start_time) < timeout_ms:
            iterations += 1

            if phase == "SAT":
                sat_result = self._sat_search(
                    time_budget=sat_time_budget,
                    iterations_remaining=self.config.max_iterations - iterations,
                )
                if sat_result.status == SolvingStatus.SAT:
                    return sat_result
                phase = "UNSAT"
            else:
                unsat_result = self._unsat_proof(
                    time_budget=unsat_time_budget,
                    iterations_remaining=self.config.max_iterations - iterations,
                )
                if unsat_result.status == SolvingStatus.UNSAT:
                    return unsat_result
                self._learn_from_conflict(unsat_result)
                phase = "SAT"

        return SolvingResult(
            status=SolvingStatus.TIMEOUT,
            time_elapsed=time_elapsed(start_time),
            iterations=iterations,
            message="Timeout reached",
        )

    def _solve_sat(self, start_time: float, timeout_ms: int) -> SolvingResult:
        """SAT-only solving."""
        result = self._sat_search(
            time_budget=timeout_ms,
            iterations_remaining=self.config.max_iterations,
        )
        result.time_elapsed = time_elapsed(start_time)
        return result

    def _solve_unsat(self, start_time: float, timeout_ms: int) -> SolvingResult:
        """UNSAT-only solving."""
        result = self._unsat_proof(
            time_budget=timeout_ms,
            iterations_remaining=self.config.max_iterations,
        )
        result.time_elapsed = time_elapsed(start_time)
        return result

    def _sat_search(
        self, time_budget: int, iterations_remaining: int
    ) -> SolvingResult:
        """Model-finding with synthesized specifications."""
        self._add_spec_constraints()

        iterations = 0
        while iterations < iterations_remaining:
            iterations += 1
            result = self.solver.check()

            if result == z3.sat:
                model = self.solver.model()
                if self._validate_model(model):
                    return SolvingResult(
                        status=SolvingStatus.SAT,
                        model=model,
                        iterations=iterations,
                    )
                else:
                    self._add_model_blocking_clause(model)

            elif result == z3.unsat:
                return SolvingResult(
                    status=SolvingStatus.UNSAT,
                    unsat_core=list(self.solver.assertions()),
                    iterations=iterations,
                )
            else:
                return SolvingResult(
                    status=SolvingStatus.UNKNOWN,
                    iterations=iterations,
                    message="Solver returned unknown",
                )

        return SolvingResult(
            status=SolvingStatus.UNKNOWN,
            iterations=iterations,
            message="Max iterations reached",
        )

    def _unsat_proof(
        self, time_budget: int, iterations_remaining: int
    ) -> SolvingResult:
        """UNSAT proof construction using specifications."""
        self._add_spec_constraints()

        iterations = 0
        while iterations < iterations_remaining:
            iterations += 1
            result = self.solver.check()

            if result == z3.unsat:
                core = self.solver.unsat_core()
                return SolvingResult(
                    status=SolvingStatus.UNSAT,
                    unsat_core=list(core) if core else [],
                    iterations=iterations,
                )

            elif result == z3.sat:
                return SolvingResult(
                    status=SolvingStatus.SAT,
                    model=self.solver.model(),
                    iterations=iterations,
                )
            else:
                return SolvingResult(
                    status=SolvingStatus.UNKNOWN,
                    iterations=iterations,
                    message="Solver returned unknown",
                )

        return SolvingResult(
            status=SolvingStatus.UNKNOWN,
            iterations=iterations,
            message="Max iterations reached",
        )

    def _add_spec_constraints(self):
        """Add synthesized spec constraints to the solver."""
        for oracle_name, spec in self.specs.items():
            for constraint_str in spec.constraints:
                try:
                    smt = f"(assert {constraint_str})"
                    assertions = z3.parse_smt2_string(smt)
                    for a in assertions:
                        self.solver.add(a)
                except Exception:
                    continue

    def _validate_model(self, model: z3.ModelRef) -> bool:
        """Validate a candidate model against real oracles."""
        for oracle_name, oracle in self.oracles.items():
            for inputs in self._find_oracle_calls(model, oracle_name):
                expected = self._execute_oracle(oracle_name, inputs)
                if expected is not None:
                    model_output = self._get_model_output(model, oracle_name, inputs)
                    if not values_equal(expected, model_output):
                        return False
        return True

    def _find_oracle_calls(
        self, model: z3.ModelRef, oracle_name: str
    ) -> List[Dict[str, Any]]:
        """Find all oracle applications in the model."""
        calls = []
        assertions = self.solver.assertions()
        for i in range(len(assertions)):  # type: ignore
            decl = assertions[i]
            if self._contains_oracle_call(decl, oracle_name):
                inputs = self._extract_inputs(decl, oracle_name)
                if inputs is not None:
                    calls.append(inputs)
        return calls

    def _contains_oracle_call(self, expr, oracle_name: str) -> bool:
        """Check if expression contains a call to the given oracle."""
        try:
            decl = getattr(expr, 'decl', None)
            if decl is not None:
                name = getattr(decl, 'name', None)
                if name == oracle_name:
                    return True
            children = getattr(expr, 'children', lambda: [])()
            for child in children:
                if self._contains_oracle_call(child, oracle_name):
                    return True
        except Exception:
            pass
        return False

    def _extract_inputs(self, expr, oracle_name: str) -> Optional[Dict[str, Any]]:
        """Extract input values from an oracle call expression."""
        try:
            decl = getattr(expr, 'decl', None)
            if decl is None:
                return None
            name = getattr(decl, 'name', None)
            if name != oracle_name:
                return None
            inputs = {}
            children = list(getattr(expr, 'children', lambda: [])())  # type: ignore
            for i, child in enumerate(children):
                try:
                    inputs[f"arg{i}"] = str(child)
                except Exception:
                    inputs[f"arg{i}"] = None
            return inputs
        except Exception:
            return None

    def _execute_oracle(
        self, oracle_name: str, inputs: Dict[str, Any]
    ) -> Any:
        """Execute oracle with given inputs."""
        cache_key = generate_cache_key(oracle_name, inputs)
        if self.cache.contains(cache_key):
            return self.cache.get(cache_key)

        oracle = self.oracles[oracle_name]
        result = self._query_llm_oracle(oracle, inputs)
        if result is not None:
            self.cache.put(cache_key, result)
        return result

    def _query_llm_oracle(
        self, oracle: OracleInfo, inputs: Dict[str, Any]
    ) -> Optional[Any]:
        """Query LLM to simulate oracle function."""
        examples_text = "\n".join(
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in (oracle.examples or [])
        )
        prompt = (
            f"Act as the following function:\n{oracle.description}\n\n"
            f"Examples:\n{examples_text}\n\n"
            f"Now, given the input: {inputs}\n\n"
            "Return ONLY the output value."
        )
        try:
            result, _, _ = self.llm.infer(prompt, is_measure_cost=False)
            return self._parse_oracle_response(result, oracle.output_type)
        except Exception:
            return None

    def _parse_oracle_response(
        self, response: str, output_type: z3.SortRef
    ) -> Any:
        """Parse LLM oracle response."""
        try:
            text = response.strip()
            if output_type == z3.IntSort():
                return int(text)
            elif output_type == z3.RealSort():
                return float(text)
            elif output_type == z3.BoolSort():
                return text.lower() in ["true", "1", "yes"]
            elif output_type == z3.StringSort():
                return text.strip('"')
            return text
        except Exception:
            return None

    def _get_model_output(
        self, model: z3.ModelRef, oracle_name: str, inputs: Dict[str, Any]
    ) -> Any:
        """Get oracle output from model."""
        for decl in model.decls():
            if decl.name() == oracle_name:
                return z3_value_to_python(model[decl])
        return None

    def _add_model_blocking_clause(self, model: z3.ModelRef):
        """Add a clause to block the current model."""
        blocking = []
        for oracle_name in self.oracles:
            calls = self._find_oracle_calls(model, oracle_name)
            for inputs in calls:
                actual = self._execute_oracle(oracle_name, inputs)
                if actual is not None:
                    oracle_func = self._find_oracle_func(oracle_name)
                    if oracle_func is not None:
                        expected_z3 = self._python_to_z3(actual)
                        blocking.append(oracle_func != expected_z3)
        if blocking:
            clause = z3.Or(blocking)
            self.solver.add(clause)
            self.learned_clauses.append(clause)

    def _find_oracle_func(self, oracle_name: str) -> Optional[z3.ExprRef]:
        """Find an oracle function application in current constraints."""
        for decl in self.solver.assertions():
            if self._contains_oracle_call(decl, oracle_name):
                return decl
        return None

    def _python_to_z3(self, value: Any) -> z3.ExprRef:
        """Convert Python value to Z3."""
        if isinstance(value, int):
            return z3.IntVal(value)
        elif isinstance(value, float):
            return z3.RealVal(value)
        elif isinstance(value, bool):
            return z3.BoolVal(value)
        elif isinstance(value, str):
            return z3.StringVal(value)
        return z3.StringVal(str(value))

    def _learn_from_conflict(self, result: SolvingResult):
        """Learn clauses from a conflicting result."""
        if result.unsat_core:
            for clause in result.unsat_core:
                if clause not in self.learned_clauses:
                    self.learned_clauses.append(clause)


def time_elapsed(start_time: float) -> int:
    """Calculate elapsed time in milliseconds."""
    return int((time.time() - start_time) * 1000)


def values_equal(val1: Any, val2: Any) -> bool:
    """Check if two values are equal."""
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return abs(val1 - val2) < 1e-10
    return val1 == val2
