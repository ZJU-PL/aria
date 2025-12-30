"""Orax: SMTO solver integrating Z3 with LLM-powered oracles (blackbox/whitebox)."""

import os
import tempfile
from typing import Dict, List, Optional, Any

import z3

from aria.ml.llm.llmtool.LLM_utils import LLM
from aria.ml.llm.llmtool.logger import Logger
from aria.ml.llm.smto.oracles import OracleInfo, WhiteboxOracleInfo, OracleType
from aria.ml.llm.smto.whitebox import WhiteboxAnalyzer, ModelEvaluator
from aria.ml.llm.smto.utils import (
    OracleCache,
    ExplanationLogger,
    z3_value_to_python,
    python_to_z3_value,
    values_equal,
    generate_cache_key
)


class OraxSolver:
    """
    SMTO solver that combines Z3 SMT solving with LLM-powered oracles.

    Supports both blackbox and whitebox modes:
    - Blackbox: Traditional SMTO where we can only observe input-output behavior
    - Whitebox: Enhanced SMTO using LLM to analyze available component information
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 provider: str = "openai",
                 cache_dir: Optional[str] = None,
                 explanation_level: str = "basic",
                 whitebox_analysis: bool = False,
                 temperature: float = 0.1,
                 system_role: str = (
                     "You are a experienced programmer and good at understanding "
                     "programs written in mainstream programming languages."
                 )):  # noqa: E501
        """
        Initialize Orax solver

        Args:
            api_key: API key for LLM provider (if None, looks for env variable)
            model: LLM model to use
            provider: LLM provider to use ('openai' or 'anthropic')
            cache_dir: Directory to cache oracle results (None for no caching)
            explanation_level: Level of explanation detail ('none', 'basic', 'detailed')
            whitebox_analysis: Whether to perform whitebox analysis on components
            temperature: Temperature for LLM generation
            system_role: System role prompt for LLM
        """
        # Set up logger
        log_file = os.path.join(tempfile.gettempdir(), "orax_solver.log")
        self.logger = Logger(log_file)

        # Set up LLM directly
        self.llm = LLM(
            online_model_name=model,
            logger=self.logger,
            temperature=temperature,
            system_role=system_role
        )

        # Set up explanations
        self.explanation_logger = ExplanationLogger(level=explanation_level)

        # Set up Z3 solver
        self.solver = z3.Solver()

        # Set up cache
        self.cache = OracleCache(cache_dir=cache_dir)

        # Set up whitebox analyzer if enabled
        self.whitebox_analysis = whitebox_analysis
        if whitebox_analysis:
            self.whitebox_analyzer = WhiteboxAnalyzer(
                llm=self.llm,
                explanation_callback=self._add_explanation
            )
            self.model_evaluator = ModelEvaluator(
                llm=self.llm,
                explanation_callback=self._add_explanation
            )

        # Set up oracle registry
        self.oracles: Dict[str, OracleInfo] = {}
        self.whitebox_models: Dict[str, str] = {}  # Symbolic models for whitebox oracles

    def _add_explanation(self, message: str, level: str = "basic"):
        """Add explanation to the logger."""
        self.explanation_logger.log(message, level)

    def get_explanations(self) -> List[Dict[str, Any]]:
        """Return logged explanations."""
        return self.explanation_logger.get_history()

    def register_oracle(self, oracle_info: OracleInfo):
        """Register an oracle with the solver."""
        self.oracles[oracle_info.name] = oracle_info

        # If whitebox analysis is enabled and this is a whitebox oracle, analyze it
        if (self.whitebox_analysis and
            isinstance(oracle_info, WhiteboxOracleInfo)):
            symbolic_model = self.whitebox_analyzer.analyze_oracle(oracle_info)
            if symbolic_model:
                self.whitebox_models[oracle_info.name] = symbolic_model

    def add_constraint(self, constraint: z3.BoolRef):
        """Add a constraint to the solver."""
        self.solver.add(constraint)

    def check(self, timeout_ms: int = 0) -> Optional[z3.ModelRef]:
        """Run SMTO solving with oracle feedback loop. Returns model or None."""
        if timeout_ms > 0:
            self.solver.set("timeout", timeout_ms)

        max_iterations = 10
        for iteration in range(max_iterations):
            # Log iteration start
            self._add_explanation(f"Starting SMTO iteration {iteration+1}/{max_iterations}")

            result = self.solver.check()
            if result == z3.unsat:
                self._add_explanation("Problem is unsatisfiable")
                return None

            model = self.solver.model()

            model_dict = {
                str(decl()): str(model[decl]) for decl in model.decls()
            }
            self._add_explanation(
                f"Candidate model found: {model_dict}", level="detailed"
            )

            is_valid = self._validate_model_with_oracles(model)

            if is_valid:
                self._add_explanation(
                    "Valid model found satisfying all oracle constraints"
                )
                return model

            # Add learned constraints from oracle feedback
            self._add_oracle_constraints(model)

            self._add_explanation(
                "Model validation failed, adding oracle constraints and "
                "trying again"
            )

        self._add_explanation("Maximum iterations reached without finding valid model")
        return None

    def _validate_model_with_oracles(self, model: z3.ModelRef) -> bool:
        """Validate model against all registered oracles."""
        for oracle_info in self.oracles.values():
            if not self._validate_oracle_calls(model, oracle_info):
                return False
        return True

    def _validate_oracle_calls(self, model: z3.ModelRef, oracle_info: OracleInfo) -> bool:
        """Validate all oracle calls in the model."""
        oracle_calls = self._find_oracle_calls(model, oracle_info.name)
        for call_inputs in oracle_calls:
            oracle_result = self._query_oracle(oracle_info, call_inputs)

            if oracle_result is None:
                self._add_explanation(
                    f"Oracle {oracle_info.name} failed to produce result for "
                    f"{call_inputs}"
                )
                return False

            model_result = self._get_model_result(model, oracle_info.name, call_inputs)

            if not values_equal(oracle_result, model_result):
                self._add_explanation(
                    f"Oracle mismatch for {oracle_info.name}{call_inputs}: "
                    f"oracle={oracle_result}, model={model_result}"
                )
                return False

        return True

    def _query_oracle(self, oracle_info: OracleInfo, inputs: Dict[str, Any]) -> Optional[Any]:
        """Query oracle based on type, with caching and fallbacks."""
        cache_key = generate_cache_key(oracle_info.name, inputs)

        if self.cache.contains(cache_key):
            self._add_explanation(
                f"Using cached result for {oracle_info.name}{inputs}",
                level="detailed"
            )
            return self.cache.get(cache_key)

        result = None

        if oracle_info.oracle_type == OracleType.FUNCTION and oracle_info.function is not None:
            result = oracle_info.function(**inputs)
        elif oracle_info.oracle_type == OracleType.LLM:
            result = self._query_llm_oracle(oracle_info, inputs)
        elif oracle_info.oracle_type == OracleType.WHITEBOX:
            if (self.whitebox_analysis and
                isinstance(oracle_info, WhiteboxOracleInfo) and
                oracle_info.symbolic_model):
                result = self.model_evaluator.evaluate_model(oracle_info, inputs)
                if result is None:
                    result = self._query_llm_oracle(oracle_info, inputs)
            else:
                result = self._query_llm_oracle(oracle_info, inputs)
        elif oracle_info.oracle_type == OracleType.EXTERNAL:
            raise NotImplementedError(
                "External oracle type not yet implemented"
            )
        else:
            raise ValueError(f"Unsupported oracle type: {oracle_info.oracle_type}")

        if result is not None:
            self.cache.put(cache_key, result)

        return result

    def _query_llm_oracle(self, oracle_info: OracleInfo,
                          inputs: Dict[str, Any]) -> Optional[Any]:
        """Query LLM to simulate oracle function."""
        examples_text = "\n".join(
            [
                f"Input: {ex['input']}\nOutput: {ex['output']}"
                for ex in oracle_info.examples
            ]
        )
        prompt = (
            f"Act as the following function:\n{oracle_info.description}\n\n"
            f"Examples:\n{examples_text}\n\n"
            f"Now, given the input: {inputs}\nWhat is the output?"
        )

        try:
            original_system_role = self.llm.systemRole
            self.llm.systemRole = "You are a precise function evaluator."

            result, _, _ = self.llm.infer(prompt, is_measure_cost=False)

            self.llm.systemRole = original_system_role

            return self._parse_llm_response(result, oracle_info.output_type)
        except Exception as e:
            self._add_explanation(f"LLM query failed: {e}")
            return None

    def _parse_llm_response(self, response: str,
                           output_type: z3.SortRef) -> Optional[Any]:
        """Parse LLM response according to expected output type."""
        try:
            response = response.strip()
            if output_type == z3.BoolSort():
                return response.lower() in ["true", "1", "yes"]
            if output_type == z3.IntSort():
                return int(response)
            if output_type == z3.RealSort():
                return float(response)
            if output_type == z3.StringSort():
                if response.startswith('"') and response.endswith('"'):
                    return response[1:-1]
                return response
            return response
        except Exception as e:
            self._add_explanation(f"Error parsing LLM response: {e}")
            return None

    def _iter_oracle_applications(self, model: z3.ModelRef, oracle_name: str):
        """Yield (inputs_dict, decl) for each application of oracle in model."""
        for decl in model.decls():
            if decl.name() != oracle_name:
                continue
            args = [model.eval(arg, True) for arg in decl.children()]
            inputs = {f"arg{i}": z3_value_to_python(arg) for i, arg in enumerate(args)}
            yield inputs, decl

    def _add_oracle_constraints(self, model: z3.ModelRef):
        """Add learned constraints from oracle feedback."""
        constraints_added = 0
        for oracle_name, oracle_info in self.oracles.items():
            for inputs, decl in self._iter_oracle_applications(model, oracle_name):
                cache_key = generate_cache_key(oracle_name, inputs)
                result = (
                    self.cache.get(cache_key)
                    if self.cache.contains(cache_key)
                    else self._query_oracle(oracle_info, inputs)
                )
                if result is None:
                    continue
                z3_actual = python_to_z3_value(result, oracle_info.output_type)
                self.solver.add(decl() == z3_actual)
                constraints_added += 1

        if constraints_added > 0:
            self._add_explanation(f"Added {constraints_added} oracle constraints")

    def _find_oracle_calls(self, model: z3.ModelRef, oracle_name: str) -> List[Dict[str, Any]]:
        """Find all applications of an oracle function in the model."""
        return [inputs for inputs, _ in self._iter_oracle_applications(model, oracle_name)]

    def _get_model_result(self, model: z3.ModelRef, oracle_name: str,
                         inputs: Dict[str, Any]) -> Optional[Any]:
        """Get the result of an oracle function application from the model."""
        for decl_inputs, decl in self._iter_oracle_applications(
            model, oracle_name
        ):
            if decl_inputs == inputs:
                return z3_value_to_python(model.eval(decl(), True))
        return None

    def clear_explanations(self):
        """Clear explanation history."""
        self.explanation_logger.clear()

    def get_symbolic_model(self, oracle_name: str) -> Optional[str]:
        """Get the symbolic model for a whitebox oracle, if any."""
        oracle_info = self.oracles.get(oracle_name)
        if (oracle_info and
            isinstance(oracle_info, WhiteboxOracleInfo) and
            oracle_info.symbolic_model):
            return oracle_info.symbolic_model
        return None
