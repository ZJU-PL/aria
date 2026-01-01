"""Whitebox oracle analysis and model evaluation."""

from typing import Dict, Optional, Any

import z3

from aria.ml.llm.llmtool.LLM_utils import LLM
from aria.ml.llm.smto.oracles import WhiteboxOracleInfo, OracleAnalysisMode


class WhiteboxAnalyzer:
    """Extract symbolic models from docs, source, or other information."""

    def __init__(self, llm: LLM, explanation_callback=None):
        """Initialize analyzer with LLM and optional explanation callback."""
        self.llm = llm
        self.explanation_callback = explanation_callback

    def analyze_oracle(self, oracle_info: WhiteboxOracleInfo) -> Optional[str]:
        """Analyze a whitebox oracle and derive a symbolic model (string)."""
        if self.explanation_callback:
            self.explanation_callback(
                f"Analyzing whitebox oracle '{oracle_info.name}' "
                f"using {oracle_info.analysis_mode.value} mode"
            )

        prompt = self._build_analysis_prompt(oracle_info)

        try:
            original_system_role = self.llm.systemRole
            self.llm.systemRole = (
                "You are an expert in software analysis and formal methods."
            )

            symbolic_model, _, _ = self.llm.infer(prompt, is_measure_cost=False)

            self.llm.systemRole = original_system_role

            oracle_info.symbolic_model = symbolic_model.strip()

            if self.explanation_callback:
                self.explanation_callback(
                    f"Successfully derived symbolic model for '{oracle_info.name}'"
                )

            return symbolic_model.strip()

        except Exception as e:
            if self.explanation_callback:
                self.explanation_callback(
                    f"Failed to analyze whitebox oracle '{oracle_info.name}': "
                    f"{str(e)}"
                )
            return None

    def _build_analysis_prompt(self, oracle_info: WhiteboxOracleInfo) -> str:
        """Build analysis prompt for a whitebox oracle."""
        examples_text = "\n".join(
            [
                f"Input: {ex['input']}\nOutput: {ex['output']}"
                for ex in oracle_info.examples
            ]
        )
        sections: list[str] = [
            (
                "As an expert in software analysis, create a precise symbolic model "
                "for the following component:"
            ),  # noqa: E501
            f"Component name: {oracle_info.name}",
            f"Input types: {[str(t) for t in oracle_info.input_types]}",
            f"Output type: {str(oracle_info.output_type)}",
            f"Description: {oracle_info.description}",
            "",
            "Examples:",
            examples_text,
        ]
        # Attach whitebox content per analysis mode
        if (
            oracle_info.analysis_mode == OracleAnalysisMode.DOCUMENTATION
            and oracle_info.documentation
        ):
            sections += ["", "Documentation:", oracle_info.documentation]
        if (
            oracle_info.analysis_mode == OracleAnalysisMode.SOURCE_CODE
            and oracle_info.source_code
        ):
            sections += ["", "Source code:", oracle_info.source_code]
        if (
            oracle_info.analysis_mode == OracleAnalysisMode.BINARY
            and oracle_info.binary_code
        ):
            sections += ["", "Binary code is available for analysis (metadata only)."]
        if oracle_info.analysis_mode == OracleAnalysisMode.MIXED:
            if oracle_info.documentation:
                sections += ["", "Documentation:", oracle_info.documentation]
            if oracle_info.source_code:
                sections += ["", "Source code:", oracle_info.source_code]
            if oracle_info.binary_code:
                sections += [
                    "",
                    "Binary code is available for analysis (metadata only).",
                ]
        if oracle_info.external_knowledge:
            sections += ["", "Additional relevant knowledge:"]
            sections += [f"- {item}" for item in oracle_info.external_knowledge]
        sections += [
            "",
            "Based on this information, please:",
            "1. Identify key behaviors and patterns",
            "2. Create a formal symbolic model capturing the input-output relationship",
            "3. Express this model using SMT expressions compatible with Z3",
            "4. Ensure the model matches all provided examples",
            "",
            (
                "Output only the model in a form directly evaluable with Z3, "
                "with no extra text:"
            ),
        ]
        return "\n".join(sections)


class ModelEvaluator:
    """Evaluate symbolic models extracted from whitebox oracles."""

    def __init__(self, llm: LLM, explanation_callback=None):
        """Initialize evaluator with LLM and optional explanation callback."""
        self.llm = llm
        self.explanation_callback = explanation_callback

    def evaluate_model(
        self, oracle_info: WhiteboxOracleInfo, inputs: Dict[str, Any]
    ) -> Optional[Any]:
        """Evaluate the symbolic model with inputs. Return result or None."""
        if not oracle_info.symbolic_model:
            return None

        try:
            result = self._evaluate_with_z3(oracle_info, inputs)
            if result is not None:
                return result

            if self.explanation_callback:
                self.explanation_callback(
                    "Direct Z3 evaluation failed, falling back to LLM evaluation"  # noqa: E501
                )

            return self._evaluate_with_llm(oracle_info, inputs)

        except Exception as e:
            if self.explanation_callback:
                self.explanation_callback(f"Model evaluation failed: {str(e)}")
            return None

    def _evaluate_with_z3(
        self, oracle_info: WhiteboxOracleInfo, inputs: Dict[str, Any]
    ) -> Optional[Any]:
        """Attempt direct Z3 evaluation of the symbolic model. Return result or None."""
        try:
            solver = z3.Solver()

            # Create variables for the inputs
            z3_vars: Dict[str, z3.ExprRef] = {}
            for i, input_type in enumerate(oracle_info.input_types):
                var_name = f"arg{i}"
                if input_type == z3.IntSort():
                    z3_vars[var_name] = z3.Int(var_name)
                elif input_type == z3.RealSort():
                    z3_vars[var_name] = z3.Real(var_name)
                elif input_type == z3.BoolSort():
                    z3_vars[var_name] = z3.Bool(var_name)
                elif input_type == z3.StringSort():
                    z3_vars[var_name] = z3.String(var_name)
                elif z3.is_bv_sort(input_type):
                    # For bit-vectors, create BitVec variable with appropriate size
                    z3_vars[var_name] = z3.BitVec(var_name, input_type.size())
                elif z3.is_fp_sort(input_type):
                    # For floating points, create FP variable with appropriate sort
                    z3_vars[var_name] = z3.FP(var_name, input_type)
                elif z3.is_array_sort(input_type):
                    # For arrays, create Array variable with appropriate domain
                    # and range
                    z3_vars[var_name] = z3.Array(
                        var_name, input_type.domain(), input_type.range()
                    )
                else:
                    return None

            # Constrain inputs
            for var_name, value in inputs.items():
                if var_name in z3_vars:
                    input_type = None
                    for i, oracle_input_type in enumerate(oracle_info.input_types):
                        if f"arg{i}" == var_name:
                            input_type = oracle_input_type
                            break

                    if input_type == z3.BoolSort():
                        solver.add(z3_vars[var_name] == z3.BoolVal(value))
                    elif input_type == z3.IntSort():
                        solver.add(z3_vars[var_name] == z3.IntVal(value))
                    elif input_type == z3.RealSort():
                        solver.add(z3_vars[var_name] == z3.RealVal(value))
                    elif input_type == z3.StringSort():
                        solver.add(z3_vars[var_name] == z3.StringVal(value))
                    elif z3.is_bv_sort(input_type):
                        # For bit-vectors, handle different value types
                        if isinstance(value, int):
                            solver.add(
                                z3_vars[var_name]
                                == z3.BitVecVal(value, input_type.size())
                            )
                        elif isinstance(value, str) and value.startswith("#b"):
                            solver.add(
                                z3_vars[var_name]
                                == z3.BitVecVal(value, input_type.size())
                            )
                        else:
                            solver.add(
                                z3_vars[var_name]
                                == z3.BitVecVal(int(value), input_type.size())
                            )
                    elif z3.is_fp_sort(input_type):
                        # For floating points
                        if isinstance(value, float):
                            solver.add(z3_vars[var_name] == z3.FPVal(value, input_type))
                        elif isinstance(value, str):
                            try:
                                solver.add(
                                    z3_vars[var_name]
                                    == z3.FPVal(float(value), input_type)
                                )
                            except Exception:
                                solver.add(
                                    z3_vars[var_name] == z3.FPVal(value, input_type)
                                )
                        else:
                            solver.add(
                                z3_vars[var_name] == z3.FPVal(float(value), input_type)
                            )
                    elif z3.is_array_sort(input_type):
                        # For arrays, this is complex - simplified implementation
                        # A full implementation would need sophisticated array handling
                        pass  # Skip for now

            # Create output variable
            if oracle_info.output_type == z3.IntSort():
                z3.Int("output")
            elif oracle_info.output_type == z3.RealSort():
                z3.Real("output")
            elif oracle_info.output_type == z3.BoolSort():
                z3.Bool("output")
            elif oracle_info.output_type == z3.StringSort():
                z3.String("output")
            elif z3.is_bv_sort(oracle_info.output_type):
                z3.BitVec("output", oracle_info.output_type.size())
            elif z3.is_fp_sort(oracle_info.output_type):
                z3.FP("output", oracle_info.output_type)
            elif z3.is_array_sort(oracle_info.output_type):
                z3.Array(
                    "output",
                    oracle_info.output_type.domain(),
                    oracle_info.output_type.range(),
                )
            else:
                return None

            # Placeholder: parsing model_str into Z3 constraints is not
            # implemented
            return None

        except Exception as e:
            if self.explanation_callback:
                self.explanation_callback(f"Z3 evaluation failed: {str(e)}")
            return None

    def _evaluate_with_llm(
        self, oracle_info: WhiteboxOracleInfo, inputs: Dict[str, Any]
    ) -> Optional[Any]:
        """Evaluate symbolic model using LLM. Return parsed result or None."""
        prompt = (
            f"Based on the following symbolic model for function "
            f"'{oracle_info.name}':\n\n"
            f"{oracle_info.symbolic_model}\n\n"
            f"Evaluate this model with these input values:\n{inputs}\n\n"
            f"Return only the resulting output value, with no additional text.\n"
        )
        try:
            original_system_role = self.llm.systemRole
            self.llm.systemRole = "You are a precise formula evaluator."

            result, _, _ = self.llm.infer(prompt, is_measure_cost=False)

            self.llm.systemRole = original_system_role

            from aria.ml.llm.smto.utils import parse_text_by_sort

            return parse_text_by_sort(result, oracle_info.output_type)

        except Exception as e:
            if self.explanation_callback:
                self.explanation_callback(f"LLM evaluation failed: {str(e)}")
            return None
