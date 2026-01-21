"""Specification synthesis from oracle artifacts.

Synthesizes SMT specifications from:
- Source code (optional)
- Documentation (nldesc)
- I/O examples

The LLM synthesizes constraints that capture oracle behavior holistically.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import z3

from aria.ml.llm.llmtool.LLM_utils import LLM


@dataclass
class SynthesizedSpec:
    """Synthesized specification for an oracle."""
    oracle_name: str
    constraints: List[str]  # SMT-LIB constraint strings
    z3_constraints: List[z3.BoolRef] = field(default_factory=list)
    confidence: float = 0.0
    raw_llm_response: str = ""
    is_validated: bool = False
    validation_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_z3(self, input_vars: Dict[str, z3.ExprRef], output_var: z3.ExprRef) -> z3.BoolRef:
        """Convert constraints to Z3 formula."""
        if not self.z3_constraints:
            self._parse_constraints(input_vars, output_var)
        return z3.And(self.z3_constraints)

    def _parse_constraints(
        self, input_vars: Dict[str, z3.ExprRef], output_var: z3.ExprRef
    ):
        """Parse constraint strings to Z3."""
        self.z3_constraints = []
        for constraint in self.constraints:
            try:
                smt = f"(assert {constraint})"
                assertions = z3.parse_smt2_string(smt, decls=input_vars)
                if assertions:
                    self.z3_constraints.append(assertions[0])
            except Exception:
                continue


def spec_from_examples(
    examples: List[Dict[str, Any]],
    oracle_name: str,
    input_types: List[z3.SortRef],
    output_type: z3.SortRef,
    nldesc: str = "",
    source_code: Optional[str] = None,
    llm: Optional[LLM] = None,
) -> SynthesizedSpec:
    """Convenience: synthesize spec from examples."""
    synthesizer = SpecSynthesizer(llm)
    return synthesizer.synthesize(
        oracle_name=oracle_name,
        input_types=input_types,
        output_type=output_type,
        nldesc=nldesc,
        examples=examples,
        source_code=source_code,
    )


# Prompt for synthesizing specification from code/examples/docs
SYNTHESIS_PROMPT = """You are an expert in formal methods and program analysis.

Your task: Given information about a function, synthesize an SMT-LIB specification
that captures its behavior.

## Input Information

**Function Name**: {name}

**Signature**: {name}({input_types}) -> {output_type}

**Description**:
{nldesc}

**Examples**:
{examples}

**Source Code** (if available):
{source_code}

## Task

Synthesize an SMT-LIB formula that constrains the function output.

The formula should:
1. Be consistent with ALL examples
2. Capture the semantic behavior described
3. Work for any valid input (not just the examples)

## Output Format

Return a JSON object:
```json
{{
  "constraints": [
    "(= output arg0)",
    "(>= output 0)",
    "(ite (> arg0 0) (= output arg0) (= output (- arg0)))"
  ],
  "confidence": 0.9,
  "description": "Returns the absolute value of arg0"
}}
```

## Constraints Format Rules

1. Use "output" to refer to the function's return value
2. Use argument names: arg0, arg1, arg2, ... (in order)
3. Use standard SMT-LIB operators: =, <, >, <=, >=, +, -, *, ite, and, or, not
4. For conditionals, use: (ite condition then_expr else_expr)
5. Make constraints as general as possible

## Examples of Good Specifications

Input: max(x, y) returns maximum
```json
{{
  "constraints": ["(and (>= output arg0) (>= output arg1) (or (= output arg0) (= output arg1)))"],
  "confidence": 1.0
}}
```

Input: abs(x) returns absolute value
```json
{{
  "constraints": ["(ite (>= arg0 0) (= output arg0) (= output (- arg0)))"],
  "confidence": 1.0
}}
```

Input: clamp(x, lo, hi) constrains x to [lo, hi]
```json
{{
  "constraints": ["(and (>= output lo) (<= output hi) (or (= output lo) (= output hi) (and (>= output lo) (<= output hi) (>= output x) (<= output x))))"],
  "confidence": 0.85
}}
```

Now synthesize the specification:
"""


class SpecSynthesizer:
    """Synthesize SMT specifications from oracle artifacts."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        max_attempts: int = 3,
        temperature: float = 0.1,
    ):
        self.llm = llm
        self.max_attempts = max_attempts
        self.temperature = temperature

    def synthesize(
        self,
        oracle_name: str,
        input_types: List[z3.SortRef],
        output_type: z3.SortRef,
        nldesc: str = "",
        examples: List[Dict[str, Any]] = None,
        source_code: Optional[str] = None,
    ) -> SynthesizedSpec:
        """Synthesize specification from oracle information."""
        if examples is None:
            examples = []

        # Format input types
        input_names = [f"arg{i}" for i in range(len(input_types))]
        type_str = ", ".join(str(t) for t in input_types)

        # Format examples
        example_lines = []
        for ex in examples:
            inp = ex.get("input", {})
            out = ex.get("output", "?")
            inp_str = ", ".join(str(inp.get(name, "?")) for name in input_names)
            example_lines.append(f"({inp_str}) -> {out}")
        examples_str = "\n".join(example_lines) if example_lines else "No examples provided"

        # Format source code
        source_str = source_code if source_code else "No source code available"

        prompt = SYNTHESIS_PROMPT.format(
            name=oracle_name,
            input_types=type_str,
            output_type=output_type,
            nldesc=nldesc or "No description provided",
            examples=examples_str,
            source_code=source_str,
        )

        for attempt in range(self.max_attempts):
            try:
                if self.llm:
                    response, _, _ = self.llm.infer(prompt, is_measure_cost=False)
                else:
                    response = ""

                if not response:
                    continue

                # Parse JSON from response
                spec_data = self._parse_response(response)
                if spec_data is None:
                    continue

                # Build SynthesizedSpec
                spec = SynthesizedSpec(
                    oracle_name=oracle_name,
                    constraints=spec_data.get("constraints", ["true"]),
                    confidence=spec_data.get("confidence", 0.5),
                    raw_llm_response=response,
                    validation_examples=examples,
                )

                # If confidence is high and we have examples, validate
                if spec.confidence > 0.8 and examples:
                    spec.is_validated = self._quick_validate(spec, examples)

                return spec

            except Exception:
                continue

        # Fallback: identity specification
        return SynthesizedSpec(
            oracle_name=oracle_name,
            constraints=["true"],
            confidence=0.0,
            raw_llm_response="",
            is_validated=False,
        )

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response as JSON."""
        # Find JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

    def _quick_validate(
        self, spec: SynthesizedSpec, examples: List[Dict[str, Any]]
    ) -> bool:
        """Quick validation that spec is consistent with examples."""
        if not examples:
            return False

        # Create Z3 variables
        input_vars = {}
        for i, ex in enumerate(examples):
            inp = ex.get("input", {})
            for j, (name, _) in enumerate(inp.items()):
                if name not in input_vars:
                    input_vars[name] = z3.Int(name)

        output_var = z3.Int("output")

        try:
            spec.to_z3(input_vars, output_var)
            formula = z3.And(spec.z3_constraints)

            for ex in examples:
                inp = ex.get("input", {})
                expected = ex.get("output")

                # Check if there's a model satisfying example + formula
                solver = z3.Solver()
                solver.add(formula)

                for name, val in inp.items():
                    if name in input_vars:
                        solver.add(input_vars[name] == val)

                solver.add(output_var == expected)

                if solver.check() != z3.sat:
                    return False

            return True

        except Exception:
            return False
