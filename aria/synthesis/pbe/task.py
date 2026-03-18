"""Typed task modeling for programming-by-example synthesis."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .expressions import Theory, ValueType, infer_value_type


def _deduplicate(values: List[Any]) -> List[Any]:
    deduplicated: List[Any] = []
    seen: Set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def validate_examples(examples: List[Dict[str, Any]]) -> None:
    """Validate that examples form a well-typed PBE task."""
    if not examples:
        raise ValueError("No examples provided")

    required_inputs: Optional[Set[str]] = None
    input_python_types: Dict[str, type] = {}
    output_python_type: Optional[type] = None

    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(
                f"Example {index + 1} must be a dictionary, got {type(example)}"
            )

        if "output" not in example:
            raise ValueError(f"Example {index + 1} is missing an 'output' field")

        input_names = {key for key in example if key != "output"}
        if required_inputs is None:
            required_inputs = input_names
        elif required_inputs != input_names:
            raise ValueError(
                "All examples must have the same input variables; "
                f"example {index + 1} differs"
            )

        for name in input_names:
            value = example[name]
            current_type = type(value)
            if name not in input_python_types:
                input_python_types[name] = current_type
            elif input_python_types[name] is not current_type:
                raise ValueError(
                    f"Input variable '{name}' has inconsistent types across examples"
                )

        current_output_type = type(example["output"])
        if output_python_type is None:
            output_python_type = current_output_type
        elif output_python_type is not current_output_type:
            raise ValueError("Example outputs must all have the same type")

    if not required_inputs:
        raise ValueError("At least one input variable is required")


def get_variable_names(examples: List[Dict[str, Any]]) -> List[str]:
    """Return variable names in deterministic order."""
    validate_examples(examples)
    return sorted(key for key in examples[0] if key != "output")


def get_theory_from_variables(
    examples: List[Dict[str, Any]], theory_hint: Optional[Theory] = None
) -> Theory:
    """Infer a synthesis theory from observed values, honoring an explicit hint."""
    if theory_hint is not None:
        return theory_hint

    validate_examples(examples)
    observed_types: Set[ValueType] = set()
    for example in examples:
        for value in example.values():
            if isinstance(value, str):
                observed_types.add(ValueType.STRING)
            elif isinstance(value, bool):
                observed_types.add(ValueType.BOOL)
            elif isinstance(value, int):
                observed_types.add(ValueType.INT)
            else:
                raise ValueError(
                    "Could not infer a supported theory from the provided examples; "
                    f"unsupported value {value!r}"
                )

    if ValueType.STRING in observed_types:
        return Theory.STRING
    if observed_types <= {ValueType.INT, ValueType.BOOL}:
        return Theory.LIA

    raise ValueError(
        "Could not infer a supported theory from the provided examples; "
        "pass a theory hint explicitly"
    )


def get_variable_types(
    examples: List[Dict[str, Any]], theory: Theory
) -> Dict[str, ValueType]:
    """Infer each input variable sort under the selected theory."""
    validate_examples(examples)
    input_types: Dict[str, ValueType] = {}
    for name in get_variable_names(examples):
        inferred_type = infer_value_type(examples[0][name], theory)
        for example in examples[1:]:
            if infer_value_type(example[name], theory) != inferred_type:
                raise ValueError(
                    f"Input variable '{name}' has inconsistent value types"
                )
        input_types[name] = inferred_type
    return input_types


def get_output_type(examples: List[Dict[str, Any]], theory: Theory) -> ValueType:
    """Infer the target output type from example outputs."""
    validate_examples(examples)
    inferred_type = infer_value_type(examples[0]["output"], theory)
    for example in examples[1:]:
        if infer_value_type(example["output"], theory) != inferred_type:
            raise ValueError("All outputs must have the same type")
    return inferred_type


def default_candidate_values(
    value_type: ValueType,
    theory: Theory,
    bitwidth: Optional[int] = None,
) -> List[Any]:
    """Return a small default input domain for counterexample search."""
    del theory
    if value_type == ValueType.BOOL:
        return [False, True]
    if value_type == ValueType.STRING:
        return ["", "a", "b", "ab", "abc"]
    if value_type == ValueType.BV:
        mask = None
        if bitwidth is not None and bitwidth > 0:
            mask = (1 << bitwidth) - 1
        values = [0, 1, 2, 3, 15, 255]
        if mask is None:
            return values
        return [value & mask for value in values]
    return [-2, -1, 0, 1, 2, 3, 5, 10]


@dataclass(frozen=True)
class VariableSignature:
    """Typed input variable metadata for a PBE task."""

    name: str
    value_type: ValueType
    bitwidth: Optional[int] = None


@dataclass(frozen=True)
class PBETask:
    """A typed synthesis task derived from input-output examples."""

    examples: Tuple[Dict[str, Any], ...]
    theory: Theory
    output_type: ValueType
    inputs: Tuple[VariableSignature, ...]
    bitwidth: int = 32

    @classmethod
    def from_examples(
        cls,
        examples: List[Dict[str, Any]],
        theory_hint: Optional[Theory] = None,
        bitwidth: int = 32,
    ) -> "PBETask":
        """Construct a task from raw examples."""
        normalized_examples = [dict(example) for example in examples]
        validate_examples(normalized_examples)

        theory = get_theory_from_variables(normalized_examples, theory_hint=theory_hint)
        input_types = get_variable_types(normalized_examples, theory)
        output_type = get_output_type(normalized_examples, theory)

        inputs = tuple(
            VariableSignature(
                name=name,
                value_type=input_types[name],
                bitwidth=bitwidth if input_types[name] == ValueType.BV else None,
            )
            for name in get_variable_names(normalized_examples)
        )
        return cls(
            examples=tuple(normalized_examples),
            theory=theory,
            output_type=output_type,
            inputs=inputs,
            bitwidth=bitwidth,
        )

    @property
    def input_types(self) -> Dict[str, ValueType]:
        """Return a mapping from variable names to their value types."""
        return {signature.name: signature.value_type for signature in self.inputs}

    @property
    def variable_bitwidths(self) -> Dict[str, int]:
        """Return bitwidths for bitvector variables."""
        return {
            signature.name: signature.bitwidth or self.bitwidth
            for signature in self.inputs
            if signature.value_type == ValueType.BV
        }

    @property
    def variable_names(self) -> List[str]:
        """Return the ordered input variable names."""
        return [signature.name for signature in self.inputs]

    def input_signature(self, variable_name: str) -> Optional[VariableSignature]:
        """Return the typed signature for a variable if present."""
        for signature in self.inputs:
            if signature.name == variable_name:
                return signature
        return None

    def observed_values(self, variable_name: str) -> List[Any]:
        """Return distinct observed values for an input variable."""
        return _deduplicate([example[variable_name] for example in self.examples])

    def candidate_values(self, variable_name: str) -> List[Any]:
        """Return a small domain for distinguishing-input search."""
        for signature in self.inputs:
            if signature.name != variable_name:
                continue
            observed = self.observed_values(variable_name)
            defaults = default_candidate_values(
                signature.value_type,
                self.theory,
                bitwidth=signature.bitwidth,
            )
            return _deduplicate(observed + defaults)
        return []

    def as_examples(self) -> List[Dict[str, Any]]:
        """Return copy-on-read examples for mutation-safe consumers."""
        return [dict(example) for example in self.examples]

    def with_example(self, example: Dict[str, Any]) -> "PBETask":
        """Return a new task extended with a labeled example."""
        examples = self.as_examples()
        examples.append(dict(example))
        return PBETask.from_examples(
            examples,
            theory_hint=self.theory,
            bitwidth=self.bitwidth,
        )

    def output_matches(self, value: Any) -> bool:
        """Return whether a Python value matches the task output type."""
        try:
            return infer_value_type(value, self.theory) == self.output_type
        except ValueError:
            return False

    def statistics(self) -> Dict[str, Any]:
        """Return task-level metadata suitable for synthesis statistics."""
        return {
            "theory": self.theory.value,
            "output_type": self.output_type.value,
            "inputs": {
                signature.name: signature.value_type.value for signature in self.inputs
            },
        }
