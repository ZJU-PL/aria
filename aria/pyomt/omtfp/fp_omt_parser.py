"""Parse SMT-LIB optimization problems with floating-point objectives."""

import re
from typing import List, Optional, Sequence, Tuple, cast

import z3


class FPOMTParser:
    """Parser for OMT problems whose objectives live in floating-point sorts."""

    def __init__(self) -> None:
        self.assertions: List[z3.ExprRef] = []
        self.objectives: List[z3.ExprRef] = []
        self.original_directions: List[str] = []
        self.objective: Optional[z3.ExprRef] = None

    def parse_with_z3(self, formula: str, is_file: bool = False) -> None:
        """Parse an SMT-LIB OMT instance with FP objectives.

        Args:
            formula: Formula string or input file path.
            is_file: Whether ``formula`` is a file path.
        """
        text = self._read_input(formula, is_file)
        commands = self._split_top_level_commands(text)

        base_commands: List[str] = []
        objective_commands: List[Tuple[str, str]] = []
        for command in commands:
            objective = self._parse_objective_command(command)
            if objective is not None:
                objective_commands.append(objective)
                continue

            if self._is_query_command(command):
                continue
            base_commands.append(command)

        if not objective_commands:
            raise ValueError("No objectives found in the supplied formula/file")

        base_script = "\n".join(base_commands)
        solver = z3.Solver()
        solver.from_string(base_script)
        assertions = solver.assertions()
        self.assertions = [cast(z3.ExprRef, assertions[index]) for index in range(len(assertions))]

        self.objectives = []
        self.original_directions = []
        for index, (direction, term_text) in enumerate(objective_commands):
            objective = self._parse_objective_term(base_script, term_text, index)
            if objective.sort_kind() != z3.Z3_FLOATING_POINT_SORT:
                raise ValueError(f"Objective is not floating-point: {term_text}")
            self.objectives.append(objective)
            self.original_directions.append(direction)

        if len(self.objectives) == 1:
            self.objective = self.objectives[0]

    @staticmethod
    def _read_input(formula: str, is_file: bool) -> str:
        if not is_file:
            return formula
        with open(formula, "r", encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _split_top_level_commands(text: str) -> Sequence[str]:
        commands: List[str] = []
        depth = 0
        start: Optional[int] = None
        in_string = False
        escape = False
        index = 0

        while index < len(text):
            char = text[index]

            if not in_string and char == ";":
                while index < len(text) and text[index] != "\n":
                    index += 1
                continue

            if char == '"' and not escape:
                in_string = not in_string

            if in_string:
                escape = char == "\\" and not escape
                index += 1
                continue

            escape = False
            if char == "(":
                if depth == 0:
                    start = index
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("Malformed SMT-LIB input: unmatched ')'")
                if depth == 0 and start is not None:
                    commands.append(text[start : index + 1].strip())
                    start = None
            index += 1

        if depth != 0:
            raise ValueError("Malformed SMT-LIB input: unbalanced parentheses")
        return commands

    @staticmethod
    def _parse_objective_command(command: str) -> Optional[Tuple[str, str]]:
        match = re.match(r"^\(\s*(maximize|minimize)\s+", command, re.DOTALL)
        if match is None:
            return None
        direction = "max" if match.group(1) == "maximize" else "min"
        term_text = command[match.end() : -1].strip()
        return direction, term_text

    @staticmethod
    def _is_query_command(command: str) -> bool:
        return bool(
            re.match(
                r"^\(\s*(check-sat|get-model|get-value|exit|push|pop)\b",
                command,
                re.DOTALL,
            )
        )

    @staticmethod
    def _parse_objective_term(
        base_script: str, term_text: str, index: int
    ) -> z3.ExprRef:
        marker = f"__aria_fp_obj_{index}"
        probe_script = (
            f"{base_script}\n"
            f"(assert (! (= {term_text} {term_text}) :named {marker}))"
        )
        parsed = z3.parse_smt2_string(probe_script)
        if len(parsed) == 0:
            raise ValueError(f"Unable to parse objective: {term_text}")

        objective_probe = cast(z3.ExprRef, parsed[-1])
        if not z3.is_eq(objective_probe) or objective_probe.num_args() != 2:
            raise ValueError(f"Unable to isolate objective term: {term_text}")
        return objective_probe.arg(0)
