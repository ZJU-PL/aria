"""Helper methods for Symbolic Abstraction analysis of straight-line programs.
"""
from typing import Dict, List, Set, Tuple, Any
import z3
from ..domains.algorithms import bilateral
from ..domains.core import ConjunctiveDomain
from ..domains.core.abstract import AbstractState


class Program:
    """Represents a straight-line program in 2-operand notation.
    """

    def __init__(self, program: str) -> None:
        """Initialize the program from a string @program.

        See example_program.py for an example of the two-operand notation we
        support.
        """
        program_lines: List[str] = [
            line.strip() for line in program.split("\n") if line.strip()]
        program_statements: List[List[str]] = [line.split(" ") for line in program_lines]

        # (inout, op, in)
        assert all(len(statement) == 3 for statement in program_statements)

        inouts, _ops, ins = zip(*program_statements)
        self.variables: List[str] = sorted(variable for variable in set(inouts) | set(ins)
                                if not variable.isdigit())
        assert all("'" not in variable for variable in self.variables)

        z3_variables: List[List[z3.ArithRef]] = [list(map(z3.Int, self.variables))]
        z3_statements: List[z3.BoolRef] = []
        for i, (operand_to, operation, operand_from) in enumerate(program_statements):
            primed = [variable + ((i + 1) * "'") for variable in self.variables]
            z3_variables.append(list(map(z3.Int, primed)))

            pre_variables = z3_variables[-2]
            post_variables = z3_variables[-1]

            try:
                operand_from_index = self.variables.index(operand_from)
                pre_operand_from: z3.ArithRef = pre_variables[operand_from_index]
            except ValueError:
                pre_operand_from = int(operand_from)

            operand_to_index = self.variables.index(operand_to)
            pre_operand_to: z3.ArithRef = pre_variables[operand_to_index]
            post_operand_to: z3.ArithRef = post_variables[operand_to_index]

            if operation == "=":
                z3_statements.append(post_operand_to == pre_operand_from)
            elif operation == "<=":
                z3_statements.append(post_operand_to <= pre_operand_from)
            elif operation == ">=":
                z3_statements.append(post_operand_to >= pre_operand_from)
            elif operation == "+=":
                z3_statements.append(
                    post_operand_to == (pre_operand_to + pre_operand_from))
            elif operation == "-=":
                z3_statements.append(
                    post_operand_to == (pre_operand_to - pre_operand_from))
            elif operation == "*=":
                z3_statements.append(
                    post_operand_to == (pre_operand_to * pre_operand_from))
            elif operation == "/=":
                z3_statements.append(
                    post_operand_to == (pre_operand_to / pre_operand_from))
            else:
                raise NotImplementedError

            for pre, post in zip(pre_variables, post_variables):
                if post is not post_operand_to:
                    z3_statements.append(pre == post)

        self.prime_depth: int = len(program_statements)
        self.z3_formula: z3.BoolRef = z3.And(*z3_statements)

    def transform(self, domain: ConjunctiveDomain, input_abstract_state: AbstractState) -> AbstractState:
        """Compute the most precise output abstract state.
        """

        def add_primes(unprimed: str) -> str:
            return unprimed + (self.prime_depth * "'")

        output_domain = domain.translate(dict({
            unprimed: add_primes(unprimed) for unprimed in domain.variables
        }))

        phi = z3.And(self.z3_formula, domain.gamma_hat(input_abstract_state))
        output_state = bilateral(output_domain, phi)

        return output_state.translate(dict({
            add_primes(unprimed): unprimed for unprimed in domain.variables
        }))
