"""Sample shrinking utilities for SLIA sampling."""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import z3

from aria.sampling.base import SamplingOptions
from aria.sampling.finite_domain.common import build_sample


def _preferred_char_order(sample_strings: Iterable[str]) -> List[str]:
    preferred = list("abcdefghijklmnopqrstuvwxyz")
    preferred.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    preferred.extend(list("0123456789"))
    preferred.extend([" ", "-", "_", ".", "/"])
    extras = sorted(
        {
            character
            for sample_string in sample_strings
            for character in sample_string
            if character not in preferred
        }
    )
    return preferred + extras


class SampleShrinker:
    """Minimize already-satisfying concrete samples while preserving constraints."""

    def __init__(
        self,
        formula: z3.ExprRef,
        string_variables: Sequence[z3.SeqRef],
        int_variables: Sequence[z3.ArithRef],
        observable_terms: Sequence[z3.ExprRef],
    ) -> None:
        self.formula = formula
        self.string_variables = list(string_variables)
        self.int_variables = list(int_variables)
        self.observable_terms = list(observable_terms)

    def shrink_selected_samples(
        self,
        samples: Sequence[Dict[str, Any]],
        fixed_terms: Sequence[z3.ExprRef],
        options: SamplingOptions,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        shrunk_samples: List[Dict[str, Any]] = []
        shrink_solver_checks = 0
        shrink_changed_samples = 0

        for sample in samples:
            shrunk_sample, solver_checks = self._shrink_sample(sample, fixed_terms, options)
            shrink_solver_checks += solver_checks
            if shrunk_sample != sample:
                shrink_changed_samples += 1
            shrunk_samples.append(shrunk_sample)

        return (
            shrunk_samples,
            {
                "shrink_passes": len(samples),
                "shrink_solver_checks": shrink_solver_checks,
                "shrink_changed_samples": shrink_changed_samples,
            },
        )

    def _shrink_sample(
        self,
        sample: Dict[str, Any],
        fixed_terms: Sequence[z3.ExprRef],
        options: SamplingOptions,
    ) -> Tuple[Dict[str, Any], int]:
        optimizer = z3.Optimize()
        optimizer.set(priority="lex")
        if options.timeout is not None:
            optimizer.set(timeout=max(1, int(options.timeout * 1000)))
        optimizer.add(self.formula)

        for term in fixed_terms:
            name = str(term)
            if name not in sample:
                continue
            optimizer.add(term == self._python_value_to_z3(term.sort(), sample[name]))

        for string_var in self.string_variables:
            optimizer.minimize(z3.Length(string_var))
        for int_var in self.int_variables:
            optimizer.minimize(z3.Abs(int_var))

        distinct_chars = sorted(
            {
                character
                for string_var in self.string_variables
                for character in str(sample.get(str(string_var), ""))
            }
        )
        for character in distinct_chars:
            usage_terms = [
                z3.If(z3.Contains(string_var, z3.StringVal(character)), 1, 0)
                for string_var in self.string_variables
            ]
            if usage_terms:
                optimizer.minimize(z3.Sum(*usage_terms))

        preferred_chars = _preferred_char_order(
            str(sample.get(str(string_var), "")) for string_var in self.string_variables
        )
        for string_var in self.string_variables:
            string_name = str(string_var)
            current_value = str(sample.get(string_name, ""))
            for index in range(len(current_value)):
                char_expr = z3.SubString(string_var, index, 1)
                rank_expr: z3.ArithRef = z3.IntVal(len(preferred_chars) + 1)
                for rank, character in reversed(list(enumerate(preferred_chars))):
                    rank_expr = z3.If(
                        char_expr == z3.StringVal(character),
                        z3.IntVal(rank),
                        rank_expr,
                    )
                position_expr = z3.If(
                    z3.IntVal(index) < z3.Length(string_var),
                    rank_expr,
                    z3.IntVal(0),
                )
                optimizer.minimize(position_expr)

        check_result = optimizer.check()
        if check_result != z3.sat:
            return dict(sample), 1

        model = optimizer.model()
        return build_sample(model, self.observable_terms), 1

    @staticmethod
    def _python_value_to_z3(sort: z3.SortRef, value: Any) -> z3.ExprRef:
        if sort == z3.StringSort():
            return z3.StringVal(str(value))
        if sort == z3.IntSort():
            return z3.IntVal(int(value))
        raise ValueError(f"Unsupported fixed-term sort for shrinking: {sort}")
