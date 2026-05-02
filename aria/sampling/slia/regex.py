"""Regex-aware feature extraction helpers for SLIA sampling."""

from typing import Any, Dict, List, Set, Tuple, cast

import z3


def _walk_exprs(formula: z3.ExprRef) -> List[z3.ExprRef]:
    """Return all reachable application nodes in deterministic DFS order."""
    stack = [formula]
    seen_ids: Set[int] = set()
    ordered: List[z3.ExprRef] = []

    while stack:
        expr = stack.pop()
        expr_id = z3.Z3_get_ast_id(expr.ctx.ref(), expr.as_ast())
        if expr_id in seen_ids:
            continue
        seen_ids.add(expr_id)
        ordered.append(expr)

        if z3.is_quantifier(expr):
            stack.append(cast(Any, expr).body())
            continue
        if not z3.is_app(expr):
            continue
        stack.extend(expr.children())

    return ordered


class RegexFeatureExtractor:
    """Extract regex-branch coverage features from concrete samples."""

    def __init__(self) -> None:
        self.regex_atoms: List[z3.BoolRef] = []
        self._regex_membership_cache: Dict[Tuple[str, str], bool] = {}

    def initialize(self, formula: z3.ExprRef) -> None:
        self.regex_atoms = self._collect_regex_atoms(formula)
        self._regex_membership_cache = {}

    def extract_features(self, sample: Dict[str, Any]) -> Set[str]:
        features: Set[str] = set()
        for atom_index, atom in enumerate(self.regex_atoms):
            string_term = atom.arg(0)
            regex_term = atom.arg(1)
            string_name = str(string_term)
            if string_name not in sample:
                continue
            string_value = sample[string_name]
            if not isinstance(string_value, str):
                string_value = str(string_value)
            label = f"regex:{atom_index}"
            features.update(
                self._regex_branch_features(label, string_value, regex_term, "root")
            )
        return features

    def _collect_regex_atoms(self, formula: z3.ExprRef) -> List[z3.BoolRef]:
        regex_atoms: List[z3.BoolRef] = []
        for expr in _walk_exprs(formula):
            if not z3.is_app(expr):
                continue
            if expr.decl().kind() == z3.Z3_OP_SEQ_IN_RE:
                regex_atoms.append(cast(z3.BoolRef, expr))
        return regex_atoms

    def _regex_branch_features(
        self,
        label: str,
        string_value: str,
        regex_term: z3.ExprRef,
        path: str,
    ) -> Set[str]:
        if not self._concrete_in_regex(string_value, regex_term):
            return set()

        kind = regex_term.decl().kind()
        features = {f"{label}:{path}:accept", f"{label}:{path}:kind={kind}"}

        if kind == z3.Z3_OP_RE_UNION:
            matched_branches: List[int] = []
            for index, child in enumerate(regex_term.children()):
                if self._concrete_in_regex(string_value, child):
                    matched_branches.append(index)
                    features.add(f"{label}:{path}:union_branch={index}")
                    features.update(
                        self._regex_branch_features(
                            label,
                            string_value,
                            child,
                            f"{path}/union[{index}]",
                        )
                    )
            features.add(f"{label}:{path}:union_count={len(matched_branches)}")
            return features

        if kind == z3.Z3_OP_RE_OPTION:
            if string_value == "":
                features.add(f"{label}:{path}:option_empty")
            child = regex_term.arg(0)
            if self._concrete_in_regex(string_value, child):
                features.add(f"{label}:{path}:option_some")
                features.update(
                    self._regex_branch_features(
                        label,
                        string_value,
                        child,
                        f"{path}/option",
                    )
                )
            return features

        if kind == z3.Z3_OP_RE_STAR:
            features.add(
                f"{label}:{path}:star={'empty' if string_value == '' else 'nonempty'}"
            )
            child = regex_term.arg(0)
            if string_value and self._concrete_in_regex(
                string_value, z3.Loop(child, 2, max(2, len(string_value)))
            ):
                features.add(f"{label}:{path}:star_multi")
            elif string_value:
                features.add(f"{label}:{path}:star_single")
            return features

        if kind == z3.Z3_OP_RE_PLUS:
            child = regex_term.arg(0)
            if self._concrete_in_regex(
                string_value, z3.Loop(child, 2, max(2, len(string_value)))
            ):
                features.add(f"{label}:{path}:plus_multi")
            else:
                features.add(f"{label}:{path}:plus_single")
            return features

        if kind == z3.Z3_OP_RE_CONCAT and len(regex_term.children()) == 2:
            left, right = regex_term.children()
            matched_splits: List[int] = []
            for split_index in range(len(string_value) + 1):
                prefix = string_value[:split_index]
                suffix = string_value[split_index:]
                if self._concrete_in_regex(prefix, left) and self._concrete_in_regex(
                    suffix, right
                ):
                    matched_splits.append(split_index)
            if matched_splits:
                features.add(f"{label}:{path}:concat_split_count={len(matched_splits)}")
                features.add(f"{label}:{path}:concat_first_split={matched_splits[0]}")
            return features

        if kind == z3.Z3_OP_SEQ_TO_RE and regex_term.num_args() == 1:
            literal = regex_term.arg(0)
            if z3.is_string_value(literal):
                literal_value = cast(Any, literal).as_string()
                features.add(f"{label}:{path}:literal_len={len(literal_value)}")
            return features

        if kind == z3.Z3_OP_RE_RANGE and regex_term.num_args() == 2:
            lower = regex_term.arg(0)
            upper = regex_term.arg(1)
            if z3.is_string_value(lower) and z3.is_string_value(upper):
                features.add(
                    f"{label}:{path}:range={cast(Any, lower).as_string()}-{cast(Any, upper).as_string()}"
                )
            return features

        return features

    def _concrete_in_regex(self, string_value: str, regex_term: z3.ExprRef) -> bool:
        cache_key = (string_value, regex_term.sexpr())
        if cache_key in self._regex_membership_cache:
            return self._regex_membership_cache[cache_key]

        solver = z3.Solver()
        solver.add(z3.InRe(z3.StringVal(string_value), regex_term))
        is_member = solver.check() == z3.sat
        self._regex_membership_cache[cache_key] = is_member
        return is_member
