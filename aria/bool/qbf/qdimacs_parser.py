"""QDIMACS parser and compatibility wrapper."""

from pathlib import Path
from typing import List, Sequence

from .model import QDIMACSInstance, QuantifierBlock


def _parse_clause_tokens(tokens: Sequence[str]) -> List[int]:
    clause = [int(token) for token in tokens]
    if not clause or clause[-1] != 0:
        raise ValueError("expected 0-terminated QDIMACS clause")
    return clause[:-1]


def _build_qdimacs_instance(
    comments: Sequence[str],
    prefix: Sequence[QuantifierBlock],
    clauses: Sequence[Sequence[int]],
    num_vars: int,
    num_clauses: int,
    normalize: bool,
) -> QDIMACSInstance:
    instance = QDIMACSInstance(
        num_vars=num_vars,
        num_clauses=num_clauses,
        prefix=[QuantifierBlock(block.kind, list(block.variables)) for block in prefix],
        clauses=[[int(literal) for literal in clause] for clause in clauses],
        comments=list(comments),
    )
    if normalize:
        instance = instance.normalized()
    instance.validate()
    return instance


def parse_qdimacs_string(content: str, normalize: bool = True) -> QDIMACSInstance:
    """Parse a QDIMACS string into a typed instance."""

    comments: List[str] = []
    prefix: List[QuantifierBlock] = []
    clauses: List[List[int]] = []
    num_vars = 0
    num_clauses = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("c"):
            comments.append(line)
            continue

        tokens = line.split()
        head = tokens[0]
        if head == "p":
            if len(tokens) != 4 or tokens[1] != "cnf":
                raise ValueError("expected 'p cnf <vars> <clauses>' header")
            num_vars = int(tokens[2])
            num_clauses = int(tokens[3])
            continue
        if head in {"a", "e"}:
            prefix.append(QuantifierBlock(head, _parse_clause_tokens(tokens[1:])))
            continue

        clauses.append(_parse_clause_tokens(tokens))

    if num_vars <= 0 and (prefix or clauses):
        inferred_vars = 0
        for block in prefix:
            inferred_vars = max(inferred_vars, *(block.variables or [0]))
        for clause in clauses:
            inferred_vars = max(inferred_vars, *(abs(lit) for lit in clause or [0]))
        num_vars = inferred_vars

    if num_clauses == 0:
        num_clauses = len(clauses)
    elif num_clauses != len(clauses):
        raise ValueError(
            f"QDIMACS clause count mismatch: header says {num_clauses}, got {len(clauses)}"
        )

    return _build_qdimacs_instance(
        comments=comments,
        prefix=prefix,
        clauses=clauses,
        num_vars=num_vars,
        num_clauses=num_clauses,
        normalize=normalize,
    )


def parse_qdimacs_file(path: str, normalize: bool = True) -> QDIMACSInstance:
    """Parse a QDIMACS file."""

    return parse_qdimacs_string(Path(path).read_text(encoding="utf-8"), normalize=normalize)


class PaserQDIMACS:
    """Backward-compatible QDIMACS parser wrapper."""

    def __init__(self, input_qbf: str):
        self.input_file = input_qbf
        self.instance = parse_qdimacs_file(input_qbf)
        self.preamble = self.instance.preamble
        self.parsed_prefix = self.instance.parsed_prefix
        self.clauses = self.instance.clause_lines
        self.all_vars = self.instance.all_vars

    def flip_and_assume(
        self, k: int, assum: Sequence[int], assertions: Sequence[Sequence[int]]
    ) -> str:
        """Flip the first ``k`` quantifier blocks and append assumptions."""

        flipped_blocks: List[QuantifierBlock] = []
        flipped_vars: List[int] = []
        for index, (kind, variables) in enumerate(self.parsed_prefix):
            if index < k:
                flipped_vars.extend(variables)
                continue
            flipped_blocks.append(QuantifierBlock(kind, variables))
        if flipped_vars:
            flipped_blocks.append(QuantifierBlock("e", flipped_vars))

        appended_clauses = self.instance.clauses + [[literal] for literal in assum] + [
            list(clause) for clause in assertions
        ]
        flipped = QDIMACSInstance(
            num_vars=self.instance.num_vars,
            num_clauses=len(appended_clauses),
            prefix=flipped_blocks,
            clauses=appended_clauses,
            comments=self.instance.comments,
        )
        return flipped.to_qdimacs()

    def sat_renumber_and_append_wrf(self, certificate, shared_vars):
        """Append certificate clauses, renumbering non-shared variables."""

        self._append_certificate(
            certificate_clauses=certificate.clauses,
            shared_vars=set(shared_vars),
            move_shared_universals=False,
        )

    def unsat_renumber_and_append_wrf(self, certificate, shared_vars):
        """Append certificate clauses and move shared universal vars inward."""

        self._append_certificate(
            certificate_clauses=certificate.clauses,
            shared_vars=set(shared_vars),
            move_shared_universals=True,
        )

    def _append_certificate(
        self,
        certificate_clauses: Sequence[Sequence[int]],
        shared_vars: set,
        move_shared_universals: bool,
    ) -> None:
        output_dir = Path("intermediate_files")
        output_dir.mkdir(parents=True, exist_ok=True)

        current_max = self.instance.num_vars
        renamed_clauses: List[List[int]] = []
        cert_vars_map = {}
        for clause in certificate_clauses:
            new_clause: List[int] = []
            for literal in clause:
                variable = abs(int(literal))
                if variable in shared_vars:
                    new_clause.append(int(literal))
                    continue
                mapped = cert_vars_map.get(variable)
                if mapped is None:
                    current_max += 1
                    mapped = current_max
                    cert_vars_map[variable] = mapped
                new_clause.append(mapped if literal > 0 else -mapped)
            renamed_clauses.append(new_clause)

        shared_universal_variables: List[int] = []
        rewritten_prefix: List[QuantifierBlock] = []
        for kind, variables in self.parsed_prefix:
            if move_shared_universals and kind == "a":
                kept = [variable for variable in variables if variable not in shared_vars]
                shared_universal_variables.extend(
                    variable for variable in variables if variable in shared_vars
                )
                if kept:
                    rewritten_prefix.append(QuantifierBlock(kind, kept))
            else:
                rewritten_prefix.append(QuantifierBlock(kind, variables))

        new_variables = list(range(self.instance.num_vars + 1, current_max + 1))
        if new_variables:
            rewritten_prefix.append(QuantifierBlock("e", new_variables))
        if shared_universal_variables:
            rewritten_prefix.append(QuantifierBlock("e", shared_universal_variables))

        appended = QDIMACSInstance(
            num_vars=current_max,
            num_clauses=self.instance.num_clauses + len(renamed_clauses),
            prefix=rewritten_prefix,
            clauses=self.instance.clauses + renamed_clauses,
            comments=self.instance.comments,
        )
        (output_dir / "appended_instance.qdimacs").write_text(
            appended.to_qdimacs(), encoding="utf-8"
        )


QDIMACSParser = PaserQDIMACS
