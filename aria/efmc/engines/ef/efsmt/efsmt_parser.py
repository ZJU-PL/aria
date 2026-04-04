"""
Parser for EFSMT (Exists-Forall SMT) formulas in SMT-LIB2 format
We assume that the formula is of the form
    forall Y. Phi(X, Y), where X is the set of existentially quantified
    variables, and Y is the set of universally quantified variables.
"""

from typing import Tuple, List
import z3
from aria.utils.z3.expr import get_variables


class ParserError(Exception):
    """Custom exception for parsing errors"""


class EFSMTParser:
    """Parser for EFSMT (Exists-Forall SMT) formulas in SMT-LIB2 format"""

    def __init__(self):
        self.logic = None
        self.supported_logics = ["BV", "LIA", "LRA", "BIA", "NRA"]

    def set_logic(self, logic: str) -> None:
        """Set the SMT logic to use for parsing

        Args:
            logic: SMT-LIB2 logic name
        Raises:
            ValueError: If logic is not supported
        """
        if logic not in self.supported_logics:
            raise ValueError(
                f"Unsupported logic: {logic}. Supported: {self.supported_logics}"
            )
        self.logic = logic

    def parse_smt2_string(
        self, inputs: str
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef], z3.ExprRef]:
        """Parse EFSMT formula from string

        Args:
            inputs: SMT-LIB2 formula string
        Returns:
            Tuple of (exists_vars, forall_vars, body)
        Raises:
            ParserError: If parsing fails
        """
        try:
            fml_vec = z3.parse_smt2_string(inputs)
            if len(fml_vec) == 1:
                fml = fml_vec[0]
            else:
                fml = z3.And(fml_vec)
            return self._ground_quantifier(fml)
        except z3.Z3Exception as e:
            raise ParserError(f"Failed to parse SMT2 string: {str(e)}") from e

    def parse_smt2_file(
        self, filename: str
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef], z3.ExprRef]:
        """Parse EFSMT formula from file

        Args:
            filename: Path to SMT-LIB2 file
        Returns:
            Tuple of (exists_vars, forall_vars, body)
        Raises:
            ParserError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        try:
            fml_vec = z3.parse_smt2_file(filename)
            if len(fml_vec) == 1:
                fml = fml_vec[0]
            else:
                fml = z3.And(fml_vec)
            return self._ground_quantifier(fml)
        except z3.Z3Exception as e:
            raise ParserError(f"Failed to parse {filename}: {str(e)}") from e
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {filename}") from exc

    def _ground_quantifier(
        self, qexpr: z3.ExprRef
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef], z3.ExprRef]:
        """Ground a quantified formula by extracting exists/forall variables
        FIXME: this function canbe very slow?

        Args:
            qexpr: Quantified Z3 expression
        Returns:
            Tuple of (exists_vars, forall_vars, body)
        Raises:
            ParserError: If expression is not properly quantified
        """
        if not z3.is_quantifier(qexpr):
            raise ParserError("Expression must be quantified")

        body = qexpr.body()
        forall_vars = []

        try:
            for i in range(qexpr.num_vars()):
                vi_name = qexpr.var_name(i)
                vi_sort = qexpr.var_sort(i)
                vi = z3.Const(vi_name, vi_sort)
                forall_vars.append(vi)

            body = z3.substitute_vars(body, *forall_vars)
            exists_vars = [x for x in get_variables(body) if x not in forall_vars]
            return exists_vars, forall_vars, body
        except Exception as e:
            raise ParserError(f"Failed to ground quantifier: {str(e)}") from e


def test_parse_multiple_formulas():
    """Test parsing multiple formulas from SMT2 string."""
    smt2_input = """
(declare-fun x1 () (_ BitVec 8))
(declare-fun x2 () (_ BitVec 8))
(declare-fun x3 () (_ BitVec 8))
(assert
    (forall ((y1 (_ BitVec 8)) (y2 (_ BitVec 8)))
        (=> (and (bvult y1 y2)
                 (bvugt y2 x1))
            (or (= (bvand x1 y1) x2)
                (and (bvule (bvadd x2 y2) x3)
                     (= (bvmul y1 x3) y2))))))
    """
    parser = EFSMTParser()
    exists_vars, forall_vars, body = parser.parse_smt2_string(smt2_input)
    print(exists_vars, forall_vars, body)


if __name__ == "__main__":
    test_parse_multiple_formulas()
