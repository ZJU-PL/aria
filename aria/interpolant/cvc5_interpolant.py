"""CVC5-based interpolant synthesis module."""
import logging
import os
import subprocess
import tempfile
from typing import Optional
import pysmt.parsing
from pysmt.fnode import FNode
from z3 import ExprRef
from aria.global_params import global_config

logger = logging.getLogger(__name__)


class CVC5InterpolantSynthesizer:
    """CVC5-based interpolant synthesizer."""
    SOLVER_NAME = "cvc5"

    def __init__(self, timeout: int = 300, verbose: bool = False) -> None:
        self.timeout = timeout
        self.verbose = verbose
        cvc5_path = global_config.get_solver_path("cvc5")
        if not cvc5_path or not os.path.exists(cvc5_path):
            raise RuntimeError("CVC5 solver not found.")
        self.cvc5_path = cvc5_path

    def interpolate(self, A: FNode, B: FNode) -> Optional[FNode]:
        """Generate an interpolant for formulas A and B."""
        if not A or not B:
            raise ValueError("Both formulas A and B must be provided")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(f"(set-logic ALL)\n{B.to_smtlib()}\n(get-interpolant A {A.to_smtlib()})\n")
            temp_file_path = temp_file.name

        try:
            if self.verbose:
                logger.debug(f"Created temporary file: {temp_file_path}")

            cmd = [self.cvc5_path, "--produce-interpolants", "--interpolants-mode=default",
                   "--sygus-enum=fast", "--check-interpolants", "--quiet", temp_file_path]

            if self.verbose:
                logger.info(f"Running CVC5: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)

            if result.returncode != 0:
                raise RuntimeError(f"CVC5 failed (code {result.returncode}): {result.stderr or ''}")

            output = result.stdout.strip()
            if not output:
                logger.warning("CVC5 returned empty output")
                return None

            if self.verbose:
                logger.info("Successfully generated interpolant")
            return pysmt.parsing.parse(output)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"CVC5 timed out after {self.timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Interpolant generation failed: {e}")
        finally:
            try:
                os.remove(temp_file_path)
                if self.verbose:
                    logger.debug(f"Removed temporary file: {temp_file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")


# Backward compatibility alias
InterpolantSynthesiser = CVC5InterpolantSynthesizer
