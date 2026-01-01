"""CVC5-based interpolant synthesis module."""

import logging
import os
import subprocess
import tempfile
from typing import Optional
import pysmt.parsing
from pysmt.fnode import FNode
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

    def __repr__(self) -> str:
        """Return string representation of the synthesizer."""
        return f"CVC5InterpolantSynthesizer(timeout={self.timeout}, verbose={self.verbose})"

    def interpolate(self, formula_a: FNode, formula_b: FNode) -> Optional[FNode]:
        """Generate an interpolant for formulas A and B."""
        if not formula_a or not formula_b:
            raise ValueError("Both formulas A and B must be provided")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smt2", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(
                f"(set-logic ALL)\n{formula_b.to_smtlib()}\n"
                f"(get-interpolant A {formula_a.to_smtlib()})\n"
            )
            temp_file_path = temp_file.name

        try:
            if self.verbose:
                logger.debug("Created temporary file: %s", temp_file_path)

            cmd = [
                self.cvc5_path,
                "--produce-interpolants",
                "--interpolants-mode=default",
                "--sygus-enum=fast",
                "--check-interpolants",
                "--quiet",
                temp_file_path,
            ]

            if self.verbose:
                logger.info("Running CVC5: %s", " ".join(cmd))

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, check=False
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"CVC5 failed (code {result.returncode}): {result.stderr or ''}"
                )

            output = result.stdout.strip()
            if not output:
                logger.warning("CVC5 returned empty output")
                return None

            if self.verbose:
                logger.info("Successfully generated interpolant")
            return pysmt.parsing.parse(output)

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"CVC5 timed out after {self.timeout} seconds") from exc
        except Exception as exc:
            raise RuntimeError(f"Interpolant generation failed: {exc}") from exc
        finally:
            try:
                os.remove(temp_file_path)
                if self.verbose:
                    logger.debug("Removed temporary file: %s", temp_file_path)
            except OSError as exc:
                logger.warning(
                    "Failed to remove temporary file %s: %s", temp_file_path, exc
                )


# Backward compatibility alias
InterpolantSynthesiser = CVC5InterpolantSynthesizer
