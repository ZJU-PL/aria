"""
C to CHC conversion using Eldarica

We can call third-party tools that allow for parsing C code and generating CHC formulas.
- Eldarica: https://github.com/uuverifiers/eldarica
- Seahorn
- ...?
"""

import os
import subprocess
import tempfile
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_c_to_chc(
    c_file_path: str,
    output_path: Optional[str] = None,
    timeout: int = 60,
    output_format: str = "prolog",
) -> Tuple[bool, str]:
    """
    Convert C code to CHC using Eldarica.

    Args:
        c_file_path: Path to the C source file
        output_path: Optional path for output CHC file. If None, returns CHC as string
        timeout: Timeout in seconds
        output_format: Output format - "prolog" or "smtlib"

    Returns:
        Tuple of (success: bool, result: str)
    """
    workspace_root = Path(__file__).parent.parent.parent
    eldarica_path = workspace_root / "bin_solvers" / "bin" / "eldarica"

    if not eldarica_path.exists():
        return False, f"Eldarica binary not found at {eldarica_path}"

    if not os.path.exists(c_file_path):
        return False, f"Input C file not found: {c_file_path}"

    try:
        # Setup output path
        temp_file = None
        if output_path is None:
            suffix = ".pl" if output_format == "prolog" else ".smt2"
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False
            )
            chc_output_path = temp_file.name
            temp_file.close()
        else:
            chc_output_path = output_path

        # Build command
        cmd = [
            str(eldarica_path),
            "-p" if output_format == "prolog" else "-sp",
            "-lbe",
            c_file_path,
        ]

        logger.debug("Running: %s", " ".join(cmd))

        # Run Eldarica
        process = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )

        if process.returncode != 0:
            error_msg = f"Eldarica failed with return code {process.returncode}"
            if process.stderr:
                error_msg += f"\nStderr: {process.stderr}"
            return False, error_msg

        stdout = process.stdout

        # Handle SMT-LIB format edge cases
        if output_format == "smtlib" and stdout.strip() in [
            "SAFE",
            "UNSAFE",
            "UNKNOWN",
        ]:
            # Try Prolog format and convert to SMT-LIB comments
            cmd_prolog = [str(eldarica_path), "-p", "-lbe", c_file_path]
            process2 = subprocess.run(
                cmd_prolog, capture_output=True, text=True, timeout=timeout, check=False
            )

            if process2.returncode == 0 and "System transitions:" in process2.stdout:
                stdout = f"; CHC from {c_file_path}\n; Result: {stdout.strip()}\n"
                stdout += "\n".join(f"; {line}" for line in process2.stdout.split("\n"))
            else:
                stdout = f"; Result: {stdout.strip()}\n; C file: {c_file_path}\n"

        # Write output
        with open(chc_output_path, "w", encoding="utf-8") as f:
            f.write(stdout)

        if output_path is None:
            result = stdout
            if temp_file:
                os.unlink(chc_output_path)
            return True, result
        return True, chc_output_path

    except subprocess.TimeoutExpired:
        return False, f"Eldarica timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Error during conversion: {str(e)}"
    finally:
        if temp_file and os.path.exists(chc_output_path) and output_path is None:
            try:
                os.unlink(chc_output_path)
            except Exception:
                pass


def get_chc_from_c(
    c_file_path: str, timeout: int = 60, output_format: str = "prolog"
) -> Optional[str]:
    """Get CHC content as string from C code."""
    success, result = convert_c_to_chc(c_file_path, None, timeout, output_format)
    return result if success else None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        chc_content = get_chc_from_c(sys.argv[1])
        print("CHC Content:" if chc_content else "Failed to convert C to CHC")
        if chc_content:
            print(chc_content)
    else:
        print("Usage: python c2chc.py <c_file>")
