"""SyGuS invariant synthesis utilities."""

import re
from typing import Dict, Optional, Any
from aria.utils.z3_plus_smtlib_solver import Z3SolverPlus


class SygusInv:
    """Solve SyGuS invariant synthesis problems (loop invariants)."""

    def __init__(self, debug: bool = False):
        """Initialize with optional debug output."""
        self.debug = debug
        self.solver = Z3SolverPlus(debug=debug)

    @staticmethod
    def _extract_define_fun_body(content: str, name: str) -> str:
        """Extract body of a define-fun named `name` returning Bool."""
        # Try pattern followed by another define-fun
        m = re.search(rf"\(define-fun\s+{name}\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)\s*\(define-fun",
                      content, re.DOTALL)
        if m:
            return m.group(2).strip()
        # Try standalone closing
        m = re.search(rf"\(define-fun\s+{name}\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)(?:\s*\n|\s*$)",
                      content, re.DOTALL)
        if m:
            return m.group(2).strip()
        # Try trailing before any following paren for post
        m = re.search(rf"\(define-fun\s+{name}\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)\s*\(",
                      content, re.DOTALL)
        if m:
            return m.group(2).strip()
        raise ValueError(f"Could not find {name} in the file")

    def parse_sygus_file(self, filepath: str) -> Dict[str, Any]:
        """Parse .sl SyGuS file and extract logic, names, and function bodies."""
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract logic
        logic = None
        logic_match = re.search(r'\(set-logic\s+([A-Z_]+)\)', content)
        if logic_match:
            logic = logic_match.group(1)

        # Extract function name and variables
        synth_inv_match = re.search(r'\(synth-inv\s+([a-zA-Z0-9_]+)\s+\(\((.*?)\)\)', content, re.DOTALL)
        if not synth_inv_match:
            raise ValueError("Could not find synth-inv declaration in the file")

        inv_fun_name = synth_inv_match.group(1)
        vars_decl = synth_inv_match.group(2)

        # Parse variable declarations
        var_list = []
        var_sorts = {}

        for var_decl in re.findall(r'([a-zA-Z0-9_]+)\s+([^)]+)', vars_decl):
            var_name, sort_str = var_decl
            var_list.append(var_name)
            var_sorts[var_name] = sort_str.strip()

        # Extract bodies
        pre_body = self._extract_define_fun_body(content, "pre_fun")
        trans_body = self._extract_define_fun_body(content, "trans_fun")
        post_body = self._extract_define_fun_body(content, "post_fun")

        return {
            'logic': logic,
            'inv_fun_name': inv_fun_name,
            'var_list': var_list,
            'var_sorts': var_sorts,
            'pre_body': pre_body,
            'trans_body': trans_body,
            'post_body': post_body
        }

    def synthesize_invariant(self, filepath: str) -> Optional[str]:
        """Synthesize invariant for given file; return SMT-LIB2 or None."""
        # Parse the SyGuS file
        problem = self.parse_sygus_file(filepath)

        # Build full SyGuS text
        var_decls = ' '.join(f"({v} {problem['var_sorts'][v]})" for v in problem['var_list'])
        pre_args = var_decls
        trans_args = ' '.join([
            *[f"({v} {problem['var_sorts'][v]})" for v in problem['var_list']],
            *[f"({v}! {problem['var_sorts'][v]})" for v in problem['var_list']],
        ])
        parts = [
            f"(set-logic {problem['logic']})",
            f"(synth-inv {problem['inv_fun_name']} ({var_decls}))",
            f"(define-fun pre_fun ({pre_args}) Bool\n    {problem['pre_body']})",
            f"(define-fun trans_fun ({trans_args}) Bool\n    {problem['trans_body']})",
            f"(define-fun post_fun ({pre_args}) Bool\n    {problem['post_body']})",
            f"(inv-constraint {problem['inv_fun_name']} pre_fun trans_fun post_fun)",
            "(check-synth)",
        ]
        sygus_content = "\n\n".join(parts)

        # Use CVC5 to solve the invariant synthesis problem
        if self.debug:
            print("Generated SyGuS problem:")
            print(sygus_content)

        result = self._invoke_cvc5_sygus_inv(sygus_content, problem['logic'])

        if self.debug:
            print("SyGuS result:", result)

        return result

    def _invoke_cvc5_sygus_inv(self, sygus_content: str, logic: str) -> Optional[str]:
        """Invoke cvc5 on the SyGuS text and return synthesized invariant or None."""
        import tempfile
        import subprocess
        import os

        cvc5_path = self._get_cvc5_path()

        with tempfile.NamedTemporaryFile(suffix='.sl', mode='w', delete=False) as tmp:
            tmp.write(sygus_content)
            tmp_path = tmp.name

        try:
            cmd = [cvc5_path, "--lang=sygus2", "--produce-models", tmp_path]
            if self.debug:
                print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                if self.debug:
                    print(f"cvc5 error ({result.returncode}):\n{result.stderr}")
                return None
            output = result.stdout.strip()
            if self.debug:
                print("cvc5 output:\n" + output)
            if "unsat" in output.lower():
                return None
            inv_match = re.search(r'\(define-fun\s+([a-zA-Z0-9_]+)\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)',
                                  output, re.DOTALL)
            if inv_match:
                inv_fun_name = inv_match.group(1)
                inv_args = inv_match.group(2)
                inv_body = inv_match.group(3).strip()
                return f"(define-fun {inv_fun_name} (({inv_args})) Bool {inv_body})"
            return output
        except subprocess.TimeoutExpired:
            if self.debug:
                print("cvc5 timeout expired")
            return None
        except Exception as e:
            if self.debug:
                print(f"Error invoking cvc5: {e}")
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _get_cvc5_path(self) -> str:
        """Resolve cvc5 binary path from global config or PATH fallback."""
        from aria.global_params.paths import global_config
        cvc5_path = global_config.get_solver_path("cvc5")
        if not cvc5_path:
            if self.debug:
                print("CVC5 not found in global config. Using 'cvc5' from PATH.")
            cvc5_path = "cvc5"
        return cvc5_path
