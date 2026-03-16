import json
import re
import logging
from typing import List, Optional, Tuple
import z3
from aria.efmc.sts import TransitionSystem

logger = logging.getLogger(__name__)


def extract_bit_width_from_sts(sts: TransitionSystem) -> int:
    for var in sts.variables:
        if z3.is_bv_sort(var.sort()):
            return var.sort().size()
    return 32


class InvariantPromptManager:
    def __init__(self, sts: TransitionSystem, bit_width: Optional[int] = None):
        self.sts = sts
        self.bit_width = bit_width or extract_bit_width_from_sts(sts)
        self.variables = [str(var) for var in self.sts.variables]

    def generate_invariant_prompt(self, program_code: str = "",
                                   failed_candidates: Optional[List[str]] = None,
                                   max_candidates: int = 5) -> str:
        failed = ""
        if failed_candidates:
            failed = "Avoid these:\n" + "\n".join(f"- {c}" for c in failed_candidates)

        prompt = f"""You are a program verification expert. Propose up to {max_candidates} \
inductive invariant candidates. Use only: {", ".join(self.variables)}.

Program:
```
{program_code}
```

{failed}

Return JSON with "candidates": ["z3 expressions"]. Examples:
- x == z3.BitVecVal(0, 32)
- z3.ULE(x, z3.BitVecVal(100, 32))
- z3.And(z3.UGE(x, z3.BitVecVal(0, 32)), z3.ULE(x, z3.BitVecVal(255, 32)))

Or SMT-LIB: (= x #x00000000), (bvule x #x00000064)
"""
        return prompt

    def parse_invariant_response(self, response: str) -> List[Tuple[str, z3.ExprRef]]:
        try:
            json_str = self._extract_json_object(response)
            if json_str is None:
                logger.warning("No JSON in response: %s...", response[:100])
                return []
            parsed = json.loads(json_str)
            candidate_strs = parsed.get("candidates", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Parse failed: %s", e)
            return []

        return [(s, self._parse_candidate_to_z3(s)) for s in candidate_strs
                if self._parse_candidate_to_z3(s) is not None]

    def _extract_json_object(self, response: str) -> Optional[str]:
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fenced:
            return fenced.group(1)

        start = response.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(response[start:], start):
            if in_string:
                escape = not escape and ch == "\\"
                if ch == '"' and not escape:
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return response[start:idx + 1]
        return None

    def _parse_candidate_to_z3(self, s: str) -> Optional[z3.ExprRef]:
        try:
            s = s.strip()
            var_map = {str(var): var for var in self.sts.variables}
            safe_globals = {"z3": z3, "__builtins__": {}, **var_map}
            expr = eval(s, safe_globals)
            if isinstance(expr, z3.ExprRef):
                return expr
        except (NameError, SyntaxError, TypeError, AttributeError):
            pass

        try:
            if s.startswith(("(", "=", "#") or s.isdigit()):
                decls = [(f"(declare-fun {var} () (_ BitVec {var.sort().size()}))"
                          if z3.is_bv_sort(var.sort())
                          else f"(declare-fun {var} () Int)")
                         for var in self.sts.variables]
                expr = z3.parse_smt2_string("\n".join(decls) + f"\n(assert {s})\n")
                if hasattr(expr, "__len__") and len(expr) > 0:
                    return expr[0]
        except (z3.Z3Exception, AttributeError):
            pass
        return None

    def create_prompt_generator(self, **kwargs):
        return lambda: self.generate_invariant_prompt(**kwargs)

    def create_response_parser(self):
        return self.parse_invariant_response
