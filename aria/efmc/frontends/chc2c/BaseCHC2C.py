#   Copyright 2025 Budapest University of Technology and Economics
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import re

import z3
from z3 import BoolSortRef


# As long as the CHC is nonrecursive, this can be used
def sanitize_identifier(identifier):
    # Ensure the first character is a letter or underscore.
    identifier = re.sub(r"^[^A-Za-z_]", "_", identifier)
    # Replace any character that is not a letter, digit, or underscore.
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", identifier)
    return identifier


class BaseCHC2C:

    def __init__(self):
        self.bitvectors = {
            1: "_Bool",
            8: "char",
            16: "short",
            32: "int",
            64: "long",
            1: "_Bool",
        }
        self.floats = {
            (8, 24): "float",
            (11, 53): "double",
            (15, 113): "long double",
        }
        self.types_lookup = {
            z3.Z3_BOOL_SORT: lambda _: "_Bool",
            z3.Z3_INT_SORT: lambda _: "int",
            z3.Z3_BV_SORT: lambda z3sort: self.bitvectors[z3sort.size()],
            z3.Z3_FLOATING_POINT_SORT: lambda z3sort: self.floats[
                (z3sort.ebits(), z3sort.sbits())
            ],
        }

    def get_cast(self, expr, signed: bool = False) -> str:
        if isinstance(expr.sort(), BoolSortRef):
            size = 1
        else:
            size = expr.sort().size()
        c_type = self.bitvectors[size]
        return f"({'signed' if signed else 'unsigned'} {c_type})"

    def expr_to_c(self, expr, bound_vars):
        """
        Recursively convert a Z3 expression to a C expression using AST traversal.
        We ensure that parentheses are used where necessary to preserve operator precedence.
        `bound_vars` is a list of variable names corresponding to bound variables in the quantifier.
        """

        def casted(expr, e, signed=False):
            return f"({self.get_cast(expr, signed)}{e})"

        if z3.is_int_value(expr):
            return str(expr.as_long())
        elif z3.is_true(expr):
            return "1"
        elif z3.is_false(expr):
            return "0"
        elif z3.is_var(expr):  # Bound variable (e.g., Var(0), Var(1), etc.)
            idx = z3.get_var_index(expr)
            if idx < len(bound_vars):
                return sanitize_identifier(bound_vars[idx][0])
            else:
                raise ValueError(
                    f"Variable index {idx} out of bounds for bound variables {bound_vars}"
                )
        elif z3.is_add(expr):
            return (
                "("
                + " + ".join(self.expr_to_c(arg, bound_vars) for arg in expr.children())
                + ")"
            )
        elif z3.is_sub(expr):
            return (
                "("
                + " - ".join(self.expr_to_c(arg, bound_vars) for arg in expr.children())
                + ")"
            )
        elif z3.is_mul(expr):
            return (
                "("
                + " * ".join(self.expr_to_c(arg, bound_vars) for arg in expr.children())
                + ")"
            )
        elif z3.is_div(expr):
            return (
                "("
                + " / ".join(self.expr_to_c(arg, bound_vars) for arg in expr.children())
                + ")"
            )
        elif z3.is_eq(expr):
            return (
                "("
                + self.expr_to_c(expr.arg(0), bound_vars)
                + " == "
                + self.expr_to_c(expr.arg(1), bound_vars)
                + ")"
            )
        elif z3.is_le(expr):
            return (
                "("
                + self.expr_to_c(expr.arg(0), bound_vars)
                + " <= "
                + self.expr_to_c(expr.arg(1), bound_vars)
                + ")"
            )
        elif z3.is_ge(expr):
            return (
                "("
                + self.expr_to_c(expr.arg(0), bound_vars)
                + " >= "
                + self.expr_to_c(expr.arg(1), bound_vars)
                + ")"
            )
        elif z3.is_lt(expr):
            return (
                "("
                + self.expr_to_c(expr.arg(0), bound_vars)
                + " < "
                + self.expr_to_c(expr.arg(1), bound_vars)
                + ")"
            )
        elif z3.is_gt(expr):
            return (
                "("
                + self.expr_to_c(expr.arg(0), bound_vars)
                + " > "
                + self.expr_to_c(expr.arg(1), bound_vars)
                + ")"
            )
        elif z3.is_and(expr):
            return (
                "("
                + " && ".join(
                    self.expr_to_c(arg, bound_vars) for arg in expr.children()
                )
                + ")"
            )
        elif z3.is_or(expr):
            return (
                "("
                + " || ".join(
                    self.expr_to_c(arg, bound_vars) for arg in expr.children()
                )
                + ")"
            )
        elif z3.is_not(expr):
            return "(!" + self.expr_to_c(expr.arg(0), bound_vars) + ")"
        elif z3.is_mod(expr):
            # TODO: exact mod behavior
            return (
                "("
                + self.expr_to_c(expr.arg(0), bound_vars)
                + " % "
                + self.expr_to_c(expr.arg(1), bound_vars)
                + ")"
            )
        elif z3.is_app(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            return None  # the subclass should handle this
        elif z3.is_app(expr):
            kind = expr.decl().kind()
            if kind == z3.Z3_OP_UMINUS:
                return "(-" + self.expr_to_c(expr.arg(0), bound_vars) + ")"
            elif kind == z3.Z3_OP_ITE:
                return (
                    "("
                    + self.expr_to_c(expr.arg(0), bound_vars)
                    + " ? "
                    + self.expr_to_c(expr.arg(1), bound_vars)
                    + " : "
                    + self.expr_to_c(expr.arg(2), bound_vars)
                    + ")"
                )
            elif kind == z3.Z3_OP_IDIV:
                return (
                    "("
                    + " / ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + ")"
                )
            elif kind == z3.Z3_OP_BNUM:
                return casted(expr, expr.as_long(), signed=True)
            elif kind == z3.Z3_OP_BIT1:
                return "1"
            elif kind == z3.Z3_OP_BIT0:
                return "0"
            elif kind == z3.Z3_OP_BNEG:
                return f"(-{casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), signed=True)})"
            elif kind == z3.Z3_OP_BADD:
                return (
                    "("
                    + " + ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + ")"
                )
            elif kind == z3.Z3_OP_BSUB:
                return (
                    "("
                    + " - ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + ")"
                )
            elif kind == z3.Z3_OP_BMUL:
                return (
                    "("
                    + " * ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + ")"
                )
            elif kind in {z3.Z3_OP_BSDIV, z3.Z3_OP_BSDIV_I}:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), True)} / {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), True)})"
            elif kind in {z3.Z3_OP_BUDIV, z3.Z3_OP_BUDIV_I}:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), False)} / {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), False)})"
            elif kind in {z3.Z3_OP_BSREM, z3.Z3_OP_BSREM_I}:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), True)} % {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), True)})"
            elif kind in {z3.Z3_OP_BUREM, z3.Z3_OP_BUREM_I}:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), False)} % {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), False)})"
            elif kind in {z3.Z3_OP_BSMOD, z3.Z3_OP_BSMOD_I}:
                a = casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), True)
                b = casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), True)
                return f"/* smod */ (({a} % {b} + {b}) % {b})"
            elif kind == z3.Z3_OP_BAND:
                return (
                    "("
                    + " & ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + ")"
                )
            elif kind == z3.Z3_OP_BOR:
                return (
                    "("
                    + " | ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + ")"
                )
            elif kind == z3.Z3_OP_BXOR:
                return (
                    "("
                    + " ^ ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + ")"
                )
            elif kind == z3.Z3_OP_BNOT:
                return f"(~{self.expr_to_c(expr.arg(0), bound_vars)})"
            elif kind == z3.Z3_OP_BNAND:
                return (
                    f"(~("
                    + " & ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + "))"
                )
            elif kind == z3.Z3_OP_BNOR:
                return (
                    f"(~("
                    + " | ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + "))"
                )
            elif kind == z3.Z3_OP_BXNOR:
                return (
                    f"(~("
                    + " ^ ".join(
                        self.expr_to_c(arg, bound_vars) for arg in expr.children()
                    )
                    + "))"
                )
            elif kind == z3.Z3_OP_CONCAT:
                left = casted(expr, self.expr_to_c(expr.arg(0), bound_vars))
                right = casted(expr, self.expr_to_c(expr.arg(1), bound_vars))
                return f"/* concat */ {left} << {expr.arg(1).size()}) | ({right})"
            elif kind == z3.Z3_OP_SIGN_EXT:
                return (
                    f"({casted(expr, self.expr_to_c(expr.arg(0), bound_vars), True)})"
                )
            elif kind == z3.Z3_OP_ZERO_EXT:
                return (
                    f"({casted(expr, self.expr_to_c(expr.arg(0), bound_vars), False)})"
                )
            elif kind == z3.Z3_OP_EXTRACT:
                high, low = expr.params()
                val = self.expr_to_c(expr.arg(0), bound_vars)
                mask = (1 << (high - low + 1)) - 1
                return f"(({val} >> {low}) & {mask})"
            elif kind == z3.Z3_OP_BCOMP:
                return f"({self.expr_to_c(expr.arg(0), bound_vars)} == {self.expr_to_c(expr.arg(1), bound_vars)})"
            elif kind == z3.Z3_OP_BSHL:
                return f"({self.expr_to_c(expr.arg(0), bound_vars)} << {self.expr_to_c(expr.arg(1), bound_vars)})"
            elif kind == z3.Z3_OP_BLSHR:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), False)} >> {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), False)})"
            elif kind == z3.Z3_OP_BASHR:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), True)} >> {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), False)})"
            elif kind == z3.Z3_OP_ROTATE_LEFT:
                val = self.expr_to_c(expr.arg(0), bound_vars)
                amt = self.expr_to_c(expr.arg(1), bound_vars)
                size = expr.arg(0).size()
                return f"(({val} << {amt}) | ({val} >> ({size} - {amt})))"
            elif kind == z3.Z3_OP_ROTATE_RIGHT:
                val = self.expr_to_c(expr.arg(0), bound_vars)
                amt = self.expr_to_c(expr.arg(1), bound_vars)
                size = expr.arg(0).size()
                return f"(({val} >> {amt}) | ({val} << ({size} - {amt})))"
            elif kind == z3.Z3_OP_CARRY:
                a = self.expr_to_c(expr.arg(0), bound_vars)
                b = self.expr_to_c(expr.arg(1), bound_vars)
                c = self.expr_to_c(expr.arg(2), bound_vars)
                return f"((({a} & {b}) | (({a} | {b}) & {c})) != 0)"
            elif kind == z3.Z3_OP_XOR3:
                a = self.expr_to_c(expr.arg(0), bound_vars)
                b = self.expr_to_c(expr.arg(1), bound_vars)
                c = self.expr_to_c(expr.arg(2), bound_vars)
                return f"(({a} ^ {b}) ^ {c})"
            elif kind == z3.Z3_OP_ULEQ:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), False)} <= {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), False)})"
            elif kind == z3.Z3_OP_SLEQ:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), True)} <= {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), True)})"
            elif kind == z3.Z3_OP_UGEQ:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), False)} >= {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), False)})"
            elif kind == z3.Z3_OP_SGEQ:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), True)} >= {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), True)})"
            elif kind == z3.Z3_OP_ULT:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), False)} < {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), False)})"
            elif kind == z3.Z3_OP_SLT:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), True)} < {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), True)})"
            elif kind == z3.Z3_OP_UGT:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), False)} > {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), False)})"
            elif kind == z3.Z3_OP_SGT:
                return f"({casted(expr.arg(0), self.expr_to_c(expr.arg(0), bound_vars), True)} > {casted(expr.arg(1), self.expr_to_c(expr.arg(1), bound_vars), True)})"

            elif kind == z3.Z3_OP_FPA_NEG:
                return f"-{self.expr_to_c(expr.arg(0), bound_vars)}"
            elif kind == z3.Z3_OP_FPA_ADD:
                return f"({self.expr_to_c(expr.arg(1), bound_vars)} + {self.expr_to_c(expr.arg(2), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_SUB:
                return f"({self.expr_to_c(expr.arg(1), bound_vars)} - {self.expr_to_c(expr.arg(2), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_MUL:
                return f"({self.expr_to_c(expr.arg(1), bound_vars)} * {self.expr_to_c(expr.arg(2), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_DIV:
                return f"({self.expr_to_c(expr.arg(1), bound_vars)} / {self.expr_to_c(expr.arg(2), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_EQ:
                return f"({self.expr_to_c(expr.arg(0), bound_vars)} == {self.expr_to_c(expr.arg(1), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_LT:
                return f"({self.expr_to_c(expr.arg(0), bound_vars)} < {self.expr_to_c(expr.arg(1), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_GT:
                return f"({self.expr_to_c(expr.arg(0), bound_vars)} > {self.expr_to_c(expr.arg(1), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_LE:
                return f"({self.expr_to_c(expr.arg(0), bound_vars)} <= {self.expr_to_c(expr.arg(1), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_GE:
                return f"({self.expr_to_c(expr.arg(0), bound_vars)} >= {self.expr_to_c(expr.arg(1), bound_vars)})"

            elif kind == z3.Z3_OP_FPA_NUM:
                return str(expr.as_decimal(17))  # or as_string() if preferred

            elif kind == z3.Z3_OP_FPA_PLUS_INF:
                return "INFINITY"
            elif kind == z3.Z3_OP_FPA_MINUS_INF:
                return "-INFINITY"
            elif kind == z3.Z3_OP_FPA_NAN:
                return "NAN"
            elif kind == z3.Z3_OP_FPA_PLUS_ZERO:
                return "0.0"
            elif kind == z3.Z3_OP_FPA_MINUS_ZERO:
                return "-0.0"

            elif kind == z3.Z3_OP_FPA_TO_FP:
                sort = expr.sort()
                float_type = self.floats.get(
                    (sort.ebits(), sort.sbits()),
                )
                return f"({float_type})({self.expr_to_c(expr.arg(1), bound_vars)})"

            elif kind == z3.Z3_OP_FPA_TO_FP_UNSIGNED:
                sort = expr.sort()
                float_type = self.floats.get(
                    (sort.ebits(), sort.sbits()),
                )
                return f"({float_type})({self.expr_to_c(expr.arg(1), bound_vars)})"
            elif kind == z3.Z3_OP_FPA_TO_UBV:
                return casted(
                    expr, self.expr_to_c(expr.arg(1), bound_vars), signed=False
                )
            elif kind == z3.Z3_OP_FPA_TO_SBV:
                return casted(
                    expr, self.expr_to_c(expr.arg(1), bound_vars), signed=True
                )
            elif kind == z3.Z3_OP_FPA_FP:
                sign = self.expr_to_c(expr.arg(0), bound_vars)
                exp = self.expr_to_c(expr.arg(1), bound_vars)
                sgn = self.expr_to_c(expr.arg(2), bound_vars)
                return str(
                    ieee754_from_components(
                        int(sign),
                        int(exp),
                        int(sgn),
                        expr.sort().ebits(),
                        expr.sort().sbits(),
                    )
                )
            raise NotImplementedError(
                f"Operator not implemented: {expr.decl()} (code: {kind})"
            )

    def smt_sort_to_c(self, sort):
        if sort.kind() not in self.types_lookup:
            raise NotImplementedError(f"Sort {sort} not supported.")
        return self.types_lookup[sort.kind()](sort)


def ieee754_from_components(
    sign: int, exponent: int, significand: int, ebits: int, sbits: int
) -> float:
    import struct

    total_bits = 1 + ebits + (sbits - 1)

    if total_bits == 32:
        # IEEE-754 single-precision
        raw = ((sign & 1) << 31) | ((exponent & 0xFF) << 23) | (significand & 0x7FFFFF)
        return struct.unpack(">f", struct.pack(">I", raw))[0]
    elif total_bits == 64:
        # IEEE-754 double-precision
        raw = (
            ((sign & 1) << 63)
            | ((exponent & 0x7FF) << 52)
            | (significand & 0xFFFFFFFFFFFFF)
        )
        return struct.unpack(">d", struct.pack(">Q", raw))[0]
    else:
        raise NotImplementedError("Unsupported bit width for float construction.")
