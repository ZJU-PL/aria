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

import textwrap

import z3

from src.BaseCHC2C import BaseCHC2C, sanitize_identifier


# This works for all CHCs, linear, nonlinear, recursive, etc.
class NonLinearCHC2C(BaseCHC2C):

    def __init__(self):
        super().__init__()
        self.var_lookup = {}

    def expr_to_c(self, expr, bound_vars):
        simple_expr = super().expr_to_c(expr, bound_vars)
        if simple_expr:
            return simple_expr
        elif z3.is_app(expr):
            func = expr.decl()
            ret = sanitize_identifier(func.name()) + "("
            ret += ", ".join(
                [
                    self.expr_to_c(expr.arg(i), bound_vars)
                    for i in range(expr.num_args())
                ]
            )
            ret += ")"
            return ret
        else:
            raise NotImplementedError(f"Expression {expr} not supported.")

    def chc_to_c_program(self, smtlib_file_content, filename):
        # Parse the CHCs using Z3
        rules = z3.parse_smt2_string(smtlib_file_content)

        function_declarations = []
        UFs = {}  # name -> bodies
        for rule in rules:
            # Check if this is a forall with an implication
            if z3.is_quantifier(rule) and rule.is_forall():
                # Extract variables and body
                bound_vars = [
                    (
                        rule.var_name(rule.num_vars() - i - 1),
                        rule.var_sort(rule.num_vars() - i - 1),
                    )
                    for i in range(rule.num_vars())
                ]
                body = rule.body()

                if z3.is_implies(body):
                    conclusion = body.arg(1)  # Right-hand side of the implication
                    func_name = (
                        "__chc_false"
                        if z3.is_false(conclusion)
                        else conclusion.decl().name()
                    )
                    if func_name not in UFs:
                        UFs[func_name] = (conclusion, [])
                        if z3.is_false(conclusion):
                            function_declarations.append("_Bool __chc_false();")
                        elif z3.is_app(conclusion):
                            func = conclusion.decl()
                            function_declarations.append(
                                f"_Bool {sanitize_identifier(str(func))}({", ".join([self.smt_sort_to_c(conclusion.arg(i).sort()) for i in range(conclusion.num_args())])});"
                            )
                    UFs[func_name][1].append((rule, bound_vars))

        functions = [
            self.create_function(conclusion, bodies)
            for head, (conclusion, bodies) in UFs.items()
        ]

        c_program = textwrap.dedent(
            f"""
        extern char __VERIFIER_nondet_char();
        extern short __VERIFIER_nondet_short();
        extern int __VERIFIER_nondet_int();
        extern long __VERIFIER_nondet_long();
        extern long long __VERIFIER_nondet_long_long();
        extern _Bool __VERIFIER_nondet__Bool();
        extern float __VERIFIER_nondet_float();
        extern double __VERIFIER_nondet_double();
        extern long double __VERIFIER_nondet_long_double();
        extern void abort(void);
        extern void __assert_fail(const char *, const char *, unsigned int, const char *);
        void reach_error() {{ __assert_fail("0", "{filename}", 0, "reach_error"); }}
        """
        )
        c_program += "// function declarations\n"
        c_program += "\n".join(function_declarations) + "\n\n"
        c_program += "// global variables\n"
        for vars in self.var_lookup.values():
            c_program += (
                "\n".join([f"{self.smt_sort_to_c(v[1])} {v[0]};" for v in vars])
                + "\n\n"
            )
        c_program += "// rules\n"
        c_program += "\n".join(functions) + "\n\n"
        c_program += "// main function\n"
        c_program += "int main() {\n  if(__chc_false()) { reach_error(); }\n}"
        return c_program

    def create_function(self, head, bodies):
        func_name = (
            "__chc_false"
            if z3.is_false(head)
            else sanitize_identifier(head.decl().name())
        )
        func_body = ""
        # Add the CHC rule as a comment
        func_body += f"/*\n{"\n".join([str(b[0]) for b in bodies])}\n*/\n"
        # Generate function body
        func_body += f"_Bool {func_name}({", ".join([self.smt_sort_to_c(head.arg(i).sort()) + f" p{i}" for i in range(head.num_args())])}) {{\n"
        multiple_rules = len(bodies) > 1
        if multiple_rules:
            func_body += "  switch(__VERIFIER_nondet_int()) {\n"
        for i, (rule, bound_vars) in enumerate(bodies):
            func_body += f"  case {i}: {{\n" if multiple_rules else "  {\n"
            modified_bound_vars = [x for x in bound_vars]
            for i, v in enumerate(rule.body().arg(1).children()):
                index = z3.get_var_index(v)
                modified_bound_vars[index] = (f"p{i}", v.sort())
            body_bound_vars = [
                bound_vars[i]
                for i in range(len(bound_vars))
                if i not in [z3.get_var_index(v) for v in rule.body().arg(1).children()]
            ]
            body = rule.body().arg(0)
            # Declare the parameters as unbound variables
            func_body += (
                "    "
                + "\n    ".join(
                    map(
                        lambda v: f"{self.smt_sort_to_c(v[1])} {sanitize_identifier(v[0])} = __VERIFIER_nondet_{self.smt_sort_to_c(v[1]).replace(' ', '_')}();",
                        [x for x in body_bound_vars if x[0] != "CHC_COMP_UNUSED"],
                    )
                )
                + "\n"
            )
            # Convert the left-hand side of the implication to C (condition)
            condition_str = self.expr_to_c(body, modified_bound_vars)
            func_body += f"    if ({condition_str}) {{ return 1; }}\n"
            if multiple_rules:
                func_body += "    break;\n"
            func_body += "  }\n"

        if multiple_rules:
            func_body += "  default: abort(); break;\n  }\n"
        func_body += "  return 0;\n"
        func_body += "}\n\n"
        return func_body
        