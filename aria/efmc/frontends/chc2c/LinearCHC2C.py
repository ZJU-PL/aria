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


class RecursiveException(Exception):
    pass


# As long as the CHC is nonrecursive or linear, this can be used
class LinearCHC2C(BaseCHC2C):

    def __init__(self):
        super().__init__()
        self.var_lookup = {}
        self.callgraph = {}
        self.current_rule = None

    def propagate_callgraph(self, func):
        name = (
            self.current_rule.name()
            if not isinstance(self.current_rule, str)
            else self.current_rule
        )
        if name not in self.callgraph:
            self.callgraph[name] = set()
        self.callgraph[name].add(func.name())

    def check_nonrecursive(self):
        nonlinear = any([len(calls) > 1 for calls in self.callgraph.values()])
        if nonlinear:
            todo = ["bottom"]
            visited = []
            while len(todo) > 0:
                next = todo.pop()
                visited.append(next)
                if next in self.callgraph:
                    for called in self.callgraph[next]:
                        if called in visited:
                            raise RecursiveException()
                        todo.append(called)

    def expr_to_c(self, expr, bound_vars):
        simple_expr = super().expr_to_c(expr, bound_vars)
        if simple_expr:
            return simple_expr
        elif z3.is_app(expr):
            func = expr.decl()
            self.propagate_callgraph(func)
            self.create_uf_vars(expr)
            return (
                "("
                + " && ".join(
                    [self.var_lookup[func][-1][0]]
                    + [
                        f"{self.var_lookup[func][i][0]} == {self.expr_to_c(expr.arg(i), bound_vars)}"
                        for i in range(expr.num_args())
                    ]
                )
                + ")"
            )
        else:
            raise NotImplementedError(f"Expression {expr} not supported.")

    def create_uf_vars(self, expr):
        func = expr.decl()
        if func not in self.var_lookup:
            if expr.num_args() > 0:
                self.var_lookup[func] = [
                    (f"{sanitize_identifier(str(func))}_{i}", expr.arg(i).sort())
                    for i in range(expr.num_args())
                ] + [(sanitize_identifier(str(func)), z3.BoolSort())]
            else:
                self.var_lookup[func] = [
                    (sanitize_identifier(str(func)), z3.BoolSort())
                ]

    def chc_to_c_program(self, smtlib_file_content, filename):
        # Parse the CHCs using Z3
        rules = z3.parse_smt2_string(smtlib_file_content)

        function_count = 0
        function_names = []
        functions = []
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
                    # Generate a unique function name
                    func_name = f"rule_{function_count}"
                    function_count += 1
                    # Create the function itself
                    func_body = self.create_function(body, bound_vars, func_name, rule)
                    # Append the function to the list
                    function_names.append(func_name)
                    functions.append(func_body)

        self.check_nonrecursive()

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
        c_program += (
            "\n".join([f"void {func_name}();" for func_name in function_names]) + "\n\n"
        )
        c_program += "// global variables\n"
        for vars in self.var_lookup.values():
            c_program += (
                "\n".join([f"{self.smt_sort_to_c(v[1])} {v[0]};" for v in vars])
                + "\n\n"
            )
        c_program += "// rules\n"
        c_program += "\n".join(functions) + "\n\n"
        c_program += "// main function\n"
        c_program += (
            "int main() {\n  while(1) {\n    switch(__VERIFIER_nondet_int()) {\n"
        )
        c_program += (
            "\n".join(
                [
                    f"      case {i}: {func}(); break;"
                    for i, func in enumerate(function_names)
                ]
            )
            + "\n"
        )
        c_program += "      default: abort(); break;\n    }\n  }\n}"
        return c_program

    def create_function(self, body, bound_vars, func_name, rule):
        condition = body.arg(0)  # Left-hand side of the implication
        conclusion = body.arg(1)  # Right-hand side of the implication
        if z3.is_false(conclusion):
            self.current_rule = "bottom"
        elif z3.is_app(conclusion):
            self.current_rule = conclusion.decl()
        func_body = ""
        # Add the CHC rule as a comment
        func_body += f"/*\n{rule}\n*/\n"
        # Generate function body
        func_body += f"void {func_name}() {{\n"
        # Declare the parameters as unbound variables
        func_body += (
            "  "
            + "\n  ".join(
                map(
                    lambda v: f"{self.smt_sort_to_c(v[1])} {sanitize_identifier(v[0])} = __VERIFIER_nondet_{self.smt_sort_to_c(v[1]).replace(' ', '_')}();",
                    [x for x in bound_vars if x[0] != "CHC_COMP_UNUSED"],
                )
            )
            + "\n"
        )
        # Convert the left-hand side of the implication to C (condition)
        condition_str = self.expr_to_c(condition, bound_vars)
        func_body += f"  if ({condition_str}) {{\n"
        # Convert the right-hand side (assignments) to C
        if z3.is_false(conclusion):
            func_body += f"    reach_error();\n"
        elif z3.is_app(conclusion):
            func = conclusion.decl()
            self.create_uf_vars(conclusion)
            for i in range(conclusion.num_args()):
                arg = conclusion.arg(i)
                var = self.var_lookup[func][i]
                func_body += f"    {var[0]} = {self.expr_to_c(arg, bound_vars)};\n"
            func_body += f"    {self.var_lookup[func][-1][0]} = 1;\n"

        else:
            raise NotImplementedError(f"Right-hand-side {conclusion} is not supported.")
        # Else branch calls abort()
        func_body += "  } else {\n    abort();\n  }\n"
        func_body += "}\n\n"
        return func_body
