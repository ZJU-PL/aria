import lark
from lark import Lark

from .Constraint import PolynomialConstraint
from .Convertor import convert_to_desired_poly, find_index_of_variable
from .DNF import DNF
from .PositiveModel import PositiveModel
from .Solver import Solver
from .UnknownVariable import UnknownVariable


class Parser:

    def __init__(self, model: PositiveModel):
        self.model = model

    def traverse_readable_tree(self, parse_tree):
        if parse_tree.data == 'start':
            for child in parse_tree.children:
                self.traverse_readable_tree(child)
        elif parse_tree.data == 'program_var':
            for child in parse_tree.children:
                self.model.program_variables.append(
                    UnknownVariable(str(child), type_of_var='program_var'))

        elif parse_tree.data == 'template_var':
            for child in parse_tree.children:
                self.model.template_variables.append(
                    UnknownVariable(str(child), type_of_var='template_var'))

        elif parse_tree.data == 'hornclause':
            lhs = self.traverse_readable_tree(parse_tree.children[0])
            rhs = self.traverse_readable_tree(parse_tree.children[1])
            self.model.add_paired_constraint(lhs, rhs)
            return
        elif parse_tree.data == 'precondition':
            dnf = self.traverse_readable_tree(parse_tree.children[0])
            self.model.preconditions.append(dnf)
            return
        elif parse_tree.data == 'dnf':
            if len(parse_tree.children) == 1:
                return self.traverse_readable_tree(parse_tree.children[0])
            else:
                if str(parse_tree.children[1]) == "AND":
                    return (self.traverse_readable_tree(parse_tree.children[0]) & self.traverse_readable_tree(
                        parse_tree.children[2]))
                else:
                    return (self.traverse_readable_tree(parse_tree.children[0]) | self.traverse_readable_tree(
                        parse_tree.children[2]))

        elif parse_tree.data == 'literal':
            literal = []
            for i in range(len(parse_tree.children)):
                literal.append(self.traverse_readable_tree(
                    parse_tree.children[i]))
            return literal
        elif parse_tree.data == 'constraint':
            return DNF([[PolynomialConstraint(
                self.traverse_readable_tree(parse_tree.children[0]) - self.traverse_readable_tree(
                    parse_tree.children[2]),
                str(parse_tree.children[1]))]])
        elif parse_tree.data == 'polynomial':
            return convert_to_desired_poly(self.traverse_readable_tree(parse_tree.children[0]),
                                           self.model.program_variables)
        elif parse_tree.data == 'expression':
            if len(parse_tree.children) == 1:
                return self.traverse_readable_tree(parse_tree.children[0])
            elif parse_tree.children[1] == '+':
                return self.traverse_readable_tree(parse_tree.children[0]) + self.traverse_readable_tree(
                    parse_tree.children[2])
            elif parse_tree.children[1] == '-':
                return self.traverse_readable_tree(parse_tree.children[0]) - self.traverse_readable_tree(
                    parse_tree.children[2])

        elif parse_tree.data == 'term':
            if len(parse_tree.children) == 1:
                return self.traverse_readable_tree(parse_tree.children[0])
            else:
                return self.traverse_readable_tree(parse_tree.children[0]) * self.traverse_readable_tree(
                    parse_tree.children[1])

        elif parse_tree.data == 'factor':
            if len(parse_tree.children) == 1:
                return self.traverse_readable_tree(parse_tree.children[0])
            elif parse_tree.children[0] == '-':
                return -self.traverse_readable_tree(parse_tree.children[1])
            elif parse_tree.children[0] == '+':
                return self.traverse_readable_tree(parse_tree.children[1])

        elif parse_tree.data == 'primary':
            if not parse_tree.children[0].__class__ is lark.Token:
                return self.traverse_readable_tree(parse_tree.children[0])
            elif parse_tree.children[0].type == 'RATIONALNUMBER':
                return Solver.get_constant_polynomial(self.model.template_variables + self.model.program_variables,
                                                      str(parse_tree.children[0]))
            else:
                deg = 1
                if len(parse_tree.children) > 1:
                    deg = int(parse_tree.children[1])
                degrees = [
                    0] * len(self.model.template_variables + self.model.program_variables)
                degrees[find_index_of_variable(str(parse_tree.children[0]),
                                               self.model.template_variables + self.model.program_variables)] = deg
                return Solver.get_degree_polynomial(self.model.template_variables + self.model.program_variables,
                                                    degrees)

    def parse_readable_file(self, poly_text: str):
        parser = Lark(r"""
                start : program_var template_var precondition* hornclause*

                program_var : "Program_var:" VAR* ";"

                template_var : "Template_var:" VAR* ";"

                precondition : "Precondition:" dnf

                hornclause : "Horn_clause:" dnf "->" dnf

                dnf : constraint | "(" dnf ")" | dnf LOGICAL_SIGN dnf


                constraint : polynomial COMP_SIGN polynomial 
                polynomial : expression
                expression : term | expression SIGN term 

                term : factor | term "*" factor

                factor : primary | SIGN factor

                primary : VAR | RATIONALNUMBER | VAR "^" RATIONALNUMBER  | "(" expression ")"

                LOGICAL_SIGN : "AND" | "OR" 
                COMP_SIGN : ">" | "=" | "<" | ">=" | "<="
                SIGN : "+" | "-" 
                VAR: /[a-zA-Z0-9_]/+
                RATIONALNUMBER : /[+-]?/ NUMBER ("/" NUMBER)?


                %import common.NUMBER
                %import common.NEWLINE -> _NL
                %import common.WS_INLINE
                %import common.WS
                %ignore WS
            """, parser="lalr")

        parse_tree = parser.parse(poly_text)
        return self.traverse_readable_tree(parse_tree)

    def traverse_smt_tree(self, parse_tree):
        if parse_tree.data == 'start':
            for child in parse_tree.children:
                self.traverse_smt_tree(child)

        elif parse_tree.data == 'instructions':
            self.model.instructions.append(parse_tree.children[0])
            return
        elif parse_tree.data == 'declare_var':
            self.model.template_variables.append(
                UnknownVariable(str(parse_tree.children[0]), type_of_var='template_var'))

        elif parse_tree.data == 'assertion':
            self.traverse_smt_tree(parse_tree.children[0])
        elif parse_tree.data == 'hornclause':
            self.model.program_variables = []
            for i in range(len(parse_tree.children) - 2):
                self.traverse_smt_tree(parse_tree.children[i])
            lhs = self.traverse_smt_tree(parse_tree.children[-2])
            rhs = self.traverse_smt_tree(parse_tree.children[-1])
            self.model.add_paired_constraint(
                lhs, rhs, self.model.program_variables)
            return
        elif parse_tree.data == 'program_variables':
            self.model.program_variables.append(UnknownVariable(
                str(parse_tree.children[0]), type_of_var='program_var'))
        elif parse_tree.data == 'precondition':
            dnf = self.traverse_smt_tree(parse_tree.children[0])
            if len(parse_tree.children) == 2:
                second_dnf = self.traverse_smt_tree(parse_tree.children[1])
                self.model.preconditions.append((dnf, second_dnf))
            else:
                self.model.preconditions.append((dnf,))
            return
        elif parse_tree.data == 'dnf':
            if len(parse_tree.children) == 1:
                return self.traverse_smt_tree(parse_tree.children[0])
            else:
                if str(parse_tree.children[0]) == "and":
                    result_dnf = DNF([])
                    for i in range(1, len(parse_tree.children)):
                        result_dnf = result_dnf & self.traverse_smt_tree(
                            parse_tree.children[i])
                    return result_dnf
                else:
                    result_dnf = DNF([])
                    for i in range(1, len(parse_tree.children)):
                        result_dnf = result_dnf | self.traverse_smt_tree(
                            parse_tree.children[i])
                    return result_dnf

        elif parse_tree.data == 'constraint':
            if parse_tree.children[0] == '=':
                return DNF(
                    [[
                        PolynomialConstraint(
                            self.traverse_smt_tree(parse_tree.children[1])
                            -
                            self.traverse_smt_tree(parse_tree.children[2]),
                            '>='),
                        PolynomialConstraint(
                            self.traverse_smt_tree(parse_tree.children[1])
                            -
                            self.traverse_smt_tree(parse_tree.children[2]),
                            '<='),

                    ]])
            return DNF(
                [[
                    PolynomialConstraint(
                        self.traverse_smt_tree(parse_tree.children[1])
                        -
                        self.traverse_smt_tree(parse_tree.children[2]),
                        str(parse_tree.children[0]))]])
        elif parse_tree.data == 'polynomial':
            return convert_to_desired_poly(self.traverse_smt_tree(parse_tree.children[0]), self.model.program_variables)
        elif parse_tree.data == 'expression':

            if len(parse_tree.children) == 1:
                if parse_tree.children[0].data == "fraction":
                    return Solver.get_constant_polynomial(self.model.template_variables + self.model.program_variables,
                                                          self.traverse_smt_tree(
                                                              parse_tree.children[0])
                                                          )
                return self.traverse_smt_tree(parse_tree.children[0])
            elif len(parse_tree.children) == 2:
                if str(parse_tree.children[0]) == '+':
                    return self.traverse_smt_tree(parse_tree.children[1])
                elif str(parse_tree.children[0]) == '-':
                    return -self.traverse_smt_tree(parse_tree.children[1])
            elif len(parse_tree.children) == 3:

                if str(parse_tree.children[0]) == '+':
                    return self.traverse_smt_tree(parse_tree.children[1]) + self.traverse_smt_tree(
                        parse_tree.children[2])
                elif str(parse_tree.children[0]) == '-':
                    return self.traverse_smt_tree(parse_tree.children[1]) - self.traverse_smt_tree(
                        parse_tree.children[2])
                elif str(parse_tree.children[0]) == '*':
                    return self.traverse_smt_tree(parse_tree.children[1]) * self.traverse_smt_tree(
                        parse_tree.children[2])
            else:
                poly = self.traverse_smt_tree(parse_tree.children[1])
                for i in range(2, len(parse_tree.children)):
                    if str(parse_tree.children[0]) == '+':
                        poly = poly + \
                            self.traverse_smt_tree(parse_tree.children[i])
                    else:
                        poly = poly * \
                            self.traverse_smt_tree(parse_tree.children[i])
                return poly
        elif parse_tree.data == 'primary':
            if type(parse_tree.children[0]) is lark.tree.Tree:
                return Solver.get_constant_polynomial(self.model.template_variables + self.model.program_variables,
                                                      self.traverse_smt_tree(parse_tree.children[0]))

            if parse_tree.children[0].type == 'VAR':
                deg = 1
                if len(parse_tree.children) > 1:
                    deg = int(parse_tree.children[1])
                degrees = [
                    0] * len(self.model.template_variables + self.model.program_variables)
                degrees[find_index_of_variable(str(parse_tree.children[0]),
                                               self.model.template_variables + self.model.program_variables)] = deg
                return Solver.get_degree_polynomial(self.model.template_variables + self.model.program_variables,
                                                    degrees)

        elif parse_tree.data == 'fraction':
            return str(self.traverse_smt_tree(parse_tree.children[0])) + '/' + str(
                self.traverse_smt_tree(parse_tree.children[1]))
        elif parse_tree.data == 'rationalnumber':
            if len(parse_tree.children) == 1:
                return str(parse_tree.children[0])
            if len(parse_tree.children) == 2:
                return str(parse_tree.children[0]) + str(parse_tree.children[1])

    def parse_smt_file(self, poly_text: str):
        parser = Lark(r"""
                start : declare_var* assertion* instructions* 

                instructions: INS
                INS : "(check-sat)" | "(get-model)"
                declare_var: "(declare-const" VAR VAR_TYPE ")"

                assertion: "(assert" precondition  ")" | "(assert" hornclause  ")"
                precondition : dnf | "(=>" dnf dnf ")" 

                hornclause : "(forall" "(" program_variables* ")" "(=>" dnf dnf ")" ")"
                program_variables : "(" VAR VAR_TYPE ")" 
                dnf : constraint | "(" LOGICAL_SIGN dnf+ ")" 


                constraint : "(" COMP_SIGN polynomial polynomial ")" 
                polynomial : expression
                expression : "(/" fraction ")" | primary | "(" SIGN expression ")" | "(" SIGN expression+ ")"

                primary : VAR | rationalnumber 

                LOGICAL_SIGN : "and" | "or" 
                COMP_SIGN : ">" | "=" | "<" | ">=" | "<="
                SIGN : "+" | "-" | "*"
                VAR: /[a-zA-Z0-9_]/+
                VAR_TYPE: "Int" | "Real" 
                rationalnumber : NUMBER | "(" SIGN NUMBER ")" | SIGN NUMBER 
                fraction: rationalnumber  rationalnumber 

                %import common.NUMBER
                %import common.NEWLINE -> _NL
                %import common.WS_INLINE
                %import common.WS
                %ignore WS
            """, parser="lalr")

        parse_tree = parser.parse(poly_text)
        return self.traverse_smt_tree(parse_tree)
