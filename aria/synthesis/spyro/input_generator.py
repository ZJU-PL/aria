import os
import functools
from aria.synthesis.spyro.spyro_parser import SpyroParser
from aria.synthesis.spyro.util import *

class InputGenerator:
    def __init__(self, code):
        # Input code
        self.__template = SpyroParser(code)
        self.__fresh_num = 0
        self.__num_atom = 1
        self.__minimize_terms = False

    def enable_minimize_terms(self):
        self.__minimize_terms = True

    def disable_minimize_terms(self):
        self.__minimize_terms = False

    def minimize_terms_enabled(self):
        return self.__minimize_terms

    def set_num_atom(self, num_atom):
        self.__num_atom = num_atom

    def num_atom(self):
        return self.__num_atom

    def __soundness_code(self):
        code = 'harness void soundness() {\n'
        code += self.__template.get_variables_with_hole() + '\n\n'
        code += self.__template.get_relations() + '\n\n'

        arguments = self.__template.get_arguments_call()
        int_arguments = self.__template.get_integer_arguments_call()

        code += '\tboolean out;\n'
        code += '\tobtained_property(' + arguments + ',out);\n'
        code += '\tassert !out;\n'

        code += '}\n\n'

        return code

    def __precision_code(self):
        code = 'harness void precision() {\n'
        code += self.__template.get_variables_with_hole() + '\n\n'

        arguments = self.__template.get_arguments_call()
        int_arguments = self.__template.get_integer_arguments_call()

        code += '\tboolean out_1;\n'
        code += '\tobtained_property(' + arguments + ',out_1);\n'
        code += '\tassert out_1;\n\n'

        code += '\tboolean out_2;\n'
        code += '\tproperty_conj(' + arguments + ',out_2);\n'
        code += '\tassert out_2;\n\n'

        code += '\tboolean out_3;\n'
        code += '\tproperty(' + arguments + ',out_3);\n'
        code += '\tassert !out_3;\n'
        code += '}\n\n'

        return code

    def __improves_predicate_code(self):
        code = 'harness void improves_predicate() {\n'
        code += self.__template.get_variables_with_hole() + '\n\n'

        arguments = self.__template.get_arguments_call()

        code += '\tboolean out_1;\n'
        code += '\tproperty_conj(' + arguments + ',out_1);\n'
        code += '\tassert out_1;\n\n'

        code += '\tboolean out_2;\n'
        code += '\tobtained_property(' + arguments + ',out_2);\n'
        code += '\tassert !out_2;\n'

        code += '}\n\n'

        return code

    def __property_code(self, maxsat=False):
        def property_gen_code(n):
            return ' || '.join([f'atom_{i}' for i in range(n)])

        property_gen_symbol = self.__template.get_generator_rules()[0][1]
        arg_call = self.__template.get_arguments_call()
        arg_defn = self.__template.get_arguments_defn()
        atom_gen = f'{property_gen_symbol}_gen({arg_call})'

        code = self.__generators() + '\n\n' + self.__compare() + '\n\n'
        code += f'generator boolean property_gen({arg_defn}) {{\n'

        code += '\tif (??) { return false; }\n'
        if self.__minimize_terms and not maxsat:
            code += f'\tint t = ??;\n'
            for i in range(self.__num_atom):
                property_gen = property_gen_code(i + 1)
                code += f'\tboolean atom_{i} = {atom_gen};\n'
                code += f'\tif (t == {i + 1}) {{ return {property_gen}; }}\n'
            code += f'\tminimize(t);\n'
        else:
            for i in range(self.__num_atom):
                code += f'\tboolean atom_{i} = {atom_gen};\n'
            property_gen = property_gen_code(self.__num_atom)
            code += f'\treturn {property_gen};\n'
        code += '}\n\n'

        code += f'void property({arg_defn}, ref boolean out) {{\n'
        code += f'\tout = property_gen({arg_call});\n'
        code += '}\n\n'

        return code

    def __obtained_property_code(self, phi):
        code = 'void obtained_property('
        code += self.__template.get_arguments_defn()
        code += ',ref boolean out) {\n'
        code += '\t' + phi + '\n'
        code += '}\n\n'

        return code

    def __prev_property_code(self, i, phi):
        code = f'void prev_property_{i}('
        code += self.__template.get_arguments_defn()
        code += ',ref boolean out) {\n'
        code += '\t' + phi + '\n'
        code += '}\n\n'

        return code

    def __property_conj_code(self, phi_list):
        code = ''

        for i, phi in enumerate(phi_list):
            code += self.__prev_property_code(i, phi) + '\n\n'

        code += 'void property_conj('
        code += self.__template.get_arguments_defn()
        code += ',ref boolean out) {\n'

        for i in range(len(phi_list)):
            code += f'\tboolean out_{i};\n'
            code += f'\tprev_property_{i}('
            code += self.__template.get_arguments_call()
            code += f',out_{i});\n\n'

        if len(phi_list) == 0:
            code += '\tout = true;\n'
        else:
            code += '\tout = ' + ' && '.join([f'out_{i}' for i in range(len(phi_list))]) + ';\n'
        code += '}\n\n'

        return code

    def __pos_examples(self, pos_examples):
        code = ''

        for i, pos_example in enumerate(pos_examples):
            code += '\n'
            code += 'harness void positive_example_{} ()'.format(i)
            code += ' {\n' + pos_example + '\n}\n\n'

        return code

    def __neg_examples_synth(self, neg_must_examples, neg_may_examples):
        code = ''

        i = 0
        for neg_example in neg_may_examples:
            code += '\n'
            code += 'harness void negative_example_{} ()'.format(i)
            code += ' {\n' + neg_example + '\n}\n\n'
            i += 1

        for neg_example in neg_must_examples:
            code += '\n'
            code += 'harness void negative_example_{} ()'.format(i)
            code += ' {\n' + neg_example + '\n}\n\n'
            i += 1

        return code

    def __neg_examples_maxsat(self, neg_must_examples, neg_may_examples):
        code = ''

        i = 0
        for neg_example in neg_may_examples:
            code += '\n'
            code += 'void negative_example_{} ()'.format(i)
            code += ' {\n' + neg_example + '\n}\n\n'
            i += 1

        for neg_example in neg_must_examples:
            code += '\n'
            code += 'harness void negative_example_{} ()'.format(i)
            code += ' {\n' + neg_example + '\n}\n\n'
            i += 1

        return code

    def __maxsat(self, num_neg_may):
        code = 'harness void maxsat() {\n'
        code += f'\tint cnt = {num_neg_may};\n'

        for i in range(num_neg_may):
            code += f'\tif (??) {{ cnt -= 1; negative_example_{i}(); }}\n'

        code += '\tminimize(cnt);\n'
        code += '}\n\n'

        return code

    def __model_check(self, neg_example):
        code = 'harness void model_check() {\n'

        neg_example = '\n'.join(neg_example.splitlines()[:-1])
        code += neg_example.replace('property', 'obtained_property')
        code += '\tassert out;\n'

        code += '\tboolean trivial_target = ??;\n'
        code += '\tassert trivial_target;\n'

        code += '}\n\n'

        return code

    def __count_generator_calls(self, cxt, expr):
        if expr[0] == 'BINOP':
            d1 = self.__count_generator_calls(cxt, expr[2])
            d2 = self.__count_generator_calls(cxt, expr[3])
            return sum_dict(d1, d2)
        elif expr[0] == 'UNARY':
            return self.__count_generator_calls(cxt, expr[2])
        elif expr[0] == 'INT' or expr[0] == 'HOLE':
            return dict()
        elif expr[0] == 'VAR' or expr[0] == 'TYPE':
            return {expr[1] : 1} if expr[1] in cxt else dict()
        elif expr[0] == 'FCALL':
            dicts = [self.__count_generator_calls(cxt, e) for e in expr[2]]
            return functools.reduce(sum_dict, dicts) if len(dicts) > 0 else dict()
        elif expr[0] == 'LAMBDA':
            return self.__count_generator_calls(cxt, expr[2])
        else:
            raise Exception(f'Unhandled case: {expr[0]}')

    def __subcall_gen(self, cxt, num_calls_prev, num_calls):
        arg_call = self.__template.get_arguments_call()

        code = ''
        for symbol, n in num_calls.items():
            if symbol not in num_calls_prev:
                for i in range(n):
                    code += f'\t{cxt[symbol]} var_{symbol}_{i} = {symbol}_gen({arg_call});\n'
            elif num_calls_prev[symbol] < n:
                for i in range(num_calls_prev[symbol], n):
                    code += f'\t{cxt[symbol]} var_{symbol}_{i} = {symbol}_gen({arg_call});\n'

        return code + '\n'

    def __subcall_gen_example(self, cxt, num_calls_prev, num_calls):
        arg_call = self.__template.get_arguments_call()
        bnds = self.__template.get_bounds()

        code = ''
        for symbol, n in num_calls.items():
            call_code = f'{symbol}_gen(bnd - 1)' if bnds[symbol] > 0 else f'{symbol}_gen()'
            if symbol not in num_calls_prev:
                for i in range(n):
                    code += f'\t{cxt[symbol]} var_{symbol}_{i} = {call_code};\n'
            elif num_calls_prev[symbol] < n:
                for i in range(num_calls_prev[symbol], n):
                    code += f'\t{cxt[symbol]} var_{symbol}_{i} = {call_code};\n'

        return code + '\n'

    def __fresh_variable(self):
        n = self.__fresh_num
        self.__fresh_num += 1

        return f'var_{n}'

    def __compare(self):
        code = 'generator boolean compare(int x, int y) {\n'

        # This seems more efficient than the regex style
        code += '\tint t = ??;\n'
        code += '\tif (t == 0) { return x == y; }\n'
        code += '\tif (t == 1) { return x <= y; }\n'
        code += '\tif (t == 2) { return x >= y; }\n'
        code += '\tif (t == 3) { return x < y; }\n'
        code += '\tif (t == 4) { return x > y; }\n'
        code += '\treturn x != y; \n'

        code += '}'

        return code

    def __expr_to_code(self, cxt, expr, out_type = 'boolean'):
        if expr[0] == 'BINOP':
            cxt1, code1, out1 = self.__expr_to_code(cxt, expr[2])
            cxt2, code2, out2 = self.__expr_to_code(cxt1, expr[3])
            return (cxt2, code1 + code2, f'{out1} {expr[1]} {out2}')
        elif expr[0] == 'UNARY':
            cxt, code, out = self.__expr_to_code(cxt, expr[2])
            return (cxt, code, f'{expr[1]} {out}')
        elif expr[0] == 'INT':
            return (cxt, '', expr[1])
        elif expr[0] == 'VAR' or expr[0] == 'TYPE':
            symbol = expr[1]
            if symbol in cxt:
                count = cxt[symbol]
                cxt[symbol] += 1
                return (cxt, '', f'var_{symbol}_{count}')
            else:
                return (cxt, '', symbol)
        elif expr[0] == 'HOLE':
            code = '??' if expr[1] == 0 else f'??({expr[1]})'
            return (cxt, '', code)
        elif expr[0] == 'FCALL':
            code = ''
            args = []
            for e in expr[2]:
                cxt, code_sub, out_sub = self.__expr_to_code(cxt, e)
                code += code_sub
                args.append(out_sub)

            if (expr[1] == 'compare'):
                args_call = ','.join(args)
                return (cxt, code, f'{expr[1]}({args_call})')
            else:
                fresh_var = self.__fresh_variable()
                args_call = ','.join(args + [fresh_var])
                code += f'\t\t{out_type} {fresh_var};\n'
                code += f'\t\t{expr[1]}({args_call});\n'
                return (cxt, code, fresh_var)
        elif expr[0] == 'LAMBDA':
            cxt, code, out = self.__expr_to_code(cxt, expr[2])
            return (cxt, code, f'({expr[1]}) -> {out}')
        else:
            raise Exception(f'Unhandled case: {expr}')

    def __rule_to_code(self, rule):
        typ = rule[0]
        symbol = rule[1]
        exprlist = rule[2]

        cxt = self.__template.get_context()
        num_calls_prev = dict()

        arg_defn = self.__template.get_arguments_defn()

        code = f'generator {typ} {symbol}_gen({arg_defn}) {{\n'
        code += '\tint t = ??;\n'

        for n, e in enumerate(exprlist):
            num_calls = self.__count_generator_calls(cxt, e)
            code += self.__subcall_gen(cxt, num_calls_prev, num_calls)
            num_calls_prev = max_dict(num_calls_prev, num_calls)

            cxt_init = {k:0 for k in cxt.keys()}
            _, e_code, e_out = self.__expr_to_code(cxt_init, e, typ)

            if (n + 1 == len(exprlist)):
                code += e_code
                code += f'\treturn {e_out};\n'
            else:
                code += f'\tif (t == {n}) {{\n'
                code += e_code
                code += f'\t\treturn {e_out};\n'
                code += '\t}\n'

        code += '}\n'

        return code

    def __generators(self):
        rules = self.__template.get_generator_rules()

        return '\n'.join([self.__rule_to_code(rule) for rule in rules]) + '\n'

    def __example_rule_to_code(self, rule):
        typ = rule[0]
        exprlist = rule[1]
        bnd = rule[2]

        cxt = {typ:typ for (typ, _, _) in self.__template.get_example_rules()}
        num_calls_prev = dict()

        if bnd > 0:
            code = f'generator {typ} {typ}_gen(int bnd) {{\n'
            code += '\tassert bnd > 0;\n'
        else:
            code = f'generator {typ} {typ}_gen() {{\n'
        code += '\tint t = ??;\n'

        for n, e in enumerate(exprlist):
            num_calls = self.__count_generator_calls(cxt, e)
            code += self.__subcall_gen_example(cxt, num_calls_prev, num_calls)
            num_calls_prev = max_dict(num_calls_prev, num_calls)

            cxt_init = {k:0 for k in cxt.keys()}
            _, e_code, e_out = self.__expr_to_code(cxt_init, e, typ)

            if (n + 1 == len(exprlist)):
                code += e_code
                code += f'\treturn {e_out};\n'
            else:
                code += f'\tif (t == {n}) {{\n'
                code += e_code
                code += f'\t\treturn {e_out};\n'
                code += '\t}\n'

        code += '}\n'

        return code

    def __example_generators(self):
        rules = self.__template.get_example_rules()
        code = '\n'.join([self.__example_rule_to_code(rule) for rule in rules]) + '\n'

        return code

    def __lam_functions(self, lam_functions):
        return "\n\n".join(lam_functions.values()) + "\n\n"

    def generate_synthesis_input(self, pos, neg_must, neg_may, lam_functions):
        code = self.__template.get_implementation()
        code += self.__lam_functions(lam_functions)
        code += self.__pos_examples(pos)
        code += self.__neg_examples_synth(neg_must, neg_may)
        code += self.__property_code()

        return code

    def generate_soundness_input(self, phi, lam_functions):
        code = self.__template.get_implementation()
        code += self.__lam_functions(lam_functions)
        code += self.__obtained_property_code(phi)
        code += self.__example_generators()
        code += self.__soundness_code()

        return code

    def generate_precision_input(self, phi, phi_list, pos, neg_must, neg_may, lam_functions):
        code = self.__template.get_implementation()
        code += self.__lam_functions(lam_functions)
        code += self.__pos_examples(pos)
        code += self.__neg_examples_synth(neg_must, neg_may)
        code += self.__property_code()
        code += self.__obtained_property_code(phi)
        code += self.__property_conj_code(phi_list)
        code += self.__example_generators()
        code += self.__precision_code()

        return code

    def generate_maxsat_input(self, pos, neg_must, neg_may, lam_functions):
        code = self.__template.get_implementation()
        code += self.__lam_functions(lam_functions)
        code += self.__pos_examples(pos)
        code += self.__neg_examples_maxsat(neg_must, neg_may)
        code += self.__example_generators()
        code += self.__property_code(maxsat=True)
        code += self.__maxsat(len(neg_may))

        return code

    def generate_improves_predicate_input(self, phi, phi_list, lam_functions):
        code = self.__template.get_implementation()
        code += self.__lam_functions(lam_functions)
        code += self.__property_conj_code(phi_list)
        code += self.__obtained_property_code(phi)
        code += self.__example_generators()
        code += self.__improves_predicate_code()

        return code

    def generate_model_check_input(self, phi, neg_example, lam_functions):
        code = self.__template.get_implementation()
        code += self.__lam_functions(lam_functions)
        code += self.__obtained_property_code(phi)
        code += self.__example_generators()
        code += self.__model_check(neg_example)

        return code
