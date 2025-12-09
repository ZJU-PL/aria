import aria.synthesis.spyro.lexer as lexer
import aria.synthesis.spyro.generator_rule_parser as generator_rule_parser
import aria.synthesis.spyro.example_rule_parser as example_rule_parser

class SpyroParser():
    def __init__(self, template):
        # Split input code into three parts
        template, variable_section = self.__split_section_from_code(template, 'var')
        template, relation_section = self.__split_section_from_code(template, 'relation')
        template, generator_section = self.__split_section_from_code(template, 'generator')
        template, example_section = self.__split_section_from_code(template, 'example')

        self.__implenetation = template
        self.__var_decls = self.__split_var_section(variable_section)
        self.__int_decls = [(typ, symbol) for typ, symbol in self.__var_decls if typ == 'int']
        self.__relations = self.__split_relation_section(relation_section)
        self.__generators = self.__split_generator_section(generator_section)
        self.__example_generators = self.__split_example_section(example_section)

    def __split_section_from_code(self, code, section_name):
        target = section_name
        section_symbol_loc = code.find(target)
        start = code.find('{', section_symbol_loc)
        end = code.find('}', start)

        section_content = code[start+1:end]
        remainder = code[:section_symbol_loc] + code[end + 1:]

        return (remainder.strip(), section_content.strip())

    def __split_var_section(self, section_content):
        decls = [decl.strip().split() for decl in section_content.split(';')]
        return [(decl[0], decl[1]) for decl in decls[:-1]]

    def __split_relation_section(self, section_content):
        return [rel.strip() for rel in section_content.split(';')[:-1]]

    def __split_generator_section(self, section_content):
        return generator_rule_parser.parser.parse(section_content)

    def __split_example_section(self, section_content):
        return example_rule_parser.parser.parse(section_content)

    def get_int_symbols(self):
        return self.__int_decls

    def get_context(self):
        return {rule[1]:rule[0] for rule in self.__generators}

    def get_generator_rules(self):
        return self.__generators

    def get_example_rules(self):
        return self.__example_generators

    def get_implementation(self):
        return self.__implenetation + '\n\n'

    def get_arguments_defn(self):
        return ','.join([typ + ' ' + symbol for typ, symbol in self.__var_decls])

    def get_integer_arguments_defn(self):
        return ','.join([typ + ' ' + symbol for typ, symbol in self.__int_decls])

    def get_arguments_call(self):
        return ','.join([symbol for _, symbol in self.__var_decls])

    def get_integer_arguments_call(self):
        return ','.join([symbol for _, symbol in self.__int_decls])

    def get_variables_with_hole(self):
        bnds = self.get_bounds()

        def decl(typ, symbol):
            hole = f'{typ}_gen({bnds[typ]})' if bnds[typ] > 0 else f'{typ}_gen()'
            return f'\t{typ} {symbol} = {hole};'

        return '\n'.join([decl(typ, symbol) for typ, symbol in self.__var_decls])

    def get_copied_variables_with_hole(self):
        bnds = self.get_bounds()

        def decl(typ, symbol):
            hole = f'{typ}_gen({bnds[typ]})' if bnds[typ] > 0 else f'{typ}_gen()'
            return f'\t{typ} {symbol}_copy = {hole};'

        return '\n'.join([decl(typ, symbol) for typ, symbol in self.__var_decls])

    def get_relations(self):
        return '\n'.join(['\t' + rel + ';' for rel in self.__relations])

    def get_bounds(self):
        return {rule[0]:rule[2] for rule in self.__example_generators}
