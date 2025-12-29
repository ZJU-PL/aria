"""Parser for Spyro synthesis output."""

from aria.synthesis.spyro.util import (
    find_linenum_starts_with,
    find_linenum_with,
    sum_dict,
    max_dict,
    union_dict
)


def replace_last_argument(text, var):
    """Replace the last argument in a function call."""
    start = text.rfind(',')
    end = text.rfind(')')

    return text[:start] + ', out' + text[end:]


class OutputParser:
    """Parser for Spyro synthesis output."""

    def __init__(self, text):
        self.__text = text.decode('utf-8')

    def __get_function_code_lines(self, function_name):
        """Get code lines for a function."""
        lines = self.__text.splitlines()

        target = 'void ' + function_name + ' ('
        start = find_linenum_starts_with(lines, target)
        end = find_linenum_starts_with(lines, '}', start)

        return lines[start+2:end]

    def get_lam_functions(self):
        """Get lambda functions from output."""

        lines = self.__text.splitlines()

        lam_functions = {}
        for n, line in enumerate(lines):
            if ('void' in line) and ('lam' in line):
                end = find_linenum_starts_with(lines, '}', n)
                lam_code = "\n".join(lines[n:end+1])
                fun_name = line[6:line.find("(")]
                lam_functions[fun_name] = lam_code

        return lam_functions

    def parse_positive_example(self):
        """Parse positive example from output."""
        soundness_code_lines = self.__get_function_code_lines('soundness')
        soundness_code_lines = [line for line in soundness_code_lines if 'copy' not in line]
        soundness_code_lines = [line for line in soundness_code_lines if 'minimize' not in line]
        soundness_code_lines = ['\t' + line.strip() for line in soundness_code_lines]

        property_call = soundness_code_lines[-2].replace('obtained_property', 'property')

        positive_example_code = '\n'.join(soundness_code_lines[:-3])

        if property_call.find('(') >= 0:
            property_call = replace_last_argument(property_call, 'out')

            positive_example_code += '\n\tboolean out;'
            positive_example_code += '\n' + property_call
            positive_example_code += '\n\tassert out;'
        else:
            positive_example_code += '\n' + property_call
            positive_example_code += '\n\tassert out;'

        positive_example_code = positive_example_code.replace("//{}", "")

        return positive_example_code

    def parse_negative_example_precision(self):
        """Parse negative example from precision output."""

        precision_code_lines = self.__get_function_code_lines('precision')
        precision_code_lines = [line for line in precision_code_lines if 'copy' not in line]
        precision_code_lines = [line for line in precision_code_lines if 'minimize' not in line]
        precision_code_lines = ['\t' + line.strip() for line in precision_code_lines]

        negative_example_code = '\n'.join(precision_code_lines[:-9])

        if precision_code_lines[-2].find("(") >= 0:
            property_call = replace_last_argument(precision_code_lines[-2], 'out')

            negative_example_code += '\n\tboolean out;'
            negative_example_code += '\n' + property_call
            negative_example_code += '\n\tassert !out;'
        else:
            negative_example_code += '\n' + precision_code_lines[-2].replace('out_3', 'out')
            negative_example_code += '\n\tassert !out;'

        negative_example_code = negative_example_code.replace("//{}", "")

        return negative_example_code

    def parse_improves_predicate(self):
        """Parse improves predicate from output."""
        precision_code_lines = self.__get_function_code_lines('improves_predicate')
        precision_code_lines = ['\t' + line.strip() for line in precision_code_lines]

        property_call = replace_last_argument(precision_code_lines[-2], 'out')

        negative_example_code = '\n'.join(precision_code_lines[:-6])
        negative_example_code += '\n\tboolean out;'
        negative_example_code += '\n' + property_call.replace("obtained_property", "property")
        negative_example_code += '\n\tassert !out;'

        negative_example_code = negative_example_code.replace("//{}", "")

        return negative_example_code

    def parse_maxsat(self, neg_examples):
        """Parse maxsat results from output."""
        maxsat_code_lines = self.__get_function_code_lines('maxsat')
        maxsat_code = '\n'.join(maxsat_code_lines)

        used_neg_examples = []
        discarded_examples = []
        for i, e in enumerate(neg_examples):
            if f'negative_example_{i}' in maxsat_code:
                used_neg_examples.append(e)
            else:
                discarded_examples.append(e)

        return used_neg_examples, discarded_examples

    def parse_property(self):
        """Parse property from output."""

        property_code_lines = self.__get_function_code_lines('property')
        property_code_lines = ['\t' + line.strip() for line in property_code_lines]

        property_code = '\n'.join(property_code_lines)
        property_code = property_code.replace("//{}", "")

        return property_code.strip()
