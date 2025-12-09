# coding: utf-8
"""
For testing the knowledge compilation engine
"""

from aria.tests import TestCase, main
from aria.bool.knowledge_compiler.dimacs_parser import parse_cnf_string
from aria.bool.knowledge_compiler.obdd import BDD_Compiler

cnf_foo3 = """
p cnf 4 4\n
1 2 3 0\n
-2 3 4 0\n
1 -4 0\n
2 3 -4 0
"""


class TestOBDD(TestCase):

    def test_obdd(self):
        clausal_form, nvars = parse_cnf_string(cnf_foo3, True)
        print(clausal_form)
        # Using separator as key
        print('================================================')
        print('Using separator as key')
        compiler = BDD_Compiler(nvars, clausal_form)
        obdd = compiler.compile(key_type='separator')
        obdd.print_info(nvars)

        # Using cutset as key
        print('================================================')
        print('Using cutset as key')
        compiler = BDD_Compiler(nvars, clausal_form)
        obdd = compiler.compile(key_type='cutset')
        obdd.print_info(nvars)

        print('End')


if __name__ == '__main__':
    main()
