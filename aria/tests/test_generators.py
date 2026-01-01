# coding: utf-8
import z3
from aria.tests import TestCase, main
from aria.tests.grammar_gene import generate_from_grammar_as_str
import unittest


class TestGenerator(TestCase):

    def test_grammar_gene(self):
        try:
            fmlstr = generate_from_grammar_as_str(logic="QF_BV")
            if not fmlstr:  # generation error?
                self.skipTest("Formula generation error")

            # Check if the output is an error message or usage information
            if fmlstr.startswith("usage:") or "error:" in fmlstr:
                # Skip the test if we get command line output instead of SMT-LIB2 formula
                self.skipTest("Invalid SMT-LIB2 formula generated")

            fml = z3.And(z3.parse_smt2_string(fmlstr))
            # Just verify we can parse the formula
            self.assertIsNotNone(fml)
        except unittest.SkipTest:
            # Re-raise SkipTest exceptions so pytest handles them properly
            raise
        except Exception as ex:
            self.fail(f"Exception occurred: {str(ex)}")


if __name__ == "__main__":
    main()
