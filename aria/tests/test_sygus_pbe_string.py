"""
Tests for the String SyGuS PBE solver
"""

import unittest
from aria.synthesis.cvc5.sygus_pbe import StringSyGuSPBE


class TestStringSyGuSPBE(unittest.TestCase):
    """Test cases for StringSyGuSPBE"""

    def setUp(self):
        self.solver = StringSyGuSPBE(debug=True)

    def test_string_transformer(self):
        """Test synthesizing a simple string transformation function"""
        examples = [("hello", "HELLO"), ("world", "WORLD"), ("test", "TEST")]

        result = self.solver.synthesize_string_transformer(examples)
        print(f"Synthesized uppercase function: {result}")
        # The result should implement an uppercase function

    def test_string_concat(self):
        """Test synthesizing a string concatenation function"""
        examples = [
            ("hello", " world", "hello world"),
            ("abc", "def", "abcdef"),
            ("test", "ing", "testing"),
        ]

        result = self.solver.synthesize_concat_function(examples)
        print(f"Synthesized concat function: {result}")
        # The result should implement string concatenation

    def test_custom_string_function(self):
        """Test synthesizing a custom string function with multiple examples"""
        # Examples for a function that takes two strings and returns their
        # concatenation with a separator
        input_examples = [["hello", "world"], ["foo", "bar"], ["test", "case"]]
        output_examples = ["hello-world", "foo-bar", "test-case"]

        result = self.solver.synthesize_from_examples(
            input_examples, output_examples, function_name="join_with_hyphen"
        )
        print(f"Synthesized join function: {result}")
        # The result should implement a join with hyphen function


if __name__ == "__main__":
    unittest.main()
