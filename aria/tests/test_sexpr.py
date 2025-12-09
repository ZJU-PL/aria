import unittest
from aria.utils.sexpr import SExprParser


class TestSExprParser(unittest.TestCase):
    """Test cases for S-expression parser."""

    def test_tokenize(self):
        """Test tokenization of S-expressions."""
        self.assertEqual(SExprParser.tokenize("(+ 1 2)"), ['(', '+', '1', '2', ')'])
        self.assertEqual(SExprParser.tokenize("()"), ['(', ')'])
        self.assertEqual(SExprParser.tokenize("(+ 1.5 foo)"), ['(', '+', '1.5', 'foo', ')'])

    def test_parse_atoms(self):
        """Test parsing of atomic values."""
        self.assertEqual(SExprParser.parse_atom("42"), 42)
        self.assertEqual(SExprParser.parse_atom("3.14"), 3.14)
        self.assertEqual(SExprParser.parse_atom("xyz"), "xyz")

    def test_parse_expressions(self):
        """Test parsing of complete S-expressions."""
        self.assertEqual(SExprParser.parse("(+ 1 2)"), ['+', 1, 2])
        self.assertEqual(SExprParser.parse("(+ 1 (* 2 3))"), ['+', 1, ['*', 2, 3]])
        self.assertEqual(SExprParser.parse("(define x 42)"), ['define', 'x', 42])
        self.assertIsNone(SExprParser.parse(""))

    def test_parse_errors(self):
        """Test error handling for invalid expressions."""
        with self.assertRaises(SExprParser.ParseError):
            SExprParser.parse("(")  # Unclosed parenthesis
        with self.assertRaises(SExprParser.ParseError):
            SExprParser.parse(")")  # Unexpected closing parenthesis
        with self.assertRaises(SExprParser.ParseError):
            SExprParser.parse("(+ 1 2) 3")  # Trailing tokens

    def test_nested_expressions(self):
        """Test parsing of deeply nested expressions."""
        expr = "(list 1 (list 2 (list 3 4)))"
        expected = ['list', 1, ['list', 2, ['list', 3, 4]]]
        self.assertEqual(SExprParser.parse(expr), expected)

    def test_mixed_types(self):
        """Test parsing of expressions with mixed atomic types."""
        expr = "(function 42 3.14 symbol)"
        expected = ['function', 42, 3.14, 'symbol']
        self.assertEqual(SExprParser.parse(expr), expected)
        
    def test_sexpr_to_string(self):
        """Test conversion of S-expressions to strings."""
        expr = ['+', 1, ['*', 2, 3]]
        self.assertEqual(SExprParser.sexpr_to_string(expr), "(+ 1 (* 2 3))")
        
    def test_sexpr_to_json(self):
        """Test conversion of S-expressions to JSON."""
        expr = ['+', 1, ['*', 2, 3]]
        self.assertEqual(SExprParser.sexpr_to_json(expr), '["+", 1, ["*", 2, 3]]')
        
    def test_parse_sexpr_string(self):
        """Test the parse_sexpr_string method."""
        self.assertEqual(SExprParser.parse_sexpr_string("(+ 1 2)"), ['+', 1, 2])


if __name__ == '__main__':
    unittest.main()
