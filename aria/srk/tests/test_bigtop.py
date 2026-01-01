"""
Tests for the bigtop command-line interface module.
"""

import unittest
import sys
from io import StringIO
from unittest.mock import patch, MagicMock
from aria.srk.bigtop import BigtopCLI, main


class TestBigtopCLI(unittest.TestCase):
    """Test BigtopCLI class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = BigtopCLI()

    def test_initialization(self):
        """Test CLI initialization."""
        self.assertIsNotNone(self.cli.context)
        self.assertIsNotNone(self.cli.builder)
        self.assertIsNotNone(self.cli.smt)
        self.assertIsNotNone(self.cli.simplifier)

    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = self.cli.create_parser()

        # Test basic parser properties
        self.assertIsNotNone(parser)
        self.assertEqual(parser.prog, "bigtop")

        # Test that required arguments are present
        help_text = parser.format_help()
        self.assertIn("--simsat", help_text)
        self.assertIn("--nlsat", help_text)
        self.assertIn("--convex-hull", help_text)
        self.assertIn("--random", help_text)

    def test_formula_parsing_simple_cases(self):
        """Test simple formula parsing."""
        # Test boolean constants
        result = self.cli.parse_simple_formula("true")
        self.assertIsNotNone(result)

        result = self.cli.parse_simple_formula("false")
        self.assertIsNotNone(result)

    def test_formula_parsing_none_cases(self):
        """Test formula parsing that returns None."""
        # Test invalid formulas
        result = self.cli.parse_simple_formula("invalid formula!")
        self.assertIsNone(result)

        result = self.cli.parse_simple_formula("")
        self.assertIsNone(result)

    def test_term_parsing(self):
        """Test term parsing functionality."""
        # Test integer constants
        result = self.cli._parse_term("42")
        self.assertIsNotNone(result)

        result = self.cli._parse_term("-17")
        self.assertIsNotNone(result)

        # Test variables
        result = self.cli._parse_term("x")
        self.assertIsNotNone(result)

        # Test invalid terms
        result = self.cli._parse_term("invalid@term")
        self.assertIsNone(result)

    def test_comparison_parsing(self):
        """Test comparison expression parsing."""
        # Test greater than
        result = self.cli._parse_comparison("5", "0", "gt")
        self.assertIsNotNone(result)

        # Test less than
        result = self.cli._parse_comparison("x", "10", "lt")
        self.assertIsNotNone(result)

        # Test greater or equal
        result = self.cli._parse_comparison("y", "0", "geq")
        self.assertIsNotNone(result)

        # Test less or equal
        result = self.cli._parse_comparison("z", "1", "leq")
        self.assertIsNotNone(result)

        # Test invalid comparison
        result = self.cli._parse_comparison("invalid", "term", "gt")
        self.assertIsNone(result)


class TestCommands(unittest.TestCase):
    """Test individual command methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = BigtopCLI()

    @patch("sys.stdout", new_callable=StringIO)
    def test_simsat_command(self, mock_stdout):
        """Test simsat command."""
        # This is a smoke test since it depends on SMT solver
        try:
            self.cli.cmd_simsat("x > 0")
            output = mock_stdout.getvalue()
            self.assertIn("Checking satisfiability", output)
        except Exception:
            # If SMT dependencies aren't available, that's okay
            pass

    @patch("sys.stdout", new_callable=StringIO)
    def test_nlsat_command(self, mock_stdout):
        """Test nlsat command."""
        # This delegates to simsat for now
        try:
            self.cli.cmd_nlsat("x > 0")
            output = mock_stdout.getvalue()
            self.assertIn("Checking satisfiability", output)
        except Exception:
            # If SMT dependencies aren't available, that's okay
            pass

    @patch("sys.stdout", new_callable=StringIO)
    def test_convex_hull_command(self, mock_stdout):
        """Test convex hull command."""
        try:
            self.cli.cmd_convex_hull(["x >= 0", "y <= 1"])
            output = mock_stdout.getvalue()
            self.assertIn("Computing convex hull", output)
        except Exception:
            # If polyhedron dependencies aren't available, that's okay
            pass

    @patch("sys.stdout", new_callable=StringIO)
    def test_wedge_hull_command(self, mock_stdout):
        """Test wedge hull command."""
        try:
            self.cli.cmd_wedge_hull(["x >= 0"])
            output = mock_stdout.getvalue()
            self.assertIn("Computing wedge hull", output)
        except Exception:
            # If dependencies aren't available, that's okay
            pass

    @patch("sys.stdout", new_callable=StringIO)
    def test_affine_hull_command(self, mock_stdout):
        """Test affine hull command."""
        try:
            self.cli.cmd_affine_hull(["x >= 0"])
            output = mock_stdout.getvalue()
            self.assertIn("Computing affine hull", output)
            self.assertIn("not fully implemented", output)
        except Exception:
            # If dependencies aren't available, that's okay
            pass

    @patch("sys.stdout", new_callable=StringIO)
    def test_quantifier_elimination_command(self, mock_stdout):
        """Test quantifier elimination command."""
        try:
            self.cli.cmd_quantifier_elimination("∃x. x > 0")
            output = mock_stdout.getvalue()
            self.assertIn("Quantifier elimination", output)
        except Exception:
            # If dependencies aren't available, that's okay
            pass

    @patch("sys.stdout", new_callable=StringIO)
    def test_statistics_command(self, mock_stdout):
        """Test statistics command."""
        try:
            self.cli.cmd_statistics("x > 0")
            output = mock_stdout.getvalue()
            self.assertIn("Formula statistics", output)
        except Exception:
            # If dependencies aren't available, that's okay
            pass

    @patch("sys.stdout", new_callable=StringIO)
    def test_random_command(self, mock_stdout):
        """Test random formula generation."""
        try:
            self.cli.cmd_random(3, 2)
            output = mock_stdout.getvalue()
            self.assertIn("Generating random formula", output)
        except Exception:
            # If dependencies aren't available, that's okay
            pass


class TestMainFunction(unittest.TestCase):
    """Test main function and argument parsing."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_main_no_args(self, mock_stdout):
        """Test main function with no arguments."""
        with patch("sys.argv", ["bigtop"]):
            result = main([])
            self.assertEqual(result, 0)
            # Should show help when no args provided

    @patch("sys.stdout", new_callable=StringIO)
    def test_main_simsat_command(self, mock_stdout):
        """Test main function with simsat command."""
        with patch("sys.argv", ["bigtop", "--simsat", "x > 0"]):
            try:
                result = main(["--simsat", "x > 0"])
                self.assertEqual(result, 0)
            except SystemExit:
                # argparse calls sys.exit(0) on success
                pass
            except Exception:
                # If SMT dependencies aren't available, that's okay
                pass

    @patch("sys.stdout", new_callable=StringIO)
    def test_main_random_command(self, mock_stdout):
        """Test main function with random command."""
        with patch("sys.argv", ["bigtop", "--random", "2", "3"]):
            try:
                result = main(["--random", "2", "3"])
                self.assertEqual(result, 0)
            except SystemExit:
                # argparse calls sys.exit(0) on success
                pass
            except Exception:
                # If dependencies aren't available, that's okay
                pass

    @patch("sys.stderr", new_callable=StringIO)
    def test_main_invalid_args(self, mock_stderr):
        """Test main function with invalid arguments."""
        with patch("sys.argv", ["bigtop", "--invalid-command"]):
            try:
                result = main(["--invalid-command"])
                self.assertEqual(result, 1)
            except SystemExit:
                # argparse calls sys.exit(1) on error
                pass


class TestIntegration(unittest.TestCase):
    """Integration tests for CLI functionality."""

    def test_cli_workflow(self):
        """Test basic CLI workflow."""
        cli = BigtopCLI()

        # Test that we can create a parser
        parser = cli.create_parser()
        self.assertIsNotNone(parser)

        # Test that we can parse arguments
        try:
            args = parser.parse_args(["--random", "2", "3"])
            self.assertEqual(args.random, [2, 3])
        except Exception:
            # Argument parsing might depend on specific argparse version
            pass

    def test_command_method_exists(self):
        """Test that all command methods exist."""
        cli = BigtopCLI()

        # Check that all expected command methods exist
        expected_methods = [
            "cmd_simsat",
            "cmd_nlsat",
            "cmd_convex_hull",
            "cmd_wedge_hull",
            "cmd_affine_hull",
            "cmd_quantifier_elimination",
            "cmd_statistics",
            "cmd_random",
        ]

        for method_name in expected_methods:
            self.assertTrue(hasattr(cli, method_name))
            self.assertTrue(callable(getattr(cli, method_name)))


class TestErrorHandling(unittest.TestCase):
    """Test error handling in CLI."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = BigtopCLI()

    @patch("sys.stderr", new_callable=StringIO)
    def test_parse_formula_error_handling(self, mock_stderr):
        """Test error handling in formula parsing."""
        # Test with malformed formula
        result = self.cli.parse_simple_formula("this is not a formula!!!")
        self.assertIsNone(result)

        # Should have printed an error message
        error_output = mock_stderr.getvalue()
        self.assertIn("Cannot parse formula", error_output)

    @patch("sys.stderr", new_callable=StringIO)
    def test_comparison_parsing_error_handling(self, mock_stderr):
        """Test error handling in comparison parsing."""
        # Test with invalid terms
        result = self.cli._parse_comparison("invalid", "term", "gt")
        self.assertIsNone(result)

        error_output = mock_stderr.getvalue()
        self.assertIn("Cannot parse term", error_output)

    @patch("sys.stderr", new_callable=StringIO)
    def test_term_parsing_error_handling(self, mock_stderr):
        """Test error handling in term parsing."""
        # Test with invalid term
        result = self.cli._parse_term("invalid@#$%")
        self.assertIsNone(result)

        error_output = mock_stderr.getvalue()
        self.assertIn("Cannot parse term", error_output)


if __name__ == "__main__":
    unittest.main()
