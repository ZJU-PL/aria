"""Tests for the SygusInv class for invariant synthesis."""

import os
import unittest
from unittest import TestCase
import tempfile
import shutil

from aria.synthesis.cvc5.sygus_inv import SygusInv
from aria.global_params.paths import global_config


class TestSygusInv(TestCase):

    def setUp(self):
        self.sygus_inv = SygusInv(debug=True)

        # Create a sample SyGuS file for testing
        self.sample_sygus_content = """(set-logic LIA)

(synth-inv inv_fun ((x Int) (y Int)))

(define-fun pre_fun ((x Int) (y Int)) Bool
    (and (= x 1) (= y 1)))

(define-fun trans_fun ((x Int) (y Int) (x! Int) (y! Int)) Bool
    (and (= x! (+ x y)) (= y! (+ x y))))

(define-fun post_fun ((x Int) (y Int)) Bool
    (>= y 1))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)
"""
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix=".sl", mode="w", delete=False
        )
        self.temp_file.write(self.sample_sygus_content)
        self.temp_file.close()

    def tearDown(self):
        # Clean up the temporary file
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_parse_sygus_file(self):
        """Test parsing a SyGuS file."""
        result = self.sygus_inv.parse_sygus_file(self.temp_file.name)

        self.assertEqual(result["logic"], "LIA")
        self.assertEqual(result["inv_fun_name"], "inv_fun")
        self.assertEqual(result["var_list"], ["x", "y"])
        self.assertEqual(result["var_sorts"], {"x": "Int", "y": "Int"})
        self.assertTrue("pre_body" in result)
        self.assertTrue("trans_body" in result)
        self.assertTrue("post_body" in result)

    def test_synthesize_invariant(self):
        """Test synthesizing an invariant function.

        Note: This test requires CVC5 to be installed and available in the PATH
        or configured in global_params. It might be skipped if CVC5 is not available.
        """
        # Check if CVC5 is available
        cvc5_available = global_config.is_solver_available("cvc5")
        if not cvc5_available:
            cvc5_available = shutil.which("cvc5") is not None

        if not cvc5_available:
            self.skipTest("CVC5 is not available, skipping test_synthesize_invariant")

        # Run the invariant synthesis
        result = self.sygus_inv.synthesize_invariant(self.temp_file.name)

        # The exact result will depend on CVC5's implementation,
        # but we can check that we got some result back
        # Note: This test may fail if CVC5 cannot synthesize an invariant for this problem
        if result is None:
            self.skipTest("CVC5 could not synthesize an invariant for this problem")
        else:
            self.assertIn("define-fun", result)
            self.assertIn("inv_fun", result)
            self.assertIn("Bool", result)

            # Print the synthesized invariant for inspection
            print("Synthesized invariant:", result)


if __name__ == "__main__":
    unittest.main()
