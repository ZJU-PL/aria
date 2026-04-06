# coding: utf-8
"""
For testing the model counting engine
"""
import z3

from aria.global_params import global_config
from aria.tests import TestCase, main
from aria.counting.bv import BVModelCounter


class TestBVCounter(TestCase):
    def test_bv_counger(self):
        mc = BVModelCounter()
        x = z3.BitVec("x", 4)
        y = z3.BitVec("y", 4)
        fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 0), z3.ULT(x - y, 10))
        mc.init_from_fml(fml)
        # Check if sharpSAT is available
        if global_config.is_solver_available("sharp_sat"):
            print("Using sharpSAT")
            result = mc.count_models(method="sharp_sat")
        else:
            print("Warning: sharpSAT not available, falling back to enumeration")
            result = mc.count_models(method="enumeration")
        print(result)
        count_value = int(result.count) if result.count is not None else -1
        self.assertTrue(count_value > 0)

    def test_bv_structured_result(self):
        mc = BVModelCounter()
        x = z3.BitVec("x", 2)
        y = z3.BitVec("y", 2)
        fml = z3.And(x == y, z3.ULE(x, 2))
        mc.init_from_fml(fml)
        result = mc.count_models(method="enumeration")
        self.assertEqual(result.status, "exact")
        self.assertTrue(result.exact)
        self.assertEqual(result.backend, "bv-enumeration")
        self.assertEqual(result.count, 3.0)
        self.assertEqual(result.metadata["num_variables"], 2)

    def test_bv_projection_result(self):
        mc = BVModelCounter()
        x = z3.BitVec("x", 2)
        y = z3.BitVec("y", 2)
        fml = z3.And(x == y, z3.ULE(x, 2))
        mc.init_from_fml(fml)
        result = mc.count_models(method="enumeration", variables=[x])
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.count, 3.0)
        self.assertEqual(result.projection, ["x"])

    def test_bv_projected_sharpsat_unsupported(self):
        mc = BVModelCounter()
        x = z3.BitVec("x", 2)
        y = z3.BitVec("y", 2)
        fml = z3.And(x == y, z3.ULE(x, 2))
        mc.init_from_fml(fml)
        result = mc.count_models(method="sharp_sat", variables=[x])
        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.count)

    def test_bv_projected_approx_unsupported(self):
        mc = BVModelCounter()
        x = z3.BitVec("x", 2)
        y = z3.BitVec("y", 2)
        fml = z3.And(x == y, z3.ULE(x, 2))
        mc.init_from_fml(fml)
        result = mc.count_models(method="approx", variables=[x])
        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.count)


if __name__ == "__main__":
    main()
