# coding: utf-8
"""
For testing the model counting engine
"""
import z3

from aria.global_params import global_config
from aria.tests import TestCase, main
from aria.counting.qfbv_counting import BVModelCounter


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
            count = mc.count_models_by_sharp_sat()
        else:
            print("Warning: sharpSAT not available, falling back to enumeration")
            count = mc.count_model_by_bv_enumeration()
        # Fix tuple comparison issue - both methods return (count, time)
        # count = mc.count_model_by_enumeration_parallel()
        print(count)
        count_value = count[0]  # Extract the actual count from the tuple
        self.assertTrue(count_value > 0)


if __name__ == "__main__":
    main()

"""
def feat_test():
    mc = BVModelCounter()
    x = z3.BitVec("x", 4)
    y = z3.BitVec("y", 4)
    fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 0), z3.ULT(x - y, 10))
    mc.init_from_fml(fml)
    # mc.init_from_file('../../benchmarks/t1.smt2')
    # mc.count_model_by_bv_enumeration()
    mc.count_models_by_sharp_sat()


if __name__ == '__main__':
    feat_test()
"""
