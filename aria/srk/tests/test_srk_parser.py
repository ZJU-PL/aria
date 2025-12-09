#!/usr/bin/env python3
"""
Test script for SRK mathematical expression parser.
"""

import sys
import os

# Add the srk module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'srk'))

from aria.srk.srkParse import MathParser, SMT2Parser
from aria.srk.syntax import Context

def test_srk_parser():
    """Test the SRK mathematical expression parser."""
    print("Testing SRK Mathematical Expression Parser")
    print("=" * 50)
    
    try:
        # Create parser
        parser = MathParser()
        
        # Test expressions
        test_expressions = [
            "x + y",
            "x * y + z",
            "x > 0",
            "x + y = z",
            "x + y > 0 and z < 1",
            "forall x. x > 0",
            "exists x. x + y = z",
            "x^2 + 2*x + 1",
            "if x > 0 then y else z",
        ]
        
        success_count = 0
        
        for expr in test_expressions:
            print(f"\nTesting: {expr}")
            try:
                result = parser.parse(expr)
                print(f"✅ Success: {type(result)}")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        print(f"\n=== Summary ===")
        print(f"Successfully parsed: {success_count}/{len(test_expressions)} expressions")
        
        if success_count == len(test_expressions):
            print("🎉 All SRK parser tests passed!")
        else:
            print("⚠️  Some SRK parser tests failed")
            
    except Exception as e:
        print(f"❌ Parser initialization failed: {e}")
        import traceback
        traceback.print_exc()

def test_smt2_parser():
    """Test the SMT2 parser."""
    print("\n" + "=" * 50)
    print("Testing SMT2 Parser")
    print("=" * 50)
    
    try:
        # Create parser
        parser = SMT2Parser()
        
        # Test SMT2 expressions
        test_expressions = [
            "(set-logic QF_LIA)",
            "(declare-fun x () Int)",
            "(assert (> x 0))",
            "(check-sat)",
        ]
        
        success_count = 0
        
        for expr in test_expressions:
            print(f"\nTesting: {expr}")
            try:
                result = parser.parse(expr)
                print(f"✅ Success: {type(result)}")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        print(f"\n=== Summary ===")
        print(f"Successfully parsed: {success_count}/{len(test_expressions)} expressions")
        
        if success_count == len(test_expressions):
            print("🎉 All SMT2 parser tests passed!")
        else:
            print("⚠️  Some SMT2 parser tests failed")
            
    except Exception as e:
        print(f"❌ SMT2 parser initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_srk_parser()
    test_smt2_parser()
