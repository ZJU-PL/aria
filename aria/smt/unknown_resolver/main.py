"""
Main entry point for SAE Unknown Formula Resolver
"""

import sys
import argparse
from pathlib import Path
import z3

from .resolver import SAEUnknownResolver

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Resolve unknown SMT formulas using SAE mutations"
    )
    parser.add_argument("formula_file", help="Path to SMT-LIB2 formula file")
    parser.add_argument("solver_binary", help="Path to SMT solver binary")
    parser.add_argument("--timeout", type=int, default=30, 
                       help="Timeout per solver call in seconds (default: 30)")
    parser.add_argument("--max-depth", type=int, default=3,
                       help="Maximum mutation depth (default: 3)")
    parser.add_argument("--max-mutations", type=int, default=20,
                       help="Maximum mutations per formula (default: 20)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.formula_file).exists():
        print(f"Error: Formula file '{args.formula_file}' not found")
        return 1
    
    if not Path(args.solver_binary).exists():
        print(f"Error: Solver binary '{args.solver_binary}' not found")
        return 1
    
    try:
        # Parse the formula using Z3
        formula = z3.And(z3.parse_smt2_file(args.formula_file))
        if not formula:
            print("Error: No formula found in file or parsing failed")
            return 1
        
        # Verify the formula is indeed unknown with the specified solver
        resolver = SAEUnknownResolver(
            solver_binary=args.solver_binary,
            timeout=args.timeout,
            max_depth=args.max_depth,
            max_mutations_per_formula=args.max_mutations,
            verbose=args.verbose
        )
        
        original_result = resolver._solve_with_external_solver(formula)
        if original_result != "unknown":
            print(f"Formula is not unknown. Solver returned: {original_result}")
            return 1
        
        print(f"Attempting to resolve unknown formula...")
        if args.verbose:
            print(f"Formula file: {args.formula_file}")
            print(f"Solver: {args.solver_binary}")
            print(f"Max depth: {args.max_depth}")
            print(f"Max mutations: {args.max_mutations}")
            print(f"Formula: {formula}")
            print()
        
        result = resolver.resolve(formula)
        
        if result != "unknown":
            print(f"SUCCESS: Resolved result = {result}")
            return 0
        else:
            print("FAILED: Could not resolve result after exhaustive search")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
    