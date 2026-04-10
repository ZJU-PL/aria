"""
LLM4Inv Demo: Guess-and-Check Invariant Inference
"""

import logging
import os
import time
from typing import Dict, Any
import z3
from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.llm4inv.bv.llm4inv_prover import LLM4InvProver

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_counter_example() -> TransitionSystem:
    x = z3.BitVec("x", 32)
    x_prime = z3.BitVec("x!", 32)
    init = x == z3.BitVecVal(0, 32)
    trans = z3.Or(
        z3.And(z3.ULT(x, z3.BitVecVal(100, 32)), x_prime == x + z3.BitVecVal(1, 32)),
        z3.And(z3.UGE(x, z3.BitVecVal(100, 32)), x_prime == x),
    )
    post = z3.ULE(x, z3.BitVecVal(100, 32))
    return TransitionSystem(variables=[x], prime_variables=[x_prime], init=init, trans=trans, post=post)


def create_array_bounds_example() -> TransitionSystem:
    i, n = z3.BitVec("i", 32), z3.BitVec("n", 32)
    i_prime, n_prime = z3.BitVec("i!", 32), z3.BitVec("n!", 32)
    init = z3.And(i == z3.BitVecVal(0, 32), z3.UGT(n, z3.BitVecVal(0, 32)))
    trans = z3.And(
        z3.Or(
            z3.And(z3.ULT(i, n), i_prime == i + z3.BitVecVal(1, 32)),
            z3.And(z3.UGE(i, n), i_prime == i),
        ),
        n_prime == n,
    )
    post = z3.ULE(i, n)
    return TransitionSystem(variables=[i, n], prime_variables=[i_prime, n_prime], init=init, trans=trans, post=post)


def run_demo(sts: TransitionSystem, name: str, **kwargs) -> Dict[str, Any]:
    logger.info("Running %s with LLM4InvProver", name)
    prover = LLM4InvProver(sts, **kwargs)
    start = time.time()
    result = prover.solve()
    stats = prover.get_statistics()
    stats.update({"example_name": name, "solve_time": time.time() - start})
    logger.info("✓ %s: %s" if result.is_safe else "✗ %s: Failed", name,
                 result.invariant if result.is_safe else "")
    return stats


def main():
    logger.info("Starting LLM4Inv Demo")
    config = {
        "timeout": 300, "max_iterations": 5, "max_candidates_per_iter": 3,
        "bit_width": 32, "llm_provider": "local", "llm_model": "gpt-5.4",
        "temperature": 0.1, "max_output_length": 4096, "measure_cost": False,
        "local_provider": "tingly", "local_base_url": "http://localhost:12580/tingly/openai",
        "local_api_key": os.environ.get("TINGLY_API_KEY", ""), "local_max_retries": 3,
    }

    examples = [("Counter", create_counter_example()), ("Array Bounds", create_array_bounds_example())]
    results = []

    for name, sts in examples:
        logger.info("\n%s", "=" * 50)
        logger.info("Testing: %s", name)
        try:
            results.append(run_demo(sts, name, **config))
        except Exception as e:
            logger.error("Error in %s: %s", name, e)

    logger.info("\n%s", "=" * 50)
    logger.info("DEMO SUMMARY")
    for stats in results:
        success = stats.get("success", False) or stats.get("is_safe", False)
        logger.info("%s: %s (%.2fs)", stats.get("example_name", "Unknown"),
                    "✓ SUCCESS" if success else "✗ FAILED", stats.get("solve_time", 0))


if __name__ == "__main__":
    main()
