/**
 * Shared-context batched satisfiability helpers implemented with the Z3 C++ API.
 *
 * The routines mirror the Python implementations in aria.scsat.cores.*:
 * - Unary checks (per-constraint satisfiability)
 * - Disjunctive over-approximation
 * - Conjunctive checks with unsat-core guided splitting
 *
 * Result encoding follows the Python code and tests:
 *   - 1 -> satisfiable
 *   - 0 -> unsatisfiable
 *   - 2 -> unknown (e.g., solver returned unknown)
 *
 * All expressions must live in the same Z3 context.
 */
#pragma once

#include <string>
#include <vector>

#include <z3++.h>

namespace aria::scsat::app {

enum class CheckResult {
    Unsat = 0,
    Sat = 1,
    Unknown = 2,
};

using ResultVector = std::vector<CheckResult>;

// Unary checks
ResultVector unary_check(const z3::expr& precond, const std::vector<z3::expr>& constraints);
ResultVector unary_check_incremental(const z3::expr& precond, const std::vector<z3::expr>& constraints);
ResultVector unary_check_cached(const z3::expr& precond, const std::vector<z3::expr>& constraints);
ResultVector unary_check_incremental_cached(const z3::expr& precond, const std::vector<z3::expr>& constraints);

// Disjunctive checks (over-approximate with caching)
ResultVector disjunctive_check_cached(const z3::expr& precond, const std::vector<z3::expr>& constraints);
ResultVector disjunctive_check_incremental_cached(const z3::expr& precond, const std::vector<z3::expr>& constraints);

// Conjunctive checks with unsat-core guided splitting
ResultVector conjunctive_check(const z3::expr& precond,
                               const std::vector<z3::expr>& constraints,
                               int algorithm = 0);
ResultVector conjunctive_check_incremental(const z3::expr& precond,
                                           const std::vector<z3::expr>& constraints,
                                           int algorithm = 0);

// Small helper for printing/debugging
std::string to_string(CheckResult result);

}  // namespace aria::scsat::app
