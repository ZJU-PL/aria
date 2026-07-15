// Unary satisfiability helpers.
#include "scsat.hpp"
#include <optional>
#include <vector>
#include <z3++.h>

#include "detail.hpp"

namespace aria::scsat::app {
namespace {

using detail::ensure_same_context;
using detail::all_indices;
using detail::map_check_result;
using detail::model_true;
using detail::propagate_model_sat;

}  // namespace

ResultVector unary_check(const z3::expr& precond, const std::vector<z3::expr>& constraints) {
    ensure_same_context(precond, constraints);
    ResultVector results;
    results.reserve(constraints.size());
    auto& ctx = precond.ctx();
    for (const auto& cnt : constraints) {
        z3::solver solver(ctx);
        solver.add(precond);
        solver.add(cnt);
        results.push_back(map_check_result(solver.check()));
    }
    return results;
}

ResultVector unary_check_incremental(const z3::expr& precond, const std::vector<z3::expr>& constraints) {
    ensure_same_context(precond, constraints);
    ResultVector results;
    results.reserve(constraints.size());
    z3::solver solver(precond.ctx());
    solver.add(precond);
    for (const auto& cnt : constraints) {
        solver.push();
        solver.add(cnt);
        results.push_back(map_check_result(solver.check()));
        solver.pop();
    }
    return results;
}

namespace {
void handle_check_result(z3::check_result res, z3::solver& solver, std::size_t idx,
                         const std::vector<z3::expr>& constraints,
                         std::vector<std::optional<CheckResult>>& results) {
    if (res == z3::sat) {
        const auto model = solver.get_model();
        results[idx] = CheckResult::Sat;
        propagate_model_sat(model, constraints, results, all_indices(constraints.size()));
    } else {
        results[idx] = (res == z3::unsat) ? CheckResult::Unsat : CheckResult::Unknown;
    }
}
}  // namespace

ResultVector unary_check_cached(const z3::expr& precond, const std::vector<z3::expr>& constraints) {
    ensure_same_context(precond, constraints);
    std::vector<std::optional<CheckResult>> results(constraints.size());
    auto& ctx = precond.ctx();
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        if (results[i].has_value()) continue;
        z3::solver solver(ctx);
        solver.add(precond);
        solver.add(constraints[i]);
        handle_check_result(solver.check(), solver, i, constraints, results);
    }
    return detail::finalize_results(results);
}

ResultVector unary_check_incremental_cached(const z3::expr& precond,
                                            const std::vector<z3::expr>& constraints) {
    ensure_same_context(precond, constraints);
    std::vector<std::optional<CheckResult>> results(constraints.size());
    z3::solver solver(precond.ctx());
    solver.add(precond);
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        if (results[i].has_value()) continue;
        solver.push();
        solver.add(constraints[i]);
        handle_check_result(solver.check(), solver, i, constraints, results);
        solver.pop();
    }
    return detail::finalize_results(results);
}

std::string to_string(CheckResult result) {
    constexpr const char* strs[] = {"unsat(0)", "sat(1)", "unknown(2)"};
    return strs[static_cast<int>(result)];
}

}  // namespace aria::scsat::app
