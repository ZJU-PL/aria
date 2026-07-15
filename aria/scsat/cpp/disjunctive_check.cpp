// Disjunctive over-approximation helpers.
#include "scsat.hpp"

#include <optional>
#include <vector>

#include <z3++.h>

#include "detail.hpp"

namespace aria::scsat::app {
namespace {

using detail::ensure_same_context;
using detail::make_disjunction;
using detail::model_true;

namespace {
std::vector<z3::expr> collect_unknown_conditions(const std::vector<z3::expr>& constraints,
                                                  const ResultVector& results) {
    std::vector<z3::expr> conditions;
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (results[i] == CheckResult::Unknown) {
            conditions.push_back(constraints[i]);
        }
    }
    return conditions;
}

void update_results_from_disjunction(z3::check_result res, z3::solver& solver,
                                     const std::vector<z3::expr>& constraints, ResultVector& results) {
    if (res == z3::unsat) {
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i] == CheckResult::Unknown) {
                results[i] = CheckResult::Unsat;
            }
        }
    } else if (res == z3::sat) {
        const auto model = solver.get_model();
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i] == CheckResult::Unknown && model_true(model, constraints[i])) {
                results[i] = CheckResult::Sat;
            }
        }
    }
}
}  // namespace

void compact_check_cached(const z3::expr& precond,
                          const std::vector<z3::expr>& constraints,
                          ResultVector& results) {
    auto conditions = collect_unknown_conditions(constraints, results);
    if (conditions.empty()) return;

    auto& ctx = precond.ctx();
    const z3::expr disj = make_disjunction(ctx, conditions);
    if (z3::eq(disj, ctx.bool_val(false))) return;

    z3::solver solver(ctx);
    solver.add(precond && disj);
    const auto res = solver.check();
    if (res == z3::unknown) return;
    update_results_from_disjunction(res, solver, constraints, results);
    compact_check_cached(precond, constraints, results);
}

void compact_check_incremental_cached(z3::solver& solver,
                                      const std::vector<z3::expr>& constraints,
                                      ResultVector& results) {
    auto conditions = collect_unknown_conditions(constraints, results);
    if (conditions.empty()) return;

    auto& ctx = solver.ctx();
    const z3::expr disj = make_disjunction(ctx, conditions);
    if (z3::eq(disj, ctx.bool_val(false))) return;

    solver.push();
    solver.add(disj);
    const auto res = solver.check();
    if (res == z3::unknown) {
        solver.pop();
        return;
    }
    update_results_from_disjunction(res, solver, constraints, results);
    solver.pop();
    compact_check_incremental_cached(solver, constraints, results);
}

}  // namespace

ResultVector disjunctive_check_cached(const z3::expr& precond, const std::vector<z3::expr>& constraints) {
    ensure_same_context(precond, constraints);
    ResultVector results(constraints.size(), CheckResult::Unknown);
    compact_check_cached(precond, constraints, results);
    return results;
}

ResultVector disjunctive_check_incremental_cached(const z3::expr& precond,
                                                  const std::vector<z3::expr>& constraints) {
    ensure_same_context(precond, constraints);
    ResultVector results(constraints.size(), CheckResult::Unknown);
    z3::solver solver(precond.ctx());
    solver.add(precond);
    compact_check_incremental_cached(solver, constraints, results);
    return results;
}

}  // namespace aria::scsat::app
