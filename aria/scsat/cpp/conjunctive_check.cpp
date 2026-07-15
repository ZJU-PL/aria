// Conjunctive satisfiability helpers with unsat-core guided splitting.
#include "scsat.hpp"

#include <algorithm>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <z3++.h>

#include "detail.hpp"

namespace aria::scsat::app {
namespace {

using detail::all_indices;
using detail::ensure_same_context;
using detail::finalize_results;
using detail::indices_from_core;
using detail::make_tracking_literals;
using detail::model_true;
using detail::propagate_model_sat;

namespace {
void handle_check_result_partial(z3::check_result res, z3::solver& solver, std::size_t idx,
                                 const std::vector<z3::expr>& constraints,
                                 std::vector<std::optional<CheckResult>>& results,
                                 const std::vector<std::size_t>& check_list) {
    if (res == z3::sat) {
        const auto model = solver.get_model();
        results[idx] = CheckResult::Sat;
        propagate_model_sat(model, constraints, results, check_list);
    } else {
        results[idx] = (res == z3::unsat) ? CheckResult::Unsat : CheckResult::Unknown;
    }
}
}  // namespace

void unary_check_cached_partial(const z3::expr& precond,
                                const std::vector<z3::expr>& constraints,
                                std::vector<std::optional<CheckResult>>& results,
                                const std::vector<std::size_t>& check_list) {
    auto& ctx = precond.ctx();
    for (std::size_t idx : check_list) {
        if (results[idx].has_value()) continue;
        z3::solver solver(ctx);
        solver.add(precond);
        solver.add(constraints[idx]);
        handle_check_result_partial(solver.check(), solver, idx, constraints, results, check_list);
    }
}

void unary_check_incremental_cached_partial(z3::solver& solver,
                                            const std::vector<z3::expr>& constraints,
                                            std::vector<std::optional<CheckResult>>& results,
                                            const std::vector<std::size_t>& check_list) {
    for (std::size_t idx : check_list) {
        if (results[idx].has_value()) continue;
        solver.push();
        solver.add(constraints[idx]);
        handle_check_result_partial(solver.check(), solver, idx, constraints, results, check_list);
        solver.pop();
    }
}

void disjunctive_check_incremental_cached_partial(z3::solver& solver,
                                                  const std::vector<z3::expr>& constraints,
                                                  std::vector<std::optional<CheckResult>>& results,
                                                  const std::vector<std::size_t>& check_list) {
    std::vector<std::size_t> pending;
    std::copy_if(check_list.begin(), check_list.end(), std::back_inserter(pending),
                 [&](std::size_t idx) { return !results[idx]; });
    if (pending.empty()) return;

    std::vector<z3::expr> conditions;
    std::transform(pending.begin(), pending.end(), std::back_inserter(conditions),
                  [&](std::size_t idx) { return constraints[idx]; });

    auto& ctx = solver.ctx();
    const z3::expr disj = detail::make_disjunction(ctx, conditions);
    if (z3::eq(disj, ctx.bool_val(false))) return;

    solver.push();
    solver.add(disj);
    const auto res = solver.check();
    if (res == z3::unsat) {
        for (auto idx : pending) results[idx] = CheckResult::Unsat;
        solver.pop();
    } else if (res == z3::sat) {
        const auto model = solver.get_model();
        std::vector<std::size_t> remaining;
        for (auto idx : pending) {
            if (model_true(model, constraints[idx])) {
                results[idx] = CheckResult::Sat;
            } else {
                remaining.push_back(idx);
            }
        }
        solver.pop();
        disjunctive_check_incremental_cached_partial(solver, constraints, results, remaining);
    } else {
        for (auto idx : pending) results[idx] = CheckResult::Unknown;
        solver.pop();
    }
}

ResultVector conjunctive_check_internal(const z3::expr& precond,
                                        const std::vector<z3::expr>& constraints,
                                        int algorithm,
                                        bool reuse_split_solver) {
    ensure_same_context(precond, constraints);
    const auto n = constraints.size();
    std::vector<std::optional<CheckResult>> results(n);
    std::vector<std::size_t> waiting;
    std::deque<std::vector<std::size_t>> queue;
    queue.push_back(all_indices(n));

    auto& ctx = precond.ctx();
    const auto tracking = make_tracking_literals(ctx, n);
    z3::solver shared_split_solver(ctx);

    while (!queue.empty()) {
        auto current_subset = queue.front();
        queue.pop_front();
        if (current_subset.empty()) {
            continue;
        }

        z3::solver* split_solver_ptr = reuse_split_solver ? &shared_split_solver : nullptr;
        std::unique_ptr<z3::solver> owned_solver;
        if (!reuse_split_solver) {
            owned_solver = std::make_unique<z3::solver>(ctx);
            split_solver_ptr = owned_solver.get();
        } else {
            split_solver_ptr->push();
        }
        auto& split_solver = *split_solver_ptr;

        z3::expr_vector assumptions(ctx);
        for (auto idx : current_subset) {
            split_solver.add(z3::implies(tracking[idx], constraints[idx]));
            assumptions.push_back(tracking[idx]);
        }
        const auto res_split = split_solver.check(assumptions);

        if (res_split == z3::sat) {
            if (reuse_split_solver) {
                split_solver.pop();
            }
            auto working_subset = current_subset;
            while (!working_subset.empty()) {
                z3::solver solver_check(ctx);
                solver_check.add(precond);
                z3::expr_vector assumptions(ctx);
                for (auto idx : working_subset) {
                    solver_check.add(z3::implies(tracking[idx], constraints[idx]));
                    assumptions.push_back(tracking[idx]);
                }
                const auto check_res = solver_check.check(assumptions);
                if (check_res == z3::sat) {
                    for (auto idx : working_subset) {
                        results[idx] = CheckResult::Sat;
                    }
                    break;
                }
                if (check_res == z3::unsat) {
                    const auto core_idx = indices_from_core(solver_check.unsat_core());
                    for (auto idx : core_idx) {
                        waiting.push_back(idx);
                        auto it = std::find(working_subset.begin(), working_subset.end(), idx);
                        if (it != working_subset.end()) {
                            working_subset.erase(it);
                        }
                    }
                } else {
                    for (auto idx : working_subset) {
                        if (!results[idx]) {
                            results[idx] = CheckResult::Unknown;
                        }
                    }
                    working_subset.clear();
                }
            }
        } else if (res_split == z3::unsat) {
            const auto core_idx = indices_from_core(split_solver.unsat_core());
            if (reuse_split_solver) {
                split_solver.pop();
            }

            std::vector<std::size_t> sat_set;
            std::copy_if(current_subset.begin(), current_subset.end(), std::back_inserter(sat_set),
                        [&](std::size_t idx) {
                            return std::find(core_idx.begin(), core_idx.end(), idx) == core_idx.end();
                        });

            if (core_idx.size() == 1) {
                waiting.push_back(core_idx.front());
                if (!sat_set.empty()) {
                    queue.push_back(std::move(sat_set));
                }
            } else if (!core_idx.empty()) {
                std::vector<std::vector<std::size_t>> subsets(core_idx.size());
                for (std::size_t i = 0; i < core_idx.size(); ++i) {
                    subsets[i].push_back(core_idx[i]);
                }
                for (std::size_t i = 0; i < sat_set.size(); ++i) {
                    subsets[i % core_idx.size()].push_back(sat_set[i]);
                }
                for (auto& subset : subsets) {
                    queue.push_back(std::move(subset));
                }
            }
        } else {
            if (reuse_split_solver) {
                split_solver.pop();
            }
            for (auto idx : current_subset) {
                if (!results[idx]) {
                    results[idx] = CheckResult::Unknown;
                }
            }
        }
    }

    if (!waiting.empty()) {
        z3::solver solver(ctx);
        solver.add(precond);
        switch (algorithm) {
            case 0:
                unary_check_cached_partial(precond, constraints, results, waiting);
                break;
            case 1:
                unary_check_incremental_cached_partial(solver, constraints, results, waiting);
                break;
            case 2:
                disjunctive_check_incremental_cached_partial(solver, constraints, results, waiting);
                break;
            default:
                throw std::invalid_argument("Invalid algorithm choice. Choose 0, 1, or 2.");
        }
    }

    return finalize_results(results);
}

}  // namespace

ResultVector conjunctive_check(const z3::expr& precond,
                               const std::vector<z3::expr>& constraints,
                               int algorithm) {
    return conjunctive_check_internal(precond, constraints, algorithm, false);
}

ResultVector conjunctive_check_incremental(const z3::expr& precond,
                                           const std::vector<z3::expr>& constraints,
                                           int algorithm) {
    return conjunctive_check_internal(precond, constraints, algorithm, true);
}

}  // namespace aria::scsat::app
