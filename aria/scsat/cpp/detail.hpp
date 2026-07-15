// Internal helpers shared by scsat algorithm implementations.
#pragma once

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <z3++.h>

#include "scsat.hpp"

namespace aria::scsat::app::detail {

inline constexpr const char* kTrackPrefix = "p_";

inline void ensure_same_context(const z3::expr& precond, const std::vector<z3::expr>& constraints) {
    for (const auto& c : constraints) {
        if (&c.ctx() != &precond.ctx()) {
            throw std::invalid_argument("All expressions must share the same Z3 context.");
        }
    }
}

inline CheckResult map_check_result(z3::check_result r) {
    if (r == z3::sat) {
        return CheckResult::Sat;
    }
    if (r == z3::unsat) {
        return CheckResult::Unsat;
    }
    return CheckResult::Unknown;
}

inline bool model_true(const z3::model& m, const z3::expr& e) {
    z3::expr evaluated = m.eval(e, true);
    return evaluated.is_true();
}

inline std::vector<std::size_t> all_indices(std::size_t n) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    return idx;
}

inline std::vector<z3::expr> make_tracking_literals(z3::context& ctx, std::size_t n) {
    std::vector<z3::expr> res;
    res.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        res.emplace_back(ctx.bool_const((std::string(kTrackPrefix) + std::to_string(i)).c_str()));
    }
    return res;
}

inline std::size_t tracking_index(const z3::expr& marker) {
    const std::string name = marker.decl().name().str();
    if (name.rfind(kTrackPrefix, 0) != 0) {
        throw std::runtime_error("Unexpected tracking literal name: " + name);
    }
    return static_cast<std::size_t>(std::stoul(name.substr(std::char_traits<char>::length(kTrackPrefix))));
}

inline std::vector<std::size_t> indices_from_core(const z3::expr_vector& core) {
    std::vector<std::size_t> indices;
    indices.reserve(core.size());
    for (unsigned i = 0; i < core.size(); ++i) {
        indices.push_back(tracking_index(core[i]));
    }
    return indices;
}

inline ResultVector finalize_results(const std::vector<std::optional<CheckResult>>& partial) {
    ResultVector out;
    out.reserve(partial.size());
    for (const auto& v : partial) {
        out.push_back(v.value_or(CheckResult::Unknown));
    }
    return out;
}

inline void propagate_model_sat(const z3::model& model,
                                const std::vector<z3::expr>& constraints,
                                std::vector<std::optional<CheckResult>>& results,
                                const std::vector<std::size_t>& indices) {
    for (auto idx : indices) {
        if (!results[idx] && model_true(model, constraints[idx])) {
            results[idx] = CheckResult::Sat;
        }
    }
}

inline z3::expr make_disjunction(z3::context& ctx, const std::vector<z3::expr>& exprs) {
    if (exprs.empty()) return ctx.bool_val(false);
    z3::expr_vector vec(ctx);
    std::for_each(exprs.begin(), exprs.end(), [&](const z3::expr& e) { vec.push_back(e); });
    return z3::mk_or(vec);
}

}  // namespace aria::scsat::app::detail
