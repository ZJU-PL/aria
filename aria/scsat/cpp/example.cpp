#include "scsat.hpp"

#include <iostream>
#include <string>
#include <vector>

using aria::scsat::app::ResultVector;
using namespace aria::scsat::app;

namespace {

void print_results(const std::string& label, const ResultVector& results) {
    std::cout << label << ": [";
    for (std::size_t i = 0; i < results.size(); ++i) {
        std::cout << i << "=" << to_string(results[i]);  // keep original order visible
        if (i + 1 < results.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

}  // namespace

int main() {
    z3::context ctx;
    const z3::expr x = ctx.bool_const("x");
    const z3::expr y = ctx.bool_const("y");

    // Shared scenario across all variants for consistent comparison
    const z3::expr precond = x && y;
    const std::vector<z3::expr> constraints = {x, y, !x};

    // Unary variants
    print_results("unary_check", unary_check(precond, constraints));
    print_results("unary_check_incremental", unary_check_incremental(precond, constraints));
    print_results("unary_check_cached", unary_check_cached(precond, constraints));
    print_results("unary_check_incremental_cached",
                  unary_check_incremental_cached(precond, constraints));

    // Disjunctive checks
    print_results("disjunctive_check_cached", disjunctive_check_cached(precond, constraints));
    print_results("disjunctive_check_incremental_cached",
                  disjunctive_check_incremental_cached(precond, constraints));

    // Conjunctive checks
    for (int algo = 0; algo <= 2; ++algo) {
        print_results("conjunctive_check algo=" + std::to_string(algo),
                      conjunctive_check(precond, constraints, algo));
        print_results("conjunctive_check_incremental algo=" + std::to_string(algo),
                      conjunctive_check_incremental(precond, constraints, algo));
    }

    return 0;
}
