from Chisq import Chisq
from Kendall import DPKendalTau
from DataUtils import read_table
from GraphUtils import *
from Utility import CIStatement
import IndependenceSolver
from IndependenceSolver import KnowledgeBase, FUNCTION_TIME_DICT, EDsanAssertError
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool
from typing import List, Set, Dict
from functools import partial
from itertools import combinations
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Dag import Dag
from datetime import datetime
import argparse

MAX_ORDER = 5
ENABLE_PREFETCHING = True
REACHABLE = {}

IndependenceSolver.CONSTRAINT_SLICING = ENABLE_PREFETCHING


def EDsan_pc_skl(
    var_num,
    independence_func,
    enable_solver=True,
    use_marginal=True,
    use_graphoid=True,
    use_slicing=True,
):
    TOTAL_CI = 0
    graph = {i: set(range(var_num)) - {i} for i in range(var_num)}
    kb = KnowledgeBase([], var_num, False)

    for node_x, node_y in combinations(range(var_num), 2):
        TOTAL_CI += 1
        is_ind = independence_func(node_x, node_y, set())
        ci = CIStatement.createByXYZ(node_x, node_y, set(), is_ind)
        kb.EDSan_ablation(ci, use_marginal, use_graphoid, use_slicing, False)
        kb.AddFact(ci)
        if is_ind:
            graph[node_x].remove(node_y)
            graph[node_y].remove(node_x)
    order = 1
    while order <= MAX_ORDER:
        edges_to_remove = set()
        for node_x in graph:
            neighbors = graph[node_x]
            for node_y in neighbors:
                if (node_y, node_x) in edges_to_remove or (
                    node_x,
                    node_y,
                ) in edges_to_remove:
                    continue
                if len(neighbors) - 1 < order:
                    continue
                print(node_x, node_y, edges_to_remove, graph, TOTAL_CI)
                cond_set_list = list(combinations(neighbors - {node_y}, order))
                ci_relation_candidate = [
                    CIStatement.createByXYZ(node_x, node_y, set(cond_set), True)
                    for cond_set in cond_set_list
                ]
                if enable_solver:
                    for ci in ci_relation_candidate:
                        TOTAL_CI += 1
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        is_ind = independence_func(x, y, z)
                        incoming_ci = CIStatement.createByXYZ(x, y, z, is_ind)
                        kb.EDSan_ablation(
                            incoming_ci, use_marginal, use_graphoid, use_slicing, False
                        )
                        kb.AddFact(incoming_ci)
                        print("CI Query", str(incoming_ci))
                        if is_ind:
                            edges_to_remove.add((node_x, node_y))
                            break
                else:
                    for ci in ci_relation_candidate:
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        TOTAL_CI += 1
                        if independence_func(x, y, z):
                            edges_to_remove.add((node_x, node_y))
                            break
        for edge in edges_to_remove:
            node_x, node_y = edge
            graph[node_x].remove(node_y)
            graph[node_y].remove(node_x)
        order += 1
    return graph, TOTAL_CI


def run_error_injection_oracle_pc(
    benchmark,
    error_rate=0.1,
    seed: int = 0,
    use_marginal=True,
    use_graphoid=True,
    use_slicing=True,
):
    dag_path = f"./benchmarks/{benchmark}_graph.txt"
    dag = read_dag(dag_path)
    oracle = ErrorInjectionOracleCI(dag=dag, error_rate=error_rate, seed=seed)
    error_detected = False
    method_name = "None"
    try:
        EDsan_pc_skl(
            dag.get_num_nodes(),
            oracle.oracle_ci,
            enable_solver=True,
            use_marginal=use_marginal,
            use_graphoid=use_graphoid,
            use_slicing=use_slicing,
        )
    # except AssertionError:
    #     error_detected = True
    #     print("Error Detected")
    #     print("======================================")
    except EDsanAssertError as e:
        error_detected = True
        method_name = e.method_name
        print(f"Error Detected: {e}")
        print("======================================")
    marginal_count = IndependenceSolver.MARGINAL_COUNT
    IndependenceSolver.MARGINAL_COUNT = 0
    return (
        error_detected,
        method_name,
        oracle.error_num,
        oracle.ci_invoke_count,
        oracle.error_injection_position,
        marginal_count,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        "-b",
        type=str,
        choices=["earthquake", "survey", "cancer", "sachs", "alarm", "insurance"],
        default="sachs",
    )
    parser.add_argument("--use_marginal", "-um", action="store_true")
    parser.add_argument("--use_graphoid", "-ug", action="store_true")
    parser.add_argument("--use_slicing", "-us", action="store_true")
    parser.add_argument("--error_ratio", "-r", type=float, default=0.05)
    parser.add_argument("--rq1", "-rq1", action="store_true")
    args = parser.parse_args()
    print(args)

    benchmark = args.benchmarks
    print("=========================================")

    result = {
        "error_detected": [],
        "error_count": [],
        "ci_count": [],
        "error_injection_position": [],
        "last_time": [],
        "method_name": [],
        "marginal_count": [],
        "args": f"{args}",
    }
    for seed in range(10):
        start_time = datetime.now()
        dag_path = f"./benchmark/{benchmark}_graph.txt"
        dag = read_dag(dag_path)
        if args.rq1:
            error_ratio = (seed + 1) / 100
            (
                error_detected,
                method_name,
                error_count,
                ci_count,
                error_injection_position,
                marginal_count,
            ) = run_error_injection_oracle_pc(
                benchmark,
                error_ratio,
                seed,
                args.use_marginal,
                args.use_graphoid,
                args.use_slicing,
            )
        else:
            (
                error_detected,
                method_name,
                error_count,
                ci_count,
                error_injection_position,
                marginal_count,
            ) = run_error_injection_oracle_pc(
                benchmark,
                args.error_ratio,
                seed,
                args.use_marginal,
                args.use_graphoid,
                args.use_slicing,
            )

        result["method_name"].append(method_name)
        result["marginal_count"].append(marginal_count)
        result["error_detected"].append(error_detected)
        result["error_count"].append(error_count)
        result["ci_count"].append(ci_count)
        result["error_injection_position"].append(error_injection_position)
        end_time = datetime.now()
        last_time = end_time - start_time
        print("Time taken: ", last_time)
        result["last_time"].append(str(last_time))

    json.dump(
        result,
        open(
            f"result/{benchmark}_ratio{args.error_ratio}_{args.use_marginal}_{args.use_graphoid}_{args.use_slicing}_EDSanPC.json",
            "w",
        ),
    )
    for key, val in FUNCTION_TIME_DICT.items():
        print(f"Function: {key}, Total time: {val[0]}, Total count: {val[1]}")

    # for error_rate in np.linspace(0.01, 0.1, 10):
    #     print("=========================================")
    #     print(f"Benchmark: {benchmark}, Error rate: {error_rate}")
    #     start_time = datetime.now()
    #     dag_path = f"./benchmark/{benchmark}_graph.txt"
    #     dag=read_dag(dag_path)
    #     error_detected, error_count, ci_count, error_injection_position = run_error_injection_oracle_pc(benchmark, error_rate)
    #
    #     print("Error detected: ", error_detected)
    #     print("Error count: ", error_count)
    #     print("CI count: ", ci_count)
    #     print("Error injection position: ", error_injection_position)
    #     result["error_rate"].append(error_rate)
    #     result["error_detected"].append(error_detected)
    #     result["error_count"].append(error_count)
    #     result["ci_count"].append(ci_count)
    #     result["error_injection_position"].append(error_injection_position)
    #
