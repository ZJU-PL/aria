from Chisq import Chisq
from Kendall import DPKendalTau
from DataUtils import read_table
from GraphUtils import *
from Utility import CIStatement
import IndependenceSolver
from IndependenceSolver import KnowledgeBase, FUNCTION_TIME_DICT, EDsanAssertError
import numpy as np
import pandas as pd
import statistics
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


def EDsan_pc_skl(var_num, independence_func, enable_solver=False):
    TOTAL_CI = 0
    graph = {i: set(range(var_num)) - {i} for i in range(var_num)}
    kb = KnowledgeBase([], var_num, False)

    for node_x, node_y in combinations(range(var_num), 2):
        TOTAL_CI += 1
        is_ind = independence_func(node_x, node_y, set())
        ci = CIStatement.createByXYZ(node_x, node_y, set(), is_ind)
        kb.EDSan(ci)
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
                # print(node_x, node_y, edges_to_remove, graph, TOTAL_CI)
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
                        kb.EDSan(incoming_ci)
                        kb.AddFact(incoming_ci)
                        print("CI Query", str(incoming_ci))
                        if is_ind:
                            edges_to_remove.add((node_x, node_y))
                            break
                else:
                    for ci in ci_relation_candidate:
                        TOTAL_CI += 1
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        is_ind = independence_func(x, y, z)
                        incoming_ci = CIStatement.createByXYZ(x, y, z, is_ind)
                        kb.AddFact(incoming_ci)
                        if is_ind:
                            edges_to_remove.add((node_x, node_y))
                            break
        for edge in edges_to_remove:
            node_x, node_y = edge
            graph[node_x].remove(node_y)
            graph[node_y].remove(node_x)
        order += 1
    return graph, TOTAL_CI, kb


def run_detection(
    kb: KnowledgeBase,
    error_rate: float,
    seed: int,
    use_marginal=True,
    use_graphoid=True,
    use_slicing=True,
):
    start_time = datetime.now()
    # kb.Perturb(error_rate, seed)
    # kb.FlipSome(seed, int(len(kb.facts) * 0.03)) # for sachs
    kb.FlipSome(seed, int(len(kb.facts) * 0.1))  # for survey
    last_ci = kb.facts.pop()
    error_detected = False
    method_name = None
    try:
        kb.EDSan_ablation(last_ci, use_marginal, use_graphoid, use_slicing)
    except EDsanAssertError as e:
        error_detected = True
        method_name = e.method_name
        print(f"Error Detected: {e}")
        print("======================================")
    end_time = datetime.now()
    last_time = (end_time - start_time).total_seconds()
    return error_detected, method_name, last_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        choices=["earthquake", "survey", "cancer", "sachs"],
        default="sachs",
    )
    parser.add_argument("--use_marginal", "-um", action="store_true")
    parser.add_argument("--use_graphoid", "-ug", action="store_true")
    parser.add_argument("--use_slicing", "-us", action="store_true")
    parser.add_argument("--error-ratio", "-r", type=float, default=0.01)
    args = parser.parse_args()
    print(args)
    print("Flip one CI")
    print("=========================================")

    method_list = []
    time_list = []
    detected_idx = []
    start_time = datetime.now()
    detected_num = 0
    total_num = 100
    dag_path = f"./benchmarks/{args.benchmark}_graph.txt"
    dag = read_dag(dag_path)
    oracle = OracleCI(dag=dag)
    est, total_ci, kb = EDsan_pc_skl(
        dag.get_num_nodes(), oracle.oracle_ci, enable_solver=False
    )
    print("Total CI", total_ci)
    print("SHD", compare_skeleton(est, dag))
    print("Start detection")
    for seed in range(total_num):
        if detected_num == 100:
            break
        error_detected, method_name, last_time = run_detection(
            kb.copy(),
            args.error_ratio,
            seed,
            args.use_marginal,
            args.use_graphoid,
            args.use_slicing,
        )
        if error_detected:
            detected_num += 1
            detected_idx.append(seed)
            method_list.append(method_name)
            time_list.append(last_time)
    end_time = datetime.now()
    print(
        f"Optimization Marginal: {args.use_marginal}, Graphoid: {args.use_graphoid}, Slicing: {args.use_slicing}"
    )
    print("All Time taken: ", end_time - start_time)
    print("Average time taken: ", sum(time_list) / len(time_list))
    print(f"Time standard deviation: {statistics.stdev(time_list)}")
    print(f"Detected {detected_num} out of {total_num} errors")
    print("Method list", method_list)
    print("Detected index", detected_idx)
    # for key, val in FUNCTION_TIME_DICT.items():
    #     print(f"Function: {key}, Total time: {val[0]}, Total count: {val[1]}")
