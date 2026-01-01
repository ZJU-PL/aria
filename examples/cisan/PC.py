import numpy as np
import pandas as pd
import copy
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

from IndependenceSolver import KnowledgeBase, FUNCTION_TIME_DICT
import IndependenceSolver

IndependenceSolver.CONSTRAINT_SLICING = ENABLE_PREFETCHING
from Utility import CIStatement
from GraphUtils import *
from DataUtils import read_table
from Kendall import DPKendalTau
from Chisq import Chisq

import functools


def logme(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)

    return wrapped


def pc_skl(var_num, independence_func, enable_solver=True):  # Legacy code

    TOTAL_CI = 0

    graph = {i: set(range(var_num)) - {i} for i in range(var_num)}
    kb = KnowledgeBase([], var_num, False)

    for node_x, node_y in combinations(range(var_num), 2):
        TOTAL_CI += 1
        if independence_func(node_x, node_y, set()):
            graph[node_x].remove(node_y)
            graph[node_y].remove(node_x)
            kb.AddFact(CIStatement.createByXYZ(node_x, node_y, set(), True))
        else:
            kb.AddFact(CIStatement.createByXYZ(node_x, node_y, set(), False))
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
                        psan_outcome = kb.SinglePSan(ci)
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        if psan_outcome is None:
                            is_ind = independence_func(x, y, z)
                            kb.AddFact(CIStatement.createByXYZ(x, y, z, is_ind))
                            print(
                                "CI Query",
                                str(CIStatement.createByXYZ(x, y, z, is_ind)),
                            )
                        else:
                            is_ind = psan_outcome.ci
                            assert is_ind == independence_func(x, y, z, True)
                            kb.AddFact(psan_outcome)
                        if is_ind:
                            edges_to_remove.add((node_x, node_y))
                            break
                else:
                    for ci in ci_relation_candidate:
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


def EDsan_pc_skl(var_num, independence_func, enable_solver=True):
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
                        kb.EDSan(incoming_ci)
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


def Psan_pc_skl(var_num, independence_func, enable_solver=True):
    TOTAL_CI = 0
    graph = {i: set(range(var_num)) - {i} for i in range(var_num)}
    kb = KnowledgeBase([], var_num, False)

    for node_x, node_y in combinations(range(var_num), 2):
        TOTAL_CI += 1
        ci = CIStatement.createByXYZ(node_x, node_y, set(), True)
        psan_outcome = kb.SinglePSan(ci)
        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
        if psan_outcome is None:
            is_ind = independence_func(x, y, z)
            kb.AddFact(CIStatement.createByXYZ(x, y, z, is_ind))
        else:
            is_ind = psan_outcome.ci
            assert is_ind == independence_func(x, y, z, True)
            kb.AddFact(psan_outcome)
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
                        psan_outcome = kb.SinglePSan(ci)
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        if psan_outcome is None:
                            is_ind = independence_func(x, y, z)
                            kb.AddFact(CIStatement.createByXYZ(x, y, z, is_ind))
                            print(
                                "CI Query",
                                str(CIStatement.createByXYZ(x, y, z, is_ind)),
                            )
                        else:
                            is_ind = psan_outcome.ci
                            # assert is_ind == independence_func(x, y, z, True)
                            kb.AddFact(psan_outcome)
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


@logme
def run_dpkt_pc(benchmark):
    dag_path = f"./benchmarks/{benchmark}_graph.txt"
    data_path = f"data/{benchmark}-10k.csv"
    dag = read_dag(dag_path)
    dpkt = DPKendalTau(read_table(data_path), dag=dag)
    est, TOTAL_CI = Psan_pc_skl(dag.get_num_nodes(), dpkt.kendaltau_ci, True)
    return est, TOTAL_CI, dpkt.ci_invoke_count, dpkt.get_eps_prime()


@logme
def run_dpkt_pc_repeat(benchmark):
    with Pool() as pool:
        result = pool.map(run_dpkt_pc, [benchmark] * 10)
    dag_path = f"./benchmarks/{benchmark}_graph.txt"
    dag = read_dag(dag_path)
    avg_shd = np.mean([compare_skeleton(rlt[0], dag) for rlt in result])
    avg_total_ci = np.mean([rlt[1] for rlt in result])
    avg_ci_count = np.mean([rlt[2] for rlt in result])
    avg_eps = np.mean([rlt[3] for rlt in result])

    return avg_shd, avg_total_ci, avg_ci_count, avg_eps


@logme
def run_chisq_pc(benchmark):
    dag_path = f"./benchmarks/{benchmark}_graph.txt"
    data_path = f"data/{benchmark}-10k.csv"
    dag = read_dag(dag_path)
    chisq = Chisq(read_table(data_path), dag=dag)
    # est, TOTAL_CI = pc_skl(dag.get_num_nodes(), chisq.chisq_ci, True)
    est, TOTAL_CI = Psan_pc_skl(dag.get_num_nodes(), chisq.chisq_ci, True)
    return est, TOTAL_CI, chisq.ci_invoke_count


@logme
def run_oracle_pc(benchmark):
    dag_path = f"./benchmarks/{benchmark}_graph.txt"
    dag = read_dag(dag_path)
    oracle = OracleCI(dag=dag)
    # est, TOTAL_CI = pc_skl(dag.get_num_nodes(), oracle.oracle_ci, True)
    est, TOTAL_CI = Psan_pc_skl(dag.get_num_nodes(), oracle.oracle_ci, True)
    # est, TOTAL_CI = EDsan_pc_skl(dag.get_num_nodes(), oracle.oracle_ci, True)
    return est, TOTAL_CI, oracle.ci_invoke_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        "-b",
        type=str,
        choices=["earthquake", "survey", "cancer", "sachs", "insurance"],
        default="earthquake",
    )
    parser.add_argument(
        "--function",
        "-f",
        type=str,
        choices=["dpkt", "chisq", "oracle"],
        default="oracle",
    )

    args = parser.parse_args()

    start_time = datetime.now()

    benchmarks = [args.benchmarks]
    print(args)
    print("=========================================")
    print(f"Benchmarks: {benchmarks}")

    result = {
        bn: {"shd": [], "#CI Test": [], "#CI Query": [], "Eps": []} for bn in benchmarks
    }

    for benchmark in benchmarks:

        dag_path = f"./benchmarks/{benchmark}_graph.txt"
        dag = read_dag(dag_path)
        # dag = read_dag(dag_path)
        # REACHABLE = {}
        # NUM_OF_CI_TEST = 0
        # TOTAL_CI = 0
        # print("reachable", REACHABLE)
        # print("NUM_OF_CI_TEST",NUM_OF_CI_TEST, "TOTAL_CI", TOTAL_CI)

        #     # kendal tau
        #     for rlt in results:
        #         est, TOTAL_CI, ci_invoke_count, eps_prime = rlt
        #         result[benchmark]["shd"].append(compare_skeleton(est, dag))
        #         result[benchmark]["#CI Test"].append(ci_invoke_count)
        #         result[benchmark]["#CI Query"].append(TOTAL_CI)
        #         result[benchmark]["Eps"].append(eps_prime)

        # # for benchmark in benchmarks:
        #     print(benchmark)
        #     print("SHD", np.mean(result[benchmark]["shd"]), np.std(result[benchmark]["shd"]))
        #     print("NUM_OF_CI_TEST", np.mean(result[benchmark]["#CI Test"]), np.std(result[benchmark]["#CI Test"]))
        #     print("TOTAL_CI", np.mean(result[benchmark]["#CI Query"]), np.std(result[benchmark]["#CI Query"]))
        #     print("Eps", np.mean(result[benchmark]["Eps"]), np.std(result[benchmark]["Eps"]))

        # est, TOTAL_CI, ci_invoke_count = run_oracle_pc(benchmark)
        # shd = compare_skeleton(est, dag)
        if args.function == "oracle":
            est, TOTAL_CI, ci_invoke_count = run_oracle_pc(benchmark)
        elif args.function == "chisq":
            est, TOTAL_CI, ci_invoke_count = run_chisq_pc(benchmark)
        else:
            raise NotImplementedError
        # est, TOTAL_CI, ci_invoke_count, eps = run_dpkt_pc(benchmark)

        print(benchmark)
        print("SHD", compare_skeleton(est, dag))
        print("NUM_OF_CI_TEST", ci_invoke_count)
        print("TOTAL_CI", TOTAL_CI)
        # print("EPS", eps)

        # SHD_list = []
        # ci_invoke_count_list = []
        # TOTAL_CI_list = []
        # eps_list = []
        # for i in range(10):
        #     est, TOTAL_CI, ci_invoke_count, eps = run_dpkt_pc(benchmark)
        #     SHD = compare_skeleton(est, dag)
        #     SHD_list.append(SHD)
        #     ci_invoke_count_list.append(ci_invoke_count)
        #     TOTAL_CI_list.append(TOTAL_CI)
        #     eps_list.append(eps)

        # print("SHD", np.mean(SHD_list))
        # print("NUM_OF_CI_TEST", np.mean(ci_invoke_count_list))
        # print("TOTAL_CI", np.mean(TOTAL_CI_list))
        # print("EPS", np.mean(eps_list))

        end_time = datetime.now()
        print("Time taken: ", end_time - start_time)

        for key, val in FUNCTION_TIME_DICT.items():
            print(f"Function: {key}, Total time: {val[0]}, Total count: {val[1]}")
