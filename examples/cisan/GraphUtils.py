from typing import *
import numpy as np
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode


def reachable(x: int, z: set, dag: Dag):

    UP = 1
    DOWN = 0

    def get_node_idx(node):
        return int(node.get_name()[1:])

    def get_parent_idx(child_idx):
        node_y = dag.get_node(f"X{child_idx}")
        return [get_node_idx(node) for node in dag.get_parents(node_y)]

    def get_children_idx(parent_idx):
        node_y = dag.get_node(f"X{parent_idx}")
        return [get_node_idx(node) for node in dag.get_children(node_y)]

    L = list(z)
    A = set()
    while len(L) != 0:
        y = L.pop(0)
        if y not in A:
            L += get_parent_idx(y)
        A.add(y)

    L = [(x, UP)]
    V = set()
    R = set()
    while len(L) != 0:
        y, d = L.pop(0)
        if (y, d) not in V:
            if y not in z:
                R.add(y)
            V.add((y, d))
            if d == UP and y not in z:
                for pa_y in get_parent_idx(y):
                    L.append((pa_y, UP))
                for ch_y in get_children_idx(y):
                    L.append((ch_y, DOWN))
            elif d == DOWN:
                if y not in z:
                    for ch_y in get_children_idx(y):
                        L.append((ch_y, DOWN))
                if y in A:
                    for pa_y in get_parent_idx(y):
                        L.append((pa_y, UP))
    return R


def choose_stepsize(var_num):
    if var_num <= 8:
        return 1
    if var_num <= 12:
        return 3
    return 4


def read_dag(dag_path: str):
    adj_mat: np.ndarray = np.loadtxt(dag_path)
    nodes = [GraphNode(f"X{i}") for i in range(adj_mat.shape[0])]
    dag = Dag(nodes)
    for (Vi, Vj), _ in np.ndenumerate(adj_mat):
        if _ == 1:
            dag.add_directed_edge(nodes[Vi], nodes[Vj])
    return dag


def compare_skeleton(graph: Dict[int, Set[int]], dag: Dag):
    est_edges = set()
    gt_edges = set()
    for node_x in graph:
        for node_y in graph:
            if node_x > node_y:
                continue
            if node_y in graph[node_x]:
                est_edges.add((node_x, node_y))
            if dag.is_adjacent_to(dag.nodes[node_x], dag.nodes[node_y]):
                gt_edges.add((node_x, node_y))
    print(gt_edges, est_edges)
    SHD = len(gt_edges - est_edges) + len(est_edges - gt_edges)
    return SHD


class OracleCI:
    def __init__(self, dag: Dag = None):

        self.dag = dag
        self.oracle_cache = {}
        self.ci_invoke_count = 0

    def oracle_ci(self, x: int, y: int, z: set[int], debug=False):
        if not debug:
            self.ci_invoke_count += 1
        f_z = frozenset(z)
        x_reachable: dict = self.oracle_cache.setdefault(x, {})
        if f_z in x_reachable:
            x_z_reachable = x_reachable[f_z]
        else:
            x_z_reachable: set = frozenset(reachable(x, z, self.dag))
        return y not in x_z_reachable


class ErrorInjectionOracleCI:
    def __init__(self, dag: Dag = None, error_rate: float = 0.1, seed: int = 0):
        self.dag = dag
        self.oracle_cache = {}
        self.error_rate = error_rate

        # temporary solution
        if len(self.dag.get_nodes()) == 5:
            total_ci = 50
        elif len(self.dag.get_nodes()) == 6:
            total_ci = 100
        else:
            total_ci = 1100
        self.error_num = int(max(1, error_rate * total_ci))
        print("Var Count: ", len(self.dag.get_nodes()))
        print("Error injection num: ", self.error_num)
        np.random.seed(seed)
        self.error_injection_position = np.random.choice(
            total_ci, self.error_num, replace=False
        ).tolist()
        print("Error injection pos: ", self.error_injection_position)
        self.ci_invoke_count = 0

    def oracle_ci(self, x: int, y: int, z: set[int], debug=False):
        if not debug:
            self.ci_invoke_count += 1
        f_z = frozenset(z)
        x_reachable: dict = self.oracle_cache.setdefault(x, {})
        if f_z in x_reachable:
            x_z_reachable = x_reachable[f_z]
        else:
            x_z_reachable: set = frozenset(reachable(x, z, self.dag))
        is_ind = y not in x_z_reachable
        if self.ci_invoke_count in self.error_injection_position:
            print("Error injection: ", x, y, z, is_ind)
            is_ind = not is_ind
        return is_ind
