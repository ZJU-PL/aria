"""
TODO: this file relies on qiskit, which is not installed in the aria environment.
"""

#!pypy3
import sys
from copy import deepcopy
from math import sqrt, ceil, log
from typing import List, Dict, Any, Tuple
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.utils import QuantumInstance
import numpy as np


class RandomCFLSolver:

    def __init__(self) -> None:
        self.correct: bool = True

    # Algorithm 2 in the paper
    def subroutine(self, n: int, sol: List[int]) -> Tuple[int, int]:
        # use the qasm_simulator to simulate Grover search
        backend = BasicAer.get_backend('qasm_simulator')
        # for each Grover search, only running one shot
        quantum_instance = QuantumInstance(backend, shots=1)
        # create oracle for Grover search
        # Note that although qiskit provides Statevector for creating oracle,
        # it takes O(N) time without QRAM model on a classical simulator.
        oracle = Statevector(sol)
        problem = AmplificationProblem(oracle, is_good_state=oracle)
        # initialization
        m: int = 1
        lab: float = 6/5
        total: int = 0
        # repeat until reaching sqrt{2^n}
        while 1:
            j = np.random.randint(0, m)
            total += j
            # run Grover search on a simulator
            grover = Grover(quantum_instance=quantum_instance,iterations=j)
            # the Grover search result is saved in 'result'
            result = grover.amplify(problem)
            # get the return value of Grover search by 'result.top_measurement'
            return_value = int(result.top_measurement,2)
            if sol[return_value] == 1:
                return return_value, total
            m = min(lab*m, sqrt(2**n))
            # stop because reaching the threshold
            if total > sqrt(2**n):
                return -1, total

    # Algorithm 3 in the paper
    def estimate(self, sol_list: List[int], answer_num: int) -> int:
        sol = deepcopy(sol_list)
        n = ceil(log(len(sol),2))
        # the number of targets that are not found is 'current_num'
        current_num: int = answer_num
        # this is the total number of iterations needed to find all targets
        total: int = 0
        # this is control variable of the while loop
        iteration: int = 0
        # targets are stored in 'return_list'
        return_list: List[int] = []
        while iteration < 3*n:
            # xx is the index found, t is the current number of quantum iterations
            xx, t = self.subroutine(n ,sol)
            # successfully find a target index
            if xx != -1:
                return_list.append(xx)
                # mark index 'xx' as found
                sol[xx] = 0
                iteration = 0
                # the number of targets that are not found is decreased
                current_num -= 1
            else:
                if current_num == 0:
                    # we need additional time to determine there is no target left
                    # because we have found all targets
                    total += t * 3 * n
                    break
                iteration += 1
            total += t
        # report we miss a target
        if current_num != 0:
            self.correct = False
            print("we miss some targets:", answer_num, len(return_list))
            print("ground truth:", sol_list)
            print("our result:", return_list)
        return total

    def solve(self, graph: List[List[List[int]]], grammar: Dict[str, Any]) -> None:
        self.__cubic_solve(graph, grammar)

    # DTC-based CFL-reachability
    def __cubic_solve(self, graph: List[List[List[int]]], grammar: Dict[str, Any]) -> None:
        total_classical: int = 0
        total_quantum: int = 0
        # graph is accessed by [i, label, j]
        size = len(graph)
        grammar_size = len(graph[0])
        Worklist: set = set()
        for i in range(size):
            for label in range(grammar_size):
                for j in range(size):
                    if graph[i][label][j] == 1:
                        Worklist.add((i,label,j))
        for X in grammar['epsilon']:
            for i in range(size):
                if graph[i][X][i] == 0:
                    graph[i][X][i] = 1
                    Worklist.add((i,X,i))
        # worklist analysis starts
        while len(Worklist) > 0:
            i, Y, j = Worklist.pop()
            if Y in grammar['single'].keys():
                for X in grammar['single'][Y]:
                    if graph[i][X][j] == 0:
                        graph[i][X][j] = 1
                        Worklist.add((i,X,j))
            # save the solution indices in this list
            # for creating oracle of Grover search
            answer_num: int = 0
            sol: List[int] = [0 for i in range(size)]
            total_classical += 2 * size
            if Y in grammar['double1'].keys():
                for X, Z in grammar['double1'][Y]:
                    for k in range(size):
                        if graph[i][X][k] == 0 and graph[j][Z][k] == 1:
                            sol[k] = 1
                            answer_num += 1
                            graph[i][X][k] = 1
                            Worklist.add((i,X,k))
            # simulate the process of Grover search
            # and store the number of quantum iterations needed for finding all targets
            total_quantum += self.estimate(sol, answer_num)
            answer_num = 0
            # save the solution indices in this list
            # for creating oracle of Grover search
            sol = [0 for i in range(size)]
            if Y in grammar['double2'].keys():
                for X, Z in grammar['double2'][Y]:
                    for k in range(size):
                        if graph[k][X][j] == 0 and graph[k][Z][i] == 1:
                            sol[k] = 1
                            answer_num += 1
                            graph[k][X][j] = 1
                            Worklist.add((k,X,j))
            # simulate the process of Grover search
            # and store the number of quantum iterations needed for finding all targets
            total_quantum += self.estimate(sol, answer_num)
        # print necessary results of the simulation
        print("input size: ", size)
        if self.correct:
            print("We successfully find all targets.")
        else:
            print("We miss some targets.")


def main(argv: List[str]) -> None:
    size = 8
    # get the input size from user input. should be 2^k
    if len(argv) > 1:
        num = int(sys.argv[1])
        if num > 0 and (num & (num - 1)) == 0:
            size = int(sys.argv[1])
        else:
            print('please input an integer = 2^k')
            return
    grammar_size = 5
    a = [0,1]
    g = np.random.choice(a,(size,grammar_size,size),p=[0.8,0.2])
    grammar = {'epsilon':[0],'single':{0:[1], 2:[3]},'double1':{1:[(0,3)],4:[(1,2)]},'double2':{3:[(0,1)],2:[(1,4)]}}
    solver = RandomCFLSolver()

    solver.solve(g, grammar)


if __name__ == '__main__':
    main(sys.argv)
