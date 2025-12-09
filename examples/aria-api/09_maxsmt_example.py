#!/usr/bin/env python3
import z3
from aria.optimization.maxsmt import MaxSMTSolver, solve_maxsmt

def basic_maxsmt():
    x, y = z3.Ints('x y')
    hard = [x >= 0, y >= 0, x + y <= 10]
    soft = [x == 5, y == 5, x + y == 8]
    w = [1.0, 1.0, 2.0]
    for alg in ["core-guided", "z3-opt"]:
        try:
            sat, m, c = solve_maxsmt(hard, soft, w, algorithm=alg)
            if sat:
                print(f"{alg}: x={m.eval(x)}, y={m.eval(y)}, cost={c}")
        except: print(f"{alg}: Failed")

def scheduling():
    tasks = [z3.Bool(f"t{i}s{j}") for i in range(3) for j in range(2)]
    hard = [z3.PbEq([(tasks[i*2 + j], 1) for j in range(2)], 1) for i in range(3)]
    soft = [tasks[0], tasks[3], z3.Not(tasks[4])]
    w = [2.0, 1.0, 1.0]
    try:
        s = MaxSMTSolver("z3-opt")
        s.add_hard_constraints(hard)
        s.add_soft_constraints(soft, w)
        sat, m, c = s.solve()
        if sat:
            for i in range(3):
                for j in range(2):
                    if z3.is_true(m.eval(tasks[i*2 + j])):
                        print(f"Task {i}->Slot {j}")
            print(f"Cost: {c}")
    except: print("Scheduling failed")

def weighted_maxsmt():
    x, y, z = z3.Ints('x y z')
    hard = [x >= 0, y >= 0, z >= 0, x + y + z <= 100]
    soft = [x >= 10, y >= 20, z >= 30, x + y >= 50]
    w = [10.0, 5.0, 1.0, 5.0]
    try:
        sat, m, c = solve_maxsmt(hard, soft, w, algorithm="z3-opt")
        if sat:
            print(f"x={m.eval(x)}, y={m.eval(y)}, z={m.eval(z)}, cost={c}")
    except: print("Weighted failed")

if __name__ == "__main__":
    basic_maxsmt(); scheduling(); weighted_maxsmt() 