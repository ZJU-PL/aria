# Minimal Bounded Model Checking (BMC) Demo
from z3 import *

def bmc_demo():
    x = Int('x')
    x1 = Int('x1')
    init = (x == 0)
    trans = (x1 == x + 1)
    prop = (x < 3)
    s = Solver()
    # Unroll 4 steps
    s.add(init)
    for k in range(4):
        s.push()
        s.add(Not(prop.substitute(x, Int(f'x{k}'))))
        if s.check() == sat:
            print(f'Counterexample at step {k}')
            return
        s.pop()
        s.add(Int(f'x{k+1}') == Int(f'x{k}') + 1)
    print('Property holds for 4 steps')

if __name__ == "__main__":
    bmc_demo()