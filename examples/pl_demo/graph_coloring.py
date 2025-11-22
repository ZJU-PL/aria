from z3 import Solver, Int, And, sat
import sys, time

def create_graph_coloring_model(graph, num_colors):
    color_var = {v: Int(f"color_{v}") for v in graph}
    bounds    = [And(0 <= c, c < num_colors) for c in color_var.values()]
    different = [color_var[v] != color_var[n]
                 for v in graph for n in graph[v] if v < n]
    return color_var, bounds + different

def solve_graph_coloring(graph, num_colors):
    color_var, constraints = create_graph_coloring_model(graph, num_colors)
    s = Solver(); s.add(constraints)
    t0 = time.time(); res = s.check(); t1 = time.time()
    return res == sat, s.model() if res == sat else None, color_var, t1 - t0

def print_coloring(model, color_var):
    buckets = {}
    for v, z3var in color_var.items():
        c = model.evaluate(z3var).as_long()
        buckets.setdefault(c, []).append(v)
    for c in sorted(buckets):
        print(f"  Color {c}: {', '.join(map(str, sorted(buckets[c])))}")

def main():
    cycle5   = {0:[1,4], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3,0]}
    petersen = {0:[1,4,5],1:[0,2,6],2:[1,3,7],3:[2,4,8],4:[0,3,9],
                5:[0,7,8],6:[1,8,9],7:[2,5,9],8:[3,5,6],9:[4,6,7]}
    k4       = {i:[j for j in range(4) if j!=i] for i in range(4)}
    examples = {"cycle5": (cycle5, "5-Cycle"),
                "petersen": (petersen, "Petersen Graph"),
                "k4": (k4, "Complete Graph K4")}

    key = sys.argv[1] if len(sys.argv) > 1 else "petersen"
    graph, name = examples.get(key, examples["petersen"])

    print(f"Graph Coloring using Z3 SMT Solver\n----------------------------------\nUsing {name}")
    for k in range(2, 5):
        print(f"\nAttempting coloring with {k} colors:")
        ok, model, color_var, dt = solve_graph_coloring(graph, k)
        if ok:
            print(f"Solution found in {dt:.4f}s")
            print_coloring(model, color_var)
        else:
            print(f"No valid coloring with {k} colors (checked in {dt:.4f}s)")

if __name__ == "__main__":
    main()
