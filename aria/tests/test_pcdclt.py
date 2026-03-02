import z3

print("Z3 version:", z3.get_version_string())

with open("../file_transfer/data/raw/parallel/QF_LIA/scrambled103783.smt2", "r") as f:
    smt2_string = f.read()
print()

fml = z3.And(z3.parse_smt2_string(smt2_string))

simplified = z3.Tactic("simplify")(fml)
simplified = z3.Tactic("elim-uncnstr")(simplified.as_expr())
simplified = z3.Tactic("solve-eqs")(simplified.as_expr())
simplified = z3.Tactic("simplify")(simplified.as_expr())
# logger.debug(simplified.as_expr().sexpr())
simplified = z3.Tactic("tseitin-cnf")(simplified.as_expr())