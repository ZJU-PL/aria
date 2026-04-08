from aria.datalog import Program


def main() -> None:
    program = Program()
    parent = program.relation("parent", 2)
    ancestor = program.relation("ancestor", 2)
    X, Y, Z = program.vars("X Y Z")

    program.fact(parent("bill", "john"))
    program.fact(parent("john", "sam"))

    program.rule(ancestor(X, Y)).when(parent(X, Y))
    program.rule(ancestor(X, Y)).when(parent(X, Z), ancestor(Z, Y))

    result = program.query(ancestor("bill", Y))

    print("rows:", result.rows())
    print("scalar_rows:", result.scalar_rows())
    print("named_rows:", result.named_rows())
    print("first:", result.first())
    print("first_value:", result.first_value())


if __name__ == "__main__":
    main()
