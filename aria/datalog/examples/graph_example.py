from aria.datalog import py_datalog


def main() -> None:
    py_datalog.clear()

    @py_datalog.program()
    def _() -> None:
        +link(1, 2)
        +link(2, 3)
        +link(2, 4)
        +link(2, 5)
        +link(5, 6)
        +link(6, 7)
        +link(7, 2)
        +link(8, 9)

        link(X, Y) <= link(Y, X)

        can_reach(X, Y) <= can_reach(X, Z) & link(Z, Y) & (X != Y)
        can_reach(X, Y) <= link(X, Y)

        (path_with_cost(X, Y, P, C)) <= (
            path_with_cost(X, Z, P2, C2)
        ) & link(Z, Y) & (X != Y) & (X._not_in(P2)) & (Y._not_in(P2)) & (
            P == P2 + [Z]
        ) & (C == C2 + 1)
        (path_with_cost(X, Y, P, C)) <= link(X, Y) & (P == []) & (C == 0)

        (shortest_path[X, Y] == min_(P, order_by=C)) <= path_with_cost(
            X, Y, P, C
        )

    print("can_reach(1, Y):", can_reach(1, Y))
    print("can_reach(8, Y):", can_reach(8, Y))
    print("shortest_path[1, 7] == P:", shortest_path[1, 7] == P)


if __name__ == "__main__":
    py_datalog.create_terms(
        "link, can_reach, path_with_cost, shortest_path, X, Y, Z, P, P2, C, C2"
    )
    main()
