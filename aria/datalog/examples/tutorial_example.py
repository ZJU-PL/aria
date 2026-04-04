from aria.datalog import pyDatalog

pyDatalog.create_terms("parent, ancestor, manager, bill, X, Y, Z, N, F, factorial")


def main() -> None:
    pyDatalog.clear()

    +parent(bill, "John Adams")
    +parent("John Adams", "Sam Adams")

    ancestor(X, Y) <= parent(X, Y)
    ancestor(X, Y) <= parent(X, Z) & ancestor(Z, Y)

    +(manager[bill] == "John Adams")

    +(factorial[1] == 1)
    (factorial[N] == F) <= (N > 1) & (F == N * factorial[N - 1])

    print("parent(bill, X):", parent(bill, X))
    print("ancestor(bill, X):", ancestor(bill, X))
    print("manager[bill] == X:", manager[bill] == X)
    print("factorial[4] == F:", factorial[4] == F)


if __name__ == "__main__":
    main()
