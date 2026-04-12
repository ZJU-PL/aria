from aria.datalog import py_datalog

py_datalog.create_atoms("salary", "manager")
py_datalog.create_atoms(
    "salary_class", "indirect_manager", "report_count", "budget", "lowest",
    "X", "Y", "Z", "N"
)


def main() -> None:
    py_datalog.clear()

    +(salary["John"] == 6800)
    +(manager["Mary"] == "John")
    +(salary["Mary"] == 6300)
    +(manager["Sam"] == "Mary")
    +(salary["Sam"] == 5900)

    salary_class[X] = salary[X] // 1000
    indirect_manager(X, Y) <= (manager[X] == Y) & (Y != None)
    indirect_manager(X, Y) <= (manager[X] == Z) & indirect_manager(Z, Y) & (
        Y != None
    )
    (report_count[X] == len_(Y)) <= indirect_manager(Y, X)
    (budget[X] == sum_(N, for_each=Y)) <= (indirect_manager(Y, X)) & (
        salary[Y] == N
    )
    (lowest[1] == min_(X, order_by=N)) <= (salary[X] == N)

    print("salary_class['John'] == Y:", salary_class["John"] == Y)
    print("salary[X] == 6300:", salary[X] == 6300)
    print("indirect_manager('Mary', X):", indirect_manager("Mary", X))
    print(
        "(salary[X] < 6000) & indirect_manager(X, 'John'):",
        (salary[X] < 6000) & indirect_manager(X, "John"),
    )
    print("report_count[X] == 2:", report_count[X] == 2)
    print("budget['John'] == N:", budget["John"] == N)
    print("lowest[1] == X:", lowest[1] == X)


if __name__ == "__main__":
    main()
