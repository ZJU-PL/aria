from aria.datalog import py_datalog

py_datalog.create_terms("X, Y")


class Employee(py_datalog.Mixin):
    def __init__(self, name: str, manager, salary: int):
        super(Employee, self).__init__()
        self.name = name
        self.manager = manager
        self.salary = salary

    def __repr__(self) -> str:
        return self.name

    @py_datalog.program()
    def Employee(self):
        Employee.salary_class[X] = Employee.salary[X] // 1000
        Employee.indirect_manager(X, Y) <= (Employee.manager[X] == Y) & (
            Y != None
        )
        Employee.indirect_manager(X, Y) <= (Employee.manager[X] == Z) & (
            Employee.indirect_manager(Z, Y)
        ) & (Y != None)
        (Employee.report_count[X] == len(Y)) <= Employee.indirect_manager(Y, X)


def main() -> None:
    john = Employee("John", None, 6800)
    mary = Employee("Mary", john, 6300)
    sam = Employee("Sam", mary, 5900)

    print("john.salary_class:", john.salary_class)
    print("Employee.salary[X] == 6300:", Employee.salary[X] == 6300)
    print("Employee.indirect_manager(mary, X):", Employee.indirect_manager(mary, X))
    print(
        "(Employee.salary_class[X] == 5) & Employee.indirect_manager(X, john):",
        (Employee.salary_class[X] == 5) & Employee.indirect_manager(X, john),
    )
    print("Employee.report_count[X] == 2:", Employee.report_count[X] == 2)
    print("sam.salary_class:", sam.salary_class)


if __name__ == "__main__":
    main()
