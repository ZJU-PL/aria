from collections import defaultdict
from functools import partial

from aria.unification import var
from aria.unification.match import VarDispatcher, match

match = partial(match, Dispatcher=VarDispatcher)

balance = defaultdict(lambda: 0)

name, amount = var("name"), var("amount")


@match({"status": 200, "data": {"name": name, "credit": amount}})
def respond(name, amount):  # noqa: F811, W0621
    balance[name] += amount


@match({"status": 200, "data": {"name": name, "debit": amount}})
def respond(name, amount):  # noqa: F811, W0621
    balance[name] -= amount


@match({"status": 404})
def respond():  # noqa: F811
    print("Bad Request")


if __name__ == "__main__":
    respond(
        {"status": 200, "data": {"name": "Alice", "credit": 100}}
    )  # pylint: disable=too-many-function-args
    respond(
        {"status": 200, "data": {"name": "Bob", "debit": 100}}
    )  # pylint: disable=too-many-function-args
    respond({"status": 404})  # pylint: disable=too-many-function-args
    print(dict(balance))
