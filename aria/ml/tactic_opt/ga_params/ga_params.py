#!/usr/bin/env python

import random
import re
import subprocess
import time

from collections import namedtuple, OrderedDict

Param = namedtuple('Param', ['key', 'value', 'ttype'])

SymbolTable = {
    "sat.gc": ["psm", "glue", "glue_psm", "dyn_psm"],
    "sat.phase": ["always_false", "always_true", "caching", "random"],
    "sat.restart": ["luby", "geometric"],
    "fixedpoint.default_relation": ["external_relation", "pentagon"],
    "fixedpoint.default_table": ["sparse", "hashtable", "bitvector", "interval"],
    "fixedpoint.default_table_checker": ["null"],
    "fixedpoint.dump_aig": [""],
    "fixedpoint.engine": ["auto-config", "datalog", "pdr", "bmc"],
    "fixedpoint.tab_selection": ["weight", "first", "var-use"],
    "nnf.mode": ["skolem", "quantifiers", "full"]
}


def random_bool(rand):
    return "true" if rand.random() < 0.5 else "false"


def random_double(rand):
    return rand.uniform(0.0, 100.0)


def random_uint(rand):
    return rand.randint(0, 4294967295)


def random_symbol(rand, key):
    values = SymbolTable[key]
    rand.shuffle(values)
    return values[0]


def mutate_param(param, rand):
    if "(bool)" == param.ttype:
        return param._replace(value=random_bool(rand))
    elif "(double)" == param.ttype:
        return param._replace(value=random_double(rand))
    elif "(unsigned int)" == param.ttype:
        return param._replace(value=random_uint(rand))
    elif "(symbol)" == param.ttype:
        return param._replace(value=random_symbol(rand, param.key))
    elif "(string)" == param.ttype:
        return param
    else:
        raise LookupError("Unsupported param type: {0}".format(param.ttype))


class Params:
    _rex = re.compile(r"(.*)\s=\s(.*)\s(\(.*\))")
    _mutate_probability = 0.02
    _rand = random.Random()
    _rand.seed()

    def __init__(self):
        self._storage = OrderedDict()
        self.fitness = 0.0

    def add(self, param):
        self._storage[param.key] = param
        return self

    def load(self, fname):
        with open(fname, "r") as ffile:
            for line in ffile:
                match = self._rex.match(line)
                self.add(Param(match.group(1), match.group(2), match.group(3)))
        return self

    def dump(self, fname):
        with open(fname, "w") as ffile:
            for param in self._storage.values():
                ffile.write("{0} = {1} {2}\n".format(param.key, param.value, param.ttype))
        return self

    def mutate(self):
        for key, param in self._storage.items():
            if self._rand.random() < self._mutate_probability:
                self._storage[key] = mutate_param(param, self._rand)
        return self

    @staticmethod
    def crossover(p1, p2):
        res = Params()
        crossover_point = p1._rand.randint(0, len(p1._storage) - 1)
        res._storage.update(list(p1._storage.items())[:crossover_point])
        res._storage.update(list(p2._storage.items())[crossover_point:])
        return res

    def __cmp__(self, obj):
        if obj is None: return 1
        if not isinstance(obj, Params): return 1
        return self._storage.__cmp__(obj._storage)


def run_tests():
    # cmd = "timeout 8s ./run-tests --gtest_output='xml:test_results.xml' --gtest_color=yes --gtest_filter=xx"
    cmd = "z3"
    start = time.perf_counter()
    # TODO:
    #  add timeout
    #  can subprocess.call spawn its own sub-threads??
    ret = subprocess.call(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.perf_counter()
    return end - start if 0 == ret else 4294967295


class GA:
    _population_size = 64
    _retain_percentage = 0.25
    _retain_size = int(_retain_percentage * _population_size) + 1

    _rand = random.Random()
    _rand.seed()

    def __init__(self):
        self._population = [Params().load("z3.params.{0}".format(i)).mutate() for i in range(self._population_size)]
        self._new = []
        self._retained = []

    def evaluate(self):
        for param in self._population:
            param.dump("z3.params")
            param.fitness = run_tests()
            self._new.append(param)

    def dump(self):
        with open("z3.results", "w") as results:
            s = sorted(self._new, key=lambda e: e.fitness)
            for i, param in enumerate(s):
                param.dump("z3.params.{0}".format(i))
                results.write("{0}: {1}\n".format(i, param.fitness))

    def retained(self):
        with open("z3.results", "w") as results:
            for i, param in enumerate(self._retained):
                param.dump("z3.params.{0}".format(i))
                results.write("{0}: {1}\n".format(i, param.fitness))

    def repopulate(self):
        self._new.extend(self._retained)
        self._new = list(set(self._new))
        self._new = sorted(self._new, key=lambda e: e.fitness)

        self._population = []
        self._retained = []

        self._retained.extend(self._new[:self._retain_size])

        while len(self._population) < self._population_size:
            i1 = self._rand.randint(0, len(self._new) - 1)
            i2 = self._rand.randint(0, len(self._new) - 1)
            if i1 == i2: continue
            if self._new[i1].fitness == 4294967295 or self._new[i2].fitness == 4294967295: continue
            self._population.append(
                Params.crossover(
                    self._new[i1],
                    self._new[i2]
                ).mutate()
            )

        self._new = []


def main():
    ga = GA()
    for i in range(128):
        ga.evaluate()
        ga.dump()
        ga.repopulate()
    ga.retained()


if __name__ == "__main__":
    main()
