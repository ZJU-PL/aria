#!/usr/bin/env python3
"""Core data models for Z3 tactic optimization."""

import json
import random
import z3


class Param:
    """Represents a parameter for a Z3 tactic."""
    def __init__(self, key, value, kind):
        self.key = key
        self.value = value
        self.kind = kind

    def _replace(self, **kwargs):
        """Create a copy with modified attributes."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


class Tactic:
    """Represents a single Z3 tactic with parameters."""
    _mutate_probability = 0.02
    _tactic_names = z3.tactics()
    _rand = random.Random()
    _rand.seed()

    def __init__(self, name):
        self.name = name
        self.params = {}
        pds = z3.Tactic(name).param_descrs()
        for i in range(pds.size()):
            pname = pds.get_name(i)
            self.params[pname] = Param(pname, None, pds.get_kind(pname))

    def clone(self):
        """Create a deep copy of this tactic."""
        res = Tactic(self.name)
        res.params = self.params.copy()
        return res

    def mutate(self):
        """Randomly mutate tactic parameters with low probability."""
        for key, param in self.params.items():
            if Tactic._rand.random() < Tactic._mutate_probability:
                self.params[key] = param._replace(value=self._random_param_value(param))
        return self

    def _random_param_value(self, param):
        """Generate random value for parameter based on its type."""
        kind_map = {
            z3.Z3_PK_BOOL: lambda: Tactic._rand.random() < 0.5,
            z3.Z3_PK_UINT: lambda: Tactic._rand.randint(0, 4294967295),
            z3.Z3_PK_DOUBLE: lambda: Tactic._rand.uniform(0.0, 1000000.0),
            z3.Z3_PK_STRING: lambda: "",
            z3.Z3_PK_SYMBOL: lambda: "",
            z3.Z3_PK_OTHER: lambda: None,
            z3.Z3_PK_INVALID: lambda: None
        }
        if param.kind in kind_map:
            return kind_map[param.kind]()
        raise LookupError(f"Unknown parameter kind: {param.kind}")

    @staticmethod
    def random():
        """Create a random tactic."""
        return Tactic(Tactic._rand.sample(Tactic._tactic_names, 1)[0])

    def __repr__(self):
        return f"{self.name} -> {self.params}"


class TacticSeq:
    """Represents a sequence of Z3 tactics for genetic algorithm evolution."""
    _max_size = 16
    _rand = random.Random()
    _rand.seed()

    def __init__(self, tactics_list=None):
        if tactics_list is None:
            tactics_list = ["simplify", "propagate-values", "solve-eqs",
                          "elim-uncnstr", "simplify", "max-bv-sharing", "smt"]
        self.storage = [Tactic(name) for name in tactics_list]
        self.fitness = 0.0

    def mutate(self):
        """Mutate all tactics in this sequence."""
        self.storage = [t.mutate() for t in self.storage]
        return self

    def to_z3_tactic(self):
        """Convert to Z3 Tactic object."""
        if not self.storage:
            return z3.Tactic('skip')

        result = None
        for tactic in self.storage:
            has_params = tactic.params and any(p.value is not None for p in tactic.params.values())
            if has_params:
                param_dict = {p.key: p.value for p in tactic.params.values() if p.value is not None}
                current = z3.With(tactic.name, **param_dict)
            else:
                current = z3.Tactic(tactic.name)
            result = current if result is None else z3.AndThen(result, current)
        return result

    def to_smtlib_apply(self):
        """Convert to SMT-LIB2 (apply ...) format."""
        if not self.storage:
            return "(apply skip)"

        tactic_parts = []
        for tactic in self.storage:
            param_list = []
            if tactic.params:
                for p in tactic.params.values():
                    if p.value is not None:
                        if p.kind == z3.Z3_PK_BOOL:
                            param_list.append(f":{p.key} {'true' if p.value else 'false'}")
                        elif p.kind in [z3.Z3_PK_UINT, z3.Z3_PK_DOUBLE]:
                            param_list.append(f":{p.key} {p.value}")
                        elif p.kind in [z3.Z3_PK_STRING, z3.Z3_PK_SYMBOL]:
                            param_list.append(f":{p.key} \\\"{p.value}\\\"")
            params_str = " " + " ".join(param_list) if param_list else ""
            tactic_parts.append(f"{tactic.name}{params_str}")

        return f"(apply ({' then '.join(tactic_parts)}))"

    @staticmethod
    def crossover(a, b):
        """Create offspring by crossing over two parent sequences."""
        res = TacticSeq()
        crossover_point = TacticSeq._rand.randint(0, len(a.storage) - 1)
        res.storage = [t.clone() for t in a.storage[:crossover_point] + b.storage[crossover_point:]]
        return res

    @staticmethod
    def random():
        """Create a random tactic sequence."""
        return TacticSeq()

    def to_string(self):
        """Return string representation of the sequence."""
        result = []
        for tactic in self.storage:
            params_str = ", ".join(f"{p.key}={p.value}" for p in tactic.params.values() if p.value is not None)
            result.append(f"{tactic.name}({params_str})" if params_str else tactic.name)
        return " -> ".join(result)

    def clone(self):
        """Create a deep copy of this sequence."""
        res = TacticSeq()
        res.storage = [tactic.clone() for tactic in self.storage]
        res.fitness = self.fitness
        return res

    def __repr__(self):
        return f"TacticSeq({len(self.storage)} tactics, fitness={self.fitness})"


class CustomJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for tactic objects."""
    def default(self, o):
        if isinstance(o, Param):
            return {"key": o.key, "value": o.value}
        if isinstance(o, Tactic):
            return {
                "name": o.name,
                "type": o.name,
                "params": [p for p in o.params.values() if p.value is not None]
            }
        if isinstance(o, TacticSeq):
            return {"tactics": o.storage, "fitness": o.fitness}
        return super().default(o)
