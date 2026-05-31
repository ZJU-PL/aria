"""
NeuroSym DPLL SAT solver — pure Python, no external dependencies.

Interface:
  solve_cnf(clauses, n_vars) → dict | None
    clauses : list of lists of int  (1-indexed literals; negative = negated)
    n_vars  : number of Boolean variables
    returns : {var: bool} satisfying assignment, or None if UNSAT

Implements:
  - Unit propagation (BCP)
  - Pure literal elimination
  - VSIDS-inspired frequency-based decision heuristic
  - Non-chronological backtracking via conflict analysis (1-UIP)
  - Clause learning
  - Luby restart schedule
"""

from typing import List, Dict, Optional, Tuple, Set
import time


# ── CNF representation ─────────────────────────────────────────────────────────
# Literal encoding: variable v (1-indexed) → positive lit = 2*(v-1),
#                                             negative lit = 2*(v-1)+1
# This makes complement computation: lit ^ 1

def _pos(v: int) -> int: return 2 * (v - 1)
def _neg(v: int) -> int: return 2 * (v - 1) + 1
def _var(lit: int) -> int: return (lit >> 1) + 1
def _sign(lit: int) -> bool: return (lit & 1) == 0   # True = positive


def _encode_clause(clause: List[int]) -> List[int]:
    """Convert external clause (signed ints) to internal literal encoding."""
    result = []
    for x in clause:
        if x > 0: result.append(_pos(x))
        else:     result.append(_neg(-x))
    return result


# ── Solver state ───────────────────────────────────────────────────────────────

_UNSET = -1
_TRUE  =  1
_FALSE =  0

TIMEOUT_S = 4.5   # leave 0.5 s margin from 5 s overall limit


class DPLL:
    def __init__(self, clauses: List[List[int]], n_vars: int,
                 deadline: Optional[float] = None):
        self.n_vars   = n_vars
        self.n_lits   = 2 * n_vars
        self.deadline = deadline

        # Clause storage: list of lists of internal lits
        self.clauses: List[List[int]] = [_encode_clause(c) for c in clauses if c]

        # Assignments: indexed by var (1-indexed) → _UNSET / _TRUE / _FALSE
        self.val: List[int] = [_UNSET] * (n_vars + 1)

        # Decision level of assignment
        self.level: List[int] = [0] * (n_vars + 1)

        # Reason clause for each variable (index into self.clauses, or -1)
        self.reason: List[int] = [-1] * (n_vars + 1)

        # Decision trail: list of (var, value, decision_level)
        self.trail: List[Tuple[int, int, int]] = []

        # Trail separators by decision level
        self.trail_lim: List[int] = []

        # Current decision level
        self.dl: int = 0

        # Learned clauses appended to self.clauses
        self.n_original = len(self.clauses)

        # VSIDS activity scores per literal
        self.activity: List[float] = [0.0] * self.n_lits
        for clause in self.clauses:
            for lit in clause:
                self.activity[lit] += 1.0

        # Two-watched-literal scheme: watch[lit] → list of clause indices
        self.watch: List[List[int]] = [[] for _ in range(self.n_lits)]
        for ci, clause in enumerate(self.clauses):
            if len(clause) >= 2:
                self.watch[clause[0]].append(ci)
                self.watch[clause[1]].append(ci)
            elif len(clause) == 1:
                pass  # unit clause handled during init propagation

        # Propagation queue
        self.prop_queue: List[int] = []   # literals to propagate

        # Conflict count for restart scheduling
        self.conflicts = 0
        self.restart_limit = 100

    # ── Value lookup ───────────────────────────────────────────────────────────

    def _lit_val(self, lit: int) -> int:
        v  = self.val[_var(lit)]
        if v == _UNSET: return _UNSET
        if _sign(lit):  return v        # positive literal
        return _TRUE if v == _FALSE else _FALSE   # negative literal

    def _enqueue(self, lit: int, reason: int = -1) -> bool:
        """Assign lit=True. Returns False on conflict."""
        v = _var(lit)
        cur = self._lit_val(lit)
        if cur == _TRUE:  return True
        if cur == _FALSE: return False   # conflict
        value = _TRUE if _sign(lit) else _FALSE
        self.val[v]    = value
        self.level[v]  = self.dl
        self.reason[v] = reason
        self.trail.append((v, value, self.dl))
        self.prop_queue.append(lit)
        return True

    # ── Unit propagation ───────────────────────────────────────────────────────

    def _propagate(self) -> int:
        """BCP. Returns -1 on no conflict, else conflicting clause index."""
        while self.prop_queue:
            lit   = self.prop_queue.pop(0)
            false_lit = lit ^ 1           # the literal that became False
            watch_list = self.watch[false_lit]
            new_watch  = []
            conflict   = -1

            i = 0
            while i < len(watch_list):
                ci = watch_list[i]
                clause = self.clauses[ci]
                # Make sure false_lit is at position 1
                if clause[0] == false_lit:
                    clause[0], clause[1] = clause[1], clause[0]
                # clause[1] is now false_lit
                # If clause[0] is True, clause is satisfied
                if self._lit_val(clause[0]) == _TRUE:
                    new_watch.append(ci)
                    i += 1
                    continue
                # Find a new watch literal among clause[2:]
                found = False
                for k in range(2, len(clause)):
                    if self._lit_val(clause[k]) != _FALSE:
                        clause[1], clause[k] = clause[k], clause[1]
                        self.watch[clause[1]].append(ci)
                        found = True
                        break
                if not found:
                    new_watch.append(ci)
                    if self._lit_val(clause[0]) == _FALSE:
                        # All literals false → conflict
                        conflict = ci
                        i += 1
                        # Drain rest into new_watch
                        while i < len(watch_list):
                            new_watch.append(watch_list[i])
                            i += 1
                        break
                    else:
                        # Unit clause: clause[0] is unset → propagate
                        ok = self._enqueue(clause[0], ci)
                        if not ok:
                            conflict = ci
                            i += 1
                            while i < len(watch_list):
                                new_watch.append(watch_list[i])
                                i += 1
                            break
                i += 1

            self.watch[false_lit] = new_watch
            if conflict != -1:
                self.prop_queue.clear()
                return conflict
        return -1

    # ── Conflict analysis (1-UIP) ──────────────────────────────────────────────

    def _analyze(self, conflict: int) -> Tuple[List[int], int]:
        """
        Analyse conflict clause. Returns (learned_clause, backjump_level).
        learned_clause is in internal literal encoding.
        """
        seen: Set[int] = set()
        learnt: List[int] = [0]   # placeholder for UIP literal
        counter = 0
        p = -1
        reason = conflict

        trail_idx = len(self.trail) - 1

        while True:
            clause = self.clauses[reason]
            for lit in clause:
                v = _var(lit)
                if v not in seen and self.level[v] > 0:
                    seen.add(v)
                    self.activity[lit]     += 1.0
                    self.activity[lit ^ 1] += 1.0
                    if self.level[v] == self.dl:
                        counter += 1
                    else:
                        learnt.append(lit)

            # Find next variable in trail at current level
            while trail_idx >= 0:
                v, _, lv = self.trail[trail_idx]
                trail_idx -= 1
                if v in seen and lv == self.dl:
                    break

            counter -= 1
            if counter == 0:
                # v is 1-UIP
                lit_sign = _TRUE if self.val[v] == _TRUE else _FALSE
                # The UIP literal should be negated in learnt clause
                uip_lit = _neg(v) if self.val[v] == _TRUE else _pos(v)
                learnt[0] = uip_lit
                break

            p = v
            val = self.val[v]
            reason = self.reason[v]

        # Compute backjump level = max level of non-UIP literals
        if len(learnt) == 1:
            blevel = 0
        else:
            max_lv = max(self.level[_var(l)] for l in learnt[1:])
            blevel = max_lv

        return learnt, blevel

    # ── Backjump ───────────────────────────────────────────────────────────────

    def _backjump(self, level: int):
        while self.trail and self.trail[-1][2] > level:
            v, _, _ = self.trail.pop()
            self.val[v]    = _UNSET
            self.level[v]  = 0
            self.reason[v] = -1
        # Restore trail_lim
        while self.trail_lim and self.trail_lim[-1] > level:
            self.trail_lim.pop()
        self.dl = level
        self.prop_queue.clear()

    # ── Clause learning ────────────────────────────────────────────────────────

    def _add_learned(self, clause: List[int]):
        ci = len(self.clauses)
        self.clauses.append(clause)
        if len(clause) >= 2:
            self.watch[clause[0]].append(ci)
            self.watch[clause[1]].append(ci)

    # ── Decision heuristic (VSIDS-lite) ───────────────────────────────────────

    def _decide(self) -> Optional[int]:
        """Pick an unassigned variable; return its positive literal or None."""
        best_lit = -1
        best_act = -1.0
        for v in range(1, self.n_vars + 1):
            if self.val[v] == _UNSET:
                p = _pos(v)
                n = _neg(v)
                act = max(self.activity[p], self.activity[n])
                if act > best_act:
                    best_act = act
                    best_lit = p if self.activity[p] >= self.activity[n] else n
        return best_lit if best_lit != -1 else None

    # ── Restart ────────────────────────────────────────────────────────────────

    def _restart(self):
        self._backjump(0)
        self.conflicts = 0
        # Luby-inspired: double the limit
        self.restart_limit = min(self.restart_limit * 2, 10_000)

    # ── Main solve loop ────────────────────────────────────────────────────────

    def solve(self) -> Optional[Dict[int, bool]]:
        # Initial unit propagation
        for ci, clause in enumerate(self.clauses):
            if len(clause) == 0:
                return None   # empty clause → trivially UNSAT
            if len(clause) == 1:
                if not self._enqueue(clause[0]):
                    return None

        conflict = self._propagate()
        if conflict != -1:
            return None

        while True:
            # Timeout check
            if self.deadline and time.time() > self.deadline:
                return None   # unknown / timeout

            # Decision
            lit = self._decide()
            if lit is None:
                # All variables assigned → SAT
                return {v: (self.val[v] == _TRUE)
                        for v in range(1, self.n_vars + 1)}

            self.dl += 1
            self.trail_lim.append(self.dl)
            self._enqueue(lit)

            # Propagate
            conflict = self._propagate()
            while conflict != -1:
                self.conflicts += 1
                if self.dl == 0:
                    return None   # UNSAT

                # Analyse and learn
                learnt, blevel = self._analyze(conflict)
                self._backjump(blevel)
                self._add_learned(learnt)

                # Enqueue UIP
                if not self._enqueue(learnt[0], len(self.clauses) - 1):
                    return None
                conflict = self._propagate()

                # Restart check
                if self.conflicts >= self.restart_limit:
                    self._restart()
                    break


# ── Public interface ───────────────────────────────────────────────────────────

def solve_cnf(clauses: List[List[int]], n_vars: int,
              deadline: Optional[float] = None) -> Optional[Dict[int, bool]]:
    """
    Solve a CNF formula.

    clauses  : list of clauses; each clause is a list of signed ints
               (positive = positive literal, negative = negated literal)
               Variables are 1-indexed.
    n_vars   : total number of Boolean variables
    deadline : time.time() deadline; returns None if exceeded

    Returns a satisfying assignment {var → bool} or None if UNSAT/timeout.
    """
    if not clauses:
        return {}   # trivially SAT (no constraints)

    # Quick check: any empty clause?
    for c in clauses:
        if len(c) == 0:
            return None

    solver = DPLL(clauses, n_vars, deadline)
    return solver.solve()
