from collections import namedtuple
from bb import BB, get_bbs, bbEntry
from copy import copy
from ast import AstAssert, AstAssume, AstHavoc, AstAssignment, AstGoto, \
  AstReturn, AstUnExpr, AstBinExpr, AstNumber, AstId, AstTrue, AstFalse

# Program = Dict[str, BB]
# Value = Union[int, bool]
# PC = Tuple[str, int]
# Store = Dict[str, Value]
# State = Tuple[PC, Store, str]
# RandF = Callable[State, str]->Value
# ChoiceF = Callable[Program, [Set[State]], Set[State]]

# A program counter is a tuple (bb_name, next_stmt)
PC = namedtuple("PC", ["bb", "next_stmt"])
# Program state is a tuple (PC, Dict[str, int], str)
State = namedtuple("State", ["pc", "store", "status"])

# Possible statuses:
STALLED="STALLED"   # reached an assume that is false
FAILED="FAILED"     # reached an assert that is false
FINISHED="FINISHED" # reached end of program
RUNNING="RUNNING"   # can still make progress

_logical_un = {
  '!':  lambda x: not x,
}

_arith_un = {
  '-':  lambda x: -x,
}

_logical_bin = {
  "&&": lambda x, y:  x and y,
  "||": lambda x, y:  x or y,
  "==>": lambda x, y:  (not x) or y,
}

_arith_bin = {
  "+": lambda x, y:  x + y,
  "-": lambda x, y:  x - y,
  "*": lambda x, y:  x * y,
  "div": lambda x, y:  x // y,
  "mod": lambda x, y:  x % y,
# These are not really arithmetic (return bool not int) but for runtime type checking
# its sufficient to put them here
  "==": lambda x, y:  x == y,
  "!=": lambda x, y:  x != y,
  "<": lambda x, y:  x < y,
  ">": lambda x, y:  x > y,
  "<=": lambda x, y:  x <= y,
  ">=": lambda x, y:  x >= y,
}

class BoogieRuntimeExc(Exception):  pass

def check(v, typ):
  """
  Raise a runtime exception of the type of v is not typ
  """
  if not type(v) == typ:
    raise BoogieRuntimeExc("Expected {} not {}.".format(typ, v))

def eval_quick(expr, store):
  # type: (AstNode, Store) -> Value
  """
  Evaluate an expression in a given environment. Boogie expressions are always
  pure so we don't modify the store. Raise a runtime error on:
    - type mismatch
    - lookup of free id
    - div by 0
  """
  if isinstance(expr, AstNumber):
    return expr.num
  elif isinstance(expr, AstTrue):
    return True
  elif isinstance(expr, AstFalse):
    return False
  elif isinstance(expr, AstId):
    try:
      v = store[expr.name]
      check(v, int) # Currently can only handle int vars
      return v
    except KeyError:
      raise BoogieRuntimeExc("Unkown id {}".format(expr.name))
  elif isinstance(expr, AstUnExpr):
    inner = eval_quick(expr.expr, store)
    op = expr.op

    if (op in _logical_un):
      check(inner, bool)
      return _logical_un[op](inner)
    
    if (op in _arith_un):
      check(inner, int)
      return _arith_un[op](inner)

    assert False, "Unknown unary op {}".format(op)
  elif isinstance(expr, AstBinExpr):
    lhs, rhs = (eval_quick(expr.lhs, store), eval_quick(expr.rhs, store))
    op = expr.op

    if (op in _logical_bin):
      check(lhs, bool)
      check(rhs, bool)
      return _logical_bin[op](lhs, rhs)
    
    if (op in _arith_bin):
      check(lhs, int)
      check(rhs, int)

      if (op == 'div' and rhs == 0):
        raise BoogieRuntimeExc("Divide by 0")

      return _arith_bin[op](lhs, rhs)
    assert False, "Unknown binary op {}".format(op)
  else:
    assert False, "Unkown expression {}".format(expr)


def stalled(s):
  # type: (State) -> bool
  """
  Return true iff this state has stalled (reached unsatisfiable assume)
  """
  return s.status == STALLED


def active(s):
  # type: (State) -> bool
  """
  Return true iff this state can make progress
  """
  return s.status == RUNNING

def finished(s):
  # type: (State) -> bool
  """
  Return true iff this state is in the finished state
  """
  return s.status == FINISHED


def interp_one(bbs, state, rand):
  # type: (Program, State, RandF) -> Iterable[State]
  """
  Given a program bbs and a current state, return the set of possible next
  states
  """
  if not active(state):
    # Never escape a failed/stalled/finished state
    yield state
    return

  ((bb, next_stmt), store, status) = state
  assert bb in bbs and 0 <= next_stmt and next_stmt <= len(bbs[bb].stmts)

  if next_stmt == len(bbs[bb].stmts):
    # At end of BB - any successor is fair game
    for s in bbs[bb].successors:
      yield State(PC(s, 0), copy(store), status)

    # If no successors we are at the end of the funciton. Yield a finished
    # state
    if (len(bbs[bb].successors) == 0):
      yield State(PC(bb, next_stmt + 1), store, FINISHED)
    return
  else:
    # Inside of a BB - interp the next statment
    stmt = bbs[bb].stmts[next_stmt]

    if isinstance(stmt, AstAssert):
      v = eval_quick(stmt.expr, store)
      check(v, bool)
      if (not v):
        status = FAILED
    elif isinstance(stmt, AstAssume):
      v = eval_quick(stmt.expr, store)
      if (not v):
        status = STALLED
    elif isinstance(stmt, AstHavoc):
      store = copy(store)
      for var_id in stmt.ids:
        store[var_id.name] = rand(store, var_id.name)
    elif isinstance(stmt, AstAssignment):
      v = eval_quick(stmt.rhs, store)
      store = copy(store)
      store[stmt.lhs.name] = v
    else:
      assert False, "Can't handle statement {}".format(stmt)

    yield State(PC(bb, next_stmt + 1), store, status)

def trace_n(bbs, state, nsteps, rand, filt):
  # type: (Program, State, int, RandF, ChoiceF) -> Tuple[List[Trace], List[Trace]]
  """
  Given a program bbs and a current state state, and number of steps nsteps
  interpret the program for up to nsteps. Return two lists - the active traces
  (traces of length nsteps that can still make progress) and inactive traces
  (traces of length <=nsteps that are finished or failed).

    :param bbs: the basic blocks of the program to interpret
    :param state: the starting state of the program
    :param nsteps:  number of steps up to which to interpret
    :param rand:  a callback with signature (State, str) -> Value for providing
                  random values to havoc
    :param filt:  callback (Program, List[States])->List[States] called when we have multiple
                  possible next states. Allows the consumer to prune the non-determinism or change
                  exploration order.

    :return:  tuple (active_traces, inactive_traces).
              active_traces - a list of traces of length nsteps that can make
                              further progress
              inative_traces - a list of traces of length UP TO nsteps that are
                               either failed or finished.
  """
  active_traces = [ [state] ] 
  inactive_traces = []

  for step in range(nsteps):
    new_traces = [ ]

    for t in active_traces:
      new_states = list(interp_one(bbs, t[-1], rand))
      # Don't care about stalled traces
      new_states = [x for x in new_states if not(stalled(x))]
      if (len(new_states) > 1):
        # If execution is non-deterministic here, allow consumer to prune the list
        # of next states
        new_states = filt(bbs, new_states)

      for st in new_states:
        new_traces.append(t + [ st ])
    
    # active_traces are just the traces of length step  
    active_traces = [t for t in new_traces if active(t[-1])]
    # inactive_traces are all FAILED/FINISHED traces of length <= step 
    inactive_traces.extend([t for t in new_traces if not active(t[-1])])

  return (active_traces, inactive_traces)

def trace_n_from_start(bbs, starting_store, nsteps, rand, filt):
  starting_state = State(PC(bbEntry(bbs), 0), starting_store, RUNNING)
  return trace_n(bbs, starting_state, nsteps, rand, filt)

if __name__ == "__main__":
  from argparse import ArgumentParser
  from lib.common.util import error
  from random import randint, choice

  p = ArgumentParser(description="interpeter")
  p.add_argument("file", type=str, help="path to desugared boogie file to interpret")
  p.add_argument("starting_env", type=str,
    help="comma separated list of starting assignments to variables." +
         "e.g. a=4,b=10,x=0")
  p.add_argument("steps", type=int, help="the number of steps for which to interpret")
  p.add_argument("--nond-mode", type=str,
    help="mode controlling what to do when execution is non-deterministic. " + 
         "Possible values:\n" +
            " all - explore all branches of the execution tree\n" +
            " random_lookahead_1 - remove all states that will stall after 1 step " +
            "and pick a random one from the rest",
    choices=["all", "random_lookahead_1"], default="all")

  args = p.parse_args()

  bbs = get_bbs(args.file)
  starting_store = {  k : int(v) for (k, v) in
    [ x.split("=") for x in args.starting_env.split(",") ]
  }

  if (args.nond_mode == "all"):
    filt_f = lambda bbs, states:  states
  elif (args.nond_mode == "random_lookahead_1"):
    def f(bbs, states):
      def lookahead_one_filter(bbs, s):
        if s.pc.next_stmt == len(bbs[s.pc.bb]):
          return True

        stmt = bbs[s.pc.bb][s.pc.next_stmt]
        if not isinstance(stmt, AstAssume):
          return True

        v = eval_quick(stmt.expr, s.store)
        check(v, bool)
        return v

      feasible_states = filter(lambda x:  lookahead_one_filter(bbs, x), states)
      return [choice(feasible_states)]
    filt_f = f
  else:
    error("Usage: unknown nond-mode: {}".format(args.nond_mode))

  rand_f = lambda state, Id:  randint(-1000, 1000)

  starting_state = State(PC(bbEntry(bbs), 0), starting_store, RUNNING)
  (active, inactive) = trace_n(bbs, starting_state, args.steps, rand_f, filt_f)

  def pp_state(st):
    return "[{},{}]: ".format(st.pc.bb, st.pc.next_stmt) + \
           ",".join([k + "={}".format(v) for (k, v) in st.store.items()])

  def pp_trace(t):
    return "->\n".join(map(pp_state, t))


  print("Active({}):".format(len(active)))
  for t in active:
    print(pp_trace(t))
  print("Inactive({}):".format(len(inactive)))
  for t in inactive:
    print(pp_trace(t))
