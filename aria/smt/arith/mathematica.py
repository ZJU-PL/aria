"""
Mathematica-based solver for NRA.
"""


import logging

from fractions import Fraction

from concurrent.futures import TimeoutError

try:
  import wolframclient
  from wolframclient.evaluation import WolframLanguageSession
  from wolframclient.language import wl, wlexpr
  from wolframclient.evaluation.kernel.path import find_default_kernel_path
except:
  pass

from pysmt.exceptions import SolverAPINotFound
from pysmt.logics import QF_NRA, QF_LRA, NRA, LRA
from pysmt import typing as types
from pysmt.solvers.solver import Solver, Converter, SolverOptions
from pysmt.solvers.qelim import QuantifierEliminator

from pysmt.solvers.eager import EagerModel
from pysmt.walkers import DagWalker
from pysmt.constants import is_pysmt_fraction, is_pysmt_integer
from pysmt.decorators import clear_pending_pop, catch_conversion_error

from pysmt.exceptions import (PysmtException, ConvertExpressionError,
                              PysmtValueError, PysmtTypeError)
from pysmt.oracles import get_logic
from pysmt.typing import REAL
from pysmt.shortcuts import (
  get_env, TRUE, FALSE,
  Real, Symbol, Pow
)

class OutOfTimeSolverError(PysmtException):
  def __init__(self, budget):
    PysmtException.__init__(self,
                            "The solver used already a maximum budget (%s)" % str(budget))


def has_kernel():
  try:
    return not (find_default_kernel_path() is None)
  except:
    return None

def exit_callback_print_time(solver, outstream):
    if (not solver.session is None):
      outstream.write("Getting out of mathematica!\n")

      if (solver.options.budget_time != 0):
        outstream.write("Mathematica time: %s\n" % str(solver.used_time))
      else:
        outstream.write("Mathematica time (no budget): %s\n" % str(solver.session.evaluate(wl.TimeUsed())))


class MathematicaSession():
  _session = None

  @staticmethod
  def get_session():
    if has_kernel() is None or not has_kernel():
      raise SolverAPINotFound

    if MathematicaSession._session is None:
      if True:
        logging.debug("Creating a session for mathematica...")
        try:
          MathematicaSession._session = WolframLanguageSession()
          MathematicaSession._session.ensure_started()
        except:
          raise SolverAPINotFound
      else:
        logging.debug("Mathematica Kernel not found")
        raise SolverAPINotFound
    else:
        logging.debug("Mathematica session already exists...")

    return MathematicaSession._session

  def terminate_session():
    if not MathematicaSession._session is None:
      MathematicaSession._session.terminate()

class MathematicaOptions(SolverOptions):
  """Options for the Mathematica Solver.
  """

  def __init__(self, **base_options):
    SolverOptions.__init__(self, **base_options)

    self.budget_time = 0
    for k,v in self.solver_options.items():
      if k == "budget_time":
        try:
          v_float = float(v)
        except:
          raise ValueError("Invalid value for %s: %s" % \
                           (str(k),str(v)))
      else:
        raise ValueError("Unrecognized option '%s'." % k)
      # Store option
      setattr(self, k, v)

  def __call__(self, solver):
    # do nothing now
    pass

# EOC MathematicaOptions


class MathematicaSolver(Solver):
  """Solver based on the Mathematica Reduce function
  """
  LOGICS = [ QF_NRA ]
  OptionsClass = MathematicaOptions

  def __init__(self, environment, logic, **options):
    Solver.__init__(self,
                    environment=environment,
                    logic=logic,
                    **options)

    self.mgr = environment.formula_manager
    self.converter = MathematicaConverter(environment=self.environment)
    self.options(self)

    self.used_time = 0

    self.backtrack = []
    self.assertions_stack = []
    self.reset_assertions()

    self._exit_callback = None

    self.session = MathematicaSession.get_session()

  @clear_pending_pop
  def reset_assertions(self):
    true_formula = self.mgr.Bool(True)
    self.assertions_stack = [true_formula]

  @clear_pending_pop
  def add_assertion(self, formula, named=None):
    self.assertions_stack.append(formula)

  @clear_pending_pop
  def solve(self, assumptions=None):
    if assumptions is not None:
      self.push()
      self.add_assertion(self.mgr.And(assumptions))
      self.pending_pop = True

    to_solve = None
    for expr in self.assertions_stack:
      if to_solve is None:
        to_solve = expr
      else:
        to_solve = self.mgr.And(to_solve, expr)

    if to_solve is None:
      to_solve = self.mgr.Bool(True)

    # Here is where we call Reduce from Mathematica
    free_vars = to_solve.get_free_variables()

    generate_model = True

    if (not generate_model):
      exists_formula = self.mgr.Exists(free_vars, to_solve)
      mathematica_exists_formula = self.converter.convert(exists_formula)

      reduce_cmd = wl.Reduce(mathematica_exists_formula, wlexpr('Reals'))
    else:
      vars_list = wl.List(*[self.converter.convert(v) for v in free_vars])

      mathematica_formula = self.converter.convert(to_solve)
      reduce_cmd = wl.FindInstance(mathematica_formula, vars_list, wlexpr('Reals'))

    budget_time = self.options.budget_time
    if (self.options.budget_time > 0):
      remaining_time = (self.options.budget_time - self.used_time)

      if (remaining_time <= 0):
        if (not None is self._exit_callback):
          self._exit_callback(self)
        raise OutOfTimeSolverError(budget_time)

      logging.debug("Dummy command...")
      timing = self.session.evaluate(wl.TimeUsed())
      logging.debug("Dummy command done...")

      timed_eval_cmd = wl.TimeConstrained(reduce_cmd, wlexpr('%s' % remaining_time))

      logging.debug("About to evaluate an expression with Mathematica...")
      exist_res = self.session.evaluate(timed_eval_cmd)
      logging.debug("Mathematica expression evaluated...")
    else:
      exist_res = self.session.evaluate(reduce_cmd)

    # Invalidate cached model
    self.latest_model = None

    # TODO: generate a pysmt model to be compliant
    if generate_model:
      if (isinstance(exist_res, wolframclient.language.expression.WLFunction) and
          isinstance(exist_res[0] , (bool))):
        if exist_res[0]:
          d = {}
          model = EagerModel(assignment=d)
          self.latest_model = model
          exist_res = True
        else:
          exist_res = False
      elif len(exist_res) > 0:

        # Construct the model
        d = {}
        for assignment in exist_res[0]:
          if (isinstance(assignment, wolframclient.language.expression.WLFunction)):
            node_type = assignment.head
            if (isinstance(node_type, wolframclient.language.expression.WLSymbol)):
              try:
                var = self.converter.back(assignment.args[0])
                value = self.converter.back(assignment.args[1])
              except NotImplementedError:
                # return result but invalidate the model
                # Need to deal with algebraic numbers...
                self.latest_model = None
                return exist_res

              d[var] = value
            else:
              raise NotImplementedError("Error parsing the models from mathematica!")
          else:
            raise NotImplementedError("Error parsing the models from mathematica!")

        model = EagerModel(assignment=d)
        self.latest_model = model
        exist_res = True
      else:
        if (self.options.budget_time > 0):
          if (exist_res == ()):
            exist_res = False
          elif (exist_res.name == '$Aborted'):
            self.used_time += remaining_time
            if (not None is self._exit_callback):
              self._exit_callback(self)
            raise OutOfTimeSolverError(self.options.budget_time)
        else:
          exist_res = False
    else:
      if (self.options.budget_time > 0):
        if (type(exist_res) != bool):
          if (exist_res.name == '$Aborted'):
            self.used_time += remaining_time
            if (not None is self._exit_callback):
              self._exit_callback(self)
            raise OutOfTimeSolverError(self.options.budget_time)

    if (self.options.budget_time > 0):
        self.used_time = self.session.evaluate(wl.TimeUsed())

    return exist_res


  def find_min(self, function, constraints, round_precision = 6):
    return self.find_optimal(function, constraints, True, round_precision)

  def find_max(self, function, constraints, round_precision = 6):
    return self.find_optimal(function, constraints, False, round_precision)

  def find_optimal(self, function, constraints, is_minimum = True, round_precision = 6):
    # Note: The method is numeric and returns floating point (i.e., it's not
    # guarnateed.
    #
    def myround(n,k):
      rounded = round(float(n),k)
      rounded = Fraction.from_float(rounded)
      rounded = rounded.limit_denominator(pow(10,k))
      return Real(rounded)

    free_vars = function.get_free_variables()
    free_vars = free_vars.union(constraints.get_free_variables())

    function_mat = self.converter.convert(function)
    constraints_mat =  self.converter.convert(constraints)
    vars_list = wl.List(*[self.converter.convert(v) for v in free_vars])

    if is_minimum:
      cmd = wl.FindMinimum(wl.List(function_mat, constraints_mat),
                           vars_list)
      infinity = wl.DirectedInfinity(-1)
    else:
      cmd = wl.FindMaximum(wl.List(function_mat, constraints_mat),
                           vars_list)
      infinity = wl.DirectedInfinity(1)

    res = self.session.evaluate(cmd)

    if (infinity == res[0]):
      # No optimal
      return None, None

    # Here we have a result, parsing optimal and model
    opt_value = myround(res[0], round_precision)

    # Build the model finding the value
    assignment = {}
    for rule in res[1]:
      assert len(rule.args) == 2
      assert (type(rule.args[1]) == float)

      var = self.converter.back(rule.args[0])
      assert not var in assignment

      value = myround(rule.args[1], round_precision)

      assignment[var] = value

    model = EagerModel(assignment=assignment)
    return opt_value, model


  def get_value(self, item):
    assert (not self.latest_model is None)
    return self.latest_model[item]

  def get_model(self):
    # We should call FindInstance to find a model (instead o reduce)
    # The main issue is to parse algebraic numbers
    assert (not self.latest_model is None)

    return self.latest_model

  @clear_pending_pop
  def push(self, levels=1):
    for _ in range(levels):
      self.backtrack.append(len(self.assertions_stack))

  @clear_pending_pop
  def pop(self, levels=1):
    for _ in range(levels):
      l = self.backtrack.pop()
      self.assertions_stack = self.assertions_stack[:l]

  def set_exit_callback(self, callback):
    self._exit_callback = callback

  def _exit(self):
    if (not None is self._exit_callback):
      self._exit_callback(self)

# EOC MathematicaSolver


class MathematicaConverter(Converter, DagWalker):
  """ Convert a pysmt formula in a mathematica formula.

  Does not implement the back conversion!
  """

  @staticmethod
  def powertotimes(term, args):
    # issue - mathematica returns negative powers for variables
    # we would need to pre-process mathematica result to
    # - remove the variables from the denominators
    # - add a condition forcing the denominator to be non-zero (a new conjunction)
    #   - Mathematica usually already has such condition in the formula

    raise NotImplementedError("Conversion of Pow operator from mathematica not supported ")

    # return Pow(term, args[0])

  def sanitize(self, identifier):
    """ Just returns the identifier removing special characters """
    return identifier.replace("_", "underscore")

  def __init__(self, environment):
    DagWalker.__init__(self)

    self.environment = environment
    self.mgr = self.environment.formula_manager
    self._get_type = environment.stc.get_type

    # todo: remember mapping of symbols
    # todo: implement back mapping

    self.back_memoization = {}
    self.back_fun = {
      # wl.True
      # mathsat.MSAT_TAG_TRUE: lambda term, args: self.mgr.TRUE(),
      # mathsat.MSAT_TAG_FALSE:lambda term, args: self.mgr.FALSE(),

      wl.Plus : self._back_adapter(self.mgr.Plus),
      wl.Times : self._back_adapter(self.mgr.Times),
      wl.Divide : self._back_adapter(self.mgr.Div),

      wl.Equal : self._back_adapter(self.mgr.Equals),
      wl.LessEqual : self._back_adapter(self.mgr.LE),
      wl.Less : self._back_adapter(self.mgr.LT),

      wl.GreaterEqual : lambda term, args: self.mgr.LE(args[1],args[0]),
      wl.Greater : lambda term, args: self.mgr.LT(args[1], args[0]),

      wl.And : self._back_adapter(self.mgr.And),
      wl.Or : self._back_adapter(self.mgr.Or),
      wl.Not : self._back_adapter(self.mgr.Not),
      wl.Equivalent : self._back_adapter(self.mgr.Iff),
      wl.Implies : self._back_adapter(self.mgr.Implies),

      wl.Exists : lambda term, args: self.mgr.Exists(),
      wl.ForAll : lambda term, args: self.mgr.ForAll() ,

      wl.Power : lambda term, args: MathematicaConverter.powertotimes(term, args) ,

      # # Symbols, Constants and UFs have TAG_UNKNOWN
      # mathsat.MSAT_TAG_UNKNOWN: self._back_tag_unknown,
    }

    return

  def back(self, expr):
    return self._walk_back(expr, self.mgr)

  def _back_adapter(self, op):
    """Create a function that for the given op.
      This is used in the construction of back_fun, to simplify the code.
    """
    def back_apply(term, args):
      return op(*args)
    return back_apply

  def _back_single_term(self, term, mgr, args):
    """Builds the pysmt formula given a term and the list of formulae
    obtained by converting the term children.
    :param term: The Mathematica term to be transformed in pysmt formulae
    :type term: mathematica term
    :param mgr: The formula manager to be sued to build the
    formulae, it should allow for type unsafety.
    :type mgr: Formula manager
    :param args: List of the pysmt formulae obtained by converting
    all the args (obtained by mathsat.msat_term_get_arg()) to
    pysmt formulae
    :type args: List of pysmt formulae
    :returns The pysmt formula representing the given term
    :rtype Pysmt formula
    """

    head = term.head

    try:
      return self.back_fun[head](term, args)
    except KeyError:
      raise ConvertExpressionError("Unsupported expression:",
                                   repr(term))

  def _walk_back(self, term, mgr):
    stack = [term]
    while len(stack) > 0:
      current = stack.pop()

      if (isinstance(current, bool)):
        res = TRUE() if current else FALSE()
        self.back_memoization[current] = res
      elif (isinstance(current, int)):
        res = Real(current)
        self.back_memoization[current] = res
      elif (isinstance(current, wolframclient.language.expression.WLSymbol)):
        name = current.name

        if (current in self.back_memoization):
          return self.back_memoization[current]
        else:
          # Assume symbol is a real variable for now

          prefix = 'Global`'
          assert(name.startswith(prefix))
          var_name = name[len(prefix):]
          res = self.mgr.Symbol(var_name, REAL)
          self.back_memoization[current] = res

      elif (isinstance(current, wolframclient.language.expression.WLFunction) and
            current.head == wl.Rational):
        res = Real(Fraction(current.args[0],current.args[1]))
        self.back_memoization[current] = res
      else:
        arity = len(current.args)
        if current not in self.back_memoization:
          self.back_memoization[current] = None
          stack.append(current)
          for i in range(arity):
            son = current.args[i]
            stack.append(son)
        elif self.back_memoization[current] is None:
          args = [self.back_memoization[current.args[i]] for i in range(arity)]
          res = self._back_single_term(current, mgr, args)
          self.back_memoization[current] = res
        else:
          # we already visited the node, nothing else to do
          pass
    return self.back_memoization[term]

  @catch_conversion_error
  def convert(self, formula):
    """Convert a PySMT formula into a Mathematica formula.

    Now we only allow symbols to be Real, and everything we use the
    Real domain.
    Mathematica has other domains (e.g., Boolean) but I'm not sure
    it implements any theory combination.

    This function might throw an exception if
    an error during conversion occurs.
    """
    # May rewrite Boolean variables to use Real variables in the
    # future.
    res = self.walk(formula)

    return res

  def walk_symbol(self, formula, **kwargs):
    # TODO check the type!
    if not formula.is_symbol():
      raise PysmtTypeError("Trying to declare as a variable something "
                           "that is not a symbol: %s" % formula)

    if not formula.symbol_type().is_real_type():
      raise ConvertExpressionError("Trying to declare a symbol that "
                                   "is not of Real type (%s : %s)" % (str(formula.symbol_type()),
                                                                      formula.symbol_name()))

    sanitized = self.sanitize(formula.symbol_name())

    # res = wlexpr(sanitized)
    res =  wolframclient.language.Global.__getattr__(sanitized)

    # If this happens then another term different from formula
    # was converted to the same res

    # if (res in self.back_memoization):
    #   print(formula != self.back_memoization[res])
    #   print(sanitized)
    #   print(res)
    #   print(self.back_memoization[res])
    #   print(formula)

    assert not (res in self.back_memoization and
                formula != self.back_memoization[res])

    self.back_memoization[res] = formula

    return res

  def walk_real_constant(self, formula, **kwargs):
    assert is_pysmt_fraction(formula.constant_value())
    frac = formula.constant_value()
    n,d = frac.numerator, frac.denominator
    rep = str(n) + "/" + str(d)

    return wlexpr(rep)

  def walk_int_constant(self, formula, **kwargs):
    raise ConvertExpressionError("Integer constants (%s) are not"
                                 "allowed!" % str(formula) )

  def walk_bool_constant(self, formula, **kwargs):
    if formula.constant_value():
      return wlexpr('True')
    else:
      return wlexpr('False')

  def walk_bv_constant(self, formula, **kwargs):
    raise ConvertExpressionError("BV constants (%s) are not"
                                 "allowed!" % str(formula) )

  def walk_plus(self, formula, args, **kwargs):
    res = wl.Plus(*args)
    return res

  def walk_minus(self, formula, args, **kwargs):
    assert(len(args) == 2)
    return wl.Plus(args[0], wl.Minus(args[1]))

  def walk_times(self, formula, args, **kwargs):
    res = wl.Times(*args)
    return res

  def walk_div(self, formula, args, **kwargs):
    return wl.Divide(args[0],args[1])

  def walk_equals(self, formula, args, **kwargs):
    return wl.Equal(args[0], args[1])

  def walk_le(self, formula, args, **kwargs):
    return wl.LessEqual(args[0], args[1])

  def walk_lt(self, formula, args, **kwargs):
    return wl.Less(args[0], args[1])

  def walk_and(self, formula, args, **kwargs):
    return wl.And(*args)

  def walk_or(self, formula, args, **kwargs):
    return wl.Or(*args)

  def walk_not(self, formula, args, **kwargs):
    return wl.Not(args[0])

  def walk_iff(self, formula, args, **kwargs):
    return wl.Equivalent(args[0], args[1])

  def walk_implies(self, formula, args, **kwargs):
    return wl.Implies(args[0], args[1])

  def walk_ite(self, formula, args, **kwargs):
    i = args[0]
    t = args[1]
    e = args[2]

    if self._get_type(formula).is_bool_type():
      impl = self.mgr.Implies(formula.arg(0), formula.arg(1))
      th = self.walk_implies(impl, [i,t])
      nif = self.mgr.Not(formula.arg(1))
      ni = self.walk_not(nif, [i])
      el = self.walk_implies(self.mgr.Implies(nif, formula.arg(2)), [ni,e])
      return wl.And(th, el)
    else:
      raise ConvertExpressionError("Trying to convert a non-boolean "
                                   "ITE statement (%s)" % formula)

  def walk_exists(self, formula, args, **kwargs):
    assert len(args) == 1
    sf = args[0]
    varset = [self.walk_symbol(x) for x in formula.quantifier_vars()]
    if len(varset) == 0:
      return sf
    return wl.Exists(varset, sf)

  def walk_forall(self, formula, args, **kwargs):
    assert len(args) == 1
    sf = args[0]
    varset = [self.walk_symbol(x) for x in formula.quantifier_vars()]
    if len(varset) == 0:
      return sf
    return wl.ForAll(varset, sf)

  def walk_function(self, formula, args, **kwargs):
    raise ConvertExpressionError("Uninterpreted functions (%s) are not "
                                 "allowed!" % str(formula) )

  def walk_toreal(self, formula, args, **kwargs):
    raise ConvertExpressionError("toreal operator (%s) are not "
                                 "allowed!" % str(formula) )
# EOC MathematicaConverter


def get_mathematica(env=get_env(), budget_time=0, exit_callback=None):
  try:
    import wolframclient
  except:
    raise SolverAPINotFound

  solver = MathematicaSolver(env, QF_NRA,
                             solver_options={"budget_time" : budget_time})

  if (not exit_callback is None):
    solver.set_exit_callback(exit_callback)

  return solver




class MathematicaQuantifierEliminator(QuantifierEliminator):

  LOGICS = [LRA, NRA]

  def __init__(self, environment, logic=None):
    """
    """
    QuantifierEliminator.__init__(self)

    self.mathematica_solver = get_mathematica(env=environment)
    self.converter = self.mathematica_solver.converter
    self.session = self.mathematica_solver.session
    self.logic = logic


  def eliminate_quantifiers(self, formula):
    """Returns a quantifier-free equivalent formula of `formula`."""
    mathematica_formula = self.converter.convert(formula)
    reduce_cmd = wl.Reduce(mathematica_formula, wlexpr('Reals'))
    result = self.session.evaluate(reduce_cmd)

    result_pysmt = self.converter.back(result)

    return result_pysmt


  def _exit(self):
      MathematicaSession.terminate_session()
