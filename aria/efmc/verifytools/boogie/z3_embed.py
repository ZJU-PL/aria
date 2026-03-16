# pylint: disable=global-variable-not-assigned,no-self-argument
import aria.efmc.verifytools.boogie.ast as ast
import z3

from threading import Condition, local
from time import sleep, time
from random import randint
from multiprocessing import Process, Queue as PQueue
from aria.efmc.verifytools.common.util import error
import Pyro4
import sys
import atexit
from signal import signal, SIGTERM, SIGINT

ctxHolder = local()


def val_to_boogie(v):
    if (isinstance(v, z3.IntNumRef)):
        return ast.AstNumber(v.as_long())
    elif (isinstance(v, z3.BoolRef)):
        return ast.AstTrue() if ast.is_true(v) else ast.AstFalse()
    elif (v == "true"):
        return ast.AstTrue()
    elif (v == "false"):
        return ast.AstFalse()
    else:
        return ast.AstNumber(int(v))


def env_to_expr(env, suff=""):
    return ast.ast_and(
        [ast.AstBinExpr(ast.AstId(k + suff), "==", val_to_boogie(v))
         for (k, v) in env.iteritems()])


def getCtx():
    global ctxHolder
    ctx = getattr(ctxHolder, "ctx", None)
    if (ctx == None):
        ctx = z3.Context()
        ctxHolder.ctx = ctx
    return ctx


class WrappedZ3Exception(BaseException):
    def __init__(s, value):
        BaseException.__init__(s)
        s._value = value


def wrapZ3Exc(f):
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except z3.z3types.Z3Exception as e:
            raise WrappedZ3Exception(e.value)

    return wrapped


class Z3ServerInstance(object):
    def __init__(s):
        s._solver = z3.Solver(ctx=getCtx()) 

    @Pyro4.expose
    @wrapZ3Exc
    def add(s, sPred):
        pred = z3.parse_smt2_string(sPred, ctx=s._solver.ctx)
        s._solver.add(pred)
        return 0;

    @Pyro4.expose
    @wrapZ3Exc
    def check(s, sComm):
        sys.stderr.write("check(" + sComm + "):" + s._solver.to_smt2() + "\n");
        return str(s._solver.check());

    @Pyro4.expose
    @wrapZ3Exc
    def model(s):
        m = s._solver.model()
        return {x.name(): m[x].as_long() for x in m}

    @Pyro4.expose
    @wrapZ3Exc
    def push(s):
        s._solver.push();
        return 0;

    @Pyro4.expose
    @wrapZ3Exc
    def pop(s):
        s._solver.pop()
        return 0


def startAndWaitForZ3Instance():
    q = PQueue()            

    def runDaemon(q):
        import os

        out = "z3_child.%d.out" % os.getpid()
        err = "z3_child.%d.err" % os.getpid()

        error("Redirecting child", os.getpid(), "streams to", out, err)

        sys.stdout.close()
        sys.stderr.close()

        sys.stdout = open(out, "w")
        sys.stderr = open(err, "w")

        daemon = Pyro4.Daemon()
        uri = daemon.register(Z3ServerInstance)
        sys.stderr.write("Notify parent of my uri: " + str(uri) + "\n")
        sys.stderr.flush()
        q.put(uri)
        # Small window for racing
        daemon.requestLoop()    

    p = Process(target=runDaemon, args=(q,))
    p.start()
    uri = q.get()
    return p, uri


class Unknown(Exception):
    def __init__(s, q):
        Exception.__init__(s, str(q) + " returned unknown.")
        s.query = q


class Crashed(Exception):
    pass


class Z3ProxySolver:
    def __init__(s, uri, proc):
        s._uri = uri
        s._proc = proc
        s._remote = Pyro4.Proxy(uri)
        s._timeout = None

    def add(s, p):
        dummy = z3.Solver(ctx=getCtx())
        dummy.add(p)
        strP = dummy.to_smt2()
        strP = strP.replace("(check-sat)\n", "");
        s._remote.add(strP)
        return None

    def push(s):
        s._remote.push()
        return None

    def pop(s):
        s._remote.pop()
        return None

    def check(s, timeout=None, comm=""):
        old_timeout = s._timeout
        s._remote._pyroTimeout = timeout
        try:
            r = s._remote.check(comm)
        except Pyro4.errors.TimeoutError:
            sys.stderr.write("Child " + str(s._proc.pid) + \
                             "Timedout. Restarting.\n")
            r = "unknown"
            s._restartRemote();
        except Exception as e:
            sys.stdout.write("Got exception: " + str(e) + "\n")
            ecode = s._proc.exitcode
            s._restartRemote()

            if (ecode == -11):  # Retry Z3 segfaults
                r = "crashed"
            else:
                r = "unknown"
        finally:
            s._remote._pyroTimeout = old_timeout

        if (r == "sat"):
            return z3.sat
        elif (r == "unsat"):
            return z3.unsat
        elif (r == "unknown"):
            raise Unknown("storing query NYI in proxy solver")
        elif (r == "crashed"):
            raise Crashed()
        else:
            raise Exception("bad reply to check: " + str(r))

    def model(s):
        return s._remote.model()

    def to_smt2(s, p):
        dummy = z3.Solver(ctx=getCtx())
        dummy.add(p)
        strP = dummy.to_smt2()
        strP = strP.replace("(check-sat)\n", "")
        strP = strP.replace(
            "; benchmark generated from python API\n" + \
            "(set-info :status unknown)\n", "")
        return strP

    def _restartRemote(s):
        # Kill Old Process
        s._proc.terminate()
        s._proc.join()
        # Restart
        s._proc, s._uri = startAndWaitForZ3Instance()
        s._remote = Pyro4.Proxy(s._uri)
        s.push()


z3ProcessPoolCond = Condition()
MAX_Z3_INSTANCES = 100
ports = set(range(8100, 8100 + MAX_Z3_INSTANCES))
z3ProcessPool = {}


def _cleanupChildProcesses():
    for proxy in z3ProcessPool:
        proxy._proc.terminate()


atexit.register(_cleanupChildProcesses)

# atexit handlers don't get called on SIGTERM.
# cleanup child z3 processes explicitly on SIGTERM
def handler(signum, frame):
  _cleanupChildProcesses()
  sys.exit(-1)

for signum in [SIGTERM, SIGINT]:
  signal(signum, handler)


def getSolver():
    try:
        z3ProcessPoolCond.acquire()
        # Repeatedly GC dead processes and see what free context we have
        # If we have none wait on the condition variable for request to
        # finish or processes to timeout and die.
        while True:
            free = [(proxy, busy) for (proxy, busy) in z3ProcessPool.iteritems()
                    if (not busy)]

            if (len(free) == 0 and len(ports) == 0):
                print("No free instances and no ports for new instances...")
                z3ProcessPoolCond.wait();
            else:
                break;

        # We either have a free z3 solver or a process died and freed
        # up a port for us to launch a new solver with.
        if (len(free) > 0):
            # print("Assigning existing z3 proxy")
            solver = free[randint(0, len(free) - 1)][0]
        else:
            # print("Creating new z3 proxy")
            p, uri = startAndWaitForZ3Instance()
            solver = Z3ProxySolver(uri, p)

        z3ProcessPool[solver] = (True, False)
        # print("z3ProcessPool has ", len(z3ProcessPool), "entries")
    finally:
        z3ProcessPoolCond.release()

    solver.push()
    return solver


def releaseSolver(solver):
    if (solver == None):
        return;
    try:
        z3ProcessPoolCond.acquire();
        solver.pop();
        z3ProcessPool[solver] = False
        z3ProcessPoolCond.notify();
    finally:
        z3ProcessPoolCond.release();


def Int(n):
    return z3.Int(n, ctx=getCtx());


def Or(*args):
    return z3.Or(*(args + (getCtx(),)))


def And(*args):
    return z3.And(*(args + (getCtx(),)))


def Not(pred):
    return z3.Not(pred, ctx=getCtx())


def Implies(a, b):
    return z3.Implies(a, b, ctx=getCtx())

def Function(name, *params):
    return z3.Function(name, *params)

def IntVal(v):
    return z3.IntVal(v, ctx=getCtx())


def BoolVal(v):
    return z3.BoolVal(v, ctx=getCtx())


def counterex(pred, timeout=None, comm=""):
    s = None
    try:
        s = getSolver()
        while True:
            try:
                s.add(Not(pred))
                res = s.check(timeout, comm)
                m = None if res == z3.unsat else s.model()
            except Crashed:
                continue;
            break;

        return m;
    finally:
        if (s):
            releaseSolver(s);


def counterexamples(pred, num, timeout=None, comm=""):
    s = None
    assert num > 0
    try:
        s = getSolver()
        s.add(Not(pred))
        counterexes = []
        while len(counterexes) < num:
            try:
                res = s.check(timeout, comm)
                if res == z3.unsat:
                    break;

                env = s.model()
                counterexes.append(env)
                s.add(Not(env_to_expr(env)))
            except Crashed:
                continue
            break

        return counterexes
    finally:
        releaseSolver(s)


def satisfiable(pred, timeout=None):
    s = None
    try:
        s = getSolver()
        s.add(pred);
        res = s.check(timeout)
        return res == z3.sat;
    finally:
        releaseSolver(s)


def unsatisfiable(pred, timeout=None):
    s = None
    try:
        s = getSolver()
        s.add(pred);
        res = s.check(timeout)
        return res == z3.unsat;
    finally:
        releaseSolver(s)


def model(pred):
    s = None
    try:
        s = getSolver();
        s.add(pred);
        assert s.check() == z3.sat
        m = s.model();
        return m;
    finally:
        releaseSolver(s);


def maybeModel(pred):
    s = None
    try:
        s = getSolver();
        s.add(pred);
        res = s.check();
        m = s.model() if res == z3.sat else None
        return m
    finally:
        releaseSolver(s);


def simplify(pred, *args, **kwargs):
    # No need to explicitly specify ctx here since z3.simplify gets it from pred
    return z3.simplify(pred, *args, **kwargs)


def implies(inv1, inv2):
    return unsatisfiable(And(inv1, Not(inv2)))


def equivalent(inv1, inv2):
    return implies(inv1, inv2) and implies(inv2, inv1)


def tautology(inv):
    return unsatisfiable(Not(inv))


class AllIntTypeEnv:
    def __getitem__(s, i):
        return Int


def expr_to_z3(expr, typeEnv):
    if isinstance(expr, ast.AstNumber):
        return IntVal(expr.num)
    elif isinstance(expr, ast.AstId):
        return typeEnv[expr.name](expr.name)
    elif isinstance(expr, ast.AstTrue):
        return BoolVal(True);
    elif isinstance(expr, ast.AstFalse):
        return BoolVal(False);
    elif isinstance(expr, ast.AstFuncExpr):
        params = map((lambda p : expr_to_z3(p, typeEnv)), expr.ops)
        intsort = map((lambda p : z3.IntSort(ctx=getCtx())), expr.ops) + [z3.IntSort(ctx=getCtx())]
        f = Function(expr.funcName.name, *intsort)
        return f(*params)
    elif isinstance(expr, ast.AstUnExpr):
        z3_inner = expr_to_z3(expr.expr, typeEnv)
        if expr.op == '-':
            return -z3_inner
        elif expr.op == '!':
            return Not(z3_inner)
        else:
            raise Exception("Unknown unary operator " + str(expr.op))
    elif isinstance(expr, ast.AstBinExpr):
        e1 = expr_to_z3(expr.lhs, typeEnv)
        e2 = expr_to_z3(expr.rhs, typeEnv)
        if expr.op == "<==>":
            return And(Implies(e1, e2), Implies(e2, e1))
        elif expr.op == "==>":
            return Implies(e1, e2)
        elif expr.op == "||":
            return Or(e1, e2)
        elif expr.op == "&&":
            return And(e1, e2)
        elif expr.op == "==":
            return e1 == e2
        elif expr.op == "!=":
            return e1 != e2
        elif expr.op == "<":
            return e1 < e2
        elif expr.op == ">":
            return e1 > e2
        elif expr.op == "<=":
            return e1 <= e2
        elif expr.op == ">=":
            return e1 >= e2
        elif expr.op == "+":
            return e1 + e2
        elif expr.op == "-":
            return e1 - e2
        elif expr.op == "*":
            return e1 * e2
        elif expr.op == "div":
            return e1 / e2
        elif expr.op == "mod":
            return e1 % e2
        else:
            raise Exception("Unknown binary operator " + str(expr.op))
    else:
        raise Exception("Unknown expression " + str(expr))


def stmt_to_z3(stmt, typeEnv):
    if (isinstance(stmt, ast.AstLabel)):
        stmt = stmt.stmt

    if (isinstance(stmt, ast.AstAssignment)):
        lvalue = typeEnv[stmt.lhs](str(stmt.lhs))
        rhs = expr_to_z3(stmt.rhs, typeEnv)
        return (lvalue == rhs)
    elif (isinstance(stmt, ast.AstAssert)):
        return (expr_to_z3(stmt.expr, typeEnv))
    elif (isinstance(stmt, ast.AstAssume)):
        return (expr_to_z3(stmt.expr, typeEnv))
    else:
        raise Exception("Can't convert " + str(stmt))


def isnum(s):
    try:
        _ = int(s)
        return True
    except ValueError:
        return False


def ids(z3expr):
    if len(z3expr.children()) == 0:
        return [z3expr] if not isnum(str(z3expr)) else []
    else:
        childIds = reduce(lambda x, y: x + y, map(ids, z3expr.children()), [])
        tmp = {str(x): x for x in childIds}
        return list(tmp.keys())


def z3_expr_to_boogie(expr):
    d = expr.decl()
    if (d.arity() == 0):
        # Literals and Identifiers
        if (isinstance(expr.sort(), z3.BoolSortRef)):
            if (z3.is_true(expr)):  # No need for explicit ctx
                return ast.AstTrue()
            elif (z3.is_false(expr)):  # No need for explicit ctx
                return ast.AstFalse()
            else:
                return ast.AstId(d.name())
        else:
            assert isinstance(expr.sort(), z3.ArithSortRef), \
                "For now only handle bools and ints"
            if (z3.is_int_value(expr)):  # No need for explicit ctx
                return ast.AstNumber(int(str(expr)))
            else:
                return ast.AstId(d.name())
    elif (d.arity() == 1):
        # Unary operators
        arg = z3_expr_to_boogie(expr.children()[0])
        boogie_op = {
            '-': '-',
            'not': '!',
        }[d.name()]
        return ast.AstUnExpr(boogie_op, arg)
    elif (d.name() == "if" and d.arity() == 2):
        # TODO: Check types of branches are bool
        c = expr.children();
        cond = z3_expr_to_boogie(c[0])
        thenB = z3_expr_to_boogie(c[1])
        return ast.AstBinExpr(cond, "==>", thenB)
    elif (d.arity() == 2):
        # Binary operators
        try:
            boogie_op, assoc = {
                "+": ("+", "left"),
                "-": ("-", "left"),
                "*": ("*", "left"),
                "div": ("div", "left"),
                "mod": ("mod", "none"),
                "=": ("==", "none"),
                "distinct": ("!=", "none"),
                "<": ("<", "none"),
                ">": (">", "none"),
                "<=": ("<=", "none"),
                ">=": (">=", "none"),
                "and": ("&&", "left"),
                "or": ("||", "left"),
                "=>": ("==>", "none"),
                "Implies": ("==>", "none"),
            }[d.name()]
        except:
            boogie_op, assoc = d.name(), "func"
        c = expr.children()
        if assoc == "func":
            try:
                pars = map(z3_expr_to_boogie, c)
                func = ast.AstId(boogie_op)
                fun = ast.AstFuncExpr(func, *pars)
            except Exception as ex:
                error(ex.message)
            return fun
        else:
            while (len(c) > 2):
                if (assoc == "none"):
                    raise Exception("Error: Expression " + str(expr) + " has " + \
                                    len(c) + " children: " + c + " but root operator " + \
                                    d.name() + " is non-associative.")

                if (assoc == "left"):
                    lhs = z3_expr_to_boogie(c[0]) \
                        if (not isinstance(c[0], ast.AstNode)) else c[0]
                    rhs = z3_expr_to_boogie(c[1])
                    c[0:2] = [ast.AstBinExpr(lhs, boogie_op, rhs)]
                else:
                    raise Exception("NYI")

            lhs = z3_expr_to_boogie(c[0]) \
                if (not isinstance(c[0], ast.AstNode)) else c[0]
            rhs = z3_expr_to_boogie(c[1])
            return ast.AstBinExpr(lhs, boogie_op, rhs)
    elif (d.name() == "if"):
        # TODO: Check types of branches are bool
        c = expr.children();
        cond = z3_expr_to_boogie(c[0])
        thenB = z3_expr_to_boogie(c[1])
        elseB = z3_expr_to_boogie(c[2])
        return ast.AstBinExpr(
            ast.AstBinExpr(cond, "==>", thenB),
            "&&",
            ast.AstBinExpr(ast.AstUnExpr("!", cond), "==>", elseB))
    else:
        raise Exception("Can't translate z3 expression " + str(expr) +
                        " to boogie.")


def to_smt2(p):
    s = getSolver();
    res = s.to_smt2(p)
    releaseSolver(s);
    return res
