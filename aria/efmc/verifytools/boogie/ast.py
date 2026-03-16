
#pylint: disable=no-self-argument,multiple-statements
from functools import reduce
from aria.efmc.verifytools.boogie.grammar import BoogieParser
from pyparsing import ParseResults
from aria.efmc.verifytools.common.ast import AstNode, replace, reduce_nodes

def _strip(arg):
    if isinstance(arg, ParseResults):
        return arg[0]
    else:
        return arg

class AstProgram(AstNode):
    def __init__(s, decls): AstNode.__init__(s, decls)
    def __str__(s): return "\n".join(map(str, s.decls))

class AstImplementation(AstNode):
    def __init__(s, name, signature, body):
        AstNode.__init__(s, name, signature, body)
    def __str__(s):
        return "implementation " + s.name + " " + str(s.signature) + str(s.body)

class AstBinding(AstNode):
    def __init__(s, names, typ):  AstNode.__init__(s, names, typ)
    def __str__(s): return ",".join(map(str, s.names)) + " : " + str(s.typ)

class AstLabel(AstNode):
    def __init__(s, label, stmt):  AstNode.__init__(s, label, stmt)
    def __str__(s): return str(s.label) + " : " + str(s.stmt)

class AstAssignment(AstNode):
    def __init__(s, lhs, rhs):  AstNode.__init__(s, lhs, rhs)
    def __str__(s): return str(s.lhs) + " := " + str(s.rhs) + ";"

class AstIntType(AstNode):
    def __init__(s):  AstNode.__init__(s)
    def __str__(s): return "int"

class AstBody(AstNode):
    def __init__(s, bindings, stmts):   AstNode.__init__(s, bindings, stmts)
    def __str__(s):
        return "{\n" + "\n".join(["var " + str(x) + ";" for x in s.bindings]) +\
                "\n" +\
                "\n".join([str(x) for x in s.stmts]) + \
                "\n}"

class AstStmt(AstNode): pass
class AstOneExprStmt(AstStmt):
    def __init__(s, expr):  AstStmt.__init__(s, expr)

class AstAssert(AstOneExprStmt):
    def __str__(s): return "assert (" + str(s.expr) + ");";

class AstAssume(AstOneExprStmt):
    def __str__(s): return "assume (" + str(s.expr) + ");";

class AstHavoc(AstStmt):
    def __init__(s, ids):  AstStmt.__init__(s, ids)
    def __str__(s): return "havoc " + ",".join(map(str, s.ids)) + ";"

# Returns is for now without argument
class AstReturn(AstStmt):
    def __init__(s):  AstStmt.__init__(s)
    def __str__(s): return "return ;"

class AstGoto(AstStmt):
    def __init__(s, labels):  AstStmt.__init__(s, labels)
    def __str__(s): return "goto " + ",".join(map(str, s.labels)) + ";"

class AstWhile(AstStmt):
    def __init__(s, condition, invariants, body):  
        AstStmt.__init__(s, condition, invariants, body)
    def __str__(s): 
        inv_str = "\n".join([f"  invariant {inv};" for inv in s.invariants]) if s.invariants else ""
        return f"while ({s.condition})\n{inv_str}\n{s.body}"

class AstBlock(AstStmt):
    def __init__(s, stmts):  AstStmt.__init__(s, stmts)
    def __str__(s): 
        return "{\n" + "\n".join([f"  {stmt}" for stmt in s.stmts]) + "\n}"

class AstUnExpr(AstNode):
    def __init__(s, op, expr):  AstNode.__init__(s, op, expr)
    def __str__(s): return s.op + str(s.expr)

class AstFalse(AstNode): 
    def __init__(s):  AstNode.__init__(s)
    def __str__(s): return "false"

class AstTrue(AstNode):
    def __init__(s):  AstNode.__init__(s)
    def __str__(s): return "true"

class AstNumber(AstNode): 
    def __init__(s, num):   AstNode.__init__(s,num)
    def __str__(s): return str(s.num)

class AstId(AstNode): 
    def __init__(s, name):  AstNode.__init__(s, name)
    def __str__(s): return str(s.name)

class AstBinExpr(AstNode):
    def __init__(s, lhs, op, rhs):  AstNode.__init__(s, lhs, op, rhs)
    def __str__(s):
        return "(" + str(s.lhs) + " " + str(s.op) + " " + str(s.rhs) + ")"

class AstFuncExpr(AstNode):
    def __init__(s, funcName, *ops):  AstNode.__init__(s, funcName, *ops)
    def __str__(s):
        return str(s.funcName) + "(" + ",".join(map(str, s.ops)) +  ")"

class AstBuilder(BoogieParser):
  def onAtom(s, prod, st, loc, toks):
    assert len(toks) == 1
    if prod == s.TRUE:
      return [ AstTrue() ]
    elif prod == s.FALSE:
      return [ AstFalse() ]
    elif prod == s.Number:
      return [ AstNumber(int(toks[0])) ]
    else:
      return [ AstId(str(toks[0])) ]

  def onUnaryOp(s, prod, st, loc, toks):
    return [ AstUnExpr(*toks) ]

  def onLABinOp(s, prod, st, loc, toks):
    if (len(toks) == 3):
      return [ AstBinExpr(*toks) ]
    else:
      assert(len(toks) > 3);
      base = AstBinExpr(*toks[:3])
      rest = [ [toks[3+2*k], toks[3+2*k+1]] for k in range((len(toks)-3)/2) ]
      return [ reduce(lambda acc,el:  AstBinExpr(acc, el[0], el[1]), \
                      rest, base)
             ]

  def onRABinOp(s, prod, st, loc, toks):
    if (len(toks) == 3):
      return [ AstBinExpr(*toks) ]
    else:
      assert(len(toks) > 3);
      toks = reversed(toks)
      base = AstBinExpr(*toks[:3])
      return [ reduce(lambda acc,el:  AstBinExpr(acc, el[0], el[1]), \
                      toks[3:], base)
             ]

  def onNABinOp(s, prod, st, loc, toks):
    assert (len(toks) == 3);
    return [ AstBinExpr(*toks) ]

  # Statements
  def onAssert(s, prod, st, loc, toks):
    assert (len(toks) == 1)
    return [ AstAssert(toks[0]) ]
  def onAssume(s, prod, st, loc, toks):
    assert (len(toks) == 1)
    return [ AstAssume(toks[0]) ]
  def onReturn(s, prod, st, loc, toks):
    assert (len(toks) == 0)
    return [ AstReturn() ]
  def onGoto(s, prod, st, loc, toks):
    assert(len(toks) > 0)
    return [ AstGoto(toks) ]
  def onAssignment(s, prod, st, loc, toks):
    assert (len(toks) == 2)
    assert (len(toks[0]) == 1)
    assert (len(toks[1]) == 1)
    return [ AstAssignment(toks[0][0], toks[1][0]) ]
  def onHavoc(s, prod, st, loc, toks):
    assert (len(toks) > 0)
    return [ AstHavoc(toks) ]
  def onWhile(s, prod, st, loc, toks):
    # The grammar is: WHILE + LPARN + WildcardExpr + RPARN + ZoM(LoopInv) + BlockStmt
    # But the parser includes the WHILE keyword in tokens, so:
    # toks = ["while", condition, [invariants...], body]
    
    if len(toks) >= 3:
        # Skip the "while" keyword (toks[0])
        condition = toks[1]  # The condition expression
        body = toks[-1]      # Last token is the body
        invariants = toks[2:-1] if len(toks) > 3 else []  # Everything in between
    else:
        condition = AstTrue()
        invariants = []
        body = AstBlock([])
    
    return [ AstWhile(condition, invariants, body) ]
  def onBlock(s, prod, st, loc, toks):
    return [ AstBlock(toks) ]
  def onProgram(s, prod, st, loc, toks): 
    return [ AstProgram(toks) ]
  def onLocalVarDecl(s, prod, st, loc, toks): 
    return [ AstBinding(map(str, toks[:-1]), toks[-1]) ]
  def onType(s, prod, st, loc, toks):
    # Currently only handle ints
    assert len(toks) == 1 and toks[0] == s.INT;
    return [ AstIntType() ];
  def onBody(s, prod, st, loc, toks):
    assert len(toks) == 2;
    return [ AstBody(toks[0], toks[1]) ]
  def onImplementationDecl(s, prod, st, loc, toks):
    attrs = toks[0]
    assert(len(attrs) == 0)
    name = str(toks[1])
    sig = toks[2]
    # For now ignore anything other than the argument list
    assert len(sig) == 3 and len(sig[0]) == 0 and len(sig[2]) == 0
    signature = None; #sig[1]
    body = toks[3][0]
    return [ AstImplementation(name, signature, body) ]
  def onLabeledStatement(s, prod, st, loc, toks):
    return AstLabel(*toks);

astBuilder = AstBuilder();

def parseExprAst(s):
  try:
    return astBuilder.parseExpr(s);
  except:
    print("Failed parsing");
    raise;

def parseAst(s):
  try:
    return astBuilder.parseProgram(s);
  except:
    print("Failed parsing");
    raise;

def expr_read(ast):
    if isinstance(ast, AstId):
        return set([ast.name])
    elif isinstance(ast, AstNumber):
        return set()
    elif isinstance(ast, AstUnExpr):
        return expr_read(ast.expr)
    elif isinstance(ast, AstBinExpr):
        return expr_read(ast.lhs).union(expr_read(ast.rhs))
    elif isinstance(ast, AstTrue) or isinstance(ast, AstFalse):
        return set()
    else:
        raise Exception("Unknown expression type " + str(ast))

def stmt_read(ast):
    if isinstance(ast, AstLabel):
        ast = ast.stmt

    if isinstance(ast, AstAssume) or isinstance(ast, AstAssert):
        return expr_read(ast.expr)
    elif isinstance(ast, AstAssignment):
        return expr_read(ast.rhs)
    elif isinstance(ast, AstHavoc):
        return set()
    elif isinstance(ast, AstWhile):
        # Read variables from condition and body
        vars_read = expr_read(ast.condition)
        if isinstance(ast.body, AstBlock):
            for stmt in ast.body.stmts:
                vars_read.update(stmt_read(stmt))
        else:
            vars_read.update(stmt_read(ast.body))
        return vars_read
    elif isinstance(ast, AstBlock):
        vars_read = set()
        for stmt in ast.stmts:
            vars_read.update(stmt_read(stmt))
        return vars_read
    else:
        raise Exception("Unknown statement: " + str(ast))

def stmt_changed(ast):
    if isinstance(ast, AstLabel):
        ast = ast.stmt

    if isinstance(ast, AstAssignment):
        return expr_read(ast.lhs)
    elif isinstance(ast, AstHavoc):
        return set(map(lambda x:  x.name, ast.ids))
    elif isinstance(ast, AstAssume) or isinstance(ast, AstAssert):
        return set([])
    elif isinstance(ast, AstWhile):
        # Variables changed in the while body
        vars_changed = set()
        if isinstance(ast.body, AstBlock):
            for stmt in ast.body.stmts:
                vars_changed.update(stmt_changed(stmt))
        else:
            vars_changed.update(stmt_changed(ast.body))
        return vars_changed
    elif isinstance(ast, AstBlock):
        vars_changed = set()
        for stmt in ast.stmts:
            vars_changed.update(stmt_changed(stmt))
        return vars_changed
    else:
        raise Exception("Unknown statement: " + str(ast))

def ast_group_bin(exprs, op, default):
    if (len(exprs) == 0):
      return default
    if (len(exprs) == 1):
      return exprs[0]

    return reduce(lambda x,y:   AstBinExpr(x, op, y), exprs[1:], exprs[0])

def ast_and(exprs): return ast_group_bin(list(exprs), "&&", AstTrue())
def ast_or(exprs): return ast_group_bin(list(exprs), "||", AstFalse())

def normalize(ast):
  # There are 2 ways to encode "-1" - as an AstUnExpr or an AstNumber. We pick
  # the AstUnExpr to be the canonical one for compatibility with the grammar
  # TODO: What happens when we parse -0?
  if (isinstance(ast, AstNumber) and ast.num < 0):
    return AstUnExpr("-", AstNumber(abs(ast.num)))
  # There are 2 ways to encode implication - as a ==> b or (!a) || b. The later
  # usually comes from the frontend, since JS lacks an explicit ==> operator.
  # We pick a ==> b to be canonical

  if (isinstance(ast, AstBinExpr) and ast.op == "||" and \
      isinstance(ast.lhs, AstUnExpr) and ast.lhs.op == "!"):
    return AstBinExpr(ast.lhs.expr, "==>", ast.rhs)

  if (isinstance(ast, AstNode)):
    return ast.__class__(*tuple(map(normalize, ast._children)))
  else:
    return ast;

def ast_constants(n):
  def cb(node, children):
    if isinstance(node, AstNumber):
      return set([node.num])
    else:
      return reduce(lambda x,y: x.union(y), children, set())
  return reduce_nodes(n, cb);


def ast_boolean_exprs(n):
  def cb(node, children):
    relOps = [ "<", ">", "<=", ">=", "==",  "!=" ]
    boolOps = [ "||", "&&", "==>", "<==>" ]
    isBoolean = (isinstance(node, AstUnExpr) and node.op == "!") or \
                (isinstance(node, AstBinExpr) and node.op in (relOps + boolOps))
    boolSubexp = reduce(lambda x,y: x.union(y), children, set([]))
    if (isBoolean):
      boolSubexp.add(node)
    return boolSubexp

  return reduce_nodes(n, cb)

def ast_primitive_boolean_exprs(n):
  def cb(node, children):
    relOps = [ "<", ">", "<=", ">=", "==",  "!=" ]
    isBoolean = (isinstance(node, AstBinExpr) and node.op in relOps)
    boolSubexp = reduce(lambda x,y: x.union(y), children, set([]))
    if (isBoolean):
      boolSubexp.add(node)
    return boolSubexp

  return reduce_nodes(n, cb)
