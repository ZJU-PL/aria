#pylint: disable=no-self-argument,unused-argument, multiple-statements
from pyparsing import DelimitedList,nums, ParserElement, infix_notation, \
        opAssoc, StringEnd
from aria.efmc.verifytools.common.parser import InfixExprParser
from pyparsing import ZeroOrMore as ZoM,\
    OneOrMore as OoM,\
    Keyword as K,\
    Suppress as S,\
    Literal as L,\
    Forward as F,\
    Optional as O,\
    Regex as R,\
    Word as W,\
    Group as G

ParserElement.enable_packrat()
csl = DelimitedList

class BoogieParser(InfixExprParser):
  # Statements
  def onAssert(s, prod, st, loc, toks): raise Exception("NYI")
  def onAssume(s, prod, st, loc, toks): raise Exception("NYI")
  def onReturn(s, prod, st, loc, toks): raise Exception("NYI")
  def onGoto(s, prod, st, loc, toks): raise Exception("NYI")
  def onAssignment(s, prod, st, loc, toks): raise Exception("NYI")
  def onHavoc(s, prod, st, loc, toks): raise Exception("NYI")
  def onProgram(s, prod, st, loc, toks): raise Exception("NYI")
  def onImplementationDecl(s, prod, st, loc, toks): raise Exception("NYI")
  def onBody(s, prod, st, loc, toks): raise Exception("NYI")
  def onLocalVarDecl(s, prod, st, loc, toks): raise Exception("NYI")
  def onType(s, prod, st, loc, toks): raise Exception("NYI")
  def onLabeledStatement(s, prod, st, loc, toks): raise Exception("NYI")
  def onWhile(s, prod, st, loc, toks): raise Exception("NYI")
  def onBlock(s, prod, st, loc, toks): raise Exception("NYI")

  def __init__(s):
    s.LT = L("<")
    s.GT = L(">")
    s.EQ = L("=")
    # Braces/Brackets
    s.LSQBR = S("[")
    s.RSQBR = S("]")
    s.LPARN = S("(")
    s.RPARN = S(")")
    s.LBRAC = S("{")
    s.RBRAC = S("}")
    # Various Delimiters
    s.SEMI = S(";")
    s.COLN = S(":")
    s.ASSGN = S(":=")
    s.STAR = S("*")

    ####### Keywords
    s.INT = K("int")
    s.BOOL = K("bool")
    s.TYPE = K("type")
    s.FINITE = K("finite")
    s.CONST = K("const")
    s.UNIQUE = K("unique")
    s.RETURNS = K("returns")
    s.FUNCTION = K("function")
    s.FALSE = K("false")
    s.FALSE.set_parse_action(\
            lambda st, loc, toks:  s.onAtom(s.FALSE, st, loc, toks))
    s.TRUE = K("true")
    s.TRUE.set_parse_action(\
            lambda st, loc, toks:  s.onAtom(s.TRUE, st, loc, toks))
    s.OLD = K("old")
    s.AXIOM = K("axiom")
    s.FORALL = K("forall")
    s.EXISTS = K("exists")
    s.VAR = K("var")
    s.PROCEDURE = K("procedure")
    s.FREE = K("free")
    s.RETURNS = K("returns")
    s.REQUIRES = K("requires")
    s.MODIFIES = K("modifies")
    s.ENSURES = K("ensures")
    s.ASSERT = K("assert")
    s.COMPLETE = K("complete")
    s.UNIQUE = K("unique")
    s.IF = K("if")
    s.ELSE = K("else")
    s.FREE = K("free")
    s.INVARIANT = K("invariant")
    s.ASSUME = K("assume")
    s.ASSERT = K("assert")
    s.HAVOC = K("havoc")
    s.CALL = K("call")
    s.WHILE = K("while")
    s.BREAK = K("break")
    s.GOTO = K("goto")
    s.RETURN = K("return")
    s.IMPLEMENTATION = K("implementation")

    s.Id = R("[a-zA-Z_][a-zA-Z0-9_#]*")
    s.Id.set_parse_action(lambda st, loc, toks:  s.onAtom(s.Id, st, loc, toks))
    s.ParentEdge = O(s.UNIQUE) + s.Id
    s.ParentInfo = S("<:") + csl(s.ParentEdge)
    s.OrderSpec = O(s.ParentInfo) + O(s.COMPLETE)
    s.StringLiteral = F(); # TODO
    s.Trigger = F(); # TODO

    ####### Expressions
    s.EquivOp = L("<==>")
    s.ImplOp = L("==>")
    s.OrOp = L("||")
    s.AndOp = L("&&")
    s.AndOrOp = s.AndOp | s.OrOp
    s.RelOp = (L("!=") | L ("<=") | L(">=") | L("<:") \
               | L("==") |  L("<") | L(">") )
    s.ConcatOp = L("++")
    s.AddOp = (L("+") | L("-"))
    s.MulOp = (L("*") | L("/") | L("div") | L("mod"))
    s.UnOp = (L("!") | L("-"))
    s.ArithUnOp = L("-")
    s.BoolUnOp = L("!")
    s.QOp = (s.FORALL | s.EXISTS)
    s.QSep = L("::")

    s.Expr = F();

    s.Number = W(nums)
    s.Number.set_parse_action(
            lambda st, loc, toks:  s.onAtom(s.Number, st, loc, toks))
    s.BitVector = R("[0-9][0-9]*bv[0-9][0-9]*")
    s.BitVector.set_parse_action(
            lambda st, loc, toks:  s.onAtom(s.BitVector, st, loc, toks))
    # TODO
    # TrigAttr = Trigger | Attribute
    s.Fun_App = s.Id + s.LPARN + csl(s.Expr) + s.RPARN
    s.Fun_App.set_parse_action(
            lambda st, loc, toks:  s.onAtom(s.Fun_App, st, loc, toks))
    s.Old = s.OLD + s.LPARN + s.Expr + s.RPARN
    s.Old.set_parse_action(
            lambda st, loc, toks:  s.onAtom(s.Old, st, loc, toks))
    s.Primitive = s.FALSE | s.TRUE | s.Number | s.BitVector
    # TODO: Handle TriggerAttrs in Quantified expressions
    #E9_Quantified = LPARN + QOp + O(TypeArgs) + csl(IdsType) \
    #        + QSep + ZoM(TrigAttr) + Expr  +  RPARN
    #s.Quantified = s.LPARN + s.QOp + O(s.TypeArgs) + \
    #        csl(s.IdsType) + s.QSep + s.Expr  +  s.RPARN

    s.Atom = s.Primitive | s.Fun_App | s.Old | s.Id #| s.Quantified
    s.ArithExpr = infix_notation(s.Atom, [
      (s.ArithUnOp, 1, opAssoc.RIGHT, \
           lambda st, loc, toks:  s.onUnaryOp(s.ArithUnOp, st, loc, toks[0])),
      (s.MulOp, 2, opAssoc.LEFT, \
           lambda st, loc, toks: s.onLABinOp(s.MulOp, st, loc, toks[0])),
      (s.AddOp, 2, opAssoc.LEFT, \
           lambda st, loc, toks: s.onLABinOp(s.AddOp, st, loc, toks[0])),
    ]);
    # TODO: Add support for Map operations
    #MapUpdate = ASSGN + Expr
    #MapOp = LSQBR + csl(Expr) + O(MapUpdate) + RSQBR | \
    #        LSQBR + Number + COLN + Number + RSQBR
    #E8 = E9 + ZoM(MapOp)
    # TODO: Add support for ConcatOp
    #E4 << E5 + ZoM(ConcatOp + E4)
    s.RelExpr = s.ArithExpr + s.RelOp + s.ArithExpr
    s.RelExpr.set_parse_action(
            lambda st, loc, toks: s.onNABinOp(s.RelExpr, st, loc, toks))

    s.BoolExpr = infix_notation((s.RelExpr | s.Atom), [
      (s.BoolUnOp, 1, opAssoc.RIGHT, \
              lambda st, loc, toks:  s.onUnaryOp(s.BoolUnOp, st, loc, toks[0])),
      (s.AndOrOp, 2, opAssoc.LEFT, \
              lambda st, loc, toks:  s.onLABinOp(s.AndOrOp, st, loc, toks[0])),
      (s.ImplOp, 2, opAssoc.LEFT, \
              lambda st, loc, toks:  s.onLABinOp(s.ImplOp, st, loc, toks[0])),
      (s.EquivOp, 2, opAssoc.LEFT, \
              lambda st, loc, toks:  s.onLABinOp(s.EquivOp, st, loc, toks[0])),
    ]);

    s.Expr << (s.BoolExpr ^ s.RelExpr ^ s.ArithExpr ) #pylint: disable=pointless-statement

    ####### Attributes
    s.AttrArg = s.Expr | s.StringLiteral
    s.Attribute = s.LBRAC + s.COLN + s.Id + csl(s.AttrArg) + s.RBRAC
    s.AttrList = ZoM(s.Attribute)

    ####### Types
    s.Type = F();
    s.BVType = R("bv[0-9][0-9]*")
    s.TypeAtom = s.INT | s.BOOL | s.BVType | s.LPARN + s.Type + s.RPARN
    s.TypeArgs = S(s.LT) + csl(s.Type) + S(s.GT)
    s.MapType = O(s.TypeArgs) + s.LSQBR + csl(s.Type) + s.RSQBR + s.Type

    s.TypeCtorArgs = F()
    s.TypeCtorArgs = s.TypeAtom + O(s.TypeCtorArgs) |\
                   s.Id + O(s.TypeCtorArgs) |\
                   s.MapType

    s.Type << (s.TypeAtom | s.MapType | s.Id + O(s.TypeCtorArgs)) #pylint: disable=expression-not-assigned
    s.Type.set_parse_action(lambda st, loc, toks: s.onType(s.Type, st, loc, toks))
    s.IdsType = csl(s.Id) + s.COLN + s.Type


    ####### Type Declarations
    s.TypeConstructor = s.TYPE + s.AttrList + O(s.FINITE) + OoM(s.Id) + s.SEMI
    s.TypeSynonym = s.TYPE + s.AttrList + OoM(s.Id) + s.EQ + s.Type + s.SEMI
    s.TypeDecl = s.TypeConstructor | s.TypeSynonym;

    ####### Constant Declarations
    s.ConstantDecl = s.CONST + O(s.Attribute) + O(s.UNIQUE) + \
            s.IdsType + s.OrderSpec;

    ####### Function Declarations
    s.FArgName = s.Id + s.COLN
    s.FArg = s.FArgName + s.Type
    s.FSig = O(s.TypeArgs) + s.LPARN + csl(s.FArg) + \
            s.RPARN + s.RETURNS + s.LPARN + s.FArg + s.RPARN
    s.FunctionDecl = s.FUNCTION + ZoM(s.Attribute) + s.Id + s.FSig + s.SEMI |\
                     s.FUNCTION + ZoM(s.Attribute) + s.Id + s.FSig + s.SEMI +\
                        s.LBRAC + s.Expr + s.RBRAC

    ####### Axiom Declarations
    s.AxiomDecl = s.AXIOM + ZoM(s.Attribute) + s.Expr;

    s.WhereClause = F() # TODO

    s.IdsTypeWhere = s.IdsType + O(s.WhereClause)
    s.VarDecl = s.VAR + ZoM(s.Attribute) + s.IdsTypeWhere;

    ####### Procedure Declarations
    s.Spec =  O(s.FREE) + s.REQUIRES + s.Expr + s.SEMI \
          | O(s.FREE) + s.MODIFIES + csl(s.Id) + s.SEMI \
          | O(s.FREE) + s.ENSURES + s.Expr + s.SEMI

    s.OutParameters = s.RETURNS + s.LPARN + csl(s.IdsTypeWhere) + s.RPARN
    s.PSig = O(s.TypeArgs) + s.LPARN + csl(s.IdsTypeWhere) + \
            s.RPARN + O(s.OutParameters)


    s.LocalVarDecl = S(s.VAR) + ZoM(s.Attribute) + csl(s.IdsTypeWhere) + \
            S(s.SEMI);
    s.LocalVarDecl.set_parse_action(
            lambda st, loc, toks: s.onLocalVarDecl(s.LocalVarDecl,
                                                   st, loc, toks))

    s.StmtList = F()
    s.WildcardExpr = s.Expr | s.STAR

    s.BlockStmt = s.LBRAC + s.StmtList + s.RBRAC

    s.LoopInv = O(s.FREE) + s.INVARIANT + s.Expr + s.SEMI

    s.IfStmt = F()
    s.Else = s.ELSE + s.BlockStmt | s.ELSE + s.IfStmt
    s.IfStmt = s.IF + s.LBRAC + s.WildcardExpr + s.RBRAC + s.BlockStmt + \
            O(s.Else)

    s.CallLhs = csl(s.Id) + s.ASSGN
    s.MapSelect = s.LSQBR + csl(s.Expr) + s.RSQBR
    s.Lhs = s.Id + ZoM(s.MapSelect)
    s.Label = s.Id | s.Number

    s.AssertStmt = S(s.ASSERT) + s.Expr + S(s.SEMI)
    s.AssertStmt.set_parse_action(
            lambda st, loc, toks: s.onAssert(s.AssertStmt, st, loc, toks))
    s.AssumeStmt = S(s.ASSUME) + O(S("{:partition}")) + s.Expr + S(s.SEMI)
    s.AssumeStmt.set_parse_action(
            lambda st, loc, toks: s.onAssume(s.AssumeStmt, st, loc, toks))
    s.ReturnStmt = S(s.RETURN) + S(s.SEMI)
    s.ReturnStmt.set_parse_action(
            lambda st, loc, toks: s.onReturn(s.ReturnStmt, st, loc, toks))
    s.GotoStmt = S(s.GOTO) + csl(s.Label) + S(s.SEMI)
    s.GotoStmt.set_parse_action(
            lambda st, loc, toks: s.onGoto(s.GotoStmt, st, loc, toks))
    s.AssignmentStmt = G(csl(s.Lhs)) + S(s.ASSGN) + G(csl(s.Expr)) + S(s.SEMI)
    s.AssignmentStmt.set_parse_action(
            lambda st, loc, toks: s.onAssignment(s.AssignmentStmt, st,
                                                 loc, toks))
    s.HavocStmt = S(s.HAVOC) + csl(s.Id) + S(s.SEMI)
    s.HavocStmt.set_parse_action(
            lambda st, loc, toks: s.onHavoc(s.HavocStmt, st, loc, toks))

    s.CallAssignStmt = s.CALL + O(s.CallLhs) + s.Id + s.LPARN + csl(s.Expr) +\
            s.RPARN + S(s.SEMI)
    s.CallForallStmt = s.CALL + s.FORALL + s.Id + s.LPARN + \
            csl(s.WildcardExpr) + s.RPARN + S(s.SEMI)

    s.WhileStmt = s.WHILE + s.LPARN + s.WildcardExpr + s.RPARN + \
            ZoM(s.LoopInv) + s.BlockStmt
    s.WhileStmt.set_parse_action(
            lambda st, loc, toks: s.onWhile(s.WhileStmt, st, loc, toks))
    s.BlockStmt.set_parse_action(
            lambda st, loc, toks: s.onBlock(s.BlockStmt, st, loc, toks))
    s.BreakStmt = s.BREAK + O(s.Id) + S(s.SEMI)

    s.Stmt = s.AssertStmt \
          | s.AssumeStmt \
          | s.HavocStmt \
          | s.AssignmentStmt \
          | s.CallAssignStmt \
          | s.CallForallStmt \
          | s.IfStmt \
          | s.WhileStmt \
          | s.BreakStmt \
          | s.ReturnStmt \
          | s.GotoStmt

    s.LStmt = F();
    s.LabeledStatement = s.Label + S(s.COLN) + s.LStmt
    s.LStmt << (s.Stmt | s.LabeledStatement) #pylint: disable=pointless-statement
    s.LEmpty = F();
    s.LEmpty <<(s.Id + s.COLN + O(s.LEmpty)) #pylint: disable=expression-not-assigned
    s.LabeledStatement.set_parse_action(
      lambda st, loc, toks: s.onLabeledStatement(s.LabeledStatement,
                                                 st, loc, toks))

    s.StmtList << (ZoM(s.LStmt) + O(s.LEmpty)) #pylint: disable=expression-not-assigned


    s.Body = S(s.LBRAC) + G(ZoM(s.LocalVarDecl)) + G(s.StmtList) + S(s.RBRAC)
    s.Body.set_parse_action(lambda st, loc, toks: s.onBody(s.Body, st, loc, toks))

    s.ProcedureDecl = \
        s.PROCEDURE + ZoM(s.Attribute) + s.Id + s.PSig + s.SEMI + ZoM(s.Spec) |\
        s.PROCEDURE + ZoM(s.Attribute) + s.Id + s.PSig + ZoM(s.Spec) + s.Body

    s.IOutParameters = s.RETURNS + s.LPARN + csl(s.IdsType) + s.RPARN
    s.ISig = G(O(s.TypeArgs)) + s.LPARN + G(O(csl(s.IdsType))) + s.RPARN +\
            G(O(s.IOutParameters))
    s.ImplementationDecl = S(s.IMPLEMENTATION) + G(ZoM(s.Attribute)) + s.Id +\
            G(s.ISig) + G(ZoM(s.Body))
    s.ImplementationDecl.set_parse_action(
      lambda st, loc, toks: s.onImplementationDecl(s.ImplementationDecl,
                                                   st, loc, toks))


    s.Decl = s.TypeDecl |\
             s.ConstantDecl |\
             s.FunctionDecl |\
             s.AxiomDecl |\
             s.VarDecl |\
             s.ProcedureDecl |\
             s.ImplementationDecl

    s.Program = ZoM(s.Decl);
    s.Program.set_parse_action(
            lambda st, loc, toks: s.onProgram(s.Program, st, loc, toks))

  def parseExpr(s, st):
    return (s.Expr + StringEnd()).parse_string(st)[0]

  def parseProgram(s, st):
    return (s.Program + StringEnd()).parse_string(st)[0]
