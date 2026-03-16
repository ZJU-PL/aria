"""Grammar parser for Daikon invariants."""

# pylint: disable=no-self-argument, invalid-name, too-many-instance-attributes, too-many-statements, abstract-method
from pyparsing import (
    Keyword as K,
    Literal as L,
    OneOrMore,
    ParserElement,
    Regex as R,
    StringEnd,
    Suppress as S,
    Word as W,
    delimitedList,
    infixNotation,
    nums,
    opAssoc,
)

from aria.efmc.verifytools.common.parser import InfixExprParser

ParserElement.enablePackrat()
csl = delimitedList


class DaikonInvParser(InfixExprParser):
    """Parser for Daikon invariant expressions."""

    def onAtom(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle atom production - to be overridden by subclasses."""
        raise NotImplementedError("Should be implemented by AstBuilder")

    def onUnaryOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle unary operator production - to be overridden by subclasses."""
        raise NotImplementedError("Should be implemented by AstBuilder")

    def onLABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle left-associative binary operator production - to be overridden by subclasses."""
        raise NotImplementedError("Should be implemented by AstBuilder")

    def onRABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle right-associative binary operator production - to be overridden by subclasses."""
        raise NotImplementedError("Should be implemented by AstBuilder")

    def onNABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle non-associative binary operator production - to be overridden by subclasses."""
        raise NotImplementedError("Should be implemented by AstBuilder")

    def onTernaryOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle ternary operator production - to be overridden by subclasses."""
        raise NotImplementedError("Should be implemented by AstBuilder")

    def onVariaryOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle variary operator production - to be overridden by subclasses."""
        raise NotImplementedError("Should be implemented by AstBuilder")

    def __init__(s):
        s.LT = L("<")
        s.GT = L(">")
        s.EQ = L("=")
        s.LPARN = S("(")
        s.RPARN = S(")")
        s.LBRAC = S("{")
        s.RBRAC = S("}")

        s.EquivOp = L("<==>")
        s.ImplOp = L("==>")
        s.OrOp = L("||")
        s.AndOp = L("&&")
        s.RelOp = L("!=") | L("<=") | L(">=") | L("<:") | L("==") | L("<") | L(">")
        s.AddOp = L("+") | L("-")
        s.MulOp = L("*") | L("/") | L("%")
        s.BWAnd = L("&")
        s.PowOp = L("**")
        s.UnOp = L("!") | L("-")

        s.FALSE = K("false")
        s.FALSE.setParseAction(lambda st, loc, toks: s.onAtom(s.FALSE, st, loc, toks))
        s.TRUE = K("true")
        s.TRUE.setParseAction(lambda st, loc, toks: s.onAtom(s.TRUE, st, loc, toks))
        s.Id = R("[a-zA-Z_][a-zA-Z0-9_#]*")
        s.Id.setParseAction(lambda st, loc, toks: s.onAtom(s.Id, st, loc, toks))
        s.Number = W(nums)
        s.Number.setParseAction(lambda st, loc, toks: s.onAtom(s.Number, st, loc, toks))

        s.Atom = s.FALSE | s.TRUE | s.Number | s.Id
        s.AndOrOp = s.AndOp | s.OrOp

        s.ArithExpr = infixNotation(
            s.Atom,
            [
                (
                    s.PowOp,
                    2,
                    opAssoc.RIGHT,
                    lambda st, loc, toks: s.onRABinOp(s.PowOp, st, loc, toks[0]),
                ),
                (
                    s.UnOp,
                    1,
                    opAssoc.RIGHT,
                    lambda st, loc, toks: s.onUnaryOp(s.UnOp, st, loc, toks[0]),
                ),
                (
                    s.MulOp,
                    2,
                    opAssoc.LEFT,
                    lambda st, loc, toks: s.onLABinOp(s.MulOp, st, loc, toks[0]),
                ),
                (
                    s.AddOp,
                    2,
                    opAssoc.LEFT,
                    lambda st, loc, toks: s.onLABinOp(s.AddOp, st, loc, toks[0]),
                ),
            ],
        )

        s.RelExpr = s.ArithExpr + s.RelOp + s.ArithExpr
        s.RelExpr.setParseAction(
            lambda st, loc, toks: s.onNABinOp(s.RelOp, st, loc, toks)
        )

        s.BoolExpr = infixNotation(
            s.RelExpr,
            [
                (
                    s.AndOrOp,
                    2,
                    opAssoc.LEFT,
                    lambda st, loc, toks: s.onLABinOp(s.AndOrOp, st, loc, toks[0]),
                ),
                (
                    s.ImplOp,
                    2,
                    opAssoc.LEFT,
                    lambda st, loc, toks: s.onLABinOp(s.ImplOp, st, loc, toks[0]),
                ),
                (
                    s.EquivOp,
                    2,
                    opAssoc.LEFT,
                    lambda st, loc, toks: s.onLABinOp(s.EquivOp, st, loc, toks[0]),
                ),
            ],
        )

        s.Expr = s.BoolExpr | s.RelExpr | s.ArithExpr

        s.IsPow2 = s.Id + S("is a power of 2")
        s.IsPow2.setParseAction(
            lambda st, loc, toks: s.onUnaryOp(s.IsPow2, st, loc, toks)
        )
        s.IsOneOf = s.Id + S("one of") + S(s.LBRAC) + csl(s.Expr) + S(s.RBRAC)
        s.IsOneOf.setParseAction(
            lambda st, loc, toks: s.onNABinOp(s.IsOneOf, st, loc, [toks[0], toks[1:]])
        )
        s.IsInRange = s.Number + S(L("<=")) + s.Id + S(L("<=")) + s.Number
        s.IsInRange.setParseAction(
            lambda st, loc, toks: s.onNABinOp(s.IsInRange, st, loc, toks)
        )
        s.IsBoolean = s.Id + S(L("is boolean"))
        s.IsBoolean.setParseAction(
            lambda st, loc, toks: s.onUnaryOp(s.IsBoolean, st, loc, toks)
        )

        s.IsEven = s.Id + S(L("is even"))
        s.IsEven.setParseAction(
            lambda st, loc, toks: s.onUnaryOp(s.IsEven, st, loc, toks)
        )

        s.IsConstMod = (
            s.Id + S(L("==")) + s.Number + S(L("(mod")) + s.Number + S(L(")"))
        )
        s.IsConstMod.setParseAction(
            lambda st, loc, toks: s.onTernaryOp(s.IsConstMod, st, loc, toks)
        )

        val_freq_pair = s.ArithExpr + S(L("[") + s.Number + L("]"))
        s.HasValues = s.Id + S(L("has values:")) + OneOrMore(val_freq_pair)
        s.HasValues.setParseAction(
            lambda st, loc, toks: s.onVariaryOp(s.HasValues, st, loc, toks)
        )

        s.JustInv = (
            s.IsPow2
            | s.IsOneOf
            | s.IsInRange
            | s.IsBoolean
            | s.IsEven
            | s.IsConstMod
            | s.HasValues
            | s.Expr
        )

        s.WarnInv = (
            S(R(r"warning: too few samples for [a-zA-Z\._]* invariant:")) + s.JustInv
            | s.JustInv
        )
        s.OneLine = s.WarnInv + StringEnd()

    def parse(s, st):
        """Parse a Daikon invariant string."""
        return s.OneLine.parseString(st)[0]
