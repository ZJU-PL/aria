"""AST classes for Daikon invariants."""

# pylint: disable=no-self-argument, unused-argument, too-few-public-methods
from functools import reduce

from aria.efmc.verifytools.common.ast import AstNode
from aria.efmc.verifytools.common.util import error
from aria.efmc.verifytools.daikon.inv_grammar import DaikonInvParser


class AstUnExpr(AstNode):
    """AST node for unary expressions."""

    def __init__(s, op, expr):
        AstNode.__init__(s, str(op), expr)

    def __str__(s):
        return s.op + str(s.expr)


class AstIsPow2(AstNode):
    """AST node for power-of-2 check."""

    def __init__(s, expr):
        AstNode.__init__(s, expr)

    def __str__(s):
        return "IsPow2(" + str(s.expr) + ")"


class AstIsOneOf(AstNode):
    """AST node for one-of check."""

    def __init__(s, expr, options):
        AstNode.__init__(s, expr, options)

    def __str__(s):
        return "IsOneOf(" + str(s.expr) + ",[" + ",".join(map(str, s.options)) + "])"


class AstIsBoolean(AstNode):
    """AST node for boolean check."""

    def __init__(s, expr):
        AstNode.__init__(s, expr)

    def __str__(s):
        return "IsBoolean(" + str(s.expr) + ")"


class AstIsEven(AstNode):
    """AST node for even check."""

    def __init__(s, expr):
        AstNode.__init__(s, expr)

    def __str__(s):
        return "IsEven(" + str(s.expr) + ")"


class AstInRange(AstNode):
    """AST node for range check."""

    def __init__(s, lower, expr, upper):
        AstNode.__init__(s, lower, expr, upper)

    def __str__(s):
        return str(s.expr) + " in [" + str(s.lower) + "," + str(s.upper) + "]"


class AstIsConstMod(AstNode):
    """AST node for constant modulo check."""

    def __init__(s, expr, remainder, modulo):
        AstNode.__init__(s, expr, remainder, modulo)

    def __str__(s):
        return (
            "IsConstMod("
            + str(s.expr)
            + ","
            + str(s.remainder)
            + ","
            + str(s.modulo)
            + ")"
        )


class AstHasValues(AstNode):
    """AST node for has-values check."""

    def __init__(s, expr, values):
        AstNode.__init__(s, expr, values)

    def __str__(s):
        return "HasValues(" + str(s.expr) + "," + str(s.values) + ")"


class AstFalse(AstNode):
    """AST node for false literal."""

    def __init__(s):
        AstNode.__init__(s)

    def __str__(s):
        return "false"


class AstTrue(AstNode):
    """AST node for true literal."""

    def __init__(s):
        AstNode.__init__(s)

    def __str__(s):
        return "true"


class AstNumber(AstNode):
    """AST node for number literal."""

    def __init__(s, num):
        AstNode.__init__(s, int(num))

    def __str__(s):
        return str(s.num)


class AstId(AstNode):
    """AST node for identifier."""

    def __init__(s, name):
        AstNode.__init__(s, str(name))

    def __str__(s):
        return s.name


class AstBinExpr(AstNode):
    """AST node for binary expressions."""

    def __init__(s, lhs, op, rhs):
        AstNode.__init__(s, lhs, str(op), rhs)

    def __str__(s):
        return "(" + str(s.lhs) + " " + s.op + " " + str(s.rhs) + ")"


class AstBuilder(DaikonInvParser):
    """AST builder for Daikon invariants."""

    def onAtom(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle atom production."""
        return [s.atomM[prod](*toks)]

    def onUnaryOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle unary operator production."""
        if prod == s.IsPow2:
            return [AstIsPow2(toks[0])]
        if prod == s.IsBoolean:
            return [AstIsBoolean(toks[0])]
        if prod == s.IsEven:
            return [AstIsEven(toks[0])]
        return [AstUnExpr(*toks)]

    def onLABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle left-associative binary operator production."""
        if len(toks) == 3:
            return [AstBinExpr(*toks)]
        assert len(toks) > 3
        base = AstBinExpr(*toks[:3])
        rest = [
            [toks[3 + 2 * k], toks[3 + 2 * k + 1]] for k in range((len(toks) - 3) // 2)
        ]
        return [reduce(lambda acc, el: AstBinExpr(acc, el[0], el[1]), rest, base)]

    def onRABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle right-associative binary operator production."""
        if len(toks) == 3:
            return [AstBinExpr(*toks)]
        assert len(toks) > 3
        toks_list = list(reversed(toks))
        base = AstBinExpr(*toks_list[:3])
        rest = [
            [toks_list[3 + 2 * k], toks_list[3 + 2 * k + 1]]
            for k in range((len(toks_list) - 3) // 2)
        ]
        return [reduce(lambda acc, el: AstBinExpr(acc, el[0], el[1]), rest, base)]

    def onNABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle non-associative binary operator production."""
        if prod == s.IsInRange:
            return [AstInRange(toks[0], toks[1], toks[2])]
        if prod == s.IsOneOf:
            return [AstIsOneOf(toks[0], toks[1])]
        assert len(toks) == 3
        return [AstBinExpr(*toks)]

    def onTernaryOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle ternary operator production."""
        if prod == s.IsConstMod:
            assert len(toks) == 3
            return [AstIsConstMod(toks[0], toks[1], toks[2])]
        raise ValueError("Unknown ternary operator: " + str(prod))

    def onVariaryOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle variary operator production."""
        if prod == s.HasValues:
            assert len(toks) > 1
            return [AstHasValues(toks[0], toks[1:])]
        raise ValueError("Unknown variary operator: " + str(prod))

    def __init__(s):
        DaikonInvParser.__init__(s)
        s.atomM = {  # pylint: disable=invalid-name
            s.TRUE: AstTrue,
            s.FALSE: AstFalse,
            s.Id: AstId,
            s.Number: AstNumber,
        }


ast_builder = AstBuilder()


def parse_expr_ast(s):
    """Parse a Daikon invariant expression string into an AST."""
    try:
        return ast_builder.parse(s)
    except Exception:  # pylint: disable=broad-except
        error("Failed parsing")
        raise
