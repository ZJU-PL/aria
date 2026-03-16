"""Convert Daikon AST expressions to Boogie AST expressions."""

import aria.efmc.verifytools.boogie.ast as bast
import aria.efmc.verifytools.daikon.inv_ast as dast


def daikon_to_boogie_expr(astn):
    """Convert a Daikon AST expression to a Boogie AST expression."""
    if isinstance(astn, dast.AstUnExpr):
        cn = daikon_to_boogie_expr(astn.expr)
        try:
            boogie_op = {
                "-": "-",
                "!": "!",
            }[astn.op]
            return bast.AstUnExpr(boogie_op, cn)
        except KeyError as exc:
            raise ValueError("Don't know how to translate " + str(astn)) from exc
    elif isinstance(astn, dast.AstBinExpr):
        ln, rn = map(daikon_to_boogie_expr, [astn.lhs, astn.rhs])

        # We can translate power operators to a constant power
        if astn.op == "**" and isinstance(astn.rhs, dast.AstNumber):
            res = ln
            for _ in range(astn.rhs.num - 1):
                res = bast.AstBinExpr(res, "*", ln)
            return res

        try:
            boogie_op = {
                "&&": "&&",
                "||": "||",
                "==": "==",
                "!=": "!=",
                "<": "<",
                ">": ">",
                "<=": "<=",
                ">=": ">=",
                "+": "+",
                "-": "-",
                "*": "*",
                "/": "/",
                "%": "mod",
                "==>": "==>",
            }[astn.op]
            return bast.AstBinExpr(ln, boogie_op, rn)
        except KeyError as exc:
            raise ValueError("Don't know how to translate " + str(astn)) from exc
    elif isinstance(astn, dast.AstId):
        return bast.AstId(astn.name)
    elif isinstance(astn, dast.AstNumber):
        return bast.AstNumber(astn.num)
    elif isinstance(astn, dast.AstTrue):
        return bast.AstTrue()
    elif isinstance(astn, dast.AstFalse):
        return bast.AstFalse()
    elif isinstance(astn, dast.AstIsOneOf):
        cn = daikon_to_boogie_expr(astn.expr)
        return bast.ast_or(
            [
                bast.AstBinExpr(cn, "==", x)
                for x in [daikon_to_boogie_expr(y) for y in astn.options]
            ]
        )
    elif isinstance(astn, dast.AstInRange):
        cn = daikon_to_boogie_expr(astn.expr)
        low = astn.lower.num
        hi = astn.upper.num
        return bast.ast_and(
            [
                bast.AstBinExpr(bast.AstNumber(low), "<=", cn),
                bast.AstBinExpr(cn, "<=", bast.AstNumber(hi)),
            ]
        )
    elif isinstance(astn, dast.AstIsBoolean):
        cn = daikon_to_boogie_expr(astn.expr)
        return bast.ast_or(
            [
                bast.AstBinExpr(cn, "==", bast.AstNumber(0)),
                bast.AstBinExpr(cn, "==", bast.AstNumber(1)),
            ]
        )
    elif isinstance(astn, dast.AstIsEven):
        cn = daikon_to_boogie_expr(astn.expr)
        return bast.AstBinExpr(
            bast.AstBinExpr(cn, "mod", bast.AstNumber(2)), "==", bast.AstNumber(0)
        )
    elif isinstance(astn, dast.AstIsConstMod):
        expr = daikon_to_boogie_expr(astn.expr)
        remainder = daikon_to_boogie_expr(astn.remainder)
        modulo = daikon_to_boogie_expr(astn.modulo)

        assert modulo.num != 0

        return bast.AstBinExpr(bast.AstBinExpr(expr, "mod", modulo), "==", remainder)
    elif isinstance(astn, dast.AstHasValues):
        if len(astn.values) > 300:
            raise ValueError(
                "Can't convert HasValues: Too many options: " + str(len(astn.values))
            )
        cn = daikon_to_boogie_expr(astn.expr)
        values = list(map(daikon_to_boogie_expr, astn.values))
        return bast.ast_or([bast.AstBinExpr(cn, "==", v) for v in values])
    else:
        raise ValueError("Don't know how to translate " + str(astn))


# Backward compatibility alias
daikonToBoogieExpr = daikon_to_boogie_expr  # pylint: disable=invalid-name
