"""Parser base classes for expression parsing."""


# pylint: disable=unused-argument, no-self-argument, too-few-public-methods
class Parser:
    """Base parser class."""


class InfixExprParser(Parser):
    """Parser for infix expressions."""

    def onAtom(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle atom production."""
        raise NotImplementedError("NYI")

    def onUnaryOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle unary operator production."""
        raise NotImplementedError("NYI")

    def onLABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle left-associative binary operator production."""
        raise NotImplementedError("NYI")

    def onRABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle right-associative binary operator production."""
        raise NotImplementedError("NYI")

    def onNABinOp(s, prod, st, loc, toks):  # pylint: disable=invalid-name
        """Handle non-associative binary operator production."""
        raise NotImplementedError("NYI")
