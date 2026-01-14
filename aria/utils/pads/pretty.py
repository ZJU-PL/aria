"""Minimal pretty printing stub for compatibility."""


class PrettyRepr:
    """Base class for pretty-printable objects."""



def pfun(name, args):
    """Format a function call for pretty printing."""
    if args:
        return f"{name}({', '.join(str(arg) for arg in args)})"
    return f"{name}()"
