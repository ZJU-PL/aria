import sys

from aria.unification.variable import var


def gen_long_chain(last_elem=None, n=None, use_lvars=False):  # noqa: N803
    """Generate a nested list of length `N` with the last element set to `last_elm`.

    Parameters
    ----------
    last_elem: object
        The element to be placed in the inner-most nested list.
    n: int
        The number of nested lists.
    use_lvars: bool
        Whether or not to add `var`s to the first elements of each nested list
        or simply integers.  If ``True``, each `var` is passed the nesting
        level integer (i.e. ``var(i)``).

    Returns
    -------
    list, dict
        The generated nested list and a ``dict`` containing the generated
        `var`s and their nesting level integers, if any.

    """
    b_struct = None
    if n is None:
        n = sys.getrecursionlimit()
    lvars = {}
    for i in range(n - 1, 0, -1):
        i_el = var(i) if use_lvars else i
        if use_lvars:
            lvars[i_el] = i
        b_struct = [i_el, last_elem if i == n - 1 else b_struct]
    return b_struct, lvars
