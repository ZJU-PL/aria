"""Miscellaneous stuff that does not really fit anywhere else.

- powerset: computes the powerset of a given set
- filldedent: dedents a string and fills it with the given width
- strlines: returns a cut-and-pastable string that, when printed, is
  equivalent to the input.
- rawlines: returns a cut-and-pastable string that, when printed, is
  equivalent to the input.
- debug_decorator: if ARIA_DEBUG is True, it will print a nice execution
  tree with arguments and results of all decorated functions, else do
  nothing.
- debug: Print ``*args`` if ARIA_DEBUG is True, else do nothing.
- func_name: Return function name of `x` (if defined) else the `type(x)`.
- _replace: Return a function that can make the replacements, given in
  ``reps``, on a string.
- replace: Return ``string`` with all keys in ``reps`` replaced with
  their corresponding values, longer strings first, irrespective of the
  order they are given.
"""

from __future__ import annotations

import re as _re
import struct
import sys
from textwrap import fill, dedent
from typing import List
import itertools
import subprocess
import threading


def powerset(elements: List):
    """Generates the powerset of the given elements set.

    E.g., powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    return itertools.chain.from_iterable(itertools.combinations(elements, r)
                                         for r in range(len(elements) + 1))



def filldedent(s, w=70, **kwargs):
    """
    Strips leading and trailing empty lines from a copy of ``s``, then dedents,
    fills and returns it.

    Empty line stripping serves to deal with docstrings like this one that
    start with a newline after the initial triple quote, inserting an empty
    line at the beginning of the string.

    Additional keyword arguments will be passed to ``textwrap.fill()``.

    See Also
    ========
    strlines, rawlines
    """
    return '\n' + fill(dedent(str(s)).strip('\n'), width=w, **kwargs)


def strlines(s, c=64, short=False):
    """Return a cut-and-pastable string that, when printed, is
    equivalent to the input.  The lines will be surrounded by
    parentheses and no line will be longer than c (default 64)
    characters. If the line contains newlines characters, the
    `rawlines` result will be returned.  If ``short`` is True
    (default is False) then if there is one line it will be
    returned without bounding parentheses.

    Examples
    ========

    >>> from aria.utils.misc import strlines
    >>> q = 'this is a long string that should be broken into shorter lines'
    >>> print(strlines(q, 40))
    (
    'this is a long string that should be b'
    'roken into shorter lines'
    )
    >>> q == (
    ... 'this is a long string that should be b'
    ... 'roken into shorter lines'
    ... )
    True

    See Also
    ========
    filldedent, rawlines
    """
    if not isinstance(s, str):
        raise ValueError('expecting string input')
    if '\n' in s:
        return rawlines(s)

    q = '"' if repr(s).startswith('"') else "'"
    q = (q,) * 2

    if '\\' in s:  # use r-string
        m = f'(\nr{q[0]}%s{q[1]}\n)'
        j = f'{q[0]}\nr{q[1]}'
        c -= 3
    else:
        m = f'(\n{q[0]}%s{q[1]}\n)'
        j = f'{q[0]}\n{q[1]}'
        c -= 2

    out = [s[i:i+c] for i in range(0, len(s), c)]

    if short and len(out) == 1:
        return (m % out[0]).splitlines()[1]  # strip bounding (\n...\n)
    return m % j.join(out)


def rawlines(s):
    """Return a cut-and-pastable string that, when printed, is equivalent
    to the input. Use this when there is more than one line in the
    string. The string returned is formatted so it can be indented
    nicely within tests; in some cases it is wrapped in the dedent
    function which has to be imported from textwrap.

    Examples
    ========

    Note: because there are characters in the examples below that need
    to be escaped because they are themselves within a triple quoted
    docstring, expressions below look more complicated than they would
    be if they were printed in an interpreter window.

    >>> from aria.utils.misc import rawlines
    >>> from aria import TableForm
    >>> s = str(TableForm([[1, 10]], headings=(None, ['a', 'bee'])))
    >>> print(rawlines(s))
    (
        'a bee\\n'
        '-----\\n'
        '1 10 '
    )
    >>> print(rawlines('''this
    ... that'''))
    dedent('''\\
        this
        that''')

    >>> print(rawlines('''this
    ... that
    ... '''))
    dedent('''\\
        this
        that
        ''')

    >>> s = \"\"\"this
    ... is a triple '''
    ... \"\"\"
    >>> print(rawlines(s))
    dedent(\"\"\"\\
        this
        is a triple '''
        \"\"\")

    >>> print(rawlines('''this
    ... that
    ...     '''))
    (
        'this\\n'
        'that\\n'
        '    '
    )

    See Also
    ========
    filldedent, strlines
    """
    lines = s.split('\n')
    if len(lines) == 1:
        return repr(lines[0])

    triple = ["'''" in s, '"""' in s]
    if any(li.endswith(' ') for li in lines) or '\\' in s or all(triple):
        rv = []
        trailing = s.endswith('\n')
        last = len(lines) - 1

        for i, li in enumerate(lines):
            if i != last or trailing:
                rv.append(repr(li + '\n'))
            else:
                rv.append(repr(li))

        return '(\n    %s\n)' % '\n    '.join(rv)
    rv = '\n    '.join(lines)
    if triple[0]:
        return f'dedent("""\\\n    {rv}""")'
    return f"dedent('''\\\n    {rv}''')"


# System architecture information
ARCH = f"{struct.calcsize('P') * 8}-bit"

# Note: PyPy does not support hash randomization
HASH_RANDOMIZATION = getattr(sys.flags, 'hash_randomization', False)

_debug_tmp: List[str] = []
_DEBUG_ITER = 0


def debug_decorator(func):
    """If ARIA_DEBUG is True, it will print a nice execution tree with
    arguments and results of all decorated functions, else do nothing.
    """
    from aria import ARIA_DEBUG

    if not ARIA_DEBUG:
        return func

    def maketree(f, *args, **kw):
        global _debug_tmp
        global _DEBUG_ITER
        oldtmp = _debug_tmp
        _debug_tmp = []
        _DEBUG_ITER += 1

        def tree(subtrees):
            def indent(s, variant=1):
                x = s.split("\n")
                r = f"+−{x[0]}\n"
                for a in x[1:]:
                    if not a:
                        continue
                    r += f"{'| ' if variant == 1 else '  '}{a}\n"
                return r

            if not subtrees:
                return ""

            result = []
            for a in subtrees[:-1]:
                result.append(indent(a))
            result.append(indent(subtrees[-1], 2))
            return ''.join(result)

        r = f(*args, **kw)

        _DEBUG_ITER -= 1
        s = f"{f.__name__}{args} = {r}\n"
        if _debug_tmp:
            s += tree(_debug_tmp)
        _debug_tmp = oldtmp
        _debug_tmp.append(s)
        if _DEBUG_ITER == 0:
            print(_debug_tmp[0])
            _debug_tmp = []
        return r

    def decorated(*args, **kwargs):
        return maketree(func, *args, **kwargs)

    return decorated


def debug(*args):
    """
    Print ``*args`` if ARIA_DEBUG is True, else do nothing.
    """
    from aria import ARIA_DEBUG
    if ARIA_DEBUG:
        print(*args, file=sys.stderr)


def debugf(string, args):
    """
    Print ``string%args`` if ARIA_DEBUG is True, else do nothing. This is
    intended for debug messages using formatted strings.
    """
    from aria import ARIA_DEBUG
    if ARIA_DEBUG:
        print(string % args, file=sys.stderr)


def func_name(x, short=False):
    """Return function name of `x` (if defined) else the `type(x)`.
    If short is True and there is a shorter alias for the result,
    return the alias.

    Examples
    ========

    >>> from aria.utils.misc import func_name
    >>> from aria import Matrix
    >>> from aria.abc import x
    >>> func_name(Matrix.eye(3))
    'MutableDenseMatrix'
    >>> func_name(x < 1)
    'StrictLessThan'
    >>> func_name(x < 1, short=True)
    'Lt'
    """
    alias = {
        'GreaterThan': 'Ge',
        'StrictGreaterThan': 'Gt',
        'LessThan': 'Le',
        'StrictLessThan': 'Lt',
        'Equality': 'Eq',
        'Unequality': 'Ne',
    }

    typ = type(x)
    type_str = str(typ)

    if type_str.startswith("<type '") or type_str.startswith("<class '"):
        typ = type_str.split("'")[1]

    rv = getattr(getattr(x, 'func', x), '__name__', typ)

    if '.' in rv:
        rv = rv.split('.')[-1]

    return alias.get(rv, rv) if short else rv


def _replace(reps):
    """Return a function that can make the replacements, given in
    ``reps``, on a string. The replacements should be given as mapping.

    Examples
    ========

    >>> from aria.utils.misc import _replace
    >>> f = _replace(dict(foo='bar', d='t'))
    >>> f('food')
    'bart'
    >>> f = _replace({})
    >>> f('food')
    'food'
    """
    if not reps:
        return lambda x: x

    pattern = _re.compile("|".join(_re.escape(k) for k in reps.keys()), _re.M)
    return lambda string: pattern.sub(lambda match: reps[match.group(0)], string)


def replace(string, *reps):
    """Return ``string`` with all keys in ``reps`` replaced with
    their corresponding values, longer strings first, irrespective
    of the order they are given.  ``reps`` may be passed as tuples
    or a single mapping.

    Examples
    ========

    >>> from aria.utils.misc import replace
    >>> replace('foo', {'oo': 'ar', 'f': 'b'})
    'bar'
    >>> replace("spamham sha", ("spam", "eggs"), ("sha","md5"))
    'eggsham md5'

    There is no guarantee that a unique answer will be
    obtained if keys in a mapping overlap (i.e. are the same
    length and have some identical sequence at the
    beginning/end):

    >>> reps = [
    ...     ('ab', 'x'),
    ...     ('bc', 'y')]
    >>> replace('abc', *reps) in ('xc', 'ay')
    True

    References
    ==========

    .. [1] https://stackoverflow.com/questions/6116978/
           how-to-replace-multiple-substrings-of-a-string
    """
    if len(reps) == 1:
        kv = reps[0]
        if isinstance(kv, dict):
            reps = kv
        else:
            return string.replace(*kv)
    else:
        reps = dict(reps)
    return _replace(reps)(string)


def translate(s, a, b=None, c=None):
    """Return ``s`` where characters have been replaced or deleted.

    SYNTAX
    ======

    translate(s, None, deletechars):
        all characters in ``deletechars`` are deleted
    translate(s, map [,deletechars]):
        all characters in ``deletechars`` (if provided) are deleted
        then the replacements defined by map are made; if the keys
        of map are strings then the longer ones are handled first.
        Multicharacter deletions should have a value of ''.
    translate(s, oldchars, newchars, deletechars)
        all characters in ``deletechars`` are deleted
        then each character in ``oldchars`` is replaced with the
        corresponding character in ``newchars``

    Examples
    ========

    >>> from aria.utils.misc import translate
    >>> abc = 'abc'
    >>> translate(abc, None, 'a')
    'bc'
    >>> translate(abc, {'a': 'x'}, 'c')
    'xb'
    >>> translate(abc, {'abc': 'x', 'a': 'y'})
    'x'

    >>> translate('abcd', 'ac', 'AC', 'd')
    'AbC'

    There is no guarantee that a unique answer will be
    obtained if keys in a mapping overlap are the same
    length and have some identical sequences at the
    beginning/end:

    >>> translate(abc, {'ab': 'x', 'bc': 'y'}) in ('xc', 'ay')
    True
    """
    # Handle delete-only case
    if a is None:
        if c is not None:
            raise ValueError(f'c should be None when a=None is passed, instead got {c}')
        if b is None:
            return s
        # Delete characters in b
        return s.translate(str.maketrans('', '', b))

    # Handle dictionary mapping case
    if isinstance(a, dict):
        mr = a.copy()
        c = b

        # Extract single-character mappings
        singles = {k: v for k, v in list(mr.items()) if len(k) == 1 and len(v) == 1}
        for k in singles:
            del mr[k]

        # Create translation tables
        if singles:
            a, b = ''.join(k for k in singles), ''.join(singles[k] for k in singles)
        else:
            a = b = ''

    # Handle character-by-character mapping
    elif len(a) != len(b):
        raise ValueError('oldchars and newchars have different lengths')
    else:
        mr = {}  # No multi-character replacements

    # Apply deletions if specified
    if c:
        s = s.translate(str.maketrans('', '', c))

    # Apply multi-character replacements
    s = replace(s, mr)

    # Apply single-character replacements
    if a:
        s = s.translate(str.maketrans(a, b))

    return s


def run_external_tool(cmd, input_content=None, timeout=300, delete_input=True):
    """
    Run an external command with optional input file content and timeout.
    Returns (success, stdout, stderr). Cleans up temp file if needed.
    """
    import os
    import tempfile
    is_timeout = [False]
    input_file = None
    try:
        if input_content is not None:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                input_file = f.name
                f.write(input_content)
            if isinstance(cmd, list):
                cmd = cmd + [input_file]
            else:
                cmd = f"{cmd} {input_file}"
        process = subprocess.Popen(
            cmd,
            stdin=None if input_content is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        timer = None
        try:
            if timeout > 0:
                timer = threading.Timer(
                    timeout,
                    lambda: (process.terminate(), is_timeout.__setitem__(0, True))
                )
                timer.start()
            stdout, stderr = process.communicate()
        finally:
            if timer:
                timer.cancel()
        if input_file and delete_input:
            try:
                os.unlink(input_file)
            except Exception:
                pass
        if is_timeout[0]:
            return False, stdout, stderr
        if process.returncode != 0:
            return False, stdout, stderr
        return True, stdout, stderr
    except Exception as e:
        if input_file and delete_input:
            try:
                os.unlink(input_file)
            except Exception:
                pass
        return False, '', str(e)
