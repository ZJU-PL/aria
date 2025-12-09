"""
Performing contextual simplification
"""
import z3
from typing import Dict, Set, Any, Generator, List


def subterms(t: z3.ExprRef) -> Set[z3.ExprRef]:
    """Extract all subterms of a given term"""
    seen: Dict[z3.ExprRef, bool] = {}

    def subterms_rec(term: z3.ExprRef) -> Generator[z3.ExprRef, None, None]:
        if z3.is_app(term):
            for ch in term.children():
                if ch in seen:
                    continue
                seen[ch] = True
                yield ch
                for sub in subterms_rec(ch):
                    yield sub

    return {s for s in subterms_rec(t)}


def are_equal(s: z3.Solver, t1: z3.ExprRef, t2: z3.ExprRef) -> bool:
    """Check if two terms are equal in the given solver context"""
    s.push()
    s.add(t1 != t2)
    r = s.check()
    s.pop()
    return r == z3.unsat


def ctx_simplify(slv: z3.Solver, mdl: z3.ModelRef, t: z3.ExprRef) -> z3.ExprRef:
    """Perform contextual simplification of a term based on a model"""
    subs: Set[z3.ExprRef] = subterms(t)
    values: Dict[z3.ExprRef, z3.ExprRef] = {s: mdl.eval(s) for s in subs}
    values[t] = mdl.eval(t)

    def simplify_rec(term: z3.ExprRef) -> z3.ExprRef:
        m_subs: Set[z3.ExprRef] = subterms(term)
        for s in m_subs:
            if s.sort().eq(term.sort()) and values[s].eq(values[term]) and are_equal(slv, s, term):
                return simplify_rec(s)
        chs: List[z3.ExprRef] = [simplify_rec(ch) for ch in term.children()]
        return term.decl()(chs)

    return simplify_rec(t)
