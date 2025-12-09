"""
FIXME:
  After bit-blasting and CNF transformation, we may have many auxiliary Boolean variables.
  If operating over the Boolean level, it seems that we need to solve the problem below:
     Exists BX ForAll BY Exists BZ . BF(BX, BY, BZ)  (where BZ is the set of auxiliary variables)
  Instead of the following problem
     Exists X ForAll Y . F(X, Y)  (where X and Y are the existential and universal quantified bit-vectors, resp.)
"""
