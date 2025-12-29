"""First-order logic language definitions for the theorem prover."""

from functools import reduce

##############################################################################
# Terms
##############################################################################


class Variable:
    """Represents a variable in first-order logic."""

    def __init__(self, name):
        """Initialize a variable with a name."""
        self.name = name
        self.time = 0

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables (this variable itself)."""
        return {self}

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms (none for variables)."""
        return set()

    def replace(self, old, new):
        """Replace old term with new term if this is old."""
        if self == old:
            return new
        return self

    def occurs(self, unification_term):  # pylint: disable=unused-argument
        """Check if unification term occurs in this variable (never)."""
        return False

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for this variable."""
        self.time = time

    def __eq__(self, other):
        """Check equality with another variable."""
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

    def __str__(self):
        """String representation of the variable."""
        return self.name

    def __hash__(self):
        """Hash the variable."""
        return hash(str(self))


class UnificationTerm:
    """Represents a unification term (metavariable) in first-order logic."""

    def __init__(self, name):
        """Initialize a unification term with a name."""
        self.name = name
        self.time = 0

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables (none for unification terms)."""
        return set()

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms (this term itself)."""
        return {self}

    def replace(self, old, new):
        """Replace old term with new term if this is old."""
        if self == old:
            return new
        return self

    def occurs(self, unification_term):
        """Check if unification term occurs in this term."""
        return self == unification_term

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for this unification term."""
        self.time = time

    def __eq__(self, other):
        """Check equality with another unification term."""
        if not isinstance(other, UnificationTerm):
            return False
        return self.name == other.name

    def __str__(self):
        """String representation of the unification term."""
        return self.name

    def __hash__(self):
        """Hash the unification term."""
        return hash(str(self))


class Function:
    """Represents a function application in first-order logic."""

    def __init__(self, name, terms):
        """Initialize a function with a name and list of terms."""
        self.name = name
        self.terms = terms
        self.time = 0

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables in this function."""
        if len(self.terms) == 0:
            return set()
        return reduce((lambda x, y: x | y),
                      [term.freeVariables() for term in self.terms])

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms in this function."""
        if len(self.terms) == 0:
            return set()
        return reduce((lambda x, y: x | y),
                      [term.freeUnificationTerms() for term in self.terms])

    def replace(self, old, new):
        """Replace old term with new term recursively."""
        if self == old:
            return new
        return Function(self.name,
                        [term.replace(old, new) for term in self.terms]
                        )

    def occurs(self, unification_term):
        """Check if unification term occurs in this function."""
        return any(term.occurs(unification_term) for term in self.terms)

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for this function and its terms."""
        self.time = time
        for term in self.terms:
            term.setInstantiationTime(time)

    def __eq__(self, other):
        """Check equality with another function."""
        if not isinstance(other, Function):
            return False
        if self.name != other.name:
            return False
        if len(self.terms) != len(other.terms):
            return False
        return all(self.terms[i] == other.terms[i]
                   for i in range(len(self.terms)))

    def __str__(self):
        """String representation of the function."""
        if len(self.terms) == 0:
            return self.name
        return self.name + '(' + ', '.join(
            [str(term) for term in self.terms]
        ) + ')'

    def __hash__(self):
        """Hash the function."""
        return hash(str(self))

##############################################################################
# Formulae
##############################################################################


class Predicate:
    """Represents a predicate in first-order logic."""

    def __init__(self, name, terms):
        """Initialize a predicate with a name and list of terms."""
        self.name = name
        self.terms = terms

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables in this predicate."""
        if len(self.terms) == 0:
            return set()
        return reduce((lambda x, y: x | y),
                      [term.freeVariables() for term in self.terms])

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms in this predicate."""
        if len(self.terms) == 0:
            return set()
        return reduce((lambda x, y: x | y),
                      [term.freeUnificationTerms() for term in self.terms])

    def replace(self, old, new):
        """Replace old term with new term recursively."""
        if self == old:
            return new
        return Predicate(self.name,
                         [term.replace(old, new) for term in self.terms]
                         )

    def occurs(self, unification_term):
        """Check if unification term occurs in this predicate."""
        return any(term.occurs(unification_term) for term in self.terms)

    def __eq__(self, other):
        """Check equality with another predicate."""
        if not isinstance(other, Predicate):
            return False
        if self.name != other.name:
            return False
        if len(self.terms) != len(other.terms):
            return False
        return all(self.terms[i] == other.terms[i]
                   for i in range(len(self.terms)))

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for all terms in this predicate."""
        for term in self.terms:
            term.setInstantiationTime(time)

    def __str__(self):
        """String representation of the predicate."""
        if len(self.terms) == 0:
            return self.name
        return self.name + '(' + ', '.join(
            [str(term) for term in self.terms]
        ) + ')'

    def __hash__(self):
        """Hash the predicate."""
        return hash(str(self))


class Not:
    """Represents a negation in first-order logic."""

    def __init__(self, formula):
        """Initialize a negation with a formula."""
        self.formula = formula

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables in this negation."""
        return self.formula.freeVariables()

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms in this negation."""
        return self.formula.freeUnificationTerms()

    def replace(self, old, new):
        """Replace old term with new term recursively."""
        if self == old:
            return new
        return Not(self.formula.replace(old, new))

    def occurs(self, unification_term):
        """Check if unification term occurs in this negation."""
        return self.formula.occurs(unification_term)

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for the formula in this negation."""
        self.formula.setInstantiationTime(time)

    def __eq__(self, other):
        """Check equality with another negation."""
        if not isinstance(other, Not):
            return False
        return self.formula == other.formula

    def __str__(self):
        """String representation of the negation."""
        return '¬' + str(self.formula)

    def __hash__(self):
        """Hash the negation."""
        return hash(str(self))


class And:
    """Represents a conjunction in first-order logic."""

    def __init__(self, formula_a, formula_b):
        """Initialize a conjunction with two formulae."""
        self.formula_a = formula_a
        self.formula_b = formula_b

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables in this conjunction."""
        return self.formula_a.freeVariables() | \
            self.formula_b.freeVariables()

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms in this conjunction."""
        return self.formula_a.freeUnificationTerms() | \
            self.formula_b.freeUnificationTerms()

    def replace(self, old, new):
        """Replace old term with new term recursively."""
        if self == old:
            return new
        return And(
            self.formula_a.replace(old, new),
            self.formula_b.replace(old, new)
        )

    def occurs(self, unification_term):
        """Check if unification term occurs in this conjunction."""
        return self.formula_a.occurs(unification_term) or \
            self.formula_b.occurs(unification_term)

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for both formulae in this conjunction."""
        self.formula_a.setInstantiationTime(time)
        self.formula_b.setInstantiationTime(time)

    def __eq__(self, other):
        """Check equality with another conjunction."""
        if not isinstance(other, And):
            return False
        return self.formula_a == other.formula_a and \
            self.formula_b == other.formula_b

    def __str__(self):
        """String representation of the conjunction."""
        return f'({self.formula_a} ∧ {self.formula_b})'

    def __hash__(self):
        """Hash the conjunction."""
        return hash(str(self))


class Or:
    """Represents a disjunction in first-order logic."""

    def __init__(self, formula_a, formula_b):
        """Initialize a disjunction with two formulae."""
        self.formula_a = formula_a
        self.formula_b = formula_b

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables in this disjunction."""
        return self.formula_a.freeVariables() | \
            self.formula_b.freeVariables()

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms in this disjunction."""
        return self.formula_a.freeUnificationTerms() | \
            self.formula_b.freeUnificationTerms()

    def replace(self, old, new):
        """Replace old term with new term recursively."""
        if self == old:
            return new
        return Or(
            self.formula_a.replace(old, new),
            self.formula_b.replace(old, new)
        )

    def occurs(self, unification_term):
        """Check if unification term occurs in this disjunction."""
        return self.formula_a.occurs(unification_term) or \
            self.formula_b.occurs(unification_term)

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for both formulae in this disjunction."""
        self.formula_a.setInstantiationTime(time)
        self.formula_b.setInstantiationTime(time)

    def __eq__(self, other):
        """Check equality with another disjunction."""
        if not isinstance(other, Or):
            return False
        return self.formula_a == other.formula_a and \
            self.formula_b == other.formula_b

    def __str__(self):
        """String representation of the disjunction."""
        return f'({self.formula_a} ∨ {self.formula_b})'

    def __hash__(self):
        """Hash the disjunction."""
        return hash(str(self))


class Implies:
    """Represents an implication in first-order logic."""

    def __init__(self, formula_a, formula_b):
        """Initialize an implication with two formulae."""
        self.formula_a = formula_a
        self.formula_b = formula_b

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables in this implication."""
        return self.formula_a.freeVariables() | \
            self.formula_b.freeVariables()

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms in this implication."""
        return self.formula_a.freeUnificationTerms() | \
            self.formula_b.freeUnificationTerms()

    def replace(self, old, new):
        """Replace old term with new term recursively."""
        if self == old:
            return new
        return Implies(
            self.formula_a.replace(old, new),
            self.formula_b.replace(old, new)
        )

    def occurs(self, unification_term):
        """Check if unification term occurs in this implication."""
        return self.formula_a.occurs(unification_term) or \
            self.formula_b.occurs(unification_term)

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for both formulae in this implication."""
        self.formula_a.setInstantiationTime(time)
        self.formula_b.setInstantiationTime(time)

    def __eq__(self, other):
        """Check equality with another implication."""
        if not isinstance(other, Implies):
            return False
        return self.formula_a == other.formula_a and \
            self.formula_b == other.formula_b

    def __str__(self):
        """String representation of the implication."""
        return f'({self.formula_a} → {self.formula_b})'

    def __hash__(self):
        """Hash the implication."""
        return hash(str(self))


class ForAll:
    """Represents a universal quantification in first-order logic."""

    def __init__(self, variable, formula):
        """Initialize a universal quantification with a variable and formula."""
        self.variable = variable
        self.formula = formula

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables in this quantification (excluding bound var)."""
        return self.formula.freeVariables() - {self.variable}

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms in this quantification."""
        return self.formula.freeUnificationTerms()

    def replace(self, old, new):
        """Replace old term with new term recursively."""
        if self == old:
            return new
        return ForAll(
            self.variable.replace(old, new),
            self.formula.replace(old, new)
        )

    def occurs(self, unification_term):
        """Check if unification term occurs in this quantification."""
        return self.formula.occurs(unification_term)

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for variable and formula."""
        self.variable.setInstantiationTime(time)
        self.formula.setInstantiationTime(time)

    def __eq__(self, other):
        """Check equality with another universal quantification."""
        if not isinstance(other, ForAll):
            return False
        return self.variable == other.variable and \
            self.formula == other.formula

    def __str__(self):
        """String representation of the universal quantification."""
        return f'(∀{self.variable}. {self.formula})'

    def __hash__(self):
        """Hash the universal quantification."""
        return hash(str(self))


class ThereExists:
    """Represents an existential quantification in first-order logic."""

    def __init__(self, variable, formula):
        """Initialize an existential quantification with a variable and formula."""
        self.variable = variable
        self.formula = formula

    def freeVariables(self):  # pylint: disable=invalid-name
        """Get all free variables in this quantification (excluding bound var)."""
        return self.formula.freeVariables() - {self.variable}

    def freeUnificationTerms(self):  # pylint: disable=invalid-name
        """Get all free unification terms in this quantification."""
        return self.formula.freeUnificationTerms()

    def replace(self, old, new):
        """Replace old term with new term recursively."""
        if self == old:
            return new
        return ThereExists(
            self.variable.replace(old, new),
            self.formula.replace(old, new)
        )

    def occurs(self, unification_term):
        """Check if unification term occurs in this quantification."""
        return self.formula.occurs(unification_term)

    def setInstantiationTime(self, time):  # pylint: disable=invalid-name
        """Set the instantiation time for variable and formula."""
        self.variable.setInstantiationTime(time)
        self.formula.setInstantiationTime(time)

    def __eq__(self, other):
        """Check equality with another existential quantification."""
        if not isinstance(other, ThereExists):
            return False
        return self.variable == other.variable and \
            self.formula == other.formula

    def __str__(self):
        """String representation of the existential quantification."""
        return f'(∃{self.variable}. {self.formula})'

    def __hash__(self):
        """Hash the existential quantification."""
        return hash(str(self))
