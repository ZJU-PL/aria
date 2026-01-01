"""Command-line interface for the first-order logic theorem prover."""

# pylint: disable=wildcard-import
from aria.fol.miniprover.language import (
    And,
    ForAll,
    Function,
    Implies,
    Not,
    Or,
    Predicate,
    ThereExists,
    Variable,
)
from aria.fol.miniprover.prover import proveFormula

##############################################################################
# Command-line interface
##############################################################################


class InvalidInputError(Exception):
    """Exception for invalid input."""

    def __init__(self, message):
        """Initialize the exception with a message."""
        self.message = message


def lex(inp):
    """Perform lexical analysis on input string."""
    # perform a lexical analysis
    tokens = []
    pos = 0
    while pos < len(inp):
        # skip whitespace
        if inp[pos] == " ":
            pos += 1
            continue

        # identifiers
        identifier = ""
        while pos < len(inp) and inp[pos].isalnum():
            identifier += inp[pos]
            pos += 1
        if len(identifier) > 0:
            tokens.append(identifier)
            continue

        # symbols
        tokens.append(inp[pos])
        pos += 1

    # return the tokens
    return tokens


def parse(
    tokens,
):  # pylint: disable=too-many-return-statements,too-many-branches,too-many-statements
    """Parse tokens into a formula."""
    # keywords
    keywords = ["not", "implies", "and", "or", "forall", "exists"]
    tokens = [(token.lower() if token in keywords else token) for token in tokens]

    # empty formula
    if len(tokens) == 0:
        raise InvalidInputError("Empty formula.")

    # ForAll
    if tokens[0] == "forall":
        dot_pos = None
        for i in range(1, len(tokens)):
            if tokens[i] == ".":
                dot_pos = i
                break
        if dot_pos is None:
            raise InvalidInputError("Missing '.' in FORALL quantifier.")
        args = []
        i = 1
        while i <= dot_pos:
            end = dot_pos
            depth = 0
            for j in range(i, dot_pos):
                if tokens[j] == "(":
                    depth += 1
                    continue
                if tokens[j] == ")":
                    depth -= 1
                    continue
                if depth == 0 and tokens[j] == ",":
                    end = j
                    break
            if i == end:
                raise InvalidInputError("Missing variable in FORALL quantifier.")
            args.append(parse(tokens[i:end]))
            i = end + 1
        if len(tokens) == dot_pos + 1:
            raise InvalidInputError("Missing formula in FORALL quantifier.")
        formula = parse(tokens[dot_pos + 1 :])
        for variable in reversed(args):
            formula = ForAll(variable, formula)
        return formula

    # ThereExists
    if tokens[0] == "exists":
        dot_pos = None
        for i in range(1, len(tokens)):
            if tokens[i] == ".":
                dot_pos = i
                break
        if dot_pos is None:
            raise InvalidInputError("Missing '.' in exists quantifier.")
        if dot_pos == 1:
            raise InvalidInputError("Missing variable in exists quantifier.")
        args = []
        i = 1
        while i <= dot_pos:
            end = dot_pos
            depth = 0
            for j in range(i, dot_pos):
                if tokens[j] == "(":
                    depth += 1
                    continue
                if tokens[j] == ")":
                    depth -= 1
                    continue
                if depth == 0 and tokens[j] == ",":
                    end = j
                    break
            if i == end:
                raise InvalidInputError("Missing variable in exists quantifier.")
            args.append(parse(tokens[i:end]))
            i = end + 1
        if len(tokens) == dot_pos + 1:
            raise InvalidInputError("Missing formula in exists quantifier.")
        formula = parse(tokens[dot_pos + 1 :])
        for variable in reversed(args):
            formula = ThereExists(variable, formula)
        return formula

    # Implies
    implies_pos = None
    depth = 0
    for i, token in enumerate(tokens):
        if token == "(":
            depth += 1
            continue
        if token == ")":
            depth -= 1
            continue
        if depth == 0 and token == "implies":
            implies_pos = i
            break
    if implies_pos is not None:
        quantifier_in_left = False
        depth = 0
        for i in range(implies_pos):
            if tokens[i] == "(":
                depth += 1
                continue
            if tokens[i] == ")":
                depth -= 1
                continue
            if depth == 0 and (tokens[i] == "forall" or tokens[i] == "exists"):
                quantifier_in_left = True
                break
        if not quantifier_in_left:
            if implies_pos in (0, len(tokens) - 1):
                raise InvalidInputError("Missing formula in IMPLIES connective.")
            return Implies(
                parse(tokens[0:implies_pos]), parse(tokens[implies_pos + 1 :])
            )

    # Or
    or_pos = None
    depth = 0
    for i, token in enumerate(tokens):
        if token == "(":
            depth += 1
            continue
        if token == ")":
            depth -= 1
            continue
        if depth == 0 and token == "or":
            or_pos = i
            break
    if or_pos is not None:
        quantifier_in_left = False
        depth = 0
        for i in range(or_pos):
            if tokens[i] == "(":
                depth += 1
                continue
            if tokens[i] == ")":
                depth -= 1
                continue
            if depth == 0 and (tokens[i] == "forall" or tokens[i] == "exists"):
                quantifier_in_left = True
                break
        if not quantifier_in_left:
            if or_pos in (0, len(tokens) - 1):
                raise InvalidInputError("Missing formula in OR connective.")
            return Or(parse(tokens[0:or_pos]), parse(tokens[or_pos + 1 :]))

    # And
    and_pos = None
    depth = 0
    for i, token in enumerate(tokens):
        if token == "(":
            depth += 1
            continue
        if token == ")":
            depth -= 1
            continue
        if depth == 0 and token == "and":
            and_pos = i
            break
    if and_pos is not None:
        quantifier_in_left = False
        depth = 0
        for i in range(and_pos):
            if tokens[i] == "(":
                depth += 1
                continue
            if tokens[i] == ")":
                depth -= 1
                continue
            if depth == 0 and (tokens[i] == "forall" or tokens[i] == "exists"):
                quantifier_in_left = True
                break
        if not quantifier_in_left:
            if and_pos in (0, len(tokens) - 1):
                raise InvalidInputError("Missing formula in AND connective.")
            return And(parse(tokens[0:and_pos]), parse(tokens[and_pos + 1 :]))

    # Not
    if tokens[0] == "not":
        if len(tokens) < 2:
            raise InvalidInputError("Missing formula in NOT connective.")
        return Not(parse(tokens[1:]))

    # Function
    if (
        tokens[0].isalnum()
        and tokens[0].lower() not in keywords
        and len(tokens) > 1
        and not any(c.isupper() for c in tokens[0])
        and tokens[1] == "("
    ):
        if tokens[-1] != ")":
            raise InvalidInputError("Missing ')' after function argument list.")
        name = tokens[0]
        args = []
        i = 2
        if i < len(tokens) - 1:
            while i <= len(tokens) - 1:
                end = len(tokens) - 1
                depth = 0
                for j in range(i, len(tokens) - 1):
                    if tokens[j] == "(":
                        depth += 1
                        continue
                    if tokens[j] == ")":
                        depth -= 1
                        continue
                    if depth == 0 and tokens[j] == ",":
                        end = j
                        break
                if i == end:
                    raise InvalidInputError("Missing function argument.")
                args.append(parse(tokens[i:end]))
                i = end + 1
        return Function(name, args)

    # Predicate
    if (
        tokens[0].isalnum()
        and tokens[0].lower() not in keywords
        and len(tokens) == 1
        and any(c.isupper() for c in tokens[0])
    ):
        return Predicate(tokens[0], [])
    if (
        tokens[0].isalnum()
        and tokens[0].lower() not in keywords
        and len(tokens) > 1
        and any(c.isupper() for c in tokens[0])
        and tokens[1] == "("
    ):
        if tokens[-1] != ")":
            raise InvalidInputError("Missing ')' after predicate argument list.")
        name = tokens[0]
        args = []
        i = 2
        if i < len(tokens) - 1:
            while i <= len(tokens) - 1:
                end = len(tokens) - 1
                depth = 0
                for j in range(i, len(tokens) - 1):
                    if tokens[j] == "(":
                        depth += 1
                        continue
                    if tokens[j] == ")":
                        depth -= 1
                        continue
                    if depth == 0 and tokens[j] == ",":
                        end = j
                        break
                if i == end:
                    raise InvalidInputError("Missing predicate argument.")
                args.append(parse(tokens[i:end]))
                i = end + 1
        return Predicate(name, args)

    # Variable
    if (
        tokens[0].isalnum()
        and tokens[0].lower() not in keywords
        and len(tokens) == 1
        and not any(c.isupper() for c in tokens[0])
    ):
        return Variable(tokens[0])

    # Group
    if tokens[0] == "(":
        if tokens[-1] != ")":
            raise InvalidInputError("Missing ')'.")
        if len(tokens) == 2:
            raise InvalidInputError("Missing formula in parenthetical group.")
        return parse(tokens[1:-1])

    raise InvalidInputError(f"Unable to parse: {tokens[0]}...")


def typecheck_term(term):  # pylint: disable=too-many-return-statements
    """Type check a term."""
    if isinstance(term, Variable):
        return
    if isinstance(term, Function):
        for subterm in term.terms:
            typecheck_term(subterm)
        return
    raise InvalidInputError(f"Invalid term: {term}.")


def typecheck_formula(formula):  # pylint: disable=too-many-return-statements
    """Type check a formula."""
    if isinstance(formula, Predicate):
        for term in formula.terms:
            typecheck_term(term)
        return
    if isinstance(formula, Not):
        typecheck_formula(formula.formula)
        return
    if isinstance(formula, And):
        typecheck_formula(formula.formula_a)
        typecheck_formula(formula.formula_b)
        return
    if isinstance(formula, Or):
        typecheck_formula(formula.formula_a)
        typecheck_formula(formula.formula_b)
        return
    if isinstance(formula, Implies):
        typecheck_formula(formula.formula_a)
        typecheck_formula(formula.formula_b)
        return
    if isinstance(formula, ForAll):
        if not isinstance(formula.variable, Variable):
            raise InvalidInputError(
                f"Invalid bound variable in " f"FORALL quantifier: {formula.variable}."
            )
        typecheck_formula(formula.formula)
        return
    if isinstance(formula, ThereExists):
        if not isinstance(formula.variable, Variable):
            raise InvalidInputError(
                f"Invalid bound variable in " f"exists quantifier: {formula.variable}."
            )
        typecheck_formula(formula.formula)
        return
    raise InvalidInputError(f"Invalid formula: {formula}.")


def check_formula(formula):
    """Check if input is a valid formula (not a term)."""
    try:
        typecheck_formula(formula)
    except InvalidInputError as formula_error:
        try:
            typecheck_term(formula)
        except InvalidInputError:
            raise InvalidInputError("Enter a formula, not a term.") from formula_error
        raise


def main():  # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks
    """Main entry point for the theorem prover."""
    print("First-Order Logic Theorem Prover")
    print("")
    print("Terms:")
    print("")
    print("  x               (variable)")
    print("  f(term, ...)    (function)")
    print("")
    print("Formulae:")
    print("")
    print("  P(term)        (predicate)")
    print("  not P          (complement)")
    print("  P or Q         (disjunction)")
    print("  P and Q        (conjunction)")
    print("  P implies Q    (implication)")
    print("  forall x. P    (universal quantification)")
    print("  exists x. P    (existential quantification)")
    print("")
    print(
        "Enter formulae at the prompt. The following commands "
        "are also available for manipulating axioms:"
    )
    print("")
    print("  axioms              (list axioms)")
    print("  lemmas              (list lemmas)")
    print("  axiom <formula>     (add an axiom)")
    print("  lemma <formula>     (prove and add a lemma)")
    print("  remove <formula>    (remove an axiom or lemma)")
    print("  reset               (remove all axioms and lemmas)")
    print("  exit                (exit the program)")

    axioms = set()
    lemmas = {}

    while True:
        try:
            inp = input("\n> ")
            commands = ["axiom", "lemma", "axioms", "lemmas", "remove", "reset", "exit"]
            tokens = [
                (token.lower() if token in commands else token) for token in lex(inp)
            ]
            for token in tokens[1:]:
                if token in commands:
                    raise InvalidInputError(f"Unexpected keyword: {token}.")
            if len(tokens) > 0 and tokens[0] == "axioms":
                if len(tokens) > 1:
                    raise InvalidInputError(f"Unexpected: {tokens[1]}.")
                for axiom in axioms:
                    print(axiom)
            elif len(tokens) > 0 and tokens[0] == "lemmas":
                if len(tokens) > 1:
                    raise InvalidInputError(f"Unexpected: {tokens[1]}.")
                for lemma in lemmas:
                    print(lemma)
            elif len(tokens) > 0 and tokens[0] == "axiom":
                formula = parse(tokens[1:])
                check_formula(formula)
                axioms.add(formula)
                print(f"Axiom added: {formula}.")
            elif len(tokens) > 0 and tokens[0] == "lemma":
                formula = parse(tokens[1:])
                check_formula(formula)
                result = proveFormula(axioms | set(lemmas.keys()), formula)
                if result:
                    lemmas[formula] = axioms.copy()
                    print(f"Lemma proven: {formula}.")
                else:
                    print(f"Lemma unprovable: {formula}.")
            elif len(tokens) > 0 and tokens[0] == "remove":
                formula = parse(tokens[1:])
                check_formula(formula)
                if formula in axioms:
                    axioms.remove(formula)
                    bad_lemmas = []
                    for lemma, dependent_axioms in lemmas.items():
                        if formula in dependent_axioms:
                            bad_lemmas.append(lemma)
                    for lemma in bad_lemmas:
                        del lemmas[lemma]
                    print(f"Axiom removed: {formula}.")
                    if len(bad_lemmas) == 1:
                        print(
                            "This lemma was proven using that "
                            "axiom and was also removed:"
                        )
                        for lemma in bad_lemmas:
                            print(f"  {lemma}")
                    if len(bad_lemmas) > 1:
                        print(
                            "These lemmas were proven using that "
                            "axiom and were also removed:"
                        )
                        for lemma in bad_lemmas:
                            print(f"  {lemma}")
                elif formula in lemmas:
                    del lemmas[formula]
                    print(f"Lemma removed: {formula}.")
                else:
                    print(f"Not an axiom: {formula}.")
            elif len(tokens) > 0 and tokens[0] == "reset":
                if len(tokens) > 1:
                    raise InvalidInputError(f"Unexpected: {tokens[1]}.")
                axioms = set()
                lemmas = {}
            elif len(tokens) > 0 and tokens[0] == "exit":
                if len(tokens) > 1:
                    raise InvalidInputError(f"Unexpected: {tokens[1]}.")
                print("Goodbye!")
                break
            else:
                formula = parse(tokens)
                check_formula(formula)
                result = proveFormula(axioms | set(lemmas.keys()), formula)
                if result:
                    print(f"Formula proven: {formula}.")
                else:
                    print(f"Formula unprovable: {formula}.")
        except InvalidInputError as e:
            print(e.message)
        except KeyboardInterrupt:
            pass
        except EOFError:
            print("")
            break


if __name__ == "__main__":
    main()
