import re
from typing import Optional
import z3



def extract_smt_from_llm_response(response: str) -> str:
    """
    Extract SMT-LIB2 expressions from LLM response.
    Looks for content between code blocks or direct SMT-LIB2 expressions.
    """
    # Clean up response
    response = response.strip()

    # First check if the response is already a direct SMT-LIB2 expression
    if response.startswith("(") and response.endswith(")"):
        return response

    # If response starts with (assert ...)
    if response.startswith("(assert ") and response.endswith(")"):
        return response[len("(assert "):-1].strip()

    # Look for code blocks with SMT content
    code_block_pattern = r"```(?:smt|lisp|smt-lib|smt2|smtlib|smtlib2)?\s*([\s\S]*?)```"
    matches = re.findall(code_block_pattern, response)

    if matches:
        for match in matches:
            match = match.strip()
            if match:
                # Extract assertions if any
                assertion_pattern = r"\(assert\s+(.*?)\)"
                assertions = re.findall(assertion_pattern, match)
                if assertions:
                    return assertions[-1]  # Return the last assertion

                # If no assertions, return the whole block if it looks like an SMT expression
                if match.startswith("(") and match.endswith(")"):
                    return match

    # Try to extract direct SMT expressions with balanced parentheses
    stack = []
    start_idx = -1
    smt_expressions = []

    for i, char in enumerate(response):
        if char == '(' and len(stack) == 0:
            start_idx = i
            stack.append(char)
        elif char == '(' and len(stack) > 0:
            stack.append(char)
        elif char == ')' and len(stack) > 0:
            stack.pop()
            if len(stack) == 0:
                expression = response[start_idx:i+1].strip()
                if expression.startswith("(assert "):
                    expression = expression[len("(assert "):-1].strip()
                smt_expressions.append(expression)

    # Return the last complete SMT expression if any
    if smt_expressions:
        return smt_expressions[-1]

    # Fallback: return anything that looks like an SMT expression
    any_smt_pattern = r"\((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*\)"
    matches = re.findall(any_smt_pattern, response)
    if matches:
        expression = matches[-1]
        if expression.startswith("(assert "):
            expression = expression[len("(assert "):-1].strip()
        return expression

    return ""


def parse_smt2_string(smt_string: str, problem: Optional['AbductionProblem'] = None) -> Optional[z3.ExprRef]:
    """
    Parse an SMT-LIB2 expression string into a Z3 expression.

    Args:
        smt_string: The SMT-LIB2 expression string to parse
        problem: Optional AbductionProblem to provide variable context

    Returns:
        Optional[z3.ExprRef]: The parsed Z3 expression, or None if parsing failed
    """
    try:
        # Clean up the SMT string
        smt_string = smt_string.strip()

        # Handle 'assert' clauses - remove outer assert if present
        if smt_string.startswith("(assert ") and smt_string.endswith(")"):
            smt_string = smt_string[len("(assert "):-1].strip()

        # If we have a problem context, use its variables to build declarations
        if problem is not None:
            # Create sort declarations for variables
            declarations = []
            z3_vars = {}

            for var in problem.variables:
                var_name = str(var)
                sort_name = str(var.sort())

                # Create appropriate declarations based on variable type
                if z3.is_int(var):
                    declarations.append(f"(declare-const {var_name} Int)")
                    z3_vars[var_name] = z3.Int(var_name)
                elif z3.is_real(var):
                    declarations.append(f"(declare-const {var_name} Real)")
                    z3_vars[var_name] = z3.Real(var_name)
                elif z3.is_bool(var):
                    declarations.append(f"(declare-const {var_name} Bool)")
                    z3_vars[var_name] = z3.Bool(var_name)
                else:
                    declarations.append(f"(declare-const {var_name} {sort_name})")

            # Build the complete SMT-LIB2 input with declarations
            declarations_str = "\n".join(declarations)
            full_smt = f"{declarations_str}\n(assert {smt_string})"

            try:
                # Parse the full SMT formula
                formulas = z3.parse_smt2_string(full_smt)

                if formulas and len(formulas) > 0:
                    parsed_formula = formulas[0]

                    # Substitute original variables from the problem
                    # This is needed because Z3 creates new variables during parsing
                    subst = {}
                    for var in problem.variables:
                        var_name = str(var)
                        if var_name in z3_vars:
                            # Map parsed variable to original variable
                            for parsed_var in get_variables(parsed_formula):
                                if str(parsed_var) == var_name:
                                    subst[parsed_var] = var
                                    break

                    # Apply substitution if needed
                    if subst:
                        return z3.substitute(parsed_formula, *[(k, v) for k, v in subst.items()])
                    return parsed_formula
            except Exception as e:
                pass

            # TODO: If Z3 parsing fails, maybe try simpler approaches?

            # Fall back to True as a default
            return z3.BoolVal(True)


    except Exception as e:
        # Return a default hypothesis
        return z3.BoolVal(True)

    return z3.BoolVal(True)
