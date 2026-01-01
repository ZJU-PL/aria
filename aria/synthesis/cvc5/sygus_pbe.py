"""SyGuS Programming-By-Example for strings."""

from typing import List, Tuple, Optional, Dict
import z3
from aria.utils.z3_plus_smtlib_solver import Z3SolverPlus


class StringSyGuSPBE:
    """SyGuS PBE solver for string-manipulation synthesis (Z3/CVC5 backend)."""

    def __init__(self, debug: bool = False):
        """Initialize with optional debug output."""
        self.debug = debug
        self.solver = Z3SolverPlus(debug=debug)

    def synthesize_from_examples(
        self,
        input_examples: List[List[str]],
        output_examples: List[str],
        function_name: str = "str_func",
    ) -> Optional[str]:
        """Synthesize a string function from examples; return SMT-LIB2 or None."""
        if len(input_examples) != len(output_examples):
            raise ValueError(
                "Number of input examples must match number of output examples"
            )

        if not input_examples:
            raise ValueError("At least one example must be provided")

        # Create the function declarations
        arity = len(input_examples[0])

        # Define the symbolic variables
        inputs = [z3.String(f"x{i}") for i in range(arity)]
        func = z3.Function(function_name, *([z3.StringSort()] * arity), z3.StringSort())

        # Add constraints from examples
        constraints = []
        for i, (inputs_ex, output_ex) in enumerate(
            zip(input_examples, output_examples)
        ):
            if len(inputs_ex) != arity:
                raise ValueError(f"Example {i} has incorrect number of inputs")

            # Convert input examples to Z3 string constants
            input_consts = [z3.StringVal(s) for s in inputs_ex]

            # Add the constraint: func(inputs) == output
            constraints.append(func(*input_consts) == z3.StringVal(output_ex))

        # Perform synthesis with SyGuS
        result = self.solver.sygus(
            [func],
            constraints,
            inputs,
            logic="QF_S",  # Quantifier-free theory of strings
            pbe=True,  # Enable PBE mode
        )

        if self.debug:
            print("SyGuS result:", result)

        return result

    def synthesize_string_transformer(
        self, examples: List[Tuple[str, str]], function_name: str = "str_transform"
    ) -> Optional[str]:
        """Synthesize single-arg string transformer from examples."""
        input_examples = [[ex[0]] for ex in examples]
        output_examples = [ex[1] for ex in examples]

        return self.synthesize_from_examples(
            input_examples, output_examples, function_name
        )

    def synthesize_concat_function(
        self,
        examples: List[Tuple[str, str, str]],
        function_name: str = "str_concat_func",
    ) -> Optional[str]:
        """Synthesize 2-arg concatenation function from examples."""
        input_examples = [[ex[0], ex[1]] for ex in examples]
        output_examples = [ex[2] for ex in examples]

        return self.synthesize_from_examples(
            input_examples, output_examples, function_name
        )

    @staticmethod
    def explain_synthesized_function(smt_function: str) -> Dict:
        """Parse SMT-LIB2 function and return a structured explanation dict."""
        if not smt_function or "define-fun" not in smt_function:
            return {
                "error": "Invalid or empty function definition",
                "input": smt_function,
            }

        result = {
            "original": smt_function,
            "name": "",
            "params": [],
            "param_types": [],
            "return_type": "",
            "body": "",
            "explanation": "",
            "python_equivalent": "",
        }

        name_start = smt_function.find("define-fun") + len("define-fun") + 1
        name_end = smt_function.find("(", name_start)
        result["name"] = smt_function[name_start:name_end].strip()

        params_start = name_end
        params_end = smt_function.find(")", params_start)
        params_section = smt_function[params_start + 1 : params_end]

        param_entries = []
        param_depth = 0
        current_param = ""

        for char in params_section:
            if char == "(":
                param_depth += 1
                current_param += char
            elif char == ")":
                param_depth -= 1
                current_param += char
                if param_depth == 0:
                    param_entries.append(current_param.strip())
                    current_param = ""
            else:
                current_param += char

        for param in param_entries:
            if param and "(" in param and ")" in param:
                param = param.strip("()")
                parts = param.split()
                if len(parts) >= 2:
                    result["params"].append(parts[0])
                    result["param_types"].append(parts[1])

        return_type_start = params_end + 1
        body_start = smt_function.find("(", return_type_start + 1)
        if body_start == -1:  # If there's no nested expression in the body
            body_start = smt_function.find(" ", return_type_start + 1)

        result["return_type"] = smt_function[return_type_start:body_start].strip()

        body_end = smt_function.rfind(")")
        result["body"] = smt_function[body_start:body_end].strip()

        try:
            result["explanation"] = StringSyGuSPBE._generate_explanation(
                result["name"], result["params"], result["body"]
            )

            result["python_equivalent"] = StringSyGuSPBE._generate_python_equivalent(
                result["name"], result["params"], result["body"]
            )
        except Exception as e:
            result["explanation"] = f"Could not generate explanation: {str(e)}"
            result["python_equivalent"] = (
                f"\n# Could not generate Python equivalent: {str(e)}"
            )

        return result

    @staticmethod
    def _generate_explanation(name: str, params: List[str], body: str) -> str:
        """Generate a brief human-readable description for common string ops."""
        # Common string operations to check for
        if "str.++" in body:
            return f"Function '{name}' concatenates strings."

        if "str.replace" in body:
            return f"Function '{name}' performs string replacement."

        if "str.substr" in body:
            return f"Function '{name}' extracts a substring from the input."

        if "str.to_upper" in body:
            return f"Function '{name}' converts strings to uppercase."

        if "str.to_lower" in body:
            return f"Function '{name}' converts strings to lowercase."

        if "str.at" in body:
            return f"Function '{name}' extracts characters at specific positions."

        if "str.contains" in body or "str.prefixof" in body or "str.suffixof" in body:
            return f"Function '{name}' checks for string containment patterns."

        if "str.indexof" in body:
            return f"Function '{name}' finds the position of substrings."

        return f"Function '{name}' takes {len(params)} parameters and performs string manipulation."

    @staticmethod
    def _generate_python_equivalent(name: str, params: List[str], body: str) -> str:
        """Generate a minimal Python equivalent for common patterns."""
        # Start with function definition
        python_func = f"def {name}({', '.join(params)}):\n"

        # Handle common string operations
        if "str.++" in body:
            # Simple concatenation case
            if body.count("str.++") == 1 and all(param in body for param in params):
                python_func += f"    return {''.join(params)}\n"
            else:
                python_func += f"    # String concatenation\n"
                python_func += f"    return ''.join([{', '.join(params)}])\n"

        elif "str.replace" in body and body.count("str.replace") == 1:
            # Simple replacement
            search_term = ""
            replace_term = ""
            if ' " ' in body and ' "-' in body:
                # Replace space with hyphen
                python_func += f"    return {params[0]}.replace(' ', '-')\n"
            else:
                python_func += f"    # String replacement operation\n"
                python_func += f"    return {params[0]}.replace(search_term, replace_term)  # Determine correct terms\n"

        elif "str.to_upper" in body:
            python_func += f"    return {params[0]}.upper()\n"

        elif "str.to_lower" in body:
            python_func += f"    return {params[0]}.lower()\n"

        elif "str.substr" in body:
            python_func += f"    # Substring extraction\n"
            python_func += (
                f"    return {params[0]}[start:end]  # Determine correct indices\n"
            )

        else:
            # Default for complex or unrecognized patterns
            python_func += f"    # Complex string operation\n"
            python_func += f"    # SMT-LIB2 body: {body}\n"
            python_func += f"    pass  # Implement based on the SMT-LIB2 definition\n"

        return python_func
