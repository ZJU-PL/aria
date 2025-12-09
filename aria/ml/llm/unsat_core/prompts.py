EXPLORER_SYSTEM = """
### ROLE ###
You are playing the role of an SMT solver. 

### INPUT FORMAT ###
Given a list of constraint clauses, your goal is to identify a subset of these clauses such that it is still unsatisfiable, i.e., containts conflicts resulting from inconsistencies or contradictions in the logical state space.

### THOUGHT-STEPS ###
Follow these thought steps to find the conflicts or contradictions:
Step 1: Read and understand the given logical formula to identify the variables, operators, and logical connectives used in the formula.
Step 2: Analyze semantic dependencies both within, and between constraints. 
Step 3: Macro-reason on these constraints to find pairs or groups that can possibly be combined and resolved. It is okay for there to be overlap among the different groups of constraints. The motivation for this grouping, however, should be to find redundant constraints in the input, which can be inferred from combining parts of these groups. Note that such redundant constraints CAN be eliminated.
Step 4: Next, try to identify constraints or groups of constraints which also satisfy another constraint or group of constraints. In such a case, the latter constraint or group CAN be eliminated.

For steps 3 and 4, try to find as many such groups as possible.

Step 5: Try to identify pairs of constraints that directly conflict with each other. These need to definitely be part of the output subset. Note that such conflicts or inconsistencies WILL arise in the given example.

Think step by step.

### OUTPUT-FORMAT ###
After identifying all such constraints that can safely be removed, output a comma-separated list of all remaining clauses. Each clause should be inserted between <c> and </c> tags, and the entire output, between <OUTPUT> and </OUTPUT> tags.

### DISCLAIMER ###
Try to minimize the input set of constraints as best as possible, while also ensuring the output subset's unsatisfiability.
"""


EXPLORER_TEMPLATE = """
Input Formula:
<FORMULA>

<EXAMPLES>

>>>
"""


EXPLORER_EXEMPLARS = """
"""


PARSER_FEEDBACK = """
There is an error in the output format.

Note that the output should be a comma-separated list of all such conflict-causing clauses, where each clause is inserted between <c> and </c> tags, and the entire output is inserted between <OUTPUT> and </OUTPUT> tags.
For instance, if the output contains the clauses C1, C2, and C3, its output should be: <OUTPUT><c>C1</c>,<c>C2</c>,<c>C3</c></OUTPUT>. If there is no such subset, the output is <OUTPUT></OUTPUT>.

Fix output based on this feedback.
"""


VALIDATOR_FEEDBACK = """
The output clause subset is Satisfiable, not Unsatisfiable.

Retry exploration and output a different clause subset (in the same output format) that is Unsatisfiable.
"""
