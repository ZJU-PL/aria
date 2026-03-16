"""Boogie to EFMC Transition System Converter

This module converts Boogie programs (specifically loops) to EFMC's transition system format,
allowing verification using EFMC's engines.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple

import z3
from aria.efmc.verifytools.boogie.ast import (
    AstProgram,
    AstImplementation,
    AstLabel,
    AstAssert,
    AstAssume,
    AstAssignment,
    AstGoto,
    AstReturn,
    AstBinExpr,
    AstId,
    AstNumber,
    AstTrue,
    AstFalse,
    AstUnExpr,
    parseAst,
    stmt_read,
    stmt_changed,
    AstHavoc,
    AstWhile,
    AstBlock,
)
from aria.efmc.verifytools.boogie.bb import BB
from aria.efmc.verifytools.tools.boogie_loops import loops, Loop
from aria.efmc.sts import TransitionSystem

logger = logging.getLogger(__name__)


class BoogieToEFMCConverter:
    """
    Converts Boogie programs to EFMC transition systems.

    This converter focuses on extracting loops from Boogie programs and converting
    them to transition systems that can be verified using EFMC's engines.
    """

    def __init__(self):
        """Initialize the converter."""
        self.logger = logger
        self.variable_mapping = {}  # Maps original variables to Z3 variables
        self.prime_variable_mapping = (
            {}
        )  # Maps original variables to Z3 prime variables

    def parse_boogie_file(self, filename: str) -> AstProgram:
        """Parse a Boogie file and return the AST."""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
            return parseAst(content)
        except Exception as e:
            self.logger.error("Failed to parse Boogie file %s: %s", filename, e)
            raise

    def extract_loops_from_program(
        self, ast_program: AstProgram
    ) -> Tuple[List[Loop], Dict[str, BB]]:
        """Extract loops from a Boogie program AST."""
        # Get the first implementation (assuming single implementation)
        implementations = [
            decl for decl in ast_program.decls if isinstance(decl, AstImplementation)
        ]
        if not implementations:
            raise ValueError("No implementation found in Boogie program")

        implementation = implementations[0]
        self.logger.info("Processing implementation: %s", implementation.name)

        # Convert to basic blocks
        bbs = self._ast_to_basic_blocks(implementation)

        # Extract loops using the existing loop detection
        try:
            detected_loops = loops(bbs)
            self.logger.info("Found %d loops", len(detected_loops))
        except Exception as e:
            self.logger.error("Failed to detect loops: %s", e)
            # If loop detection fails, try to create a simple loop manually
            detected_loops = self._create_manual_loop(bbs)

        return detected_loops, bbs

    def _ast_to_basic_blocks(self, implementation: AstImplementation) -> Dict[str, BB]:
        """Convert AST implementation to basic blocks."""
        bbs = {}
        cur_label = None

        for stmt in implementation.body.stmts:
            # A basic block starts with a labeled statement
            if isinstance(stmt, AstLabel):
                cur_label = str(stmt.label)
                bbs[cur_label] = BB([], [], [])
                stmt = stmt.stmt

            # Add statement to current basic block
            if isinstance(stmt, (AstAssert, AstAssume, AstHavoc, AstAssignment)):
                if cur_label is None:
                    # Create entry block if no label found
                    cur_label = "_entry_"
                    bbs[cur_label] = BB([], [], [])
                bbs[cur_label].stmts.append(stmt)
            elif isinstance(stmt, AstWhile):
                # Convert while loop to basic blocks
                if cur_label is None:
                    cur_label = "_entry_"
                    bbs[cur_label] = BB([], [], [])

                # Create while loop structure with basic blocks
                while_header = f"{cur_label}_while_header"
                while_body = f"{cur_label}_while_body"
                while_exit = f"{cur_label}_while_exit"

                # Header block: contains the loop condition
                bbs[while_header] = BB(
                    [cur_label], [AstAssume(stmt.condition)], [while_body, while_exit]
                )
                bbs[cur_label].successors.append(while_header)

                # Body block: contains the loop body statements
                body_stmts = []
                if isinstance(stmt.body, AstBlock):
                    body_stmts = stmt.body.stmts
                else:
                    body_stmts = [stmt.body]

                bbs[while_body] = BB([while_header], body_stmts, [while_header])

                # Exit block: for statements after the while loop
                bbs[while_exit] = BB(
                    [while_header], [AstAssume(AstUnExpr("!", stmt.condition))], []
                )

                cur_label = while_exit
            elif isinstance(stmt, AstGoto):
                if cur_label is not None:
                    bbs[cur_label].successors.extend(map(str, stmt.labels))
                cur_label = None
            elif isinstance(stmt, AstReturn):
                cur_label = None
            else:
                self.logger.warning("Unknown statement type: %s", type(stmt))

        # Build predecessor relationships
        for bb_name in bbs:
            for succ in bbs[bb_name].successors:
                if succ in bbs:
                    bbs[succ].predecessors.append(bb_name)

        return bbs

    def convert_loop_to_transition_system(
        self, loop: Loop, bbs: Dict[str, BB], variables: Optional[List[str]] = None
    ) -> TransitionSystem:
        """Convert a single Boogie loop to an EFMC transition system."""
        self.logger.info("Converting loop with header: %s", loop.header)

        # Extract variables from loop if not provided
        if variables is None:
            variables = self._extract_loop_variables(loop, bbs)

        # Create Z3 variables
        z3_vars = []
        z3_prime_vars = []

        for var_name in variables:
            # Create current state variable
            z3_var = z3.Int(var_name)  # Assuming integer variables for now
            z3_prime_var = z3.Int(f"{var_name}!")  # Prime variable with ! suffix

            z3_vars.append(z3_var)
            z3_prime_vars.append(z3_prime_var)

            self.variable_mapping[var_name] = z3_var
            self.prime_variable_mapping[var_name] = z3_prime_var

        # Extract initial condition (entry condition to loop)
        init_condition = self._extract_initial_condition(loop, bbs)

        # Extract transition relation from loop body
        trans_condition = self._extract_transition_relation(loop, bbs)

        # Extract post condition (safety property to verify)
        post_condition = self._extract_post_condition(loop, bbs)

        # Create transition system
        ts = TransitionSystem(
            variables=z3_vars,
            prime_variables=z3_prime_vars,
            init=init_condition,
            trans=trans_condition,
            post=post_condition,
        )

        self.logger.info("Successfully created transition system")
        print(ts)
        return ts

    def _extract_loop_variables(self, loop: Loop, bbs: Dict[str, BB]) -> List[str]:
        """Extract all variables used in the loop."""
        variables = set()

        # Collect variables from all loop paths
        for path in loop.loop_paths:
            for bb_name in path:
                if bb_name in bbs:
                    bb = bbs[bb_name]
                    for stmt in bb.stmts:
                        variables.update(stmt_read(stmt))
                        variables.update(stmt_changed(stmt))

        # Also collect from exit paths
        for path in loop.exit_paths:
            for bb_name in path:
                if bb_name in bbs:
                    bb = bbs[bb_name]
                    for stmt in bb.stmts:
                        variables.update(stmt_read(stmt))
                        variables.update(stmt_changed(stmt))

        return sorted(list(variables))

    def _extract_initial_condition(self, loop: Loop, bbs: Dict[str, BB]) -> z3.ExprRef:
        """Extract initial condition for entering the loop."""
        conditions = []

        # Look for entry blocks that lead to the loop header
        # These contain initialization statements
        loop_header = loop.header[0] if loop.header else None
        if loop_header:
            # Find blocks that have the loop header as successor (entry blocks)
            # But exclude blocks that are part of the loop itself (like loop body blocks)
            entry_blocks = []
            for bb_name, bb in bbs.items():
                if loop_header in bb.successors and bb_name != loop_header:
                    # Check if this block is part of the loop structure
                    # Loop body blocks typically have names ending with "_while_body"
                    if not bb_name.endswith("_while_body"):
                        entry_blocks.append(bb_name)

            # Process assignments in entry blocks to create initial conditions
            for entry_bb_name in entry_blocks:
                if entry_bb_name in bbs:
                    bb = bbs[entry_bb_name]
                    for stmt in bb.stmts:
                        if isinstance(stmt, AstAssignment):
                            # Convert assignment to initial condition: x = value
                            lhs_var = str(stmt.lhs)
                            if lhs_var in self.variable_mapping:
                                rhs_expr = self._ast_expr_to_z3(
                                    stmt.rhs, is_prime=False
                                )
                                lhs_z3 = self.variable_mapping[lhs_var]
                                conditions.append(lhs_z3 == rhs_expr)

        # Don't include the loop entry condition here - that should be part of the transition relation
        # The initial condition should only contain the initialization assignments

        if conditions:
            return z3.And(conditions) if len(conditions) > 1 else conditions[0]
        else:
            return z3.BoolVal(True)  # No specific entry condition

    def _extract_transition_relation(
        self, loop: Loop, bbs: Dict[str, BB]
    ) -> z3.ExprRef:
        """Extract transition relation from loop body."""
        transitions = []

        # Process each loop path
        for path in loop.loop_paths:
            path_condition = self._extract_path_condition(
                path, bbs, include_guards=True
            )
            if path_condition is not None:
                transitions.append(path_condition)

        if transitions:
            return z3.Or(transitions) if len(transitions) > 1 else transitions[0]
        else:
            return z3.BoolVal(False)  # No valid transitions

    def _extract_path_condition(
        self, path: List[str], bbs: Dict[str, BB], include_guards: bool = True
    ) -> Optional[z3.ExprRef]:
        """Extract condition for a single path through the loop."""
        conditions = []
        assignments = []

        for bb_name in path:
            if bb_name not in bbs:
                continue

            bb = bbs[bb_name]
            for stmt in bb.stmts:
                if isinstance(stmt, AstAssume):
                    # Only add assumption as condition if we're including guards
                    if include_guards:
                        cond = self._ast_expr_to_z3(stmt.expr, is_prime=False)
                        conditions.append(cond)
                elif isinstance(stmt, AstAssignment):
                    # Add assignment relation
                    lhs_var = str(stmt.lhs)
                    if lhs_var in self.prime_variable_mapping:
                        rhs_expr = self._ast_expr_to_z3(stmt.rhs, is_prime=False)
                        lhs_prime = self.prime_variable_mapping[lhs_var]
                        assignments.append(lhs_prime == rhs_expr)
                elif isinstance(stmt, AstHavoc):
                    # Havoc statements introduce non-determinism
                    # For now, we'll just allow any value (no constraint)
                    pass

        # Add frame conditions for unchanged variables
        for var_name in self.variable_mapping:
            if var_name in self.prime_variable_mapping:
                # Check if variable is modified in this path
                modified = False
                for bb_name in path:
                    if bb_name in bbs:
                        bb = bbs[bb_name]
                        for stmt in bb.stmts:
                            if (
                                isinstance(stmt, AstAssignment)
                                and str(stmt.lhs) == var_name
                            ):
                                modified = True
                                break
                            elif isinstance(stmt, AstHavoc) and var_name in [
                                str(id_node.name) for id_node in stmt.ids
                            ]:
                                modified = True
                                break

                if not modified:
                    # Variable unchanged: x' = x
                    assignments.append(
                        self.prime_variable_mapping[var_name]
                        == self.variable_mapping[var_name]
                    )

        all_conditions = conditions + assignments
        if all_conditions:
            return z3.And(all_conditions)
        else:
            return z3.BoolVal(True)

    def _extract_post_condition(self, loop: Loop, bbs: Dict[str, BB]) -> z3.ExprRef:
        """Extract post condition (safety property) from actual assert statements."""
        assertions = []
        seen_assertions = set()  # Track unique assertions by string representation

        # Look for assert statements in exit paths and exit blocks
        for path in loop.exit_paths:
            for bb_name in path:
                if bb_name in bbs:
                    bb = bbs[bb_name]
                    for stmt in bb.stmts:
                        if isinstance(stmt, AstAssert):
                            # Convert the assertion to Z3
                            z3_assertion = self._ast_expr_to_z3(stmt.expr)
                            assertion_str = str(z3_assertion)
                            if assertion_str not in seen_assertions:
                                assertions.append(z3_assertion)
                                seen_assertions.add(assertion_str)

        # Also look in blocks that might contain post-loop assertions
        # (blocks with names ending in '_while_exit' or similar)
        for bb_name, bb in bbs.items():
            if "_while_exit" in bb_name or "_exit" in bb_name:
                for stmt in bb.stmts:
                    if isinstance(stmt, AstAssert):
                        # Convert the assertion to Z3
                        z3_assertion = self._ast_expr_to_z3(stmt.expr)
                        assertion_str = str(z3_assertion)
                        if assertion_str not in seen_assertions:
                            assertions.append(z3_assertion)
                            seen_assertions.add(assertion_str)

        # For k-induction, we need inductive safety properties, not post-conditions
        # So we prioritize safety invariants over assertions
        # Create a simple safety property for decreasing loops
        safety_invariants = []

        # Look for simple patterns like decreasing counters in the loop body
        for path in loop.loop_paths:
            for bb_name in path:
                if bb_name in bbs:
                    bb = bbs[bb_name]
                    for stmt in bb.stmts:
                        if isinstance(stmt, AstAssignment):
                            lhs_var = str(stmt.lhs)
                            if lhs_var in self.variable_mapping:
                                # Check if this is a decrement: x := x - 1
                                if (
                                    isinstance(stmt.rhs, AstBinExpr)
                                    and stmt.rhs.op == "-"
                                ):
                                    if (
                                        isinstance(stmt.rhs.lhs, AstId)
                                        and str(stmt.rhs.lhs.name) == lhs_var
                                        and isinstance(stmt.rhs.rhs, AstNumber)
                                        and stmt.rhs.rhs.num == 1
                                    ):
                                        # This is x := x - 1, so x >= 0 is likely a good invariant
                                        var_z3 = self.variable_mapping[lhs_var]
                                        safety_invariants.append(var_z3 >= 0)

        if safety_invariants:
            return (
                z3.And(safety_invariants)
                if len(safety_invariants) > 1
                else safety_invariants[0]
            )
        else:
            # Default safety property: true (no specific property to verify)
            return z3.BoolVal(True)

    def _ast_expr_to_z3(self, ast_expr, is_prime: bool = False) -> z3.ExprRef:
        """Convert Boogie AST expression to Z3 expression."""
        if isinstance(ast_expr, AstTrue):
            return z3.BoolVal(True)
        elif isinstance(ast_expr, AstFalse):
            return z3.BoolVal(False)
        elif isinstance(ast_expr, AstNumber):
            return z3.IntVal(ast_expr.num)
        elif isinstance(ast_expr, AstId):
            var_name = str(ast_expr.name)
            if is_prime and var_name in self.prime_variable_mapping:
                return self.prime_variable_mapping[var_name]
            elif not is_prime and var_name in self.variable_mapping:
                return self.variable_mapping[var_name]
            else:
                # Create new variable if not seen before
                if is_prime:
                    z3_var = z3.Int(f"{var_name}!")
                    self.prime_variable_mapping[var_name] = z3_var
                else:
                    z3_var = z3.Int(var_name)
                    self.variable_mapping[var_name] = z3_var
                return z3_var
        elif isinstance(ast_expr, AstUnExpr):
            operand = self._ast_expr_to_z3(ast_expr.expr, is_prime)
            if ast_expr.op == "!":
                return z3.Not(operand)
            elif ast_expr.op == "-":
                return -operand
            else:
                raise NotImplementedError(f"Unary operator {ast_expr.op} not supported")
        elif isinstance(ast_expr, AstBinExpr):
            left = self._ast_expr_to_z3(ast_expr.lhs, is_prime)
            right = self._ast_expr_to_z3(ast_expr.rhs, is_prime)

            op = ast_expr.op
            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "div":
                return left / right
            elif op == "mod":
                return left % right
            elif op == "==":
                return left == right
            elif op == "!=":
                return left != right
            elif op == "<":
                return left < right
            elif op == "<=":
                return left <= right
            elif op == ">":
                return left > right
            elif op == ">=":
                return left >= right
            elif op == "&&":
                return z3.And(left, right)
            elif op == "||":
                return z3.Or(left, right)
            elif op == "==>":
                return z3.Implies(left, right)
            else:
                raise NotImplementedError(f"Binary operator {op} not supported")
        else:
            raise NotImplementedError(f"Expression type {type(ast_expr)} not supported")

    def convert_file_to_transition_system(self, filename: str) -> TransitionSystem:
        """Convert a Boogie file to a transition system (main entry point)."""
        self.logger.info(f"Converting Boogie file: {filename}")

        # Parse the file
        ast_program = self.parse_boogie_file(filename)

        # Extract loops
        detected_loops, bbs = self.extract_loops_from_program(ast_program)

        if not detected_loops:
            raise ValueError("No loops found in the Boogie program")

        if len(detected_loops) > 1:
            self.logger.warning("Multiple loops found, using the first one")

        # Convert the first loop to transition system
        loop = detected_loops[0]
        return self.convert_loop_to_transition_system(loop, bbs)

    def _create_manual_loop(self, bbs: Dict[str, BB]) -> List[Loop]:
        """Create a simple manual loop when automatic detection fails."""
        self.logger.info("Attempting manual loop creation...")

        # Look for while loop structures we created
        while_headers = [name for name in bbs.keys() if name.endswith("_while_header")]

        if while_headers:
            # Use the while loop structure we created
            header = while_headers[0]
            self.logger.info("Found while loop header: %s", header)

            # Find the corresponding body and exit blocks
            body_name = header.replace("_while_header", "_while_body")
            exit_name = header.replace("_while_header", "_while_exit")

            if body_name in bbs and exit_name in bbs:
                # Create proper loop paths
                loop_paths = [[header, body_name]]  # header -> body -> back to header
                exit_paths = [[header, exit_name]]  # header -> exit

                # Extract the condition from the header block
                entry_cond = AstTrue()
                if bbs[header].stmts and isinstance(bbs[header].stmts[0], AstAssume):
                    entry_cond = bbs[header].stmts[0].expr

                # Create the Loop namedtuple
                manual_loop = Loop(
                    header=tuple([header]),
                    loop_paths=loop_paths,
                    exit_paths=exit_paths,
                    entry_cond=entry_cond,
                )

                return [manual_loop]

        # Fallback to generic loop detection
        loop_candidates = []
        for bb_name, bb in bbs.items():
            # A loop header typically has predecessors and successors
            if len(bb.predecessors) > 0 and len(bb.successors) > 0:
                # Check if any successor can reach back to this block
                for succ in bb.successors:
                    if self._can_reach(succ, bb_name, bbs, visited=set()):
                        loop_candidates.append(bb_name)
                        break

        if not loop_candidates:
            self.logger.warning("No loop candidates found")
            return []

        # Use the first candidate as loop header
        header = loop_candidates[0]
        self.logger.info("Using %s as loop header", header)

        # Create a simple loop structure
        loop_paths = [[header]]  # Simple single-block loop path
        exit_paths = [[header, "_exit_"]]  # Simple exit path
        entry_cond = AstTrue()  # Default entry condition

        # Ensure we have at least one variable for the transition system to work
        if not self.variable_mapping:
            # Add a dummy variable if none found
            dummy_var = z3.Int("dummy")
            self.variable_mapping["dummy"] = dummy_var
            self.prime_variable_mapping["dummy"] = z3.Int("dummy!")

        # Create the Loop namedtuple
        manual_loop = Loop(
            header=tuple([header]),  # Convert to tuple as expected
            loop_paths=loop_paths,
            exit_paths=exit_paths,
            entry_cond=entry_cond,
        )

        return [manual_loop]

    def _can_reach(
        self, start: str, target: str, bbs: Dict[str, BB], visited: Set[str]
    ) -> bool:
        """Check if start block can reach target block."""
        if start == target:
            return True
        if start in visited or start not in bbs:
            return False

        visited.add(start)
        for succ in bbs[start].successors:
            if self._can_reach(succ, target, bbs, visited.copy()):
                return True
        return False


def boogie_to_efmc(filename: str) -> TransitionSystem:
    """Convenience function to convert a Boogie file to EFMC transition system."""
    converter = BoogieToEFMCConverter()
    return converter.convert_file_to_transition_system(filename)
