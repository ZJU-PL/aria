# coding: utf-8
"""
Demo script for the complete Dissolve algorithm implementation.

This script demonstrates the key features of the full Dissolve implementation:
- Dilemma rule engine
- UBTree with subsumption
- Scheduler architecture
- Complete algorithm flow
"""

from aria.bool.dissolve.dissolve import (
    DissolveConfig,
    DilemmaTriple, DilemmaQuery,
    DilemmaEngine, UBTree, Scheduler
)

def demo_dilemma_engine():
    """Demonstrate the Dilemma rule engine."""
    print("=== Dilemma Engine Demo ===")

    engine = DilemmaEngine()

    # Add some equivalences
    engine.add_equivalence(1, 1)  # x1 ≡ 1
    engine.add_equivalence(2, 0)  # x2 ≡ 0

    print(f"Variable 1 ≡ {engine.get_equivalence(1)}")
    print(f"Variable 2 ≡ {engine.get_equivalence(2)}")
    print(f"Are variables 1 and 2 equivalent? {engine.is_equivalent(1, 2)}")

    # Demonstrate dilemma triple
    triple = DilemmaTriple(3, 0, 1)  # v3 with values (0, 1)
    print(f"Dilemma triple: v{triple.variable} with values ({triple.value_a}, {triple.value_b})")

def demo_ubtree():
    """Demonstrate the UBTree data structure."""
    print("\n=== UBTree Demo ===")

    ubtree = UBTree()

    # Insert some clauses
    clause1 = [1, 2]  # (x1 ∨ x2)
    clause2 = [-1, 3]  # (¬x1 ∨ x3)
    clause3 = [2, 3]  # (x2 ∨ x3) - should subsume clause1 in some contexts

    node1 = ubtree.insert_clause(clause1, 0)
    node2 = ubtree.insert_clause(clause2, 0)
    node3 = ubtree.insert_clause(clause3, 0)

    print(f"Inserted clause {clause1}, node subsumed: {node1.subsumed_by is not None}")
    print(f"Inserted clause {clause2}, node subsumed: {node2.subsumed_by is not None}")
    print(f"Inserted clause {clause3}, node subsumed: {node3.subsumed_by is not None}")

    # Get best clauses for next round
    best = ubtree.get_best_clauses_for_round(0, 10)
    print(f"Best clauses for round 0: {best}")

def demo_dilemma_queries():
    """Demonstrate dilemma query generation."""
    print("\n=== Dilemma Query Generation Demo ===")

    # Create a simple config for demonstration
    config = DissolveConfig(k_split_vars=2)

    # Create a minimal Dissolve instance (just for demo)
    # pylint: disable=too-few-public-methods,duplicate-code
    class DemoDissolve:
        """Demo class for demonstrating dilemma query generation."""
        def __init__(self):
            self.cfg = config
            self.variable_scores = {1: 10, 2: 8, 3: 5}
            self.decision_polarities = {1: 0.7, 2: 0.3, 3: 0.5}

        def _select_split_variables(self):
            candidates = list(range(1, self.cfg.k_split_vars + 1))
            if self.variable_scores:
                candidates.sort(key=lambda v: -self.variable_scores.get(v, 0))
            return candidates[:self.cfg.k_split_vars]

        def _assumptions_from_mask(self, vars_to_split, mask):
            assumps = []
            for i, v in enumerate(vars_to_split):
                bit = (mask >> i) & 1
                polarity = self.decision_polarities.get(v, 0.5)
                if bit == 1:
                    assumps.append(v if polarity >= 0.5 else -v)
                else:
                    assumps.append(-v if polarity >= 0.5 else v)
            return assumps

        def _create_dilemma_triple(self, vars_to_split, mask):
            if not vars_to_split:
                return DilemmaTriple(1, 0, 1)
            primary_var = vars_to_split[0]
            value_a = (mask >> 0) & 1
            value_b = 1 - value_a
            return DilemmaTriple(primary_var, value_a, value_b)

    demo = DemoDissolve()

    # Generate queries manually for demo
    queries = []
    # pylint: disable=protected-access
    split_vars = demo._select_split_variables()
    for mask in range(1 << len(split_vars)):
        assumptions = demo._assumptions_from_mask(split_vars, mask)
        dilemma_triple = demo._create_dilemma_triple(split_vars, mask)
        query = DilemmaQuery(
            assumptions=assumptions,
            dilemma_triple=dilemma_triple,
            round_id=0,
            query_id=mask
        )
        queries.append(query)

    print(f"Generated {len(queries)} dilemma queries:")
    for i, query in enumerate(queries):
        print(f"Query {i}: assumptions={query.assumptions}")
        if query.dilemma_triple:
            dt = query.dilemma_triple
            print(f"  Dilemma: v{dt.variable} with values ({dt.value_a}, {dt.value_b})")

def demo_scheduler():
    """Demonstrate the scheduler architecture."""
    print("\n=== Scheduler Architecture Demo ===")

    scheduler = Scheduler(num_workers=2)
    print(f"Created scheduler with {scheduler.num_workers} workers")
    print(f"Idle workers: {len(scheduler.idle_workers)}")
    print(f"Active queries: {len(scheduler.active_queries)}")

def main():
    """Run all demos."""
    print("Complete Dissolve Algorithm Implementation Demo")
    print("=" * 50)

    try:
        demo_dilemma_engine()
        demo_ubtree()
        demo_dilemma_queries()
        demo_scheduler()

        print("\n" + "=" * 50)
        print("All components demonstrated successfully!")
        print("\nThe implementation includes:")
        print("- Complete Dilemma rule engine with propagation rules")
        print("- UBTree with subsumption for clause management")
        print("- Dilemma-based query generation")
        print("- Producer/consumer scheduler architecture")
        print("- Sophisticated variable selection heuristics")
        print("- Full integration following the paper's Algorithm 1-3")

    except (RuntimeError, ValueError, AttributeError) as e:
        print(f"Demo error: {e}")
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()

if __name__ == "__main__":
    main()
