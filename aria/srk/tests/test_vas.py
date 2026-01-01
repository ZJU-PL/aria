"""
Tests for the Vector Addition Systems (VAS) and Petri Nets module.
"""

import unittest
from fractions import Fraction
from aria.srk.vas import (
    Transformer,
    VectorAdditionSystem,
    PetriNet,
    Place,
    Transition,
    ReachabilityResult,
    make_vas,
    make_petri_net,
    producer_consumer_petri_net,
)
from aria.srk.linear import QQVector


class TestTransformer(unittest.TestCase):
    """Test transformer operations."""

    def test_creation(self):
        """Test transformer creation."""
        a = QQVector({0: Fraction(1), 1: Fraction(0)})
        b = QQVector({0: Fraction(2), 1: Fraction(-1)})

        transformer = Transformer(a, b)

        self.assertEqual(transformer.a.entries[0], Fraction(1))
        self.assertEqual(transformer.a.entries[1], Fraction(0))
        self.assertEqual(transformer.b.entries[0], Fraction(2))

    def test_invalid_transformer(self):
        """Test transformer with invalid 'a' vector."""
        # 'a' vector should only contain 0s and 1s
        a = QQVector({0: Fraction(2)})  # Invalid: 2 is not 0 or 1
        b = QQVector({0: Fraction(1)})

        with self.assertRaises(ValueError):
            Transformer(a, b)

    def test_apply(self):
        """Test transformer application."""
        a = QQVector({0: Fraction(1), 1: Fraction(0)})
        b = QQVector({0: Fraction(2), 1: Fraction(-1)})

        transformer = Transformer(a, b)
        state = QQVector({0: Fraction(3), 1: Fraction(4)})

        new_state = transformer.apply(state)

        # x0' = x0 + 2 = 3 + 2 = 5
        # x1' = -1 (since a1 = 0)
        self.assertEqual(new_state.entries[0], Fraction(5))
        self.assertEqual(new_state.entries[1], Fraction(-1))


class TestVectorAdditionSystem(unittest.TestCase):
    """Test Vector Addition System operations."""

    def test_creation(self):
        """Test VAS creation."""
        transformers = [
            Transformer(QQVector({0: Fraction(1)}), QQVector({0: Fraction(1)})),
            Transformer(QQVector({0: Fraction(0)}), QQVector({0: Fraction(-1)})),
        ]

        vas = VectorAdditionSystem(transformers, 1)
        self.assertEqual(len(vas.transformers), 2)
        self.assertEqual(vas.dimension, 1)

    def test_add_transformer(self):
        """Test adding transformers to VAS."""
        vas = VectorAdditionSystem([], 1)
        transformer = Transformer(
            QQVector({0: Fraction(1)}), QQVector({0: Fraction(1)})
        )

        new_vas = vas.add_transformer(transformer)
        self.assertEqual(len(new_vas.transformers), 1)

    def test_applicable_transformers(self):
        """Test finding applicable transformers."""
        transformers = [
            Transformer(
                QQVector({0: Fraction(1)}), QQVector({0: Fraction(1)})
            ),  # requires x0 >= 0
            Transformer(
                QQVector({0: Fraction(0)}), QQVector({0: Fraction(-1)})
            ),  # always applicable
        ]

        vas = VectorAdditionSystem(transformers, 1)

        # State with x0 = 1 (both applicable)
        state1 = QQVector({0: Fraction(1)})
        applicable1 = vas.is_applicable(state1)
        self.assertEqual(len(applicable1), 2)

        # State with x0 = -1 (only second applicable)
        state2 = QQVector({0: Fraction(-1)})
        applicable2 = vas.is_applicable(state2)
        self.assertEqual(len(applicable2), 1)

    def test_step(self):
        """Test one step of VAS execution."""
        transformers = [
            Transformer(QQVector({0: Fraction(1)}), QQVector({0: Fraction(1)}))
        ]

        vas = VectorAdditionSystem(transformers, 1)
        state = QQVector({0: Fraction(1)})

        next_states = vas.step(state)
        self.assertEqual(len(next_states), 1)

        next_state = list(next_states)[0]
        self.assertEqual(next_state.entries[0], Fraction(2))

    def test_reachability(self):
        """Test reachability analysis."""
        # Simple VAS: x -> x+1
        transformers = [
            Transformer(QQVector({0: Fraction(1)}), QQVector({0: Fraction(1)}))
        ]

        vas = VectorAdditionSystem(transformers, 1)
        start = QQVector({0: Fraction(0)})
        target = QQVector({0: Fraction(5)})

        result = vas.reachability(start, target, max_steps=10)
        self.assertEqual(result, ReachabilityResult.REACHABLE)


class TestPetriNet(unittest.TestCase):
    """Test Petri net operations."""

    def test_creation(self):
        """Test Petri net creation."""
        places = [Place("p1", 1), Place("p2", 0)]
        transitions = [Transition("t1", {"p1": 1}, {"p2": 1})]

        net = PetriNet(places, transitions)
        self.assertEqual(len(net.places), 2)
        self.assertEqual(len(net.transitions), 1)

    def test_initial_marking(self):
        """Test initial marking."""
        places = [Place("p1", 2), Place("p2", 1)]
        transitions = []

        net = PetriNet(places, transitions)
        initial = net.initial_marking()

        self.assertEqual(initial[places[0]], 2)
        self.assertEqual(initial[places[1]], 1)

    def test_transition_enabled(self):
        """Test transition enabling."""
        places = [Place("p1", 1), Place("p2", 0)]
        transition = Transition("t1", {"p1": 1}, {"p2": 1})

        marking = {"p1": 1, "p2": 0}

        self.assertTrue(transition.is_enabled(marking))

        marking2 = {"p1": 0, "p2": 0}
        self.assertFalse(transition.is_enabled(marking2))

    def test_fire_transition(self):
        """Test firing transitions."""
        places = [Place("p1", 1), Place("p2", 0)]
        transition = Transition("t1", {"p1": 1}, {"p2": 1})

        marking = {"p1": 1, "p2": 0}
        new_marking = transition.fire(marking)

        self.assertEqual(new_marking["p1"], 0)
        self.assertEqual(new_marking["p2"], 1)

    def test_petri_net_step(self):
        """Test one step of Petri net execution."""
        places = [Place("p1", 1), Place("p2", 0)]
        transitions = [Transition("t1", {"p1": 1}, {"p2": 1})]

        net = PetriNet(places, transitions)
        marking = {"p1": 1, "p2": 0}

        next_markings = net.step(marking)
        self.assertEqual(len(next_markings), 1)

        next_marking = list(next_markings)[0]
        self.assertEqual(next_marking["p1"], 0)
        self.assertEqual(next_marking["p2"], 1)

    def test_producer_consumer(self):
        """Test producer-consumer Petri net."""
        net = producer_consumer_petri_net()

        initial = net.initial_marking()
        self.assertEqual(initial[net.places[0]], 1)  # producer
        self.assertEqual(initial[net.places[1]], 0)  # buffer
        self.assertEqual(initial[net.places[2]], 0)  # consumer

        # Should be able to produce
        enabled = net.enabled_transitions(initial)
        self.assertEqual(len(enabled), 1)
        self.assertEqual(enabled[0].name, "produce")

        # Fire the produce transition
        new_marking = enabled[0].fire(initial)
        self.assertEqual(new_marking[net.places[0]], 0)  # producer consumed
        self.assertEqual(new_marking[net.places[1]], 1)  # buffer produced
        self.assertEqual(new_marking[net.places[2]], 0)  # consumer unchanged

    def test_vas_conversion(self):
        """Test conversion from Petri net to VAS."""
        net = producer_consumer_petri_net()
        vas = net.to_vas()

        self.assertEqual(vas.dimension, 3)
        self.assertEqual(len(vas.transformers), 2)  # produce and consume


if __name__ == "__main__":
    unittest.main()
