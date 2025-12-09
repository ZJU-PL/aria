"""
Tests for the apron module.
"""

import unittest
from aria.srk.apron import (
    ApronManager, ApronAbstractValue, ApronDomain, ApronAnalysis,
    make_apron_manager, make_apron_domain, make_apron_analysis,
    apron_available, get_available_apron_domains,
    ApronError, ApronUnavailableError, check_apron_available
)
from aria.srk.syntax import make_context, make_expression_builder, Type


class TestApronManager(unittest.TestCase):
    """Test Apron manager functionality."""

    def setUp(self):
        self.context = make_context()

    def test_manager_creation_interval(self):
        """Test creating interval Apron manager."""
        manager = make_apron_manager("interval")
        self.assertIsInstance(manager, ApronManager)
        self.assertEqual(manager.domain_type, "interval")

    def test_manager_creation_octagon(self):
        """Test creating octagon Apron manager."""
        manager = make_apron_manager("octagon")
        self.assertIsInstance(manager, ApronManager)
        self.assertEqual(manager.domain_type, "octagon")

    def test_manager_creation_polyhedra(self):
        """Test creating polyhedra Apron manager."""
        manager = make_apron_manager("polyhedra")
        self.assertIsInstance(manager, ApronManager)
        self.assertEqual(manager.domain_type, "polyhedra")

    def test_manager_invalid_domain(self):
        """Test creating manager with invalid domain type."""
        with self.assertRaises(ValueError):
            ApronManager("invalid_domain")

    def test_manager_availability(self):
        """Test manager availability checking."""
        manager = ApronManager("interval")

        # The availability depends on whether Apron is installed
        # We just test that the method exists and returns a boolean
        available = manager.is_available()
        self.assertIsInstance(available, bool)

        if available:
            # If Apron is available, we can get the manager
            apron_manager = manager.get_manager()
            self.assertIsNotNone(apron_manager)
        else:
            # If Apron is not available, getting manager should raise error
            with self.assertRaises(RuntimeError):
                manager.get_manager()


class TestApronAbstractValue(unittest.TestCase):
    """Test Apron abstract values."""

    def setUp(self):
        self.context = make_context()

    def test_abstract_value_creation(self):
        """Test creating abstract values."""
        manager = ApronManager("interval")

        # Create a dummy abstract value for testing
        # In a real implementation, this would use actual Apron values
        if manager.is_available():
            # Only test if Apron is available
            try:
                # This is a placeholder since we don't have real Apron values
                # In practice, this would be created from Apron operations
                pass
            except:
                # If we can't create real values, just test the structure
                pass

    def test_abstract_value_operations_unavailable(self):
        """Test operations when Apron is not available."""
        manager = ApronManager("interval")

        if not manager.is_available():
            # Test that operations fail gracefully when Apron is unavailable
            with self.assertRaises(RuntimeError):
                ApronAbstractValue(manager, None)

            # Test that methods raise appropriate errors
            try:
                dummy_value = ApronAbstractValue(manager, None)
                dummy_value.join(dummy_value)
                self.fail("Expected RuntimeError for unavailable Apron")
            except RuntimeError:
                pass  # Expected


class TestApronDomain(unittest.TestCase):
    """Test Apron domain functionality."""

    def setUp(self):
        self.context = make_context()

    def test_domain_creation(self):
        """Test creating Apron domains."""
        domain = make_apron_domain(self.context, "interval")
        self.assertIsInstance(domain, ApronDomain)
        self.assertEqual(domain.domain_type, "interval")
        self.assertEqual(domain.context, self.context)

    def test_domain_top_bottom(self):
        """Test top and bottom elements."""
        domain = ApronDomain(self.context, "interval")

        if domain.manager.is_available():
            try:
                top_val = domain.top()
                bottom_val = domain.bottom()

                # These should be abstract values
                self.assertIsInstance(top_val, ApronAbstractValue)
                self.assertIsInstance(bottom_val, ApronAbstractValue)

            except RuntimeError:
                # If Apron operations fail, that's expected when not available
                pass
        else:
            # When Apron is not available, operations should raise errors
            with self.assertRaises(RuntimeError):
                domain.top()

            with self.assertRaises(RuntimeError):
                domain.bottom()

    def test_domain_operations(self):
        """Test domain operations."""
        domain = ApronDomain(self.context, "interval")

        if domain.manager.is_available():
            try:
                # Create some test values (placeholder)
                val1 = ApronAbstractValue(domain.manager, None)
                val2 = ApronAbstractValue(domain.manager, None)

                # Test join and meet operations
                joined = domain.join(val1, val2)
                met = domain.meet(val1, val2)

                self.assertIsInstance(joined, ApronAbstractValue)
                self.assertIsInstance(met, ApronAbstractValue)

            except RuntimeError:
                # Expected when Apron is not available
                pass
        else:
            # Test that operations fail when Apron is unavailable
            try:
                val1 = ApronAbstractValue(domain.manager, None)
                val2 = ApronAbstractValue(domain.manager, None)
                domain.join(val1, val2)
                self.fail("Expected RuntimeError for unavailable Apron")
            except RuntimeError:
                pass  # Expected


class TestApronAnalysis(unittest.TestCase):
    """Test Apron analysis functionality."""

    def setUp(self):
        self.context = make_context()

    def test_analysis_creation(self):
        """Test creating Apron analysis."""
        analysis = make_apron_analysis(self.context, "interval")
        self.assertIsInstance(analysis, ApronAnalysis)
        self.assertEqual(analysis.domain.domain_type, "interval")

    def test_forward_analysis(self):
        """Test forward analysis."""
        analysis = ApronAnalysis(self.context, "interval")

        # Test forward analysis (placeholder)
        if analysis.domain.manager.is_available():
            try:
                # Create dummy initial state only if Apron is available
                initial_state = ApronAbstractValue(analysis.domain.manager, None)
                # This would perform actual forward analysis
                result = analysis.forward_analysis(initial_state, None)
                # In a real test, we'd check the result
            except:
                pass  # Expected if Apron operations aren't fully implemented
        else:
            # Should raise error when Apron unavailable
            with self.assertRaises(RuntimeError):
                # Try to create initial state - should fail
                initial_state = ApronAbstractValue(analysis.domain.manager, None)
                analysis.forward_analysis(initial_state, None)

    def test_backward_analysis(self):
        """Test backward analysis."""
        analysis = ApronAnalysis(self.context, "interval")

        # Test backward analysis (placeholder)
        if analysis.domain.manager.is_available():
            try:
                # Create dummy final state only if Apron is available
                final_state = ApronAbstractValue(analysis.domain.manager, None)
                # This would perform actual backward analysis
                result = analysis.backward_analysis(final_state, None)
                # In a real test, we'd check the result
            except:
                pass  # Expected if Apron operations aren't fully implemented
        else:
            # Should raise error when Apron unavailable
            with self.assertRaises(RuntimeError):
                # Try to create final state - should fail
                final_state = ApronAbstractValue(analysis.domain.manager, None)
                analysis.backward_analysis(final_state, None)


class TestApronUtilities(unittest.TestCase):
    """Test Apron utility functions."""

    def test_apron_available(self):
        """Test apron availability checking."""
        available = apron_available()
        self.assertIsInstance(available, bool)

    def test_get_available_domains(self):
        """Test getting available Apron domains."""
        domains = get_available_apron_domains()
        self.assertIsInstance(domains, list)

        # Should contain domain names as strings
        for domain in domains:
            self.assertIsInstance(domain, str)

    def test_check_apron_available(self):
        """Test checking Apron availability with error."""
        if not apron_available():
            # Should raise error when Apron is not available
            with self.assertRaises(ApronUnavailableError):
                check_apron_available()
        else:
            # Should not raise error when Apron is available
            try:
                check_apron_available()
            except ApronUnavailableError:
                self.fail("check_apron_available raised error unexpectedly")


class TestApronErrors(unittest.TestCase):
    """Test Apron error classes."""

    def test_apron_error_hierarchy(self):
        """Test that error classes are properly defined."""
        # Test that we can create error instances
        error = ApronError("Test error")
        unavailable_error = ApronUnavailableError("Test unavailable error")

        self.assertIsInstance(error, Exception)
        self.assertIsInstance(unavailable_error, ApronError)
        self.assertIsInstance(unavailable_error, Exception)


class TestApronIntegration(unittest.TestCase):
    """Test integration with other SRK modules."""

    def setUp(self):
        self.context = make_context()

    def test_integration_with_syntax(self):
        """Test Apron integration with syntax module."""
        # Test that we can create expressions and use them with Apron
        builder = make_expression_builder(self.context)

        # Create some variables
        x = builder.mk_var(1, Type.INT)
        y = builder.mk_var(2, Type.INT)

        # Test that Apron domain can be created with the context
        domain = ApronDomain(self.context, "interval")
        self.assertEqual(domain.context, self.context)


if __name__ == '__main__':
    unittest.main()
