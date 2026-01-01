#!/usr/bin/env python3
"""
Simple test script for WMI functionality.

This demonstrates basic WMI usage with different density functions.
"""

import sys

# Add the aria path so we can import it

try:
    import z3
    from aria.prob import wmi_integrate, WMIOptions, UniformDensity, GaussianDensity

    def test_uniform_triangle():
        """Test WMI with uniform density over a triangular region."""
        print("Testing WMI with uniform density over triangle...")

        x, y = z3.Reals("x y")
        # Triangle: x >= 0, y >= 0, x + y <= 1
        formula = z3.And(x >= 0, y >= 0, x + y <= 1)

        # Uniform density over [0,1] × [0,1]
        density = UniformDensity({"x": (0, 1), "y": (0, 1)})

        # Use region-based method for bounded regions (should be more stable)
        options = WMIOptions(method="region", num_samples=100, random_seed=42)
        result = wmi_integrate(formula, density, options)

        expected_msg = "expected ~1.0 - uniform density over constraint"
        print(f"WMI result for triangle: {result:.4f} ({expected_msg})")
        return result

    def test_gaussian_triangle():
        """Test WMI with Gaussian density over a triangular region."""
        print("Testing WMI with Gaussian density over triangle...")

        x, y = z3.Reals("x y")
        formula = z3.And(x >= 0, y >= 0, x + y <= 1)

        # Gaussian centered at (0.5, 0.5) with small variance
        gaussian = GaussianDensity(
            {"x": 0.5, "y": 0.5}, {"x": {"x": 0.1}, "y": {"y": 0.1}}
        )

        options = WMIOptions(method="region", num_samples=100, random_seed=42)
        result = wmi_integrate(formula, gaussian, options)

        print(f"WMI result for Gaussian: {result:.4f}")
        return result

    def test_region_vs_sampling():
        """Test region-based vs sampling-based WMI."""
        print("Testing region-based vs sampling-based WMI...")

        x, y = z3.Reals("x y")
        formula = z3.And(x >= 0, y >= 0, x + y <= 1)
        density = UniformDensity({"x": (0, 1), "y": (0, 1)})

        # Test sampling method
        options_sampling = WMIOptions(
            method="sampling", num_samples=100, random_seed=42
        )
        result_sampling = wmi_integrate(formula, density, options_sampling)

        # Test region method (uses enhanced sampling)
        options_region = WMIOptions(method="region", num_samples=100, random_seed=42)
        result_region = wmi_integrate(formula, density, options_region)

        print(f"Sampling method: {result_sampling:.4f}")
        print(f"Region method: {result_region:.4f}")
        expected = "Expected ~1.0 for both methods (uniform density over constraint)"
        print(expected)

    if __name__ == "__main__":
        print("WMI Test Script")
        print("=" * 50)

        try:
            test_uniform_triangle()
            print()
            test_gaussian_triangle()
            print()
            test_region_vs_sampling()
            print("\nWMI tests completed successfully!")

        except (ValueError, ImportError, RuntimeError) as e:
            print(f"Error during WMI testing: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure aria and its dependencies are properly installed.")
    sys.exit(1)
