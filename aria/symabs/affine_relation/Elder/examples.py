"""Examples demonstrating the Elder abstract domains implementation.

This module provides examples showing how to use the MOS, KS, and AG domains
together with the conversion algorithms from Elder et al.'s paper.
"""

import numpy as np
import z3
from .matrix_ops import Matrix
from .mos_domain import MOS, alpha_mos, create_z3_variables
from .ks_domain import KS
from .ag_domain import AG
from .conversions import mos_to_ks, ks_to_mos, ag_to_ks, ks_to_ag, ag_to_mos


def example_identity_relation():
    """Example: Identity relation x' = x"""
    print("=== Example: Identity Relation x' = x ===")

    # Create MOS element for identity transformation
    k = 2  # Two variables
    w = 32

    # Identity matrix: [I 0; 0 1] where I is 2x2 identity
    identity_data = np.eye(k + 1, dtype=object)
    identity_matrix = Matrix(identity_data, 2**w)

    mos_identity = MOS([identity_matrix], w)
    print(f"MOS element: {mos_identity}")
    print(f"Concretization: {mos_identity.concretize()}")

    # Convert to KS
    ks_identity = mos_to_ks(mos_identity)
    print(f"KS element: {ks_identity}")
    print(f"KS concretization: {ks_identity.concretize()}")

    # Convert to AG
    ag_identity = ks_to_ag(ks_identity)
    print(f"AG element: {ag_identity}")
    print(f"AG concretization: {ag_identity.concretize()}")

    print()


def example_affine_transformation():
    """Example: Affine transformation x' = x + 1"""
    print("=== Example: Affine Transformation x' = x + 1 ===")

    k = 1  # One variable
    w = 32

    # Transformation matrix: [1 1; 0 1] (x' = x + 1)
    transform_data = np.array([[1, 1], [0, 1]], dtype=object)
    transform_matrix = Matrix(transform_data, 2**w)

    mos_transform = MOS([transform_matrix], w)
    print(f"MOS element: {mos_transform}")
    print(f"Concretization: {mos_transform.concretize()}")

    # Convert to other domains
    ks_transform = mos_to_ks(mos_transform)
    print(f"KS concretization: {ks_transform.concretize()}")

    ag_transform = ks_to_ag(ks_transform)
    print(f"AG concretization: {ag_transform.concretize()}")

    print()


def example_ks_constraints():
    """Example: KS domain with explicit constraints"""
    print("=== Example: KS Domain with Constraints ===")

    k = 2  # Two variables
    w = 32

    # Constraint: x + y' = 0 (x = -y')
    constraint_data = np.zeros((1, 2*k + 1), dtype=object)
    constraint_data[0, 0] = 1    # x coefficient
    constraint_data[0, 2] = 1    # y' coefficient (index k + 1 = 3 for k=2)
    constraint_data[0, 4] = 0    # Constant term

    ks_constraint = KS(Matrix(constraint_data, 2**w), w)
    print(f"KS element: {ks_constraint}")
    print(f"Concretization: {ks_constraint.concretize()}")

    # Convert to MOS
    mos_from_ks = ks_to_mos(ks_constraint)
    print(f"MOS from KS: {mos_from_ks}")
    print(f"MOS concretization: {mos_from_ks.concretize()}")

    print()


def example_ag_generators():
    """Example: AG domain with generators"""
    print("=== Example: AG Domain with Generators ===")

    k = 2  # Two variables
    w = 32

    # Generator: x + y' = 0 (represents the relation x = -y')
    generator_data = np.zeros((1, 2*k + 1), dtype=object)
    generator_data[0, 0] = 1    # x coefficient
    generator_data[0, 2] = 1    # y' coefficient
    generator_data[0, 4] = 0    # Constant term

    ag_generator = AG(Matrix(generator_data, 2**w), w)
    print(f"AG element: {ag_generator}")
    print(f"Concretization: {ag_generator.concretize()}")

    # Convert to MOS using shatter
    mos_from_ag = ag_to_mos(ag_generator)
    print(f"MOS from AG: {mos_from_ag}")
    print(f"MOS concretization: {mos_from_ag.concretize()}")

    print()


def example_domain_equivalence():
    """Example: Demonstrating equivalence between domains"""
    print("=== Example: Domain Equivalence ===")

    k = 1  # One variable
    w = 32

    # Start with MOS: x' = x * 2
    scale_data = np.array([[2, 0], [0, 1]], dtype=object)
    scale_matrix = Matrix(scale_data, 2**w)
    mos_original = MOS([scale_matrix], w)

    print("Original MOS:")
    print(f"  Concretization: {mos_original.concretize()}")

    # Convert MOS -> KS -> AG -> MOS
    ks_version = mos_to_ks(mos_original)
    print(f"KS version concretization: {ks_version.concretize()}")

    ag_version = ks_to_ag(ks_version)
    print(f"AG version concretization: {ag_version.concretize()}")

    mos_final = ag_to_mos(ag_version)
    print(f"Final MOS concretization: {mos_final.concretize()}")

    # Check if they're equivalent (they should be for this simple case)
    print(f"MOS equivalent: {mos_original == mos_final}")

    print()


def example_alpha_function():
    """Example: Using the α function for symbolic abstraction"""
    print("=== Example: Alpha Function ===")

    # Simple QFBV formula: (x = y + 1) ∧ (x' = x * 2)
    # Using Z3 expressions directly

    variables = ['x', 'y']
    pre_vars, post_vars = create_z3_variables(variables)

    # Create the formula using Z3 expressions
    phi = z3.And(
        pre_vars[0] == pre_vars[1] + 1,  # x = y + 1
        post_vars[0] == pre_vars[0] * 2   # x' = x * 2
    )

    mos_result = alpha_mos(phi, pre_vars, post_vars)
    print(f"Alpha function result: {mos_result}")
    print(f"Concretization: {mos_result.concretize()}")

    print()


def run_all_examples():
    """Run all examples."""
    print("Elder Abstract Domains Implementation Examples")
    print("=" * 50)
    print()

    example_identity_relation()
    example_affine_transformation()
    example_ks_constraints()
    example_ag_generators()
    example_domain_equivalence()
    example_alpha_function()

    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()
