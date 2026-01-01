"""
Base classes for AllSMT solvers.

This module provides the abstract base class for all AllSMT solver implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Any, TypeVar, Generic

# Type variable for model types
ModelTypeVar = TypeVar("ModelTypeVar")


class AllSMTSolver(ABC, Generic[ModelTypeVar]):
    """
    Abstract base class for AllSMT solvers.

    This class defines the interface that all AllSMT solver implementations must follow.
    """

    def __init__(self) -> None:
        """Initialize the AllSMT solver with common state."""
        self._models: List[ModelTypeVar] = []
        self._model_count: int = 0
        self._model_limit_reached: bool = False

    @abstractmethod
    def solve(
        self, expr: Any, keys: List[Any], model_limit: int = 100
    ) -> List[ModelTypeVar]:
        """
        Enumerate all satisfying models for the given expression over the specified keys.

        Args:
            expr: The expression/formula to solve
            keys: The variables to track in the models
            model_limit: Maximum number of models to generate (default: 100)

        Returns:
            List of models satisfying the expression
        """

    def _reset_model_storage(self) -> None:
        """Reset model storage before solving."""
        self._models = []
        self._model_count = 0
        self._model_limit_reached = False

    def _add_model(self, model: ModelTypeVar, model_limit: int) -> bool:
        """
        Add a model to the storage and check if limit is reached.

        Args:
            model: The model to add
            model_limit: Maximum number of models to generate

        Returns:
            True if model limit has been reached, False otherwise
        """
        self._model_count += 1
        self._models.append(model)

        # Check if we've reached the model limit
        if self._model_count >= model_limit:
            self._model_limit_reached = True
            return True
        return False

    def get_model_count(self) -> int:
        """
        Get the number of models found in the last solve call.

        Returns:
            int: The number of models
        """
        return self._model_count

    @property
    def models(self) -> List[ModelTypeVar]:
        """
        Get all models found in the last solve call.

        Returns:
            List of models
        """
        return self._models

    @abstractmethod
    def _format_model_verbose(self, model: ModelTypeVar) -> None:
        """
        Print detailed information about a single model.

        Args:
            model: The model to print
        """

    def print_models(self, verbose: bool = False) -> None:
        """
        Print all models found in the last solve call.

        Args:
            verbose: Whether to print detailed information about each model
        """
        if not self._models:
            print("No models found.")
            return

        for i, model in enumerate(self._models):
            if verbose:
                print(f"Model {i + 1}:")
                self._format_model_verbose(model)
            else:
                print(f"Model {i + 1}: {model}")

        if self._model_limit_reached:
            print(
                f"Model limit reached. Found {self._model_count} models "
                f"(there may be more)."
            )
        else:
            print(f"Total number of models: {self._model_count}")
