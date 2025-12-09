"""
Base classes for AllSMT solvers.

This module provides the abstract base class for all AllSMT solver implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Union, TypeVar, Generic

# Type variable for model types
ModelType = TypeVar('ModelType')


class AllSMTSolver(ABC, Generic[ModelType]):
    """
    Abstract base class for AllSMT solvers.

    This class defines the interface that all AllSMT solver implementations must follow.
    """

    @abstractmethod
    def solve(self, expr: Any, keys: List[Any], model_limit: int = 100) -> List[ModelType]:
        """
        Enumerate all satisfying models for the given expression over the specified keys.

        Args:
            expr: The expression/formula to solve
            keys: The variables to track in the models
            model_limit: Maximum number of models to generate (default: 100)

        Returns:
            List of models satisfying the expression
        """
        pass

    @abstractmethod
    def get_model_count(self) -> int:
        """
        Get the number of models found in the last solve call.

        Returns:
            int: The number of models
        """
        pass

    @property
    @abstractmethod
    def models(self) -> List[ModelType]:
        """
        Get all models found in the last solve call.

        Returns:
            List of models
        """
        pass

    @abstractmethod
    def print_models(self, verbose: bool = False) -> None:
        """
        Print all models found in the last solve call.

        Args:
            verbose: Whether to print detailed information about each model
        """
        pass
