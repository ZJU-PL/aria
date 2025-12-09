"""
Base classes for samplers.

This module provides the abstract base classes for all sampler implementations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Set
import z3


class Logic(Enum):
    """Supported SMT logics for sampling."""
    QF_BOOL = "QF_BOOL"  # Quantifier-free Boolean logic
    QF_BV = "QF_BV"  # Quantifier-free bit-vector logic
    QF_LRA = "QF_LRA"  # Quantifier-free linear real arithmetic
    QF_LIA = "QF_LIA"  # Quantifier-free linear integer arithmetic
    QF_NRA = "QF_NRA"  # Quantifier-free non-linear real arithmetic
    QF_NIA = "QF_NIA"  # Quantifier-free non-linear integer arithmetic
    QF_LIRA = "QF_LIRA"  # Quantifier-free linear integer and real arithmetic
    QF_ALL = "QF_ALL"  # All supported logics


class SamplingMethod(Enum):
    """Available sampling methods."""
    ENUMERATION = "enumeration"  # Simple enumeration of models
    MCMC = "mcmc"  # Markov Chain Monte Carlo
    REGION = "region"  # Region-based sampling
    SEARCH_TREE = "search_tree"  # Search tree-based sampling
    HASH_BASED = "hash_based"  # Hash-based sampling
    DIKIN_WALK = "dikin_walk"  # Dikin walk for continuous domains


class SamplingResult:
    """Result of a sampling operation."""

    def __init__(self,
                 samples: List[Dict[str, Any]],
                 stats: Optional[Dict[str, Any]] = None):
        """
        Initialize a sampling result.
        
        Args:
            samples: List of samples, where each sample is a dictionary mapping variable names to values
            stats: Optional statistics about the sampling process
        """
        self.samples = samples
        self.stats = stats or {}
        self.success = len(samples) > 0

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, Any]:
        """Get a specific sample by index."""
        return self.samples[index]

    def __iter__(self):
        """Iterate over the samples."""
        return iter(self.samples)

    def __str__(self) -> str:
        """String representation of the sampling result."""
        if not self.success:
            return "SamplingResult(success=False, samples=[])"

        sample_str = f"{len(self.samples)} samples"
        if len(self.samples) > 0:
            sample_str += f", first sample: {self.samples[0]}"

        return f"SamplingResult(success={self.success}, {sample_str})"


class SamplingOptions:
    """Options for sampling."""

    def __init__(self,
                 method: SamplingMethod = SamplingMethod.ENUMERATION,
                 num_samples: int = 1,
                 timeout: Optional[float] = None,
                 random_seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize sampling options.
        
        Args:
            method: The sampling method to use
            num_samples: The number of samples to generate
            timeout: Optional timeout in seconds
            random_seed: Optional random seed for reproducibility
            **kwargs: Additional method-specific options
        """
        self.method = method
        self.num_samples = num_samples
        self.timeout = timeout
        self.random_seed = random_seed
        self.additional_options = kwargs

    def __str__(self) -> str:
        """String representation of the sampling options."""
        return (f"SamplingOptions(method={self.method.value}, "
                f"num_samples={self.num_samples}, "
                f"timeout={self.timeout}, "
                f"random_seed={self.random_seed}, "
                f"additional_options={self.additional_options})")


class Sampler(ABC):
    """Abstract base class for all samplers."""

    @abstractmethod
    def supports_logic(self, logic: Logic) -> bool:
        """
        Check if this sampler supports the given logic.
        
        Args:
            logic: The logic to check
            
        Returns:
            True if the sampler supports the logic, False otherwise
        """
        pass

    @abstractmethod
    def init_from_formula(self, formula: z3.ExprRef) -> None:
        """
        Initialize the sampler with a formula.
        
        Args:
            formula: The Z3 formula to sample from
        """
        pass

    @abstractmethod
    def sample(self, options: SamplingOptions) -> SamplingResult:
        """
        Generate samples according to the given options.
        
        Args:
            options: The sampling options
            
        Returns:
            A SamplingResult containing the generated samples
        """
        pass

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """
        Get the sampling methods supported by this sampler.
        
        Returns:
            A set of supported sampling methods
        """
        return {SamplingMethod.ENUMERATION}  # Default implementation

    def get_supported_logics(self) -> Set[Logic]:
        """
        Get the logics supported by this sampler.
        
        Returns:
            A set of supported logics
        """
        pass
