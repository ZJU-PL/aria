"""Utility functions for SMTO implementation.

Enhanced to support bit-vectors, floating points, and arrays.
"""

import hashlib
import json
import os
import time
from typing import Dict, Any, Optional, List

import z3


def z3_value_to_python(z3_val) -> Any:
    """Convert a Z3 value to its corresponding Python value."""
    if z3.is_int_value(z3_val):
        return z3_val.as_long()
    if z3.is_real_value(z3_val):
        return float(z3_val.as_fraction())
    if z3.is_bool_value(z3_val):
        return z3.is_true(z3_val)
    if z3.is_string_value(z3_val):
        return z3_val.as_string()
    if z3.is_bv_value(z3_val):
        # For bit-vectors, return as integer for simplicity
        # Could also return as binary string if needed
        return z3_val.as_long()
    if z3.is_fp_value(z3_val):
        # For floating points, return as float
        # Note: This is a simplified conversion - more sophisticated
        # handling might be needed for special values (NaN, inf, etc.)
        try:
            return float(z3_val.as_fraction())
        except Exception:
            return str(z3_val)
    if z3.is_array_value(z3_val):
        # For arrays, this is complex as we need to handle the array structure
        # For now, return a simplified representation
        # A full implementation would need to handle array indexing and values
        return f"Array({z3_val})"
    return str(z3_val)


def python_to_z3_value(py_val, sort: z3.SortRef):
    """Convert a Python value to a Z3 value of the specified sort."""
    if sort == z3.IntSort():
        return z3.IntVal(py_val)
    if sort == z3.RealSort():
        return z3.RealVal(py_val)
    if sort == z3.BoolSort():
        return z3.BoolVal(py_val)
    if sort == z3.StringSort():
        return z3.StringVal(py_val)
    if z3.is_bv_sort(sort):
        # For bit-vectors, we need to know the bitwidth
        bitwidth = sort.size()
        if isinstance(py_val, int):
            return z3.BitVecVal(py_val, bitwidth)
        if isinstance(py_val, str) and py_val.startswith('#b'):
            # Handle binary string representation
            return z3.BitVecVal(py_val, bitwidth)
        # Default to integer interpretation
        return z3.BitVecVal(int(py_val), bitwidth)
    if z3.is_fp_sort(sort):
        # For floating points, we need to know the exponent and significand bits
        if isinstance(py_val, float):
            return z3.FPVal(py_val, sort)
        if isinstance(py_val, str):
            # Try to parse as float string
            try:
                return z3.FPVal(float(py_val), sort)
            except Exception:
                return z3.FPVal(py_val, sort)
        return z3.FPVal(float(py_val), sort)
    if z3.is_array_sort(sort):
        # For arrays, this is more complex - we need domain and range sorts
        # For now, return a placeholder - full implementation would need
        # to handle array construction based on py_val structure
        domain_sort = sort.domain()
        range_sort = sort.range()
        # This is a simplified implementation - a full version would need
        # to handle the array structure in py_val
        return z3.K(domain_sort, python_to_z3_value(py_val, range_sort))
    raise ValueError(f"Unsupported sort: {sort}")


def values_equal(val1, val2) -> bool:
    """Check if two values are equal, handling Z3 values."""
    if z3.is_expr(val1) and z3.is_expr(val2):
        return z3.eq(val1, val2)
    # Handle special cases for bit-vectors, floating points, and arrays
    if z3.is_bv_value(val1) and z3.is_bv_value(val2):
        return val1.as_long() == val2.as_long()
    if z3.is_fp_value(val1) and z3.is_fp_value(val2):
        # For floating points, use fpEQ for proper NaN handling
        return z3.fpEQ(val1, val2)
    if z3.is_array_value(val1) and z3.is_array_value(val2):
        # For arrays, this would need more sophisticated comparison
        # For now, use string comparison as a fallback
        return str(val1) == str(val2)
    return val1 == val2


def generate_cache_key(oracle_name: str, inputs: Dict) -> str:
    """Generate a stable hash cache key for oracle inputs."""
    key_data = f"{oracle_name}_{json.dumps(inputs, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()


class OracleCache:
    """Cache for oracle query results."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache; optionally persistent if cache_dir is provided."""
        self.cache_dir = cache_dir
        self.cache: Dict[str, Any] = {}

        if cache_dir and os.path.exists(cache_dir):
            self._load_cache()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        return self.cache.get(key)

    def put(self, key: str, value: Any):
        """Store a value in the cache and persist if configured."""
        self.cache[key] = value
        self._save_cache()

    def contains(self, key: str) -> bool:
        """Return True if a key is in the cache."""
        return key in self.cache

    def _save_cache(self):
        """Save cache to disk if persistence is enabled."""
        if not self.cache_dir:
            return

        os.makedirs(self.cache_dir, exist_ok=True)

        serializable_cache = {}
        for key, value in self.cache.items():
            if isinstance(value, (int, float, bool, str, type(None))):
                serializable_cache[key] = value
            else:
                serializable_cache[key] = str(value)

        cache_file = os.path.join(self.cache_dir, "oracle_cache.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(serializable_cache, f)

    def _load_cache(self):
        """Load cache from disk."""
        cache_file = os.path.join(self.cache_dir, "oracle_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)


class ExplanationLogger:
    """Logger for SMTO solver explanations."""

    def __init__(self, level: str = "basic"):
        """Initialize with explanation level: 'none' | 'basic' | 'detailed'."""
        self.level = level
        self.history: List[Dict[str, Any]] = []

    def log(self, message: str, level: str = "basic"):
        """Log message if allowed by configured level."""
        if self.level == "none":
            return

        if level == "detailed" and self.level != "detailed":
            return

        self.history.append({
            "timestamp": time.time(),
            "message": message,
            "level": level
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the explanation history entries."""
        return self.history

    def clear(self):
        """Clear the explanation history."""
        self.history = []


def parse_text_by_sort(text: str, sort: z3.SortRef) -> Any:
    """Parse plain text into a Python value according to a Z3 sort."""
    text = text.strip()
    if sort == z3.BoolSort():
        return text.lower() in ["true", "1", "yes"]
    if sort == z3.IntSort():
        return int(text)
    if sort == z3.RealSort():
        return float(text)
    if sort == z3.StringSort():
        if text.startswith('"') and text.endswith('"'):
            return text[1:-1]
        return text
    if z3.is_bv_sort(sort):
        # Handle bit-vector text representations
        if text.startswith('#b'):
            # Binary representation like #b1010
            return int(text[2:], 2)
        if text.startswith('#x'):
            # Hexadecimal representation like #xA
            return int(text[2:], 16)
        # Try to parse as regular integer
        try:
            return int(text)
        except Exception:
            return text
    if z3.is_fp_sort(sort):
        # Handle floating point text representations
        try:
            return float(text)
        except Exception:
            return text
    if z3.is_array_sort(sort):
        # For arrays, this is complex - for now return the text as-is
        # A full implementation would need to parse array syntax
        return text
    return text
