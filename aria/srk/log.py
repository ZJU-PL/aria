"""
Logging utilities for SRK.

This module provides logging functionality for debugging and monitoring
the symbolic reasoning operations.
"""

import logging
import sys
from typing import Optional, Dict, Any
from contextlib import contextmanager


class SRKLogger:
    """Custom logger for SRK operations."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(f"srk.{name}")
        self.logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)


# Global loggers for different SRK components
syntax_logger = SRKLogger("syntax")
polynomial_logger = SRKLogger("polynomial")
smt_logger = SRKLogger("smt")
abstract_logger = SRKLogger("abstract")
linear_logger = SRKLogger("linear")
interval_logger = SRKLogger("interval")
iteration_logger = SRKLogger("iteration")
quantifier_logger = SRKLogger("quantifier")
interpretation_logger = SRKLogger("interpretation")
transition_logger = SRKLogger("transition")
wedge_logger = SRKLogger("wedge")
termination_logger = SRKLogger("termination")


class LogContext:
    """Context manager for logging with additional context."""

    def __init__(self, logger: SRKLogger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
        self.old_context = {}

    def __enter__(self):
        # Store old context and set new context
        self.old_context = getattr(self.logger.logger, '_context', {})
        self.logger.logger._context = self.context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old context
        self.logger.logger._context = self.old_context


def log_function_call(func_name: str, logger: SRKLogger, args: tuple = (), kwargs: dict = None):
    """Log a function call."""
    kwargs = kwargs or {}
    logger.debug(f"Calling {func_name}", extra={
        'function': func_name,
        'args': args,
        'kwargs': kwargs
    })


def log_function_result(func_name: str, logger: SRKLogger, result: Any, error: Optional[Exception] = None):
    """Log a function result."""
    if error:
        logger.error(f"Function {func_name} failed", extra={
            'function': func_name,
            'error': str(error),
            'error_type': type(error).__name__
        })
    else:
        logger.debug(f"Function {func_name} completed", extra={
            'function': func_name,
            'result_type': type(result).__name__
        })


@contextmanager
def log_performance(logger: SRKLogger, operation: str):
    """Context manager for logging performance metrics."""
    import time

    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"Operation {operation} completed", extra={
            'operation': operation,
            'duration': duration
        })


def configure_logging(level: str = "INFO", format_string: Optional[str] = None):
    """Configure logging for all SRK loggers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Update all SRK loggers
    for logger_name in [
        'srk.syntax', 'srk.polynomial', 'srk.smt', 'srk.abstract',
        'srk.linear', 'srk.interval', 'srk.iteration', 'srk.quantifier',
        'srk.interpretation', 'srk.transition', 'srk.wedge', 'srk.termination'
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)

        # Update handlers
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
            if format_string and hasattr(handler, 'formatter'):
                formatter = logging.Formatter(format_string)
                handler.setFormatter(formatter)


def enable_debug_logging():
    """Enable debug logging for all SRK components."""
    configure_logging("DEBUG")


def disable_logging():
    """Disable logging for all SRK components."""
    configure_logging("CRITICAL")


# Performance monitoring
class PerformanceMonitor:
    """Monitor performance of SRK operations."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, float]] = {}

    def record_operation(self, operation: str, duration: float) -> None:
        """Record an operation's duration."""
        if operation not in self.metrics:
            self.metrics[operation] = {'count': 0, 'total_time': 0, 'avg_time': 0}

        metrics = self.metrics[operation]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['avg_time'] = metrics['total_time'] / metrics['count']

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        return self.metrics.copy()

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


# Global performance monitor
performance_monitor = PerformanceMonitor()


@contextmanager
def monitor_performance(operation: str):
    """Context manager for monitoring operation performance."""
    import time

    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        performance_monitor.record_operation(operation, duration)


# Utility functions for logging common operations
def log_expression_creation(expr_type: str, logger: SRKLogger):
    """Log expression creation."""
    logger.debug(f"Created {expr_type} expression")


def log_smt_query(query_type: str, formula_count: int, logger: SRKLogger):
    """Log SMT query."""
    logger.debug(f"SMT query: {query_type}", extra={
        'query_type': query_type,
        'formula_count': formula_count
    })


def log_polynomial_operation(op: str, poly1_size: int, poly2_size: int, result_size: int, logger: SRKLogger):
    """Log polynomial operation."""
    logger.debug(f"Polynomial {op}", extra={
        'operation': op,
        'input1_size': poly1_size,
        'input2_size': poly2_size,
        'result_size': result_size
    })


def log_cache_operation(cache_type: str, operation: str, key_size: int, logger: SRKLogger):
    """Log cache operation."""
    logger.debug(f"Cache {operation}", extra={
        'cache_type': cache_type,
        'operation': operation,
        'key_size': key_size
    })


# Convenience functions for getting loggers
def get_syntax_logger() -> SRKLogger:
    """Get the syntax logger."""
    return syntax_logger


def get_polynomial_logger() -> SRKLogger:
    """Get the polynomial logger."""
    return polynomial_logger


def get_smt_logger() -> SRKLogger:
    """Get the SMT logger."""
    return smt_logger


def get_abstract_logger() -> SRKLogger:
    """Get the abstract logger."""
    return abstract_logger


class Log:
    """Logging class that provides OCaml-style logging interface."""

    def __init__(self, name: str = "default"):
        self.logger = SRKLogger(name)

    def log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message with specified level."""
        if level.lower() == "debug":
            self.logger.debug(message, **kwargs)
        elif level.lower() == "info":
            self.logger.info(message, **kwargs)
        elif level.lower() == "warning" or level.lower() == "warn":
            self.logger.warning(message, **kwargs)
        elif level.lower() == "error":
            self.logger.error(message, **kwargs)
        elif level.lower() == "critical":
            self.logger.critical(message, **kwargs)
        else:
            self.logger.info(message, **kwargs)

    def logf(self, message: str, level: str = "info") -> None:
        """Log a formatted message (OCaml-style interface)."""
        logf(message, level)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)


# Default logger instance (similar to OCaml's included module)
default_logger = Log("default")


def logf(message: str, level: str = "info") -> None:
    """Log a formatted message."""
    logger = get_smt_logger()  # Use SMT logger as default

    if level.lower() == "debug":
        logger.debug(message)
    elif level.lower() == "info":
        logger.info(message)
    elif level.lower() == "warning" or level.lower() == "warn":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    elif level.lower() == "critical":
        logger.critical(message)
    else:
        logger.info(message)
