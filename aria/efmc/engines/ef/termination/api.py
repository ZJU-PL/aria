"""
High-level API for termination analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import z3

from aria.efmc.sts import TransitionSystem
from .termination_prover import TerminationProver

logger = logging.getLogger(__name__)


def prove_termination_with_ranking_functions(
    sts: TransitionSystem,
    template_names: List[str] = None,
    timeout: Optional[int] = None,
    **kwargs,
) -> Tuple[bool, Optional[z3.ExprRef], str]:
    """Try to prove termination using various ranking function templates."""
    if template_names is None:
        template_names = [
            "bv_linear_ranking",
            "bv_lexicographic_ranking",
            "bv_conditional_ranking",
        ]

    for template_name in template_names:
        logger.info("Trying ranking template: %s", template_name)

        try:
            prover = TerminationProver(sts, **kwargs)
            prover.set_ranking_template(template_name, **kwargs)
            result = prover.prove_termination(timeout=timeout)

            if result.result:
                logger.info("Termination proven with template: %s", template_name)
                # pylint: disable=protected-access
                ranking_function = prover._extract_ranking_function()
                return True, ranking_function, template_name

        except Exception as e:
            logger.warning("Template %s failed: %s", template_name, e)
            continue

    logger.info("Could not prove termination with any template")
    return False, None, ""


def prove_non_termination_with_recurrence_sets(
    sts: TransitionSystem,
    template_names: List[str] = None,
    timeout: Optional[int] = None,
    **kwargs,
) -> Tuple[bool, Optional[z3.ExprRef], str]:
    """Try to prove non-termination using various recurrence set templates."""
    if template_names is None:
        template_names = [
            "bv_linear_recurrence",
            "bv_interval_recurrence",
            "bv_disjunctive_recurrence",
        ]

    for template_name in template_names:
        logger.info("Trying recurrence template: %s", template_name)

        try:
            prover = TerminationProver(sts, **kwargs)
            prover.set_recurrence_template(template_name, **kwargs)
            result = prover.prove_non_termination(timeout=timeout)

            if result.result:
                logger.info("Non-termination proven with template: %s", template_name)
                # pylint: disable=protected-access
                recurrence_set = prover._extract_recurrence_set()
                return True, recurrence_set, template_name

        except Exception as e:
            logger.warning("Template %s failed: %s", template_name, e)
            continue

    logger.info("Could not prove non-termination with any template")
    return False, None, ""


def analyze_termination(
    sts: TransitionSystem,
    ranking_templates: List[str] = None,
    recurrence_templates: List[str] = None,
    timeout: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Comprehensive termination analysis using ranking functions and recurrence sets."""
    results = {
        "termination_proven": False,
        "non_termination_proven": False,
        "ranking_function": None,
        "recurrence_set": None,
        "ranking_template_used": None,
        "recurrence_template_used": None,
        "errors": [],
    }

    # Try to prove termination first
    try:
        term_success, ranking_func, ranking_template = (
            prove_termination_with_ranking_functions(
                sts, ranking_templates, timeout, **kwargs
            )
        )

        if term_success:
            results.update(
                {
                    "termination_proven": True,
                    "ranking_function": ranking_func,
                    "ranking_template_used": ranking_template,
                }
            )
            logger.info("Termination proven - skipping non-termination analysis")
            return results

    except Exception as e:
        error_msg = f"Error in termination proving: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    # If termination couldn't be proven, try to prove non-termination
    try:
        non_term_success, recurrence_set, recurrence_template = (
            prove_non_termination_with_recurrence_sets(
                sts, recurrence_templates, timeout, **kwargs
            )
        )

        if non_term_success:
            results.update(
                {
                    "non_termination_proven": True,
                    "recurrence_set": recurrence_set,
                    "recurrence_template_used": recurrence_template,
                }
            )

    except Exception as e:
        error_msg = f"Error in non-termination proving: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    return results
