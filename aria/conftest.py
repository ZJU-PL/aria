"""Pytest collection guards for optional dependency stacks."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


SKIP_FILES = set()

if not _has_module("hypothesis"):
    SKIP_FILES.update(
        [
            "bool/nnf/tests/test_nnf.py",
        ]
    )

if not _has_module("pysmt"):
    SKIP_FILES.update(
        [
            "monabs/tests/test_pysmt_monabs.py",
            "pyomt/tests/test_bvopt_iter_search.py",
            "tests/test_bool_counting.py",
            "tests/test_cli_pyomt.py",
        ]
    )

if not _has_module("multipledispatch"):
    SKIP_FILES.update(
        [
            "unification/tests/test_benchmarks.py",
            "unification/tests/test_core.py",
            "unification/tests/test_match.py",
            "unification/tests/test_more.py",
            "unification/tests/test_utils.py",
            "unification/tests/test_variable.py",
        ]
    )

if not _has_module("httpx"):
    SKIP_FILES.update(
        [
            "tests/test_llmtools.py",
            "ml/llm/smto/examples/test_parser.py",
            "ml/llm/smto/examples/test_solver_integration.py",
        ]
    )

if not _has_module("numpy"):
    SKIP_FILES.update(
        [
            "symabs/affine_relation/Elder/test_elder.py",
            "symabs/affine_relation/Elder/test_elder2.py",
            "symabs/affine_relation/Elder/test_elder3.py",
        ]
    )

if not _has_module("psutil"):
    SKIP_FILES.add("smt/pcdclt/tests/test_process_cleanup.py")




def pytest_ignore_collect(collection_path: Path, config) -> bool:
    del config
    base = Path(__file__).parent
    try:
        rel = str(collection_path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return False
    return rel in SKIP_FILES
