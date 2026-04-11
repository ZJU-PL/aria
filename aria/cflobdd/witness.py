"""Witness helpers for reachability relations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


def extract_edge_path(witness: Optional[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    if not witness:
        return []

    kind = witness.get("kind")
    if kind == "edge":
        edge = witness.get("edge")
        return [tuple(edge)] if edge is not None else []

    if kind in {"compose", "join", "quantified_compose"}:
        left = extract_edge_path(witness.get("left"))
        right = extract_edge_path(witness.get("right"))
        if left or right:
            return left + right
        return extract_edge_path(witness.get("child"))

    child = witness.get("child")
    if child is not None:
        return extract_edge_path(child)

    children = witness.get("children")
    if children is not None:
        path: List[Tuple[int, int]] = []
        for child_witness in children:
            path.extend(extract_edge_path(child_witness))
        return path

    return []


def build_witness_tree(witness: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not witness:
        return {"kind": "empty"}

    kind = str(witness.get("kind", "unknown"))
    node: Dict[str, Any] = {
        "kind": kind,
        "details": {
            key: value
            for key, value in witness.items()
            if key not in {"child", "children", "left", "right"}
        },
        "children": [],
    }

    for key in ("child", "left", "right"):
        nested = witness.get(key)
        if nested is not None:
            node["children"].append(build_witness_tree(nested))

    children = witness.get("children")
    if children is not None:
        for nested in children:
            node["children"].append(build_witness_tree(nested))

    return node


def format_witness(witness: Optional[Mapping[str, Any]], indent: int = 0) -> str:
    if not witness:
        return "<no witness>"

    prefix = " " * indent
    kind = witness.get("kind", "unknown")
    details: Dict[str, Any] = {
        key: value
        for key, value in witness.items()
        if key not in {"child", "children", "left", "right"}
    }
    lines = [f"{prefix}{kind}: {details}"]

    for key in ("child", "left", "right"):
        nested = witness.get(key)
        if nested is not None:
            lines.append(format_witness(nested, indent + 2))

    children = witness.get("children")
    if children is not None:
        for nested in children:
            lines.append(format_witness(nested, indent + 2))

    return "\n".join(lines)
