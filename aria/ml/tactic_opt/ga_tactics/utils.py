#!/usr/bin/env python3
"""Utility functions for tactic sequence I/O."""

import json
from .models import CustomJsonEncoder, TacticSeq


def load_tactic_sequence(filename):
    """Load a tactic sequence from JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    tactics_list = [t.get("name") for t in data.get("tactics", []) if t.get("name")]
    tactic_seq = TacticSeq(tactics_list)
    tactic_seq.fitness = data.get("fitness", 0.0)
    return tactic_seq


def save_tactic_sequence(tactic_seq, filename):
    """Save a tactic sequence to JSON file."""
    with open(filename, "w") as f:
        json.dump(tactic_seq, f, cls=CustomJsonEncoder, indent=2)
