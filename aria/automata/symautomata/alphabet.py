"""This module configures that alphabet."""

import os.path
from typing import List, Optional


def _load_alphabet(filename: str) -> List[str]:
    """
    Load a file containing the characters of the alphabet.
    Every unique character contained in this file will be used as a symbol
    in the alphabet.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return list(set(f.read()))


def createalphabet(alphabetinput: Optional[str] = None) -> List[str]:
    """
    Creates a sample alphabet containing printable ASCII characters
    """
    if alphabetinput and os.path.isfile(alphabetinput):
        return _load_alphabet(alphabetinput)
    if alphabetinput:
        alpha = []
        setlist = alphabetinput.split(",")
        for alphaset in setlist:
            a = int(alphaset.split("-")[0])
            b = int(alphaset.split("-")[1])
            for i in range(a, b):
                alpha.append(chr(i))
        return alpha
    alpha: List[str] = []
    for i in range(32, 127):
        alpha.append(chr(i))
    return alpha
