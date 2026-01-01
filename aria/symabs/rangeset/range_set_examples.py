from __future__ import annotations

import z3

from aria.symabs.rangeset.range_set_abstraction import (
    minimum,
    maximum,
    range_abstraction,
    set_abstraction,
)


def demo_unsigned():
    x = z3.BitVec("x", 8)
    fml = z3.Or(
        x == 2,
        x == 3,
        x == 5,
        x == 6,
        x == 8,
        x == 9,
        x == 12,
        x == 13,
        x == 15,
    )

    print("min:", minimum(fml, x, signed=False))
    print("max:", maximum(fml, x, signed=False))
    print("range:", range_abstraction(fml, x, signed=False))
    print("set (8 steps):", set_abstraction(fml, x, signed=False, steps=8))


if __name__ == "__main__":
    demo_unsigned()
