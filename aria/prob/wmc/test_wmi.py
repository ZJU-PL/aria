import pytest
import z3

from aria.prob import UniformDensity, WMIOptions, wmi_integrate


def test_wmi_result_is_float_like():
    x = z3.Real("x")
    density = UniformDensity({"x": (0, 1)})
    result = wmi_integrate(
        z3.And(x >= 0, x <= 0.5),
        density,
        WMIOptions(method="region", num_samples=6000, random_seed=3),
    )

    assert float(result) == pytest.approx(0.5, abs=0.05)
    assert "{:.2f}".format(result) != ""
