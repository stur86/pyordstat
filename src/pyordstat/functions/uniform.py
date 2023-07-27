"""Order statistics for uniform distributions."""
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pyordstat.continuous import CallableDistrFunc, ContinuousOrderStatistics


def _uniform_pdf(x: NDArray[np.number], a: float, b: float) -> NDArray[np.number]:
    ab = b - a
    return np.where((x >= a) & (x <= b), 1 / ab, 0)


def _uniform_cdf(x: NDArray[np.number], a: float, b: float) -> NDArray[np.number]:
    ab = b - a
    ans = np.zeros_like(x)
    ans[x >= b] = 1.0
    domain = (x >= a) & (x <= b)
    ans[domain] = (x[domain] - a) / ab
    return ans


class UniformOrderStatistics(ContinuousOrderStatistics):
    """Uniform order statistics distribution."""

    def __init__(self, a: float, b: float) -> None:
        """Create a uniform order statistics distribution.

        Args:
            a (float): Lower bound of the uniform distribution.
            b (float): Upper bound of the uniform distribution.
        """
        pdf = cast(CallableDistrFunc, _uniform_pdf)
        cdf = cast(CallableDistrFunc, _uniform_cdf)
        super().__init__(pdf, cdf, a=a, b=b)
