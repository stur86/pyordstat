"""Normal order statistics distribution."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from pyordstat.continuous import CallableDistrFunc, ContinuousOrderStatistics


def _normal_pdf(x: NDArray[np.number], mu: float, sigma: float) -> NDArray[np.number]:
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def _normal_cdf(x: NDArray[np.number], mu: float, sigma: float) -> NDArray[np.number]:
    return (1 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2


class NormalOrderStatistics(ContinuousOrderStatistics):
    """Normal order statistics distribution."""

    def __init__(self, mu: float, sigma: float):
        """Create a normal order statistics distribution.

        Args:
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution.
        """
        pdf = cast(CallableDistrFunc, _normal_pdf)
        cdf = cast(CallableDistrFunc, _normal_cdf)
        super().__init__(pdf, cdf, mu=mu, sigma=sigma)
