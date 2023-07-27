"""Order statistics for discrete distributions with finite support."""
import numpy as np
from numpy.typing import NDArray

from pyordstat.base import BaseOrderStatistics
from pyordstat.core import ordstat_cdf


class FiniteOrderStatistics(BaseOrderStatistics):
    """Order statistics distribution for discrete distributions with finite support."""

    _x: NDArray[np.number]
    _pdf: NDArray[np.number]
    _cdf: NDArray[np.number]

    def __init__(self, x: NDArray[np.number], pdf: NDArray[np.number]) -> None:
        """Create a discrete order statistics distribution.

        Requires the values of the random variable and the probability density function.
        The x values can be in any order, but the pdf values must be in the same order
        as the x values, and they will be sorted in ascending order.

        Args:
            x (NDArray[np.number]): Support of the distribution.
            pdf (NDArray[np.number]): Probability density function of the distribution.
        """
        isort = np.argsort(x)
        self._x = np.asarray(x)[isort]
        pdf = np.asarray(pdf)[isort]
        cdf = np.cumsum(pdf)

        # Normalize
        pdf = pdf / cdf[-1]
        cdf = cdf / cdf[-1]

        super().__init__(pdf, cdf)

    @property
    def x(self) -> NDArray[np.number]:
        """Support of the distribution."""
        return self._x

    @property
    def pdf(self) -> NDArray[np.number]:
        """Probability density function."""
        return self._pdf

    @property
    def cdf(self) -> NDArray[np.number]:
        """Cumulative distribution function."""
        return self._cdf

    def order_statistic_pdf(self, n: int, k: int) -> NDArray[np.number]:
        """Order statistic probability density function.

        Args:
            n (int): Sample size.
            k (int): Order statistic to calculate.

        Returns
        -------
            NDArray[np.number]: Probability density function of the k-th order statistic.
        """
        cdf = self.order_statistic_cdf(n, k)
        return np.diff(cdf, prepend=0)

    def order_statistic_cdf(self, n: int, k: int) -> NDArray[np.number]:
        """Order statistic cumulative distribution function.

        Args:
            n (int): Sample size.
            k (int): Order statistic to calculate.

        Returns
        -------
            NDArray[np.number]: Cumulative distribution function of the k-th order statistic.
        """
        return ordstat_cdf(self._cdf, n, k)
