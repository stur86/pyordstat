"""Order statistics for discrete distributions of known PMF and CDF."""
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from pyordstat.base import BaseOrderStatistics, CallableDistrFunc
from pyordstat.core import ordstat_cdf


class DiscreteOrderStatistics(BaseOrderStatistics):
    """Discrete order statistics."""

    _pdf: CallableDistrFunc
    _cdf: CallableDistrFunc

    def __init__(
        self, pmf: CallableDistrFunc, cdf: CallableDistrFunc, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize discrete order statistics.

        Args:
            pmf (CallableDistrFunc): Probability mass function of the distribution.
            cdf (CallableDistrFunc): Cumulative distribution function of the distribution.
            *args (Any): Additional arguments to be passed to the PMF and CDF.
            **kwargs (Any): Additional keyword arguments to be passed to the PMF and CDF.
        """

        def bundled_pmf(x: NDArray[np.number]) -> NDArray[np.number]:
            return pmf(x, *args, **kwargs)

        def bundled_cdf(x: NDArray[np.number]) -> NDArray[np.number]:
            return cdf(x, *args, **kwargs)

        super().__init__(
            cast(CallableDistrFunc, bundled_pmf),
            cast(CallableDistrFunc, bundled_cdf),
        )

    @property
    def pdf(self) -> CallableDistrFunc:
        """Probability density function of order statistics."""
        raise NotImplementedError("PDF not implemented for discrete order statistics.")

    @property
    def pmf(self) -> CallableDistrFunc:
        """Probability mass function of order statistics."""
        return self._pdf

    def order_statistic_pmf(self, x: NDArray[np.number], n: int, k: int) -> NDArray[np.number]:
        """Probability mass function of order statistics.

        Args:
            x (NDArray[np.number]): Values to evaluate the PMF at.
            n (int): Sample size.
            k (int): Order statistic.

        Returns
        -------
            NDArray[np.number]: PMF values.
        """
        cdf_1 = self.order_statistic_cdf(x, n, k)
        cdf_0 = self.order_statistic_cdf(x - 1, n, k)

        return cdf_1 - cdf_0

    def order_statistic_cdf(self, x: NDArray[np.number], n: int, k: int) -> NDArray[np.number]:
        """Cumulative distribution function of order statistics.

        Args:
            x (NDArray[np.number]): Values to evaluate the CDF at.
            n (int): Sample size.
            k (int): Order statistic.

        Returns
        -------
            NDArray[np.number]: CDF values.
        """
        # Guarantee that x are integers
        x = np.asarray(x, dtype=int)
        cdf = self._cdf(x)
        return ordstat_cdf(cdf, n, k)
