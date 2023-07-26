"""Continuous order statistics distributions."""
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from pyordstat.base import BaseOrderStatistics, CallableDistrFunc
from pyordstat.core import ordstat_cdf, ordstat_pdf


class ContinuousOrderStatistics(BaseOrderStatistics):
    """Continuous order statistics distribution."""

    _pdf: CallableDistrFunc
    _cdf: CallableDistrFunc

    def __init__(
        self,
        pdf: CallableDistrFunc,
        cdf: CallableDistrFunc,
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> None:
        """Create a continuous order statistics distribution.

        Requires a probability density function and a cumulative distribution function,
        plus any additional arguments to pass to the distribution functions.

        Args:
            pdf (CallableDistrFunc): _Probability density function._
            cdf (CallableDistrFunc): _Cumulative distribution function._
            *args (Tuple[Any, ...]): _Additional arguments to pass to the distribution functions._
            **kwargs (Dict[str, Any]): Additional keyword arguments to pass to the distribution functions.
        """
        super().__init__(pdf, cdf, *args, **kwargs)

    @property
    def pdf(self) -> CallableDistrFunc:
        """Probability density function."""
        return self._pdf

    @property
    def cdf(self) -> CallableDistrFunc:
        """Cumulative distribution function."""
        return self._cdf

    def order_statistic_pdf(self, x: NDArray[np.number], n: int, k: int) -> NDArray[np.number]:
        """Calculate the k-th order statistic PDF of a sample of size n.

        Return the k-th order statistic probability density function of a sample of size n
        from a continuous distribution, given the values in x.

        Args:
            x (NDArray[np.number]): Values of the random variable to calculate the order statistic for.
            n (int): Number of samples.
            k (int): Order statistic to calculate.

        Raises
        ------
            ValueError: If k is not between 1 and n.

        Returns
        -------
            NDArray[np.number]: Order statistic PDF.
        """
        if (k <= 0) or (k > n):
            raise ValueError("k must be between 1 and n.")

        pdf_vals = self.pdf(x, *self._args, **self._kwargs)
        cdf_vals = self.cdf(x, *self._args, **self._kwargs)

        return ordstat_pdf(pdf_vals, cdf_vals, n, k)

    def order_statistic_cdf(self, x: NDArray[np.number], n: int, k: int) -> NDArray[np.number]:
        """Calculate the k-th order statistic PDF of a sample of size n.

        Return the k-th order statistic probability density function of a sample of size n
        from a continuous distribution, given the values in x.

        Args:
            x (NDArray[np.number]): Values of the random variable to calculate the order statistic for.
            n (int): Number of samples.
            k (int): Order statistic to calculate.

        Raises
        ------
            ValueError: If k is not between 1 and n.

        Returns
        -------
            NDArray[np.number]: Order statistic CDF.
        """
        if (k <= 0) or (k > n):
            raise ValueError("k must be between 1 and n.")

        cdf_vals = self.cdf(x, *self._args, **self._kwargs)

        return ordstat_cdf(cdf_vals, n, k)
