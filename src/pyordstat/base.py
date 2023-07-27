"""Base class for order statistics distributions."""
from abc import ABC
from typing import Any, Protocol, Union

import numpy as np
from numpy.typing import NDArray


class CallableDistrFunc(Protocol):
    """Callable distribution function protocol."""

    def __call__(self, x: NDArray[np.number], *args: Any, **kwds: Any) -> NDArray[np.number]:
        """Call distribution function."""


StatDistrFunc = Union[CallableDistrFunc, NDArray[np.number]]


class BaseOrderStatistics(ABC):
    """Base class for order statistics distributions."""

    _pdf: StatDistrFunc
    _cdf: StatDistrFunc

    def __init__(
        self,
        pdf: StatDistrFunc,
        cdf: StatDistrFunc,
    ) -> None:
        """Initialise base order statistics distribution.

        Args:
            pdf (StatDistrFunc): Probability density function.
            cdf (StatDistrFunc): Cumulative distribution function.
        """
        self._pdf = pdf
        self._cdf = cdf

    @property
    def pdf(self) -> StatDistrFunc:
        """Probability density function."""
        return self._pdf

    @property
    def cdf(self) -> StatDistrFunc:
        """Cumulative distribution function."""
        return self._cdf
