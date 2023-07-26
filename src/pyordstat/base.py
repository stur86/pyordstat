"""Base class for order statistics distributions."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Tuple, Union

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
    _args: Tuple[Any, ...]
    _kwargs: Dict[str, Any]

    def __init__(self, pdf: StatDistrFunc, cdf: StatDistrFunc,
                 *args: Tuple[Any,...], **kwargs: Dict[str, Any]) -> None:
        """Initialise base order statistics distribution.

        Args:
            pdf (StatDistrFunc): Probability density function.
            cdf (StatDistrFunc): Cumulative distribution function.
            *args (Tuple[Any,...]): Additional arguments to pass to the distribution functions.
            **kwargs (Dict[str, Any]): Additional keyword arguments to pass to the distribution functions.
        """
        self._pdf = pdf
        self._cdf = cdf
        self._args = args
        self._kwargs = kwargs

    @property
    def pdf(self) -> StatDistrFunc:
        """Probability density function."""
        return self._pdf

    @property
    def cdf(self) -> StatDistrFunc:
        """Cumulative distribution function."""
        return self._cdf

    def _get_pdf(self, x: NDArray[np.number]) -> NDArray[np.number]:
        return self._pdf(x, *self._args, **self._kwargs)

    def _get_cdf(self, x: NDArray[np.number]) -> NDArray[np.number]:
        return self._cdf(x, *self._args, **self._kwargs)

    @abstractmethod
    def order_statistic(self, x: NDArray[np.number], n: int, k: int) -> NDArray[np.number]:
        """Order statistics.

        Args:
            x (NDArray[np.number]): Random variable.
            n (int): Sample size.
            k (int): Order statistic.

        Returns
        -------
            NDArray[np.number]: Order statistic.
        """
