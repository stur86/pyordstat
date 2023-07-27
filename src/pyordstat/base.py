"""Base class for order statistics distributions."""
from abc import ABC
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

    def __init__(
        self,
        pdf: StatDistrFunc,
        cdf: StatDistrFunc,
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> None:
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
