"""pyordstat package."""
from pyordstat.continuous import ContinuousOrderStatistics
from pyordstat.discrete import DiscreteOrderStatistics
from pyordstat.finite import FiniteOrderStatistics

__all__ = ["ContinuousOrderStatistics", "FiniteOrderStatistics", "DiscreteOrderStatistics"]
