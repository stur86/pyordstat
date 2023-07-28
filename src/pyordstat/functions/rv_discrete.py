"""Order statistics for specific discrete distributions."""
from scipy import stats
from scipy.stats import rv_discrete

from pyordstat.discrete import DiscreteOrderStatistics


class RVDiscrOrderStatistics(DiscreteOrderStatistics):
    """Base class for order statistics based on rv_discrete distributions."""

    def __init__(self, distribution: rv_discrete) -> None:
        super().__init__(distribution.pmf, distribution.cdf)


class RVBinomialStatistics(RVDiscrOrderStatistics):
    """Binomial distribution order statistics."""

    def __init__(self, n: int, p: float) -> None:
        """Initialize binomial distribution order statistics.

        Args:
            n (int): Number of trials.
            p (float): Probability of success.
        """
        distr = stats.binom(n, p)
        super().__init__(distr)


class RVGeomStatistics(RVDiscrOrderStatistics):
    """Geometric distribution order statistics."""

    def __init__(self, p: float) -> None:
        """Initialize geometric distribution order statistics.

        Args:
            p (float): Probability of success.
        """
        distr = stats.geom(p)
        super().__init__(distr)
