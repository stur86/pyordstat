"""Order statistics for specific continuous distributions."""
from scipy import stats
from scipy.stats import rv_continuous

from pyordstat.continuous import ContinuousOrderStatistics


class RVContOrderStatistics(ContinuousOrderStatistics):
    """Base class for order statistics based on rv_continuous distributions."""

    def __init__(self, distribution: rv_continuous) -> None:
        super().__init__(distribution.pdf, distribution.cdf)


class RVUniformStatistics(RVContOrderStatistics):
    """Uniform distribution order statistics."""

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """Initialize uniform distribution order statistics.

        Args:
            loc (float, optional): Starting point of distribution. Defaults to 0.0.
            scale (float, optional): Scale of distribution. Defaults to 1.0.
        """
        distr = stats.uniform(loc=loc, scale=scale)
        super().__init__(distr)


class RVNormalStatistics(RVContOrderStatistics):
    """Normal distribution order statistics."""

    def __init__(self, loc: float, scale: float) -> None:
        """Initialize normal distribution order statistics.

        Args:
            loc (float, optional): Starting point of distribution. Defaults to 0.0.
            scale (float, optional): Scale of distribution. Defaults to 1.0.
        """
        distr = stats.norm(loc=loc, scale=scale)
        super().__init__(distr)
