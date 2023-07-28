"""Core functionality for order statistics distributions."""
import numpy as np
from numpy.typing import NDArray
from scipy.special import binom


def ordstat_pdf(
    pdf: NDArray[np.number], cdf: NDArray[np.number], n: int, k: int
) -> NDArray[np.number]:
    """Compute the k-th order statistic PDF of a sample of size n.

    Return the k-th order statistic probability density function of a sample of size n,
    given the values of PDF and CDF.

    Args:
        pdf (NDArray[np.number]): Values of the PDF.
        cdf (NDArray[np.number]): Values of the CDF.
        n (int): Sample size.
        k (int): Order statistic to calculate.

    Returns
    -------
        NDArray[np.number]: Order statistic PDF.
    """
    return k * binom(n, k) * (cdf ** (k - 1)) * ((1 - cdf) ** (n - k)) * pdf


def ordstat_cdf(cdf: NDArray[np.number], n: int, k: int) -> NDArray[np.number]:
    """Compute the k-th order statistic CDF of a sample of size n.

    Return the k-th order statistic cumulative distribution function of a sample of size n,
    given the values of CDF.

    Args:
        cdf (NDArray[np.number]): Values of the CDF.
        n (int): Sample size.
        k (int): Order statistic to calculate.

    Returns
    -------
        NDArray[np.number]: Order statistic CDF.
    """
    if k == 1:
        # Special case
        return 1 - (1 - cdf) ** n

    j = np.arange(k, n + 1)
    bc = binom(n, j)
    is_scalar = np.isscalar(cdf)
    cdf = np.atleast_1d(cdf)
    cdf1 = cdf[None, :] ** j[:, None]
    cdf2 = (1 - cdf[None, :]) ** (n - j)[:, None]

    ans = np.sum(bc[:, None] * cdf1 * cdf2, axis=0)

    if is_scalar:
        return ans[0]

    return ans
