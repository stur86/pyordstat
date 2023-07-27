"""Tests for continuous order statistics."""
import numpy as np
import pytest
from numpy.typing import NDArray

from pyordstat.general import OrderStatistics


def test_c_ordstat():
    """Test continuous order statistics."""

    # For a simple example, use exponential distribution
    def pdf(x: NDArray[np.number], scale: float) -> NDArray[np.number]:
        return np.exp(-x / scale) / scale

    def cdf(x: NDArray[np.number], scale: float) -> NDArray[np.number]:
        return 1 - np.exp(-x / scale)

    scale = 1.0

    exp_ordstat = OrderStatistics(pdf, cdf, scale)

    x = np.linspace(0, 10, 20)

    assert np.allclose(exp_ordstat.pdf(x, scale), pdf(x, scale))
    assert np.allclose(exp_ordstat.cdf(x, scale), cdf(x, scale))

    # Case 1: n = 2, k = 1
    pdf_2_1 = exp_ordstat.order_statistic_pdf(x, 2, 1)
    cdf_2_1 = exp_ordstat.order_statistic_cdf(x, 2, 1)

    assert np.allclose(pdf_2_1, 2 * pdf(x, scale) * (1 - cdf(x, scale)))
    assert np.allclose(cdf_2_1, 1 - (1 - cdf(x, scale)) ** 2)

    # Case 2: n = 3, k = 2
    pdf_3_2 = exp_ordstat.order_statistic_pdf(x, 3, 2)
    cdf_3_2 = exp_ordstat.order_statistic_cdf(x, 3, 2)

    # Expected values
    pdf_3_2_targ = 6 * pdf(x, scale) * cdf(x, scale) * (1 - cdf(x, scale))
    cdf_3_2_targ = 3 * cdf(x, scale) ** 2 * (1 - cdf(x, scale)) + cdf(x, scale) ** 3

    assert np.allclose(pdf_3_2, pdf_3_2_targ)
    assert np.allclose(cdf_3_2, cdf_3_2_targ)

    # Test errors
    with pytest.raises(ValueError, match="k must be between 1 and n."):
        exp_ordstat.order_statistic_pdf(x, 2, 3)

    with pytest.raises(ValueError, match="k must be between 1 and n."):
        exp_ordstat.order_statistic_cdf(x, 2, 0)
