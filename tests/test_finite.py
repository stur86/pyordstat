"""Tests for finite order statistics."""
import numpy as np

from pyordstat.finite import FiniteOrderStatistics


def test_finite():
    """Test finite order statistics."""
    coin = FiniteOrderStatistics([0, 1.0], [0.5, 0.5])

    assert np.allclose(coin.x, [0, 1])
    assert np.allclose(coin.pdf, [0.5, 0.5])
    assert np.allclose(coin.cdf, [0.5, 1.0])

    pdf_2 = coin.order_statistic_pdf(2, 1)
    cdf_2 = coin.order_statistic_cdf(2, 1)

    assert np.allclose(pdf_2, [0.75, 0.25])
    assert np.allclose(cdf_2, [0.75, 1.0])
