"""Test functions."""
import numpy as np

from pyordstat.functions import RVNormalStatistics, RVUniformStatistics


def test_uniform():
    """Test uniform order statistics."""
    uniform = RVUniformStatistics(0, 1)

    assert np.isclose(uniform.pdf(0.5), 1.0)
    assert np.isclose(uniform.cdf(0.5), 0.5)
    assert np.isclose(uniform.pdf(2.0), 0.0)
    assert np.isclose(uniform.cdf(2.0), 1.0)

    pdf_2_1 = uniform.order_statistic_pdf(0.5, 2, 1)
    cdf_2_1 = uniform.order_statistic_cdf(0.5, 2, 1)

    assert np.isclose(pdf_2_1, 1.0)
    assert np.isclose(cdf_2_1, 0.75)


def test_normal():
    """Test normal order statistics."""
    normal = RVNormalStatistics(0, 1)

    pdf_val = 0.3520653267642995
    cdf_val = 0.6914624612740131
    assert np.isclose(normal.pdf(0.5), pdf_val)
    assert np.isclose(normal.cdf(0.5), cdf_val)

    assert np.isclose(normal.order_statistic_pdf(0.5, 2, 1), 2 * pdf_val * (1 - cdf_val))
    assert np.isclose(normal.order_statistic_cdf(0.5, 2, 1), 2 * cdf_val - cdf_val**2)
