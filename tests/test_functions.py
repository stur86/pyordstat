"""Test functions."""
import numpy as np

from pyordstat.functions import UniformOrderStatistics


def test_uniform():
    """Test uniform order statistics."""
    uniform = UniformOrderStatistics(0, 1)

    assert np.isclose(uniform.pdf(0.5), 1.0)
    assert np.isclose(uniform.cdf(0.5), 0.5)
    assert np.isclose(uniform.pdf(2.0), 0.0)
    assert np.isclose(uniform.cdf(2.0), 1.0)

    pdf_2_1 = uniform.order_statistic_pdf(0.5, 2, 1)
    cdf_2_1 = uniform.order_statistic_cdf(0.5, 2, 1)

    assert np.isclose(pdf_2_1, 1.0)
    assert np.isclose(cdf_2_1, 0.75)
