# pyordstat

Order statistics for Python 📊

## Introduction

A package providing order statistics for Python. The package implements [the well-known formulas for order statistics](https://en.wikipedia.org/wiki/Order_statistic) in a format that is amenable to being used with either infinite distributions expressed as functions, or finite distributions expressed as lists of values and their probability mass. It is also compatible with the format of [SciPy's `stats` module](https://docs.scipy.org/doc/scipy/reference/stats.html).

### What are order statistics?

Order statistics are the statistics of the k-th variable in a sample of N elements. Given a probability distribution from which identical and independent samples are drawn, the k-th order statistic is the k-th smallest value in the sample. For example, the median is the 50th order statistic in a sample of 100. This package provides functions for calculating the probability density function (PDF) and cumulative distribution function (CDF) of order statistics for arbitrary distributions.

## Installation

To add and install this package as a dependency of your project, run `poetry add pyordstat`.

## Usage

### Continuous distributions

Use the class `ContinuousOrderStatistics` to calculate the PDF and CDF of order statistics defined on a continuous support. The class takes callable PDF and CDF functions as its first arguments, and any successive parameters are passed to the PDF and CDF functions. For example, to calculate the PDF and CDF of the median of a standard normal distribution, you would do the following:

```python
from pyordstat import ContinuousOrderStatistics

def normal_pdf(x, mu, sigma):
    return 1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * (x/mu) ** 2/ sigma ** 2)

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (2 ** 0.5 * sigma)))

order_stats = ContinuousOrderStatistics(normal_pdf, normal_cdf, mu, sigma)


pdf_4_2 = order_stats.order_statistics_pdf(4, 2)
```

### Discrete distributions

For discrete distributions, use the class `DiscreteOrderStatistics`. This class takes callable PMF (Probability Mass Function) and CDF functions as its first arguments, and any successive parameters are passed to the PMF and CDF functions. It works the same as `ContinuousOrderStatistics`, but the `_pdf` in methods is replaced by `_pmf` (as it's more appropriate to talk about probability mass functions for discrete distributions).

### Finite distributions

For finite distributions, use the class `FiniteOrderStatistics`. This class allows you to just pass the values of the support and their probability mass as arrays, and it will calculate the CDF for you. For example, to calculate the PDF and CDF of the median for a sample of 4 drawn from a discrete distribution with support `[1, 2, 3, 4, 5]` and probability mass `[0.1, 0.2, 0.3, 0.2, 0.2]`, you would do the following:

```python

from pyordstat import FiniteOrderStatistics

support = np.array([1, 2, 3, 4, 5])
pmf = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

order_stats = FiniteOrderStatistics(support, pmf)

pdf_4_2 = order_stats.order_statistics_pdf(4, 2)
```

### Scipy compatibility

The classes `RVContOrderStatistics` and `RVDiscrOrderStatistics` will accept an instance of a SciPy `rv_continuous` or `rv_discrete` distribution respectively. 
For convenience, a few common use cases of this are included in the `pyordstat.functions` module.

## Contributing

Contributions are welcome! This project was based off the Poetry Cookiecutter template found [here](https://github.com/radix-ai/poetry-cookiecutter). Recommended steps when developing are:

* set up your poetry environment with `poetry install`
* set up the pre-commit hooks with `pre-commit install` (make sure to have Ruff, Black, and MyPy installed in the main environment)
* lint the code with `poe lint`
* run the tests with `poe test`
* rebuild the docs with `poe docs`

Any contributions should be sent as a PR to the `develop` branch. Please make sure to include tests for any new functionality, and to update the docs accordingly.

## License

This project is licensed under the terms of the MIT license.