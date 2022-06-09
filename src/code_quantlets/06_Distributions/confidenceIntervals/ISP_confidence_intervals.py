""" Calculation of Confidence Intervals for different statistics
    - normal mean value
    - normal variability
    - Binomial p
    - Poisson mu
"""

# author: Thomas Haslwanter
# date:   June-2022

# Import the standard packages
import numpy as np
from scipy import stats
import numbers
from typing import Tuple


def mean(values: np.ndarray,
        alpha: float =0.05,
        ci_type: str ='two-sided',
        num_and_sigma: Tuple[int, float] =None) -> np.ndarray:
    """ Confidence interval for the mean value

    Parameters
    ----------
    values : sample values. If "values" is only one value, "num_and_sigma" also
        has to be specified
    alpha : significance threshold
    ci_type : Has to be "two-sided", "upper", or "lower"
    num_and_sigma : number of samples, and standard deviation. Only used if
            "values" is a single value

    Returns
    -------
    cis : If ci_type is "upper" or "lower": only upper/lower confidence limit
        If ci_type is "two-sided": lower and upper confidence limit

    Example
    -------
    cis = mean(data, ci_type='lower', alpha=0.01)
    """

    mean = np.mean(values)

    if num_and_sigma != None:
        sigma_known = True
        assert isinstance(values, numbers.Number)
        n, sigma = num_and_sigma
        df = n-1
        sem = sigma/np.sqrt(n)
    else:
        sigma_known = False
        sem = stats.sem(values, ddof=1)
        df = len(values) - 1

    if sigma_known:
        distribution = stats.norm(mean, sem)
    else:
        distribution = stats.t(df, mean, sem)

    if ci_type == 'two-sided':
        ci_lower = distribution.ppf(alpha/2)
        ci_upper = distribution.ppf(1-alpha/2)
        return np.array([ci_lower, ci_upper])


    elif ci_type == 'lower':
        ci_lower = distribution.ppf(alpha)
        return np.array([ci_lower])

    elif ci_type == 'upper':
        ci_upper = distribution.ppf(1-alpha)
        return np.array([ci_upper])

    else:
        print('Do not know the type {0}'.format(ci_type))


def binomial(n_success: int, n_total: int,
        alpha: float =0.05, ci_type: str ='two-sided') -> np.ndarray:
    """ Exact CI for the estimated probability of a binomial distribution.

    Parameters
    ----------
    n_success : number of "successes"
    n_total : total number of samples
    alpha : significance threshold
    ci_type : Has to be "two-sided", "upper", or "lower"

    Returns
    -------
    cis : If ci_type is "upper" or "lower": only upper/lower confidence limit
        If ci_type is "two-sided": lower and upper confidence limit

    Example
    -------
    cis = binomial(2, 120, alpha=0.01, ci_type='upper')
    """

    if ci_type == 'two-sided':
        p_lower = 1 - \
                stats.beta(n_total - n_success + 1, n_success).ppf(1-alpha/2)
        p_upper = 1 - \
        stats.beta(n_total - n_success, n_success + 1).ppf(alpha/2)
        return np.array([p_lower, p_upper])

    if ci_type == 'lower':
        p_lower = 1 - \
                stats.beta(n_total - n_success + 1, n_success).ppf(1-alpha)
        return np.array([p_lower])

    if ci_type == 'upper':
        p_upper = 1 - stats.beta(n_total - n_success, n_success + 1).ppf(alpha)
        return np.array([p_upper])


def binomial_approx(n_success: int, n_total: int,
        alpha: float =0.05, ci_type: str ='two-sided') -> np.ndarray:
    """Approximate CI for estimated probability of the binomial distribution.

    Parameters
    ----------
    n_success : number of "successes"
    n_total : total number of samples
    alpha : significance threshold
    ci_type : Has to be "two-sided", "upper", or "lower"

    Returns
    -------
    cis : float or Vector
        If ci_type is "upper" or "lower": upper/lower confidence limit
        If ci_type is "two-sided": ndarray, containing lower and upper
                                   confidence limit

    Example
    -------
    cis = binomial_approx(30, 1200, alpha=0.05, ci_type='two-sided')
    """

    p_hat = n_success/n_total
    sigma = np.sqrt(p_hat*(1-p_hat)/n_total)
    nd = stats.norm()
    correction = 1/(2*n_total)
    if ci_type == 'two-sided':
        p_lower = p_hat + (correction + sigma * nd.ppf(alpha/2))
        p_upper = p_hat + (correction + sigma * nd.ppf(1-alpha/2))
        return np.array([p_lower, p_upper])

    if ci_type == 'lower':
        p_lower = p_hat + (correction + sigma * nd.ppf(alpha))
        return np.array([p_lower])

    if ci_type == 'upper':
        p_upper = p_hat + (correction + sigma * nd.ppf(1-alpha))
        return np.array([p_upper])


def binomial_newton(n_success: int, n_total: int,
        alpha: float =0.05, ci_type:str ='two-sided') -> np.ndarray:
    """Exact CI for the estimated probability of the binomial distribution.
    The result is equivalent to "ci_binomial"; but it shows how the numerical
    evaluation can be done.

    Parameters
    ----------
    n_success : number of "successes"
    n_total : total number of samples
    alpha : significance threshold
    ci_type : Has to be "two-sided", "upper", or "lower"

    Returns
    -------
    cis : If ci_type is "upper" or "lower": upper/lower confidence limit
        If ci_type is "two-sided": lower and upper confidence limit

    Example
    -------
    cis = binomial_newton(30, 1200, alpha=0.05, ci_type='two-sided')

    Notes
    -----
    http://www.sigmazone.com/binomial_confidence_interval.htm
    """

    from scipy.optimize import newton

    p_hat = n_success/n_total

    def root_func(p, n, k, alpha):
        """Function to be minimized"""
        bd = stats.binom(n, p)
        val = bd.cdf(k) - alpha
        return val

    if ci_type == 'two-sided':
        p_upper = newton( root_func, p_hat,
                args=(n_total, n_success, alpha/2, ) )
        p_lower = newton( root_func, p_hat,
                args=(n_total, n_success-1, 1-alpha/2, ))
        return np.array([p_lower, p_upper])

    if ci_type == 'lower':
        p_lower = newton(root_func, p_hat, args=(n_total, n_success, 1-alpha, ))
        return np.array([p_lower])

    if ci_type == 'upper':
        p_upper = newton(root_func, p_hat, args=(n_total, n_success, alpha, ))
        return np.array([p_upper])


def poisson(n: int, alpha: float =0.05,
        ci_type:str ='two-sided') -> np.ndarray:
    """Exact CI for estimated mean of the Poisson distribution

    Parameters
    ----------
    n : number of observations
    alpha : significance threshold
    ci_type : Has to be "two-sided", "upper", or "lower"


    Returns
    -------
    cis : If ci_type is "upper" or "lower": upper/lower confidence limit
        If ci_type is "two-sided": lower and upper confidence limit

    Example
    -------
    cis = poisson(expected, alpha=0.05, ci_type='two-sided')

    Note
    ----
    http://onbiostatistics.blogspot.co.at/2014/03/computing-confidence-interval-for.html
    """

    if ci_type == 'two-sided':
        p_upper = stats.chi2(2*(n+1)).ppf(1-alpha/2) / 2
        p_lower = stats.chi2(2*n).ppf(alpha/2) / 2
        return np.array([p_lower, p_upper])

    if ci_type == 'lower':
        p_lower = newton(root_func, mu, args=(mu, 1-alpha, ))
        return np.array([p_lower])

    if ci_type == 'upper':
        p_upper = newton(root_func, mu, args=(mu, alpha, ))
        return np.array([p_upper])


def poisson_newton(mu: float, alpha: float=0.05,
        ci_type: str ='two-sided') -> np.ndarray:
    """Exact confidence interval for the estimated probability of the Poisson
       distribution - numerical calculation

    Parameters
    ----------
    mu : expected value
    alpha : significance threshold
    ci_type : Has to be "two-sided", "upper", or "lower"


    Returns
    -------
    cis : If ci_type is "upper" or "lower": upper/lower confidence limit
        If ci_type is "two-sided": lower and upper confidence limit

    Example
    -------
    cis = poisson(expected, alpha=0.05, ci_type='two-sided')

    Notes
    -----
    http://statpages.info/confint.html
    """

    from scipy.optimize import newton

    def root_func(x, m, alpha):
        """Function to be minimized"""
        bd = stats.poisson(x)
        val = bd.cdf(m)-alpha
        return val

    if ci_type == 'two-sided':
        p_upper = newton(root_func, mu, args=(mu, alpha/2, ))
        p_lower = newton(root_func, mu, args=(mu-1, 1-alpha/2, ))
        return np.array([p_lower, p_upper])

    if ci_type == 'lower':
        p_lower = newton(root_func, mu, args=(mu, 1-alpha, ))
        return np.array([p_lower])

    if ci_type == 'upper':
        p_upper = newton(root_func, mu, args=(mu, alpha, ))
        return np.array([p_upper])


def sigma(sigma_data: np.ndarray, alpha: float =0.05,
        ci_type: str ='two-sided', num: int =None) -> np.ndarray:
    """Confidence interval for the variability

    Parameters
    ----------
    sigma_data : sample values, or exact confidence interval
    alpha : significance threshold
    ci_type : Has to be "two-sided", "upper", or "lower"
    num : Number of samples; required if "sigma_data" contains the exact
        confidence interval

    Returns
    -------
    cis : float or Vector
        If ci_type is "upper" or "lower": one-sided upper/lower confidence limit
        If ci_type is "two-sided": lower and upper confidence limit

    Example
    -------
    cis = sigma(data, alpha=0.01)
    """


    if num != None:
        # Population standard deviation is known
        # Timischl, p 113: "Zufallsstreubereich"
        sigma_known = True
        assert isinstance(sigma_data, numbers.Number)
        sigma_pop = sigma_data
        n = num
    else:
        # Population standard deviation is unknown, and the
        # sample standard deviation is calculated
        # Timischl, p 135: "Vertrauensbereich"
        sigma_known = False
        std = np.std(sigma_data, ddof=1)
        n = len(sigma_data)
    df = n-1
    cd = stats.chi2(df)

    if ci_type == 'two-sided':
        if sigma_known:
            ci_lower = sigma_pop * np.sqrt( cd.ppf(alpha/2)/df )
            ci_upper = sigma_pop * np.sqrt( cd.ppf(1-alpha/2)/df )
        else:
            ci_lower = std * np.sqrt(df/cd.ppf(1-alpha/2))
            ci_upper = std * np.sqrt(df/cd.ppf(alpha/2))
        return np.array([ci_lower, ci_upper])

    elif ci_type == 'upper':
        # Nach unten abgegrenzter Vertrauensbereich
        ci_lower = 0
        if sigma_known:
            ci_upper = sigma_pop * np.sqrt( cd.ppf(1-alpha)/df )
        else:
            ci_upper = std * np.sqrt(df/cd.ppf(alpha))
        return np.array([ci_upper])

    elif ci_type == 'lower':
        if sigma_known:
            ci_lower = sigma_pop * np.sqrt( cd.ppf(alpha)/df )
        else:
            ci_lower = std * np.sqrt(df/cd.ppf(1-alpha))
        return np.array([ci_lower])


if __name__ == '__main__':
    alpha = 0.05
    ci_level = int( (1-alpha)*100 )
    cis = binomial(5, 100, alpha=alpha, ci_type='two-sided')
    np.set_printoptions(precision = 10)
    print(f'The {ci_level}% CI for the Binomial probability is: {cis} (exact)')

    # All the examples below are taken from the book Timischl:
    # Qualitaetssicherung, 4 ed

    # Timischl 3.2
    cis = binomial(1, 20, alpha=0.01, ci_type='upper')
    print(f'The confidence limit for the Binomial probability: {cis}')

    cis = binomial(30, 1200, alpha=0.01)
    print(f'The confidence limit for the Binomial probability: {cis}')

    cis = binomial(30, 1200, alpha=0.05, ci_type='two-sided')
    print(f'The confidence limit for the Binomial probability: {cis} (exact)')

    cis = binomial_newton(30, 1200, alpha=0.05, ci_type='two-sided')
    print(f'The CI limit for the Binomial probability: {cis} (numerical)')

    cis = binomial_approx(30, 1200, alpha=0.05, ci_type='two-sided')
    print(f'The CI for the Binomial probability: {cis} (approximate)')

    # Timischl 2.57
    cis = sigma(sigma_data=30, num=10, ci_type='upper')
    print(f'The upper limit for sigma from an exact inputs is: {cis}')
    cis = sigma(sigma_data=30, num=10, ci_type='upper', alpha=0.01)
    print(f'The upper limit for sigma from an exact inputs is: {cis}')

    # Timischl 3.3
    cis = binomial(2, 120, alpha=0.01, ci_type='upper')
    print(f'The confidence limit for the Binomial probability: {cis}')

    # Timischl 3.4
    expected = 12
    cis = poisson(expected, alpha=0.01, ci_type='two-sided')
    print(f'The confidence limit for the Poisson mean: {cis}')

    # Other example, numerical
    (expected, total) = (14, 400)
    cis = poisson_newton(expected, alpha=0.05, ci_type='two-sided')
    print(f'The CI for the Poisson mean: {cis} (numerical)')
    print(f'The CI for the Poisson probability: {cis/total} (numerical)')

    # Timischl, 3.7
    data = [89, 104.1, 92.3, 106.2, 96.3, 107.8, 102.5, 121.2,
            98, 109.4, 99.7, 111.6, 101.4, 113.8, 116.6]

    cis = mean(data, ci_type='lower', alpha=0.01)
    print(f'The confidence limit for the mean: {cis}')

    # Timischl 3.9
    data = [49.93, 50.03, 49.99, 50.08, 49.96, 50.03]
    cis = sigma(data, alpha=0.01)
    print(f'The confidence limit for sigma: {cis}')

    # Other tests
    cis = mean(values=75, num_and_sigma = (10, 12))
    print(f'The confidence limit for the mean from exact inputs is: {cis}')

    """
    """
