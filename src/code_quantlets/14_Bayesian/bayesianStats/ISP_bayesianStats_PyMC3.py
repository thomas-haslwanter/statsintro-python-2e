"""Example of PyMC - The Challenger Disaster
This example uses Bayesian methods to find the  mean and the 95% confidence
intervals for the likelihood of an O-ring failure in a space shuttle, as a
function of the ambient temperature.
Input data are the recorded O-ring performances of the space shuttles
before 1986.

Important note
--------------
This module requires PyMC3. I only have been able to install
either of these packages using Anaconda!!

"""

# author: Thomas Haslwanter, date: Nov-2022

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
import os

# additional packages
import pymc3 as pm
import theano.tensor as tt

from scipy.stats.mstats import mquantiles
from typing import Tuple

# additional packages
import sys
sys.path.append(os.path.join('..', '..', 'Utilities'))

try:
# Import formatting commands if directory "Utilities" is available
    from ISP_mystyle import setFonts, showData

except ImportError:
# Ensure correct performance otherwise
    def setFonts(*options):
        return
    def showData(*options):
        plt.show()
        return

sns.set_context('poster')


def logistic(x: np.ndarray, beta:float, alpha:float=0) -> np.ndarray:
    """Logistic Function"""

    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))


def getData() -> Tuple[np.ndarray, np.ndarray]:
    """Get and show the O-ring data

    Results
    -------
    temperature : temperature data
    failureData : corresponding failure status
    """

    inFile = 'challenger_data.csv'

    challenger_data = np.genfromtxt(inFile, skip_header=1, usecols=[1, 2],
                                    missing_values='NA', delimiter=',')

    # drop the NA values
    challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

    temperature = challenger_data[:, 0]
    failureData = challenger_data[:, 1]  # defect or not?
    return (temperature, failureData)


def show_and_save(temperature: np.ndarray, failures: np.ndarray) -> None:
    """Shows the input data, and saves the resulting figure

    Parameters
    ----------
    temperature : temperature data
    failureData : corresponding failure status

    """

    # Plot it, as a function of tempature
    plt.figure(figsize=(12,4))
    setFonts()
    sns.set_style('darkgrid')
    np.set_printoptions(precision=3, suppress=True)

    plt.scatter(temperature, failures, s=150, color="k", alpha=0.5)
    # plt.margins(x=10, y=0.2)
    plt.xlim([52, 82])
    plt.ylim([-0.1, 1.1])
    plt.yticks([0, 1])
    plt.ylabel("Damage Incident?")
    plt.xlabel("Outside Temperature [F]")
    plt.title("Defects of the Space Shuttle O-Rings vs temperature")
    plt.tight_layout()

    outFile = 'Challenger_ORings.png'
    showData(outFile)


def mcmc_simulations(temperature: np.ndarray, failures: np.ndarray) -> Tuple:
    """Perform the MCMC-simulations

    Parameters
    ----------
    temperature : temperature data
    failureData : corresponding failure status

    Returns
    -------
    alpha_post : posterior distribution of alpha values
    beta_post :  posterior distribution of beta values
    """

    # Define the prior distributions for alpha and beta
    # 'value' sets the start parameter for the simulation
    # The second parameter for the normal distributions is the "precision",
    # i.e. the inverse of the standard deviation

    def logistic(x, beta, alpha=0):
        """ Define the model-function for the temperature """
        return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))  

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, tau=0.001, testval=0)
        alpha = pm.Normal("alpha", mu=0, tau=0.001, testval=0)
        p = pm.Deterministic("p", 1.0/(1. + tt.exp(beta*temperature + alpha)))

    # connect the probabilities in `p` with our observations through a
    # Bernoulli random variable.
    with model:
        observed = pm.Bernoulli("bernoulli_obs", p, observed=failures)
        
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(120000, step=step, start=start)
        burned_trace = trace[100000::2]

    alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
    beta_samples = burned_trace["beta"][:, None]

    return(alpha_samples, beta_samples)


def show_sim_results(alpha_samples, beta_samples) -> None:
    """Show the results of the simulations, and save them to an outFile

    Parameters
    ----------
    alpha_samples: posterior distribution of alpha values
    beta_samples :  posterior distribution of beta values
    """

    plt.figure(figsize=(12.5, 6))
    sns.set_style('darkgrid')
    setFonts(18)

    # Histogram of the samples:
    plt.subplot(211)
    plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
    plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
             label=r"posterior of $\beta$", color="#7A68A6", density=True)
    plt.legend()

    plt.subplot(212)
    plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
             label=r"posterior of $\alpha$", color="#A60628", density=True)
    plt.legend()

    outFile = 'Challenger_Parameters.png'
    showData(outFile)


def calc_prob(alpha_samples, beta_samples, temperature, failures):
    """Calculate the mean probability, and the CIs

    Parameters
    ----------

    Returns
    -------

    """

    # Calculate the probability as a function of time
    t = np.linspace(temperature.min() - 5, temperature.max() + 5, 50)[:, None]
    p_t = logistic(t.T, beta_samples, alpha_samples)

    mean_prob_t = p_t.mean(axis=0)

    # --- Calculate CIs ---
    # vectorized bottom and top 2.5% quantiles for "confidence interval"
    quantiles = mquantiles(p_t, [0.025, 0.975], axis=0)

    return (t, mean_prob_t, p_t, quantiles)


def show_prob(linearTemperature, temperature, failures,
                      mean_prob_t, p_t, quantiles) -> None:
    """Show the posterior probabilities, and save the resulting figures

    Parameters
    ----------
    linearTemperature :
    temperature :
    failures :
    mean_prob_t :
    p_t :
    quantiles :

    """

    # --- Show the probability curve ----
    plt.figure(figsize=(12.5, 4))
    setFonts(18)

    plt.plot(linearTemperature, mean_prob_t, lw=3,
            label="Average posterior\n probability of defect")
    plt.plot(linearTemperature, p_t[0, :], ls="--",
            label="Realization from posterior")
    plt.plot(linearTemperature, p_t[-2, :], ls="--",
            label="Realization from posterior")
    plt.scatter(temperature, failures, color="k", s=50, alpha=0.5)
    plt.title('Posterior expected value of probability of defect, ' + \
              'plus realizations')
    plt.legend(loc="lower left")
    plt.ylim(-0.1, 1.1)
    plt.xlim(linearTemperature.min(), linearTemperature.max())
    plt.ylabel("Probability")
    plt.xlabel("Temperature [F]")

    outFile = 'Challenger_Probability.png'
    showData(outFile)

    # --- Draw CIs ---
    setFonts()
    sns.set_style('darkgrid')

    plt.fill_between(linearTemperature[:, 0], *quantiles, alpha=0.7,
                     color="#7A68A6")

    plt.plot(linearTemperature[:, 0], quantiles[0], label="95% CI",
            color="#7A68A6", alpha=0.7)

    plt.plot(linearTemperature, mean_prob_t, lw=1, ls="--", color="k",
             label="average posterior \nprobability of defect")

    plt.xlim(linearTemperature.min(), linearTemperature.max())
    plt.ylim([-0.1, 1.1])
    plt.legend(loc="lower left")
    plt.scatter(temperature, failures, color="k", s=50, alpha=0.5)
    plt.xlabel("Temperature [F]")
    plt.ylabel("Posterior Probability Estimate")

    outFile = 'Challenger_CIs.png'
    showData(outFile)


if __name__=='__main__':
    (temperature, failures) = getData()
    show_and_save(temperature, failures)
    (alpha, beta) = mcmc_simulations(temperature, failures)
    show_sim_results(alpha, beta)
    (linearTemperature, mean_p, p, quantiles) = calc_prob(alpha, beta,
                                                          temperature, failures)
    show_prob(linearTemperature, temperature, failures, mean_p, p, quantiles)

