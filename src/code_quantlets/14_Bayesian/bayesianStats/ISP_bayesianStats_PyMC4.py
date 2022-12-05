"""Example of PyMC4 - The Challenger Disaster
This example uses Bayesian methods to find the  mean and the 95% confidence
intervals for the likelihood of an O-ring failure in a space shuttle, as a
function of the ambient temperature.
Input data are the recorded O-ring performances of the space shuttles
before 1986.

Important note
--------------
This module requires PyMC4. To install it you probably have to use Anaconda.
With `conda', be sure to ALWAYS use
`conda <command> -c conda-forge <parameters>`
to install the virtual environment, and the desired packages.

E.g.:
`conda create -c conda-forge -n pymc4_env "pymc>=4"
conda install -c conda-forge jupyter
conda activate pymc4_env`
"""

# author: Thomas Haslwanter, date: Dec-2022

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
import os

# additional packages, for PyMC4
import pymc as pm
import aesara.tensor as tt
import arviz as az

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


def logistic(x: np.ndarray, beta: np.ndarray, alpha: np.ndarray =0) -> np.ndarray:
    """Logistic Function
    Parameters
    ----------
    x : values for the x-axis
    beta: can be a number or a 2D-matrix; controls the steepness of the logistic
          function
    alpha: can be a number or a 2D-matrix; shifts the logistic function

    Returns
    -------
    values for logistic function(s)
    """

    return 1.0 / (1.0 + np.exp(beta @ x + alpha))


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

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, tau=0.001, initval=0)
        alpha = pm.Normal("alpha", mu=0, tau=0.001, initval=0)
        p = pm.Deterministic("p", 1.0/(1. + tt.exp(beta*temperature + alpha)))

        # connect the probabilities in `p` with our observations through a
        # Bernoulli random variable.
        observed = pm.Bernoulli("bernoulli_obs", p, observed=failures)
        
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(120000, step=step, initvals=start)

    fig, axs = plt.subplots(2,1)
    az.plot_posterior(trace, var_names='alpha', ax=axs[0])
    az.plot_posterior(trace, var_names='beta', ax=axs[1])

    outFile = 'Challenger_Parameters.png'
    showData(outFile)

    # Get the sampled variable traces for alpha and beta:
    # Note that PyMC4 requires no more "burn-in", and that the sampled variables
    # are stored as 2D Arrays.
    # Also, there may be a more elegant way to get the sampled variables from 
    # the arviz.InferenceData object than the one used here
    alpha, beta = az.extract(trace, var_names=['alpha', 'beta']).to_array()
    alpha_samples = np.atleast_2d(alpha).T
    beta_samples = np.atleast_2d(beta).T

    return(alpha_samples, beta_samples)


def calc_prob(alpha_samples: np.ndarray,
              beta_samples: np.ndarray,
              temperature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the mean probability, and the CIs

    Parameters
    ----------
    alpha_samples : (n x 1)-vector, sampled alpha values
    beta_samples :  (n x 1)-vector, sampled beta values
    temperature :   (m x 1)-vector (for the failures)

    Returns
    -------
    t : temperature values, for a continuous plot
    mean_prob_t : mean failure probability
    p_t : realized traces of the posterior distribution
    quantiles : 95%-quantiles for the probability of failure
    """

    # Calculate the probability as a function of time
    t = np.linspace(temperature.min() - 5, temperature.max() + 5, 50)[:, None]
    p_t = logistic(t.T, beta_samples, alpha_samples)

    mean_prob_t = p_t.mean(axis=0)

    # --- Calculate CIs ---
    # vectorized bottom and top 2.5% quantiles for "confidence interval"
    quantiles = mquantiles(p_t, [0.025, 0.975], axis=0)

    return (t, mean_prob_t, p_t, quantiles)


def show_prob(linearTemperature, temperature,
              failures, mean_prob_t, p_t, quantiles) -> None:
    """Show the posterior probabilities, and save the resulting figures

    Parameters
    ----------
    linearTemperature : temperature values, for a continuous plot
    temperature : temperature of failed starts
    failures :    failed starts
    mean_prob_t : mean failure probability
    p_t : realized traces of the posterior distribution
    quantiles : 95%-quantiles for the probability of failure
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
    (linearTemperature, mean_p, p, quantiles) = calc_prob(alpha, beta,
                                                          temperature)
    show_prob(linearTemperature, temperature, failures, mean_p, p, quantiles)

