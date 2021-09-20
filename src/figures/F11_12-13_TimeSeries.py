""" AR- and MA-plots, for Time-Series-Analysis """

# author: Thomas Haslwanter, date: Sept-2021

# Standard packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# modules from 'statsmodels'
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels import tsa

# additional packages

# additional packages
# Import formatting commands if directory "Utilities" is available
import os
import sys
sys.path.append(os.path.join('..', 'Code_Quantlets', 'Utilities'))

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


def arma_plot(arma_list: list, title: str) -> None:
    """ Generate a plot for two different arma values

    Paramters
    ---------
    arma_list : List of 2 tuples, containing at AR- and MA-parameters
    title : Title-text for figure
    """

    fig, axs = plt.subplots(3,2, constrained_layout=True)

    for (ii, arma) in enumerate(arma_list):
        ar = arma[0]
        ma = arma[1]
        n_samples = 1000
        # Generate the data
        np.random.seed(123)      # To make it reproducible
        arma_process = tsa.arima_process.ArmaProcess(ar, ma)
        y = arma_process.generate_sample(n_samples)

        # Fit the model
        if ar is None:
            model = ARIMA(y, order=(0,0,1))
        else:
            model = ARIMA(y, order=(1,0,0))

        print(f'ARMA = {arma}')
        model_fit = model.fit()
        print(model_fit.summary())

        # Plotdata,  ACF and PACF
        axs[0, ii].plot(y, lw=0.5)
        axs[0, ii].axhline(0, ls='dotted')
        axs[0, ii].set_title(f'ar={ar}, ma={ma}')
        plot_acf(y, ax=axs[1, ii], lw=0.5, markersize=3)
        plot_pacf(y, ax=axs[2, ii], lw=0.5, markersize=3)
        # plt.tight_layout()

    out_file = title + '.jpg'
    fig.suptitle(title)
    showData(out_file)
    
    plt.show()


if __name__== '__main__':
    # Generate plots showing first order AR and MA models
    # Define the parameters
    ars = [([1,  0.9], None),
           ([1, -0.9], None) ]
    mas = [(None, [1,  0.9]),
           (None, [1, -0.9])]

    arma_plot(ars, 'AR_Models')
    arma_plot(mas, 'MA_Models')
