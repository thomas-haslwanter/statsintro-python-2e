""" Solution for Exercises 'Peak Observations' in Chapter 12
Requires the package 'xlrd' to be installed, with

`pip install xlrd`

"""

# author: Thomas Haslwanter, date: Dec-2021

# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import statsmodels.formula.api as sm

def getModelData(show: bool=True) ->None:
    """ Get the data from an Excel-file

    Parameters
    ----------
    show : boolean flag, controlling the display
    """

    # First, define the in-file and get the data
    in_file = '..\..\data\AvgTemp.xls'

    # When the data are neatly organized, they can be read in
    # directly with the pandas-functions:
    # with "ExcelFile" you open the file ...
    xls = pd.ExcelFile(in_file)

    # ... and with "parse" you get get the data from the file,
    # from the specified Excel-sheet
    data = xls.parse('Tabelle1')

    if show:
        data.plot('year', 'AvgTmp')
        plt.xlabel('Year')
        plt.ylabel('Average Temperature')
        plt.show()

    return data


def correlation(data):
    """ Exercise Peak observations - Correlation ----------------
       Calculate and show the different correlation coefficients
    """

    pearson = data['year'].corr(data['AvgTmp'],
            method = 'pearson')
    spearman = data['year'].corr(data['AvgTmp'],
            method = 'spearman')
    tau = data['year'].corr(data['AvgTmp'],
            method = 'kendall')

    print(f'Pearson correlation coefficient: {pearson:4.3f}')
    print(f'Spearman correlation coefficient: {spearman:4.3f}')
    print(f'Kendall tau: {tau:4.3f}')


def normality_check(data):
    """ Exercise Peak observations - Normality Check  """

    # Fit the model
    model = sm.ols('AvgTmp ~ year', data)
    results = model.fit()

    # Normality check --------------------------------------------
    res_data = results.resid    # Get the values for the residuals

    # QQ-plot, for a visual check
    stats.probplot(res_data, plot=plt)
    plt.show()

    # Normality test, for a quantitative check:
    _, pVal = stats.normaltest(res_data)
    if pVal < 0.05:
        print('WARNING: The data are not normally distributed ' +
              f'(p = {pVal})')
    else:
        print('Data are normally distributed.')


def regression(data):
    """ Exercise Peak observations - Regression """

    # Regression -------------------------------------------------
    # For "ordinary least square" models, you can do the model
    # with the formula-approach from statsmodels:
    # offsets are automatically included in the model
    model = sm.ols('AvgTmp ~ year', data)
    results = model.fit()
    print(results.summary())

    # Visually, the confidence intervals can be shown using seaborn
    sns.lmplot('year', 'AvgTmp', data)
    plt.show()

    # Is the inclination significant?
    ci = results.conf_int()

    # This line is a bit tricky: if both are above or both below
    # zero the product is positive:
    # we look at the coefficient that describes the correlation
    # with "year"
    if np.prod(ci.loc['year'])>0:
        print('The slope is significant')


if __name__=='__main__':
    data = getModelData()

    correlation(data)
    regression(data)
    normality_check(data)
