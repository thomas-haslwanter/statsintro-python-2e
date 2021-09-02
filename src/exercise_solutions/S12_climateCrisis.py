""" Solution to Exercise 'Climate Crisis'
  of the chapter 'Linear Regresison Models' """

# author:   Thomas Haslwanter
# date:     Sept-2021

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
""" Time Series Analysis of  global CO2-levels """

# modules from 'statsmodels'
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.formula.api as smf


def get_CO2_data() -> pd.DataFrame:
    """Read in data, and return them as a pandas DataFrame

    Returns
    -------
    df : time stamped recordings of CO2-levels at Mauna Loa, Hawaii
    """
    
    # Get the data, display a few values, and show the data
    url = 'https://www.esrl.noaa.gov/gmd/webdata/ccgg/trends/co2/co2_mm_mlo.txt'
    df = pd.read_csv(url,
                     skiprows=53,
                     delim_whitespace=True,
                     names = ['year', 'month', 'time', 'co2', 'deseasoned',
                               'nr_days', 'std_days', 'uncertainty'])

    ##  show CO2-levels as a function of time
    #df.plot('time', 'co2')
    #plt.show()

    return df


def decompose(df: pd.DataFrame) -> np.array:
    """ Make a seasonal decomposition of the input data

    Parameters
    ----------
    df : time stamped recordings of CO2-levels at Mauna Loa, Hawaii

    Returns
    -------
    decomposed : 
    """
    # Seasonal decomposition
    result_add = seasonal_decompose(df['co2'], model='additive', period=12,
            extrapolate_trend='freq')
    
    ## Show the decomposed data
    #result_add.plot()
    #plt.show()

    return result_add.trend


def find_best_fit(df: pd.DataFrame) -> None:
    """ Take the trend-data from the CO2 measurements, and find the best fit
     
    Parameters
    ----------
    df : 'year' in years (decimal), and 'co2': trend of the CO2-data
    """
    
    # Fit the models, and show the results
    linear = smf.ols('co2 ~ year', df).fit()
    quadratic = smf.ols('co2 ~ year+I(year**2)', df).fit()
    cubic = smf.ols('co2 ~ year+I(year**2)+I(year**3)', df).fit()    
    
    df['linear'] = linear.predict()
    df['quadratic'] = quadratic.predict()
    df['cubic'] = cubic.predict()
    
    # Show the data
    df.plot('year', ['co2', 'linear', 'quadratic', 'cubic'])
    
    # Select the best fit
    aics = [linear.aic, quadratic.aic, cubic.aic]
    index = np.argmin(aics)
    
    print(f'The best fit is of the order {index+1}.')
    
    plt.show()
    return 


if __name__ == '__main__':
    data = get_CO2_data()
    trend = decompose(data)
    
    time_co2 = pd.concat({'year': data.time, 'co2': trend}, axis=1)
    find_best_fit(time_co2)
