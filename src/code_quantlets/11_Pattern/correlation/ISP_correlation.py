""" Calculation and visualization of correlation matrix """

# author: thomas haslwantere; date: aug-2021

# Import the required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import pingouin as pg

# additional packages
# Import formatting commands if directory "Utilities" is available
import os
import sys
sys.path.append(os.path.join('..',  '..', 'Utilities'))
try:
    from ISP_mystyle import setFonts, showData 
    
except ImportError:
# Ensure correct performance otherwise
    def setFonts(*options):
        return
    def showData(*options):
        plt.show()
        return
    
def get_data() -> pd.DataFrame:
    """
    Returns
    -------
    df : iris-data (ML-dataset)
    """

    # The "iris" dataset is one of the most common examples
    # in pattern recognition
    df = sns.load_dataset('iris')

    return df


def corr_coeff(iris: pd.DataFrame) -> None:
    """ Calculate Pearson's correlation coefficient

    Parameters
    ----------
    iris : iris-data (ML-dataset)
    """

    # Pearson Correlation, first with scipy ...
    r = stats.pearsonr(iris.petal_length, iris.petal_width)
    print(f'r^2 = {r[0]**2:5.3f}')

    # ... then with pingouin
    result = pg.corr(iris.petal_length, iris.petal_width)
    print(result.round(3))

    # And finally, show the data:
    plt.plot(iris.petal_length, iris.petal_width, '.')
    plt.title(f'Iris data: r={r[0]:5.3f}')
    plt.xlabel('Petal Length [cm]')
    plt.ylabel('Petal Width [cm]')
    showData('correlation.jpg')


def covariance_matrix(df: pd.DataFrame) -> None:
    """ Show the covariance matrix, and print the correlation matrix

    Parameters
    ----------
    df : iris-data (ML-dataset)
    """

    # Covariance matrix
    sns.pairplot(df, hue="species", size=2.5)
    showData('multiScatterplot.jpg')

    # The corresponding values of the covariance matrix are not very useful ...
    cov_matrix = df.cov()
    print(f'The covariance matrix is\n {cov_matrix}')

    # ... but the correlation matrix is!
    corr_matrix = df.corr()
    print('Correlation matrix:')
    print(corr_matrix)


if __name__ == '__main__':
    data = get_data()
    corr_coeff(data)
    covariance_matrix(data)

    
