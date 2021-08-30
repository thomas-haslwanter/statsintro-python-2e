""" Solution to Exercise 'Logistic Regression' of the chapter 'GLM' """

# author:   Thomas Haslwanter
# date:     Aug-2021

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# additional packages
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial

sns.set_context('notebook')


def prepare_fit(in_data: np.ndarray) -> pd.DataFrame:
    """ Use the sepal-length as index, and count occurences
    of 'setosa' and 'other species' for each length
    
    Parameters
    ----------
    in_data : the iris data
    
    Returns
    -------
    df : 'num_setosa' and 'num_others', for each sepal length
    """
    
    # Create a dataframe, with suitable columns for the fit
    df = pd.DataFrame()
    df['length'] = np.unique(in_data.sepal_length)
    df['num_setosa'] = 0
    df['num_others'] = 0
    df.index = df.length    # make the 'length' the index
    
    # Count the number of 'setosa' and 'others', for each length
    for cur_length in df.length:
        df.loc[cur_length, 'num_setosa'] = \
        ((iris.sepal_length == cur_length) & (iris.species == 'setosa')).sum()
        
        df.loc[cur_length, 'num_others'] = \
        ((iris.sepal_length == cur_length) & (iris.species != 'setosa')).sum()
    
    # Just to check the total number
    df['total'] = df.num_setosa + df.num_others
    
    return df


def logistic(x: np.ndarray, beta:float, alpha:float=0) -> np.ndarray:
    """ Logistic Function """
    
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))


def show_results(iris_data: np.ndarray, model) -> None:
    """ Show the original data, and the resulting logit-fit
    
    Paramters
    ---------
    iris_data : input data
    model : model results (statsmodels.genmod.generalized_linear_model.GLM)
    
    """
    
    sepal_length = iris_data.sepal_length
    
    # First plot the original data
    plt.figure()
    sns.set_style('darkgrid')
    np.set_printoptions(precision=3, suppress=True)
    
    plt.scatter(iris.sepal_length, iris.species=='setosa',
                    s=50, color="k", alpha=0.2)
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylabel("Setosa")
    plt.xlabel("Sepal Length (cm)")
    plt.title("Probability of 'setosa', as function of sepal-length")
    plt.tight_layout
    
    # Plot the fit
    x = np.linspace(4, 8, 100)
    alpha = model.params[0]
    beta = model.params[1]
    y = logistic(x, beta, alpha)
    
    plt.plot(x, y,'r')
    plt.xlim([4, 8])
    
    out_file = 'setosa.jpg'
    plt.savefig(out_file, dpi=200)
    plt.show()
    print(f'Results save to {out_file}')
    
    return(x, y)
    
    
def find_probabilities(x: np.arange, px: np.arange, lengths: list) -> None:
    """ Find the probability that flower is a 'setosa'
    
    Parameters
    ----------
    x : sepal length
    px: corresponding probability os being 'setosa'
    lengths : values of interest
    """
    
    # find the closest x-value
    for length in lengths:
        index = np.max(np.where(x<length))
        print(f'For a length of {length:4.2f} cm, the probability of' +
              f'being a "setosa" is {px[index]:5.3f}')
        
    # Find maximum length of having at least a 10% chance of being a 'setosa'
    chance = 10/100 # [%]
    
    max_index = np.max(np.where(px>chance))
    print(f'The maximum length where you still have a {chance*100:3.0f}%  ' +
           f'chance of being a "setosa" is {x[max_index]:3.1f}cm.')
    
    
    
if __name__ == '__main__':
    # Get the data
    iris = sns.load_dataset('iris')
    
    # Count occurences of 'setosa' and 'others', for each length
    df_fit = prepare_fit(iris)
    
    # fit the model
    # --- >>> START stats <<< ---
    model = glm('num_others + num_setosa ~ length',
                data=df_fit, family=Binomial()).fit()
    # --- >>> STOP stats <<< ---
    
    print(model.summary())
    (x, px) = show_results(iris, model)
    find_probabilities(x, px, [5, 6])
    
    
