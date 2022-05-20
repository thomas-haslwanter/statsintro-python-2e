"""Multiple Regression
- Shows how to calculate the best fit to a plane in 3D, and how to find the
  corresponding statistical parameters.
- Demonstrates how to make a 3d plot.
- Example of multiscatterplot, for visualizing correlations in three- to
  six-dimensional datasets.
"""

# author: Thomas Haslwanter, date: Dec-2021

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# additional packages
import sys
import os
from typing import Tuple, List

sys.path.append(os.path.join('..', '..', 'Utilities'))

try:
# Import formatting commands if directory "Utilities" is available
    from ISP_mystyle import showData

except ImportError:
# Ensure correct performance otherwise
    def showData(*options):
        plt.show()
        return


def scatterplot() -> None:
    """Fancy scatterplots, using the package "seaborn" """

    df = sns.load_dataset("iris")
    sns.pairplot(df, hue="species", size=2.5)
    showData('multiScatterplot.jpg')


if __name__ == '__main__':
    scatterplot()
