""" Figure for a an example of a T-test for a mean value """

# author: Thomas Haslwanter, date: June-2022

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# additional packages
# Import formatting commands if directory "Utilities" is available
import os
import sys
sys.path.append(os.path.join('..', 'Code_Quantlets', 'Utilities'))
try:
    from ISP_mystyle import setFonts, showData

except ImportError:
# Ensure correct performance otherwise
    def setFonts(*options):
        return
    def showData(*options):
        plt.show()
        return

sns.set_context('notebook')
sns.set_style('ticks')
setFonts()

# Generate the data
np.random.seed(1234)
nd = stats.norm(100, 20)
weights = nd.rvs(10)

# Make the plot
plt.plot(weights, 'o')
plt.axhline(110, ls='-', label='Target')
plt.axhline(np.mean(weights), ls='--', label='Measured mean')
plt.xlim(-0.2, 9.2)
plt.ylim(50, 130)
plt.xlabel('Cookiebag-Nr')
plt.ylabel('Weight [g]')
plt.legend()

outFile = 'fig_ExampleTtest.jpg'
showData(outFile)
