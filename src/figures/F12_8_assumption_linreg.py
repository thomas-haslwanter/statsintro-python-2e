""" Assumptions in linear regression

Show the effect of assuming that the predictors are known exactly, and that all the
variability comes from the dependent variable.
"""

# author:   Thomas Haslwanter
# date:     June-2022

# Import the basic packages
import numpy as np
import matplotlib.pyplot as plt
import os

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


# Generate a simple data set:
x = np.arange(5)
y = [3, 5, 6, 5, 7]

# Fit linear models y(x), and x(y):
fit_x_independent = np.polyfit(x, y, 1)
fit_y_independent = np.polyfit(y, x, 1)

# Plot the data
x_independent = np.linspace(0, 4, 100)
y_dependent = np.polyval(fit_x_independent, x_independent)

y_independent = np.linspace(3, 7, 100)
x_dependent = np.polyval(fit_y_independent, y_independent)

plt.plot(x, y, 'o', label='xy-data')
plt.plot(x_independent, y_dependent, label='y(x)')
plt.plot(x_dependent, y_independent, label='x(y)')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-0.2, 4.2])
plt.ylim([2.8, 7.1])

ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])

# Show the residuals
y_fit = np.polyval(fit_x_independent, x)
x_fit = np.polyval(fit_y_independent, y)
for ii in range(len(x)):
    plt.vlines(x[ii], y[ii], y_fit[ii], ls='dashed', colors='C1')
    plt.hlines(y[ii], x[ii], x_fit[ii], ls='dotted', colors='C2')

outFile = 'assumptions_linear_regression.jpg'
showData(outFile)
