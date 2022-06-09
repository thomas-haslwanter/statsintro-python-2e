""" Scatterplot of normally distributed data, with Standard Deviation
and Standard Error"""

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

from F7_10_roc import arrow_bidir

# Generate the data
np.random.seed(123)
x = np.random.randn(100) + 5

# Calculate "standard devation" and "standard error" locations
sdVal = np.std(x, ddof=1)
seVal = stats.sem(x)
sd = np.mean(x) + sdVal*np.r_[-1, 1]
se = np.mean(x) + seVal*np.r_[-1, 1]

# Set up the plot
sns.set_style('ticks')
sns.set_context('notebook')
sns.color_palette('muted')
setFonts()

# Plot the data
plt.plot(x,'.', color='C0')
plt.axhline(np.mean(x), ls='dotted', color='C2')
plt.axhline(sd[0], ls='--', color='C0')
plt.axhline(sd[1], ls='--', color='C0')

# Lines for SEM
lh = []
lh.append( plt.axhline(se[0], ls='dashed', color='C1') )
lh.append( plt.axhline(se[1], ls='--', color='C1') )

# make the lines long-dashed
dashes = [8, 2] # 20 points on, 5 off
for ii in range(len(lh)):
    lh[ii].set_dashes(dashes)

curAxis = plt.gca()
curAxis.set_xticklabels( () )

# Make the arrows
plt.arrow(10, np.mean(x), 0, sdVal, width=0.2,
        length_includes_head=True, head_length=0.2, head_width=1, color='C0')

plt.arrow(10, np.mean(x), 0, -sdVal, width=0.2,
        length_includes_head=True, head_length=0.2, head_width=1, color='C0')

plt.arrow(35, np.mean(x)-4*seVal, 0, 3*seVal, width=0.2,
        length_includes_head=True, head_length=0.1, head_width=1, color='C1')

plt.arrow(35, np.mean(x)+4*seVal, 0, -3*seVal, width=0.2,
        length_includes_head=True, head_length=0.1, head_width=1, color='C1')

fs = 16
#plt.text(10, 5.5, '$\pm$ 1SD', family='sans-seriv', fontsize=fs)
#plt.text(35, 5.2, '$\pm$ 1SEM', family='sans-seriv', fontsize=fs)
plt.text(10, 5.5, '$\pm$ 1SD', fontsize=fs, color='C0')
plt.text(35, 5.2, '$\pm$ 1SEM', fontsize=fs, color='C1')
plt.annotate('mean', (70,np.mean(x)),xycoords='data', fontsize=fs, color='C2',
                xytext=(75, 5.5), textcoords='data',
                arrowprops=dict(facecolor='C2', shrink=0.05))

# Save and show
outFile = ('standardError.jpg')
showData(outFile)
