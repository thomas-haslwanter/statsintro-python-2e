""" Figure showing a linear correlation """

# author: Thomas Haslwanter, date: Dec-2021

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
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


# Get the data
iris = sns.load_dataset('iris')

# Make the plot
setFonts(20)
iris.plot('petal_length', 'petal_width', kind='scatter', figsize=(8,6))

# Show and save it
out_file = 'Correlation.jpg'
showData(out_file)
