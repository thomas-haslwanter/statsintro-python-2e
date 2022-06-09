""" Display correlated data """

# author: Thomas Haslwanter, date: June-2022

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

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

# additional packages
# Import formatting commands if directory "Utilities" is available
import os
import sys
sys.path.append(os.path.join('..', '..', 'Utilities'))
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
iris = datasets.load_iris()
df = pd.DataFrame(np.column_stack( (iris.data[:,2:], iris.target) ),
        columns = ['petal_length', 'petal_width', 'class'])

# Make the plot
sns.set_context('notebook')
sns.set_style('ticks')
setFonts()

df.plot('petal_length', 'petal_width', kind='scatter')
# Save and show
outFile = 'Correlation.jpg'
showData(outFile)

# To see the plot in color
df.plot('petal_length', 'petal_width', kind='scatter',
        c='class', cmap='viridis')
plt.show()

