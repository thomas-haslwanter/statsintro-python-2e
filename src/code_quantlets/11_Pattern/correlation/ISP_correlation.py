""" Calculation and visualization of correlation matrix """

# author: thomas haslwantere; date: march-2021

# Import the required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

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
    
# The "iris" dataset is one of the most common examples in pattern recognition
df = sns.load_dataset('iris')

# Pearson Correlation
r = stats.pearsonr(df.petal_length, df.petal_width)
print(f'r^2 = {r[0]**2:5.3f}')

plt.plot(df.petal_length, df.petal_width, '.')
plt.title(f'Iris data: r={r[0]:5.3f}')
plt.xlabel('Petal Length [cm]')
plt.ylabel('Petal Width [cm]')
showData('correlation.jpg')

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
