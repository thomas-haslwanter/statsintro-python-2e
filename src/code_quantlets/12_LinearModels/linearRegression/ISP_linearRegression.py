""" Demonstration of linear regression using pingouin """

# author: Thomas Haslwanter, date: Aug-2021

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg

import os
import sys
sys.path.append(os.path.join('..', '..', 'Utilities'))
try:
# Import formatting commands if directory "Utilities" is available
    from ISP_mystyle import setFonts, showData 
    
except ImportError:
# Ensure correct performance otherwise
    def setFonts(*options):
        return
    def showData(*options):
        plt.show()
        return

# Generate some data
np.random.seed(12345)
x = np.random.randn(100)*30
y = np.random.randn(100)*10
z = 3 + 0.4*x + 0.05*y + 10*np.random.randn(len(x))

# Put them into a DataFrame
df = pd.DataFrame({'x':x, 'y':y, 'z':z})

# Show the data
fig, axs = plt.subplots(1,2)
df.plot('x', 'z', kind='scatter', ax=axs[0])
df.plot('y', 'z', kind='scatter', ax=axs[1])
plt.tight_layout()
out_file = 'regression_pg.jpg'
showData(out_file)

# plt.show()

# Simple linear regression
results = pg.linear_regression(df.x, df.z)
print(results.round(2))

# Multiple linear regression
results = pg.linear_regression(df[['x', 'y']], df['z'])
print(results.round(2))
