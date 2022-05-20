""" Demonstration of linear regression using pingouin """

# author: Thomas Haslwanter, date: Aug-2021

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg

# ... for the 3d plot ...
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

if __name__ == '__main__':
    # Generate some data
    np.random.seed(12345)
    x = np.random.randn(100)*30
    y = np.random.randn(100)*10
    z = 3 + 0.4*x + 0.05*y + 10*np.random.randn(len(x))

    # Put them into a DataFrame
    df = pd.DataFrame({'x':x, 'y':y, 'z':z})

    # Simple linear regression
    results = pg.linear_regression(df.x, df.z)
    print(results.round(2))
    print(f'p = {results.pval.values[1]:4.1e}\n')

    # Multiple linear regression
    results = pg.linear_regression(df[['x', 'y']], df['z'])
    print(results.round(2))

    # Show the data
    fig, axs = plt.subplots(1,2, figsize=(6,3))
    df.plot('x', 'z', kind='scatter', ax=axs[0])
    df.plot('y', 'z', kind='scatter', ax=axs[1])

    p = np.polyfit(df.x, df.z, 1)
    x = np.linspace(df.x.min(), df.x.max(), 100)
    axs[0].plot(x, np.polyval(p, x), color='C1', ls='dashed')

    plt.tight_layout()
    out_file = 'regression_pg.jpg'
    showData(out_file)

    # --------- 3D plot ---------------
    x = np.linspace(df.x.min(),df.x.max(),101)
    y = np.linspace(df.y.min(),df.y.max(),101)
    (X,Y) = np.meshgrid(x,y)
    Z = results.coef[0] + results.coef[1]*X + results.coef[2]*Y
    # Set the color
    myCmap = cm.GnBu_r

    # If you want a colormap from seaborn use:
    #from matplotlib.colors import ListedColormap
    #myCmap = ListedColormap(sns.color_palette("Blues", 20))

    # Plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X,Y,Z, cmap=myCmap, rstride=2, cstride=2,
        linewidth=0, antialiased=False, alpha=0.3)
    ax.view_init(20,-120)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf, shrink=0.6)
    ax.plot(df.x, df.y, df.z, 'o')

    out_file = 'regression_3d.jpg'
    showData(out_file)

    # plt.show()

