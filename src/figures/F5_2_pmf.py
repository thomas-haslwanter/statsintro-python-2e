""" Probability Mass Function (PMF), for throwing dice """

# author:   Thomas Haslwanter
# date:     Dec-2021

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import os
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

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

pmf = np.zeros(10)
pmf[1:7] = 1/6

plt.plot(4, 1/6, 'o', ms=14, color='C1', alpha=0.5)
plt.plot(pmf, 'o')

# Annotate it
xi, yi = 4, 1/6
xy_text = (5, 0.35)
plt.annotate('PMF(4)',
             xy = (xi, yi),
             xytext = xy_text,
             arrowprops=dict(facecolor='black', shrink=0.05) )

plt.axhline(y=0, ls='dotted')
plt.axvline(x=0, ls='dotted')

plt.xlabel('n')
plt.ylabel('PMF(n)')
plt.ylim(-0.02, 0.4)
plt.tight_layout()

out_file = 'pmf.jpg'
showData(out_file)
