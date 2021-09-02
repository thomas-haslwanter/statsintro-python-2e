""" Figure explaining the power of a test """

# author: Thomas Haslwanter, date: Sept-2021

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
    
setFonts(16)
# Plot a normal distribution, and mark tc
x = np.linspace(-4, 4, 200)
nd = stats.norm()
y = nd.pdf(x)
plt.plot(x,y, ls='dotted')

# Plot the second distribution ...
x2 = x+0.3
plt.plot(x2,y)
plt.axhline(ls='dashed')
plt.axvline(x=0, ymin=0, ymax=1, ls='dashed')

# ... including the "true positive" areas
alpha=0.05
tc = nd.isf(alpha/2)
large = x2>tc
small = x2<-tc
plt.fill_between(x2[large], y[large], color='C1')
plt.fill_between(x2[small], y[small], color='C1')

# Format the plot
plt.plot([tc, tc], [0, nd.pdf(tc)], color='C0')
plt.plot([-tc, -tc], [0, nd.pdf(tc)], color='C0')

xlabels = ['ref-tc', 'ref', 'ref+tc']
xticks = [-tc, 0, tc]
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
ax.set_yticks(np.linspace(0, 0.4, 5))
plt.ylim(0, 0.42)

out_file = 'show_power.jpg'
#plt.show()

showData(out_file)
