""" Demonstration of the hypergometrical distribution """

# Copyright(c) 2021, Thomas Haslwanter.
# All rights reserved, under the CC BY-SA 4.0 International License

# Import the required packages
import numpy as np
from scipy.stats import hypergeom
import matplotlib.pyplot as plt

# Define the problme
[M, n, N] = [20, 7, 12]

# The solution
rv = hypergeom(M, n, N)
x = np.arange(0, n+1)
pmf_dogs = rv.pmf(x)

# Show the results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, pmf_dogs, 'bo')
ax.vlines(x, 0, pmf_dogs, lw=2)
ax.set_xlabel('# of dogs in our group of chosen animals')
ax.set_ylabel('hypergeom PMF')
out_file = 'hypergeometric.jpg'
plt.savefig(out_file, dpi=200)
print(f'Image saved to {out_file}')
plt.show()
