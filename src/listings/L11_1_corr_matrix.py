""" Plotting a diagonal correlation matrix

With permission from Michael Waskom, from
http://seaborn.pydata.org/examples/many_pairwise_correlations.html
"""

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Generate a large random dataset
# The syntax here is sligthly different from the previously
# used np.random.seed. For details, see
# https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array
rs = np.random.RandomState(1234)
df = pd.DataFrame(data=rs.normal(size=(100, 26)),
                  columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom colormap
cmap = sns.color_palette("viridis", as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .5})
out_file = 'many_pairwise_correlations.jpg'
plt.savefig(out_file, dpi=300)
print(f'Correlation-matrix saved to {out_file}')

plt.show()
