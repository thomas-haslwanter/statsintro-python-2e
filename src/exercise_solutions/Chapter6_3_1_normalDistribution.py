"""Solutions to 6.3.1 Examples of Normal Distributions"""

# author: Thomas Haslwanter, date: June-2022

# Import standard packages
import numpy as np
from scipy import stats

# Example 1: A man with "183 cm" height
(avg_size, std) = (175, 6)
nd = stats.norm(avg_size, std)
p = nd.cdf(184) - nd.cdf(183)
print('The chance that a randomly selected man is 183 cm ' +
       f'tall is {p*100:.1f}%')

# Example 2: Cans with a weight of at least 250g
std = 4
nd = stats.norm()
below = nd.ppf(0.01)    # 1% can be below 250 g
production_weight = 250 - std*below
print(f'The production weight for the cans is {production_weight:5.1f}g.')

# Example 3
(avg_man, std_man) = (175, 6)
(avg_woman, std_woman) = (168, 3)
std_diff = np.sqrt(std_man**2 + std_woman**2)
nd = stats.norm(avg_man - avg_woman, std_diff)
p = nd.cdf(0)
print('The chance that a man is smaller than his female partner ' + \
        f'is {p*100:.1f}%')
