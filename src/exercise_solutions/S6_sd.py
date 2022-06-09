""" Solution to Exercise "Sample Standard Deviation" """

# author: Thomas Haslwanter, date: June-2022

import numpy as np

x = np.linspace(1, 10, 10)
std = np.std(x, ddof=1)
print('The standard deviation of the numbers from 1 to 10 ' +
      f'is {std:4.2f}')
