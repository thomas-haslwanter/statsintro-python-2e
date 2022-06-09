"""Solution for Exercise "Continuous Distribution Functions" """

# author: Thomas Haslwanter, date: June-2022

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# T-distibution ----------------------------------------------
# Enter the data
x = [52, 70, 65, 85, 62, 83, 59]
""" Note that "x" is a Python "list", not an array!
 Arrays come with the numpy package, and have to contain all
 elements of the same type.
 Lists can mix different types, e.g. "x = [1, 'a', 2]"
 """

# Generate the t-distribution: note that the degrees of freedom
# is the length of the data minus 1.
# In Python, the length of an object x is given by "len(x)"
num = len(x)
dof = num - 1
mean = np.mean(x)
alpha = 0.01

td = stats.t(dof, loc=mean, scale=stats.sem(x))
ci = td.interval(1-alpha)
# This is equivalent to:
# ci = td.ppf([alpha/2, 1-alpha/2])

print(f'mean_weight = {mean:3.1f} kg, 99%CI = [{ci[0]:3.1f}, {ci[1]:3.1f}] kg')

# Chi2-distribution, with 3 DOF -----------------------------
# Define the normal distribution
nd = stats.norm()

# Generate three sets of random variates from this distribution
num_data = 1000
data = np.random.randn(num_data, 3)

# Show histogram of the sum of the squares of these random data
plt.hist(np.sum(data**2, axis=1), bins=100, density=True)

# Superpose it with the exact chi-square distribution
x = np.arange(0, 18, 0.1)
chi2 = stats.chi2(df=3)
pdf = chi2.pdf(x)
plt.plot(x, pdf, lw=3)
plt.xlabel('x')
plt.ylabel('Probability(x)')
plt.show()

# F-distribution ---------------------------------------------
# Enter the data
femurs_1 = [32.0, 32.5, 31.5, 32.1, 31.8]
femurs_2 = [33.2, 33.3, 33.8, 33.5, 34.0]

# Do the calculations
fval = np.var(femurs_1, ddof=1) / np.var(femurs_2, ddof=1)
fd = stats.distributions.f(len(femurs_1),len(femurs_2))
pval = fd.cdf(fval)

# Show the results
print(f'The p-value of the F-distribution = {pval:5.3f}.')
if pval>0.025 and pval<0.975:
    print('The precisions of the two machines are equal.')
else:
    print('The precisions of the two machines are NOT equal.')

# Uniform distribution ---------------------------------------------
ud = stats.uniform(0, 1)
data = ud.rvs(1000)
plt.plot(data, '.')
plt.title('Uniform Distribution')
for ci in [0.95, 0.999]:
    print(f'The {ci*100:.1f}-% confidence interval is {np.float16(ud.interval(ci))}')

