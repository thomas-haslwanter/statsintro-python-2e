""" Equivalence between T-Test and Statistical Model """

# author: Thomas Haslwanter, date: Sept-2021


# Import standard packages
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as sm

# Generate normally distributed data around 'reference'
np.random.seed(123)
reference = 5
data = reference + np.random.randn(100)

# t-test
(t, pVal) = stats.ttest_1samp(data, reference)
print('The probability that the sample mean is different ' +
     f'than {reference} is {pVal:5.3f}.')

# Equivalent linear model
df = pd.DataFrame({'data': data-reference})
result = sm.ols(formula='data ~ 1', data=df).fit()
print(result.summary())
